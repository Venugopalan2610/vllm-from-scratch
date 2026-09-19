"""Reference solution, stage 27 - the server, for real."""

import asyncio
import collections
import json
import math
import time
import uuid
from contextlib import asynccontextmanager

from app.s13_sampler import SamplingParams
from app.s16_metrics import poisson_arrivals

COUNTERS = {"num_requests", "num_finished", "total_tokens", "preemptions"}


# ---------------------------------------------------------------- engine


class EngineAdapter:
    """The engine behind a thread. The event loop calls add_request and abort.
    They only append a command to a deque, which is atomic. The worker thread
    calls step(), which applies the commands first."""

    def __init__(self, engine):
        self.engine = engine
        self.commands = collections.deque()

    def add_request(self, rid, prompt, max_tokens, **options):
        # Stamp the arrival now. The engine sees the request one step later,
        # and TTFT must include that wait.
        arrival = self.engine.clock()
        self.commands.append(lambda: self.engine.add_request(
            rid, prompt, max_tokens, arrival=arrival, **options))

    def abort(self, rid):
        self.commands.append(lambda: self.engine.abort(rid))

    def has_work(self):
        return bool(self.commands) or self.engine.has_work()

    def step(self):
        while self.commands:
            self.commands.popleft()()
        return self.engine.step() if self.engine.has_work() else []


class RequestHandle:
    """The queue that carries the text of one request to its client."""

    def __init__(self, rid):
        self.rid = rid
        self.queue = asyncio.Queue()


class ServingEngine:
    """One loop drives the engine. step() runs in a worker thread, so the
    event loop stays free while the GPU works."""

    def __init__(self, adapter, idle_sleep=0.001):
        self.adapter = adapter
        self.idle_sleep = idle_sleep
        self.handles = {}
        self.running = False
        self.task = None

    async def start(self):
        if self.task is None:
            self.running = True
            self.task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _deliver(self, events):
        for rid, text, finished in events:
            handle = self.handles.get(rid)
            if handle is None:
                continue
            await handle.queue.put((text, finished))
            if finished:
                self.handles.pop(rid, None)

    async def _loop(self):
        while self.running:
            if not self.adapter.has_work():
                await asyncio.sleep(self.idle_sleep)
                continue
            await self._deliver(await asyncio.to_thread(self.adapter.step))

    async def generate(self, prompt, max_tokens, rid=None, **options):
        rid = rid or uuid.uuid4().hex
        handle = self.handles[rid] = RequestHandle(rid)
        self.adapter.add_request(rid, prompt, max_tokens, **options)
        try:
            while True:
                text, finished = await handle.queue.get()
                if text:
                    yield text
                if finished:
                    return
        finally:
            if self.handles.pop(rid, None) is not None:
                self.adapter.abort(rid)           # the client left early

    def result(self, rid):
        """-> (finish_reason, prompt_tokens, completion_tokens)."""
        seq = self.adapter.engine.seqs.get(rid)
        if seq is None:
            return "stop", 0, 0
        return (seq.finish_reason or "stop", len(seq.prompt_ids),
                len(seq.output_ids))


# ---------------------------------------------------------------- the API


def request_options(body):
    """The OpenAI request fields that the engine understands."""
    params = SamplingParams(
        temperature=float(body.get("temperature", 1.0)),
        top_k=int(body.get("top_k") or 0),
        top_p=float(body.get("top_p", 1.0)),
        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
        seed=body.get("seed"))
    stop = body.get("stop") or ()
    options = {"params": params,
               "stop": (stop,) if isinstance(stop, str) else tuple(stop)}
    # Only a stage 26 engine knows json_mode. Pass it only when asked.
    if (body.get("response_format") or {}).get("type") == "json_object":
        options["json_mode"] = True
    return options


def chat_prompt(tokenizer, messages):
    extra = {}
    if "qwen3" in getattr(tokenizer, "name_or_path", "").lower():
        extra["enable_thinking"] = False        # answer, do not think aloud
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, **extra)


def choice(text, finish_reason, is_chat, is_delta):
    if not is_chat:
        return {"index": 0, "text": text, "finish_reason": finish_reason}
    key = "delta" if is_delta else "message"
    return {"index": 0, key: {"role": "assistant", "content": text},
            "finish_reason": finish_reason}


def response(rid, model_name, is_chat, is_chunk, choices, token_usage=None):
    kind = "chat.completion" if is_chat else "text_completion"
    body = {"id": f"cmpl-{rid[:12]}", "object": kind + (".chunk" if is_chunk else ""),
            "created": int(time.time()), "model": model_name, "choices": choices}
    if token_usage is not None:
        body["usage"] = token_usage
    return body


def usage(prompt_tokens, completion_tokens):
    return {"prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def prometheus_text(snapshot, prefix="vllm"):
    """A metrics snapshot in the Prometheus text format."""
    lines = []
    for key, value in sorted(snapshot.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        name = f"{prefix}_{key}"
        lines.append(f"# TYPE {name} {'counter' if key in COUNTERS else 'gauge'}")
        lines.append(f"{name} {'NaN' if math.isnan(value) else value}")
    return "\n".join(lines) + "\n"


class Completions:
    """One completion request, whole or streamed."""

    def __init__(self, serving, model_name, is_chat):
        self.serving = serving
        self.model_name = model_name
        self.is_chat = is_chat

    def _generate(self, rid, prompt, body):
        return self.serving.generate(prompt, int(body.get("max_tokens", 16)),
                                     rid, **request_options(body))

    async def whole(self, prompt, body):
        rid = uuid.uuid4().hex
        text = "".join([part async for part in self._generate(rid, prompt, body)])
        finish_reason, prompt_tokens, completion_tokens = self.serving.result(rid)
        return response(rid, self.model_name, self.is_chat, False,
                        [choice(text, finish_reason, self.is_chat, False)],
                        usage(prompt_tokens, completion_tokens))

    def _event(self, rid, text, finish_reason):
        chunk = response(rid, self.model_name, self.is_chat, True,
                         [choice(text, finish_reason, self.is_chat, True)])
        return f"data: {json.dumps(chunk)}\n\n"

    async def stream(self, prompt, body):
        rid = uuid.uuid4().hex
        async for part in self._generate(rid, prompt, body):
            yield self._event(rid, part, None)
        yield self._event(rid, "", self.serving.result(rid)[0])
        yield "data: [DONE]\n\n"


def build_app(llm_engine, metrics, model_name="vllm-from-scratch"):
    from fastapi import FastAPI, Request
    from fastapi.responses import (JSONResponse, PlainTextResponse,
                                   StreamingResponse)

    serving = ServingEngine(EngineAdapter(llm_engine))
    tokenizer = llm_engine.tokenizer

    @asynccontextmanager
    async def lifespan(app):
        await serving.start()
        yield
        await serving.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.serving = serving

    async def complete(prompt, body, is_chat):
        handler = Completions(serving, model_name, is_chat)
        if body.get("stream", False):
            return StreamingResponse(handler.stream(prompt, body),
                                     media_type="text/event-stream")
        return JSONResponse(await handler.whole(prompt, body))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.get("/metrics")
    async def metrics_text():
        return PlainTextResponse(prometheus_text(metrics.snapshot()))

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        return await complete(body.get("prompt", ""), body, is_chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        prompt = chat_prompt(tokenizer, body.get("messages", []))
        return await complete(prompt, body, is_chat=True)

    return app


async def run_load(client, prompts, rate_per_s, max_tokens, seed=0):
    """Send prompts at Poisson arrival times. -> wall-clock seconds."""
    arrivals = poisson_arrivals(rate_per_s, len(prompts), seed=seed)
    start = time.perf_counter()

    async def send(arrival, prompt):
        delay = arrival - (time.perf_counter() - start)
        if delay > 0:
            await asyncio.sleep(delay)
        reply = await client.post("/v1/completions",
                                  json={"prompt": prompt, "max_tokens": max_tokens})
        reply.raise_for_status()

    await asyncio.gather(*[send(a, p) for a, p in zip(arrivals, prompts)])
    return time.perf_counter() - start
