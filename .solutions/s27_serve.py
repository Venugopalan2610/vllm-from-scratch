"""Reference solution, stage 27 - the server, for real."""

import asyncio
import collections
import json
import math
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import struct
import time
import uuid
from contextlib import asynccontextmanager

from app.s13_sampler import SamplingParams
from app.s16_metrics import poisson_arrivals

COUNTERS = {"num_requests", "num_finished", "total_tokens", "preemptions"}

MAX_PROMPT_TOKENS = 4096
MAX_MAX_TOKENS = 4096
MAX_CONCURRENT_REQUESTS = 256


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


class SharedMemoryEventRing:
    """A ring of fixed binary records in POSIX shared memory, for the output
    events of the engine process. It removes the pickle and the pipe of
    mp.Queue. vLLM V1 also runs its engine in its own process; at the time of
    writing it talks to it over ZeroMQ sockets with a compact encoding."""
    SLOT_STRUCT = struct.Struct("32sii64s")
    HEADER_STRUCT = struct.Struct("II")  # head, tail

    def __init__(self, capacity=1024, name=None, create=True):
        self.capacity = capacity
        self.slot_size = self.SLOT_STRUCT.size
        self.header_size = self.HEADER_STRUCT.size
        self.total_size = self.header_size + self.capacity * self.slot_size
        if create:
            self.shm = SharedMemory(create=True, size=self.total_size)
            self.name = self.shm.name
            self.HEADER_STRUCT.pack_into(self.shm.buf, 0, 0, 0)
        else:
            self.name = name
            self.shm = SharedMemory(name=name, create=False)
        self.lock = mp.Lock()

    def put(self, rid: str, text: str, finished: bool):
        rid_bytes = rid.encode("ascii", errors="ignore")[:32].ljust(32, b"\x00")
        text_bytes = text.encode("utf-8", errors="ignore")[:64]
        finish_code = 1 if finished else 0
        with self.lock:
            head, tail = self.HEADER_STRUCT.unpack_from(self.shm.buf, 0)
            if (head + 1) % self.capacity == tail:
                return False  # ring buffer full
            offset = self.header_size + (head % self.capacity) * self.slot_size
            self.SLOT_STRUCT.pack_into(
                self.shm.buf, offset,
                rid_bytes, finish_code, len(text_bytes), text_bytes.ljust(64, b"\x00")
            )
            self.HEADER_STRUCT.pack_into(self.shm.buf, 0, (head + 1) % self.capacity, tail)
            return True

    def get_all(self):
        events = []
        with self.lock:
            head, tail = self.HEADER_STRUCT.unpack_from(self.shm.buf, 0)
            while tail != head:
                offset = self.header_size + (tail % self.capacity) * self.slot_size
                rid_bytes, finish_code, text_len, text_bytes = self.SLOT_STRUCT.unpack_from(self.shm.buf, offset)
                rid = rid_bytes.rstrip(b"\x00").decode("ascii", errors="ignore")
                text = text_bytes[:text_len].decode("utf-8", errors="ignore")
                finished = bool(finish_code)
                events.append((rid, text, finished))
                tail = (tail + 1) % self.capacity
            self.HEADER_STRUCT.pack_into(self.shm.buf, 0, head, tail)
        return events

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self):
        try:
            self.shm.unlink()
        except Exception:
            pass


class ProcessEngineWorker:
    """Run the engine loop in a dedicated child OS process (vLLM V1 architecture).
    This separates the GPU step loop from web serving, detokenization, and Python GC."""

    def __init__(self, cmd_queue, resp_queue, engine, shm_ring=None):
        self.cmd_queue = cmd_queue
        self.resp_queue = resp_queue
        self.engine = engine
        self.shm_ring = shm_ring

    def run_loop(self):
        while True:
            while not self.cmd_queue.empty():
                try:
                    cmd, args, kwargs = self.cmd_queue.get_nowait()
                    if cmd == "stop":
                        return
                    elif cmd == "add_request":
                        self.engine.add_request(*args, **kwargs)
                    elif cmd == "abort":
                        self.engine.abort(*args, **kwargs)
                except Exception:
                    break
            if self.engine.has_work():
                events = self.engine.step()
                if events:
                    if self.shm_ring is not None:
                        for rid, text, finished in events:
                            self.shm_ring.put(rid, text, finished)
                    elif self.resp_queue is not None:
                        self.resp_queue.put(events)
            else:
                time.sleep(0.001)


class ProcessEngineAdapter:
    """Frontend adapter sending commands across multiprocessing queues to the engine process."""

    def __init__(self, engine, use_shm=False):
        self.engine = engine
        self.use_shm = use_shm
        self.cmd_queue = mp.Queue()
        self.resp_queue = mp.Queue() if not use_shm else None
        self.shm_ring = SharedMemoryEventRing(create=True) if use_shm else None
        self.worker = ProcessEngineWorker(self.cmd_queue, self.resp_queue, engine,
                                         shm_ring=self.shm_ring)
        self.process = None

    def start(self):
        self.process = mp.Process(target=self.worker.run_loop, daemon=True)
        self.process.start()

    def stop(self):
        if self.process and self.process.is_alive():
            self.cmd_queue.put(("stop", (), {}))
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
        if self.shm_ring is not None:
            self.shm_ring.close()
            self.shm_ring.unlink()

    def add_request(self, rid, prompt, max_tokens, **options):
        arrival = self.engine.clock()
        self.cmd_queue.put(("add_request", (rid, prompt, max_tokens),
                            dict(arrival=arrival, **options)))

    def abort(self, rid):
        self.cmd_queue.put(("abort", (rid,), {}))

    def has_work(self):
        return self.engine.has_work()

    def step(self):
        if self.shm_ring is not None:
            return self.shm_ring.get_all()
        events = []
        while not self.resp_queue.empty():
            try:
                events.extend(self.resp_queue.get_nowait())
            except Exception:
                break
        return events


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

    def abort(self, rid):
        self.handles.pop(rid, None)
        self.adapter.abort(rid)

    def result(self, rid):
        """-> (finish_reason, prompt_tokens, completion_tokens)."""
        seq = self.adapter.engine.seqs.get(rid)
        if seq is None:
            return "stop", 0, 0
        return (seq.finish_reason or "stop", len(seq.prompt_ids),
                len(seq.output_ids))

    def logprobs(self, rid, tokenizer=None):
        seq = self.adapter.engine.seqs.get(rid)
        if seq is None or not getattr(seq, "output_logprobs", None):
            return None
        tokens = [tokenizer.decode([t]) if tokenizer else str(t) for t, _, _ in seq.output_logprobs]
        token_logprobs = [lp for _, lp, _ in seq.output_logprobs]
        top_logprobs = [{tokenizer.decode([k]) if tokenizer else str(k): v for k, v in top.items()}
                        for _, _, top in seq.output_logprobs]
        return {"tokens": tokens, "token_logprobs": token_logprobs, "top_logprobs": top_logprobs}


# ---------------------------------------------------------------- the API


def request_options(body):
    """The OpenAI request fields that the engine understands."""
    logprobs_k = body.get("top_logprobs") or (5 if body.get("logprobs") is True else body.get("logprobs"))
    params = SamplingParams(
        temperature=float(body.get("temperature", 1.0)),
        top_k=int(body.get("top_k") or 0),
        top_p=float(body.get("top_p", 1.0)),
        min_p=float(body.get("min_p") or 0.0),
        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
        seed=body.get("seed"),
        logprobs=int(logprobs_k) if logprobs_k else None)
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


def choice(text, finish_reason, is_chat, is_delta, logprobs=None):
    if not is_chat:
        c = {"index": 0, "text": text, "finish_reason": finish_reason}
        if logprobs is not None:
            c["logprobs"] = logprobs
        return c
    key = "delta" if is_delta else "message"
    c = {"index": 0, key: {"role": "assistant", "content": text},
         "finish_reason": finish_reason}
    if logprobs is not None:
        c["logprobs"] = logprobs
    return c


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
        logprobs = self.serving.logprobs(rid, getattr(self.serving.adapter.engine, "tokenizer", None))
        return response(rid, self.model_name, self.is_chat, False,
                        [choice(text, finish_reason, self.is_chat, False, logprobs)],
                        usage(prompt_tokens, completion_tokens))

    def _event(self, rid, text, finish_reason):
        chunk = response(rid, self.model_name, self.is_chat, True,
                         [choice(text, finish_reason, self.is_chat, True)])
        return f"data: {json.dumps(chunk)}\n\n"

    async def stream(self, prompt, body, request=None):
        rid = uuid.uuid4().hex
        try:
            async for part in self._generate(rid, prompt, body):
                if request is not None and await request.is_disconnected():
                    self.serving.abort(rid)
                    return
                yield self._event(rid, part, None)
            yield self._event(rid, "", self.serving.result(rid)[0])
            yield "data: [DONE]\n\n"
        finally:
            self.serving.abort(rid)


def build_app(llm_engine, metrics, model_name="vllm-from-scratch", multiprocess=False):
    from fastapi import FastAPI, Request
    from fastapi.responses import (JSONResponse, PlainTextResponse,
                                   StreamingResponse)

    adapter = ProcessEngineAdapter(llm_engine) if multiprocess else EngineAdapter(llm_engine)
    serving = ServingEngine(adapter)
    tokenizer = llm_engine.tokenizer

    @asynccontextmanager
    async def lifespan(app):
        if hasattr(adapter, "start"):
            adapter.start()
        await serving.start()
        yield
        await serving.stop()
        if hasattr(adapter, "stop"):
            adapter.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.serving = serving

    async def complete(prompt, body, is_chat, request=None):
        handler = Completions(serving, model_name, is_chat)
        if body.get("stream", False):
            return StreamingResponse(handler.stream(prompt, body, request=request),
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
        max_tokens = int(body.get("max_tokens", 16))
        if max_tokens > MAX_MAX_TOKENS:
            return JSONResponse(
                {"error": {"message": f"max_tokens {max_tokens} exceeds "
                           f"limit {MAX_MAX_TOKENS}", "type": "invalid_request_error"}},
                status_code=400)
        prompt = body.get("prompt", "")
        if len(tokenizer.encode(prompt)) > MAX_PROMPT_TOKENS:
            return JSONResponse(
                {"error": {"message": f"prompt exceeds limit {MAX_PROMPT_TOKENS}",
                           "type": "invalid_request_error"}},
                status_code=400)
        return await complete(prompt, body, is_chat=False, request=request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        max_tokens = int(body.get("max_tokens", 16))
        if max_tokens > MAX_MAX_TOKENS:
            return JSONResponse(
                {"error": {"message": f"max_tokens {max_tokens} exceeds "
                           f"limit {MAX_MAX_TOKENS}", "type": "invalid_request_error"}},
                status_code=400)
        prompt = chat_prompt(tokenizer, body.get("messages", []))
        if len(tokenizer.encode(prompt)) > MAX_PROMPT_TOKENS:
            return JSONResponse(
                {"error": {"message": f"prompt exceeds limit {MAX_PROMPT_TOKENS}",
                           "type": "invalid_request_error"}},
                status_code=400)
        return await complete(prompt, body, is_chat=True, request=request)

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
