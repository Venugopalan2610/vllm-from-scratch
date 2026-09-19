"""Stage 27 - the server, for real.

`./vc lore 27` for the insight. `./vc test 27` to check yourself.

Stage 15 built the server around a fake engine, with two request fields:
prompt and max_tokens. Here the server runs your stage 26 engine, which has
every part that you built, and it speaks the API that real clients send.

A real step holds the GPU for 5 to 50 ms, and your stage 15 loop calls it
inside the event loop. For that time no HTTP request moves. So step() moves
to a worker thread.

WHAT YOU ARE BUILDING

    EngineAdapter(llm_engine)
        .add_request(rid, prompt, max_tokens, **opts)  from the event loop
        .abort(rid)                                    from the event loop
        .has_work()
        .step()                                        from a worker thread
    RequestHandle(rid)                                given: one queue
    ServingEngine(adapter)
        .start(), .stop()                             given
        ._loop()      await asyncio.to_thread(self.adapter.step)
        .generate(prompt, max_tokens, rid=None, **options)  an async generator
        .result(rid) -> (finish_reason, prompt_tokens, completion_tokens)

ServingEngine does the job of your stage 15 AsyncLLMEngine, and it does not
subclass it. A subclass would depend on the private names inside your stage
15 file. Depend only on the contract: add_request, abort, has_work, step.
    request_options(body) -> dict of keyword arguments for add_request
    chat_prompt(tokenizer, messages) -> the prompt text
    choice, response, usage          one small function for each JSON part
    Completions(serving, model_name, is_chat)
        .whole(prompt, body) -> the response
        .stream(prompt, body) -> server-sent events, then [DONE]
    prometheus_text(snapshot) -> str
    build_app(llm_engine, metrics, model_name) -> FastAPI app, with
        GET  /health, /v1/models, /metrics
        POST /v1/completions, /v1/chat/completions
        and the ServingEngine in app.state.serving
    async run_load(client, prompts, rate_per_s, max_tokens) -> seconds

THE REQUEST FIELDS

    temperature, top_p, top_k, seed, repetition_penalty -> SamplingParams
    stop            a string or a list of strings
    stream          server-sent events, one delta for each chunk, then [DONE]
    response_format {"type": "json_object"} -> json_mode=True (stage 26).
                    Pass json_mode only when the request asks for it, so that
                    an engine without stage 26 still serves every other call.

A response carries "usage" (prompt, completion and total tokens) and the real
"finish_reason": "stop" or "length". A chat response puts the text in
choices[0].message.content, and a chat stream puts it in
choices[0].delta.content.

A chat request holds messages. tokenizer.apply_chat_template(messages,
tokenize=False, add_generation_prompt=True) turns them into the prompt. For
Qwen3, also pass enable_thinking=False, or the model writes its reasoning
into the answer.

TWO THREADS, ONE ENGINE

The event loop calls add_request and abort. The worker thread runs step. If
both touch the queues of the engine, one of them sees a list in the middle of
a change. Do not add a lock: the event loop then waits for the lock, which is
the stall that you started to remove. So add_request and abort only append a
command to a collections.deque, which is atomic. step() applies the commands
first, and then it steps.

TRAPS

  - Stamp the arrival time in add_request, and pass it to the engine as
    arrival=. The engine sees the request up to one step later. TTFT must
    include that wait, or your metrics flatter you.
  - A disconnect in stage 15 calls abort. Now abort must free REAL blocks.
  - Prometheus text is one "name value" line for each number, with a
    "# TYPE name gauge" or "# TYPE name counter" line before it.
"""

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
        raise NotImplementedError("stage 27: implement EngineAdapter.add_request")

    def abort(self, rid):
        raise NotImplementedError("stage 27: implement EngineAdapter.abort")

    def has_work(self):
        raise NotImplementedError("stage 27: implement EngineAdapter.has_work")

    def step(self):
        raise NotImplementedError("stage 27: implement EngineAdapter.step")


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
        raise NotImplementedError("stage 27: implement ServingEngine._deliver")

    async def _loop(self):
        raise NotImplementedError("stage 27: implement ServingEngine._loop")

    async def generate(self, prompt, max_tokens, rid=None, **options):
        raise NotImplementedError("stage 27: implement ServingEngine.generate")
        yield

    def result(self, rid):
        """-> (finish_reason, prompt_tokens, completion_tokens)."""
        raise NotImplementedError("stage 27: implement ServingEngine.result")


# ---------------------------------------------------------------- the API


def request_options(body):
    """The OpenAI request fields that the engine understands."""
    raise NotImplementedError("stage 27: implement request_options")


def chat_prompt(tokenizer, messages):
    raise NotImplementedError("stage 27: implement chat_prompt")


def choice(text, finish_reason, is_chat, is_delta):
    raise NotImplementedError("stage 27: implement choice")


def response(rid, model_name, is_chat, is_chunk, choices, token_usage=None):
    raise NotImplementedError("stage 27: implement response")


def usage(prompt_tokens, completion_tokens):
    raise NotImplementedError("stage 27: implement usage")


def prometheus_text(snapshot, prefix="vllm"):
    """A metrics snapshot in the Prometheus text format."""
    raise NotImplementedError("stage 27: implement prometheus_text")


class Completions:
    """One completion request, whole or streamed."""

    def __init__(self, serving, model_name, is_chat):
        self.serving = serving
        self.model_name = model_name
        self.is_chat = is_chat

    def _generate(self, rid, prompt, body):
        raise NotImplementedError("stage 27: implement Completions._generate")

    async def whole(self, prompt, body):
        raise NotImplementedError("stage 27: implement Completions.whole")

    def _event(self, rid, text, finish_reason):
        raise NotImplementedError("stage 27: implement Completions._event")

    async def stream(self, prompt, body):
        raise NotImplementedError("stage 27: implement Completions.stream")
        yield


def build_app(llm_engine, metrics, model_name="vllm-from-scratch"):
    raise NotImplementedError("stage 27: implement build_app")


async def run_load(client, prompts, rate_per_s, max_tokens, seed=0):
    """Send prompts at Poisson arrival times. -> wall-clock seconds."""
    raise NotImplementedError("stage 27: implement run_load")
