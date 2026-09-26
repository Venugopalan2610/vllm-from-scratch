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

API GAPS

    This server does NOT support: logprobs, n > 1 (multiple completions),
    min_p, bad_words, beam search, or tool calls. The most interesting
    missing feature is n > 1: it forces a fork of the sequence and
    copy-on-write, which stage 09 builds but the engine never uses. That
    is a gap in the course, not only in the API.

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

WHAT THIS DOES NOT DO

    vLLM V1 moved the engine into its own PROCESS because Python work
    (detokenization, HTTP handling, output processing) was stealing the
    GPU between steps. That is the largest engineering lesson of the last
    two years. This stage uses asyncio.to_thread, which removes the GIL
    stall but not the process boundary. The engine and the HTTP handler
    share one Python heap. A separate engine process with a queue is the
    next step, and it is a project of its own.

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
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import struct
import time
import uuid
from contextlib import asynccontextmanager

from app.s13_sampler import SamplingParams
from app.s16_metrics import poisson_arrivals

COUNTERS = {"num_requests", "num_finished", "total_tokens", "preemptions"}


# ---------------------------------------------------------------- limits
# Without these, one client can fill the whole block pool or exhaust memory
# with a single request. Even for teaching code, that is the first thing a
# reviewer asks about a server.

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
        raise NotImplementedError("stage 27: implement EngineAdapter.add_request")

    def abort(self, rid):
        raise NotImplementedError("stage 27: implement EngineAdapter.abort")

    def has_work(self):
        raise NotImplementedError("stage 27: implement EngineAdapter.has_work")

    def step(self):
        raise NotImplementedError("stage 27: implement EngineAdapter.step")


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
        raise NotImplementedError("stage 27: implement ServingEngine._deliver")

    async def _loop(self):
        raise NotImplementedError("stage 27: implement ServingEngine._loop")

    async def generate(self, prompt, max_tokens, rid=None, **options):
        raise NotImplementedError("stage 27: implement ServingEngine.generate")
        yield

    def abort(self, rid):
        raise NotImplementedError("stage 27: implement ServingEngine.abort")

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

    async def stream(self, prompt, body, request=None):
        raise NotImplementedError("stage 27: implement Completions.stream")
        yield


def build_app(llm_engine, metrics, model_name="vllm-from-scratch", multiprocess=False):
    raise NotImplementedError("stage 27: implement build_app")


async def run_load(client, prompts, rate_per_s, max_tokens, seed=0):
    """Send prompts at Poisson arrival times. -> wall-clock seconds."""
    raise NotImplementedError("stage 27: implement run_load")
