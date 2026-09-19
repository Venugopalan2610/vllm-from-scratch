"""Reference solution, stage 15 - async engine + OpenAI-compatible API."""

import asyncio
import json
import time
import uuid


class RequestHandle:
    def __init__(self, rid):
        self.rid = rid
        self.queue = asyncio.Queue()
        self.aborted = False


class AsyncLLMEngine:
    """Runs a synchronous step engine from a background task.

    The step engine must provide:
        add_request(rid, prompt, max_tokens)
        step() -> list[(rid, delta_text, finished)]
        has_work() -> bool
        abort(rid)
    """

    def __init__(self, step_engine, idle_sleep=0.001):
        self.engine = step_engine
        self.idle_sleep = idle_sleep
        self._handles = {}
        self._task = None
        self._running = False

    @property
    def num_active(self):
        return len(self._handles)

    async def start(self):
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _deliver(self, outputs):
        for rid, delta, finished in outputs:
            handle = self._handles.get(rid)
            if handle is None:
                continue
            await handle.queue.put((delta, finished))
            if finished:
                self._handles.pop(rid, None)

    async def _loop(self):
        """One loop for the whole server. It never waits on HTTP."""
        while self._running:
            if not self.engine.has_work():
                await asyncio.sleep(self.idle_sleep)
                continue
            await self._deliver(self.engine.step())
            await asyncio.sleep(0)      # let the event loop run

    async def generate(self, prompt, max_tokens, rid=None):
        rid = rid or str(uuid.uuid4())
        handle = RequestHandle(rid)
        self._handles[rid] = handle
        self.engine.add_request(rid, prompt, max_tokens)
        try:
            while True:
                delta, finished = await handle.queue.get()
                if delta:
                    yield delta
                if finished:
                    return
        finally:
            if rid in self._handles:
                # The client left during the stream: free its KV.
                await self.abort(rid)

    async def abort(self, rid):
        self._handles.pop(rid, None)
        self.engine.abort(rid)


class CompletionIds:
    """The fields that every part of one completion shares."""

    def __init__(self, model_name):
        self.id = f"cmpl-{uuid.uuid4().hex[:12]}"
        self.created = int(time.time())
        self.model_name = model_name

    def body(self, object_type, text, finish_reason):
        return {"id": self.id, "object": object_type, "created": self.created,
                "model": self.model_name,
                "choices": [{"index": 0, "text": text,
                             "finish_reason": finish_reason}]}

    def event(self, text, finish_reason):
        chunk = self.body("text_completion.chunk", text, finish_reason)
        return f"data: {json.dumps(chunk)}\n\n"


async def whole_completion(engine, ids, prompt, max_tokens):
    text = ""
    async for delta in engine.generate(prompt, max_tokens):
        text += delta
    return ids.body("text_completion", text, "stop")


async def completion_events(engine, ids, prompt, max_tokens):
    async for delta in engine.generate(prompt, max_tokens):
        yield ids.event(delta, None)
    yield ids.event("", "stop")
    yield "data: [DONE]\n\n"


def create_app(engine, model_name="vllm-from-scratch"):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    @asynccontextmanager
    async def lifespan(app):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 16))
        ids = CompletionIds(model_name)
        if body.get("stream", False):
            return StreamingResponse(
                completion_events(engine, ids, prompt, max_tokens),
                media_type="text/event-stream")
        return JSONResponse(await whole_completion(engine, ids, prompt,
                                                   max_tokens))

    return app
