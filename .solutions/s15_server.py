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
    """Drives a synchronous step-based engine from a background task.

    The injected `step_engine` must provide:
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
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self):
        """One loop for the whole server. Never blocks on HTTP."""
        while self._running:
            if not self.engine.has_work():
                await asyncio.sleep(self.idle_sleep)
                continue
            for rid, delta, finished in self.engine.step():
                h = self._handles.get(rid)
                if h is None:
                    continue
                await h.queue.put((delta, finished))
                if finished:
                    self._handles.pop(rid, None)
            await asyncio.sleep(0)      # yield to the event loop

    async def generate(self, prompt, max_tokens, rid=None):
        rid = rid or str(uuid.uuid4())
        h = RequestHandle(rid)
        self._handles[rid] = h
        self.engine.add_request(rid, prompt, max_tokens)
        try:
            while True:
                delta, finished = await h.queue.get()
                if delta:
                    yield delta
                if finished:
                    return
        finally:
            if rid in self._handles:
                # client went away mid-stream: tell the engine to release KV
                self._handles.pop(rid, None)
                self.engine.abort(rid)

    async def abort(self, rid):
        self._handles.pop(rid, None)
        self.engine.abort(rid)


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
        stream = bool(body.get("stream", False))
        cid = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if not stream:
            text = ""
            async for delta in engine.generate(prompt, max_tokens):
                text += delta
            return JSONResponse({
                "id": cid, "object": "text_completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": text,
                             "finish_reason": "stop"}],
            })

        async def sse():
            try:
                async for delta in engine.generate(prompt, max_tokens):
                    chunk = {
                        "id": cid, "object": "text_completion.chunk",
                        "created": created, "model": model_name,
                        "choices": [{"index": 0, "text": delta,
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {
                    "id": cid, "object": "text_completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": "",
                                 "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                raise

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app
