"""Stage 15 - async engine and an OpenAI-compatible API.

`./vc lore 15` for the insight. `./vc test 15` to check yourself.

Two rules, and they point in opposite directions:

    the engine loop must never block on HTTP
    HTTP must never block on the GPU

So: one background task runs step() forever, and each HTTP request is a
consumer on a queue that the loop feeds. Real vLLM V1 pushes this further and
puts the engine in a separate PROCESS, because Python overhead on the API side
was measurably stalling the GPU between steps.

The tests inject a fake step-engine, so this stage is about the plumbing, not
the model.
"""

import asyncio
import json
import time
import uuid


class AsyncLLMEngine:
    """Wraps a synchronous step-based engine.

    The injected `step_engine` provides:
        add_request(rid, prompt, max_tokens)
        step() -> list[(rid, delta_text, finished)]
        has_work() -> bool
        abort(rid)

    Required:
        .num_active                      in-flight requests
        async start() / stop()
        async generate(prompt, max_tokens, rid=None) -> async iterator of str
        async abort(rid)
    """

    def __init__(self, step_engine, idle_sleep=0.001):
        raise NotImplementedError("stage 15: implement AsyncLLMEngine")

    async def _loop(self):
        """The engine loop. One per server, running forever.

            while running:
                if not engine.has_work():  await sleep(idle);  continue
                for rid, delta, finished in engine.step():
                    push (delta, finished) onto that request's queue
                await asyncio.sleep(0)      # yield, or you starve the server

        That trailing yield matters. A tight loop with no await point never
        returns control to the event loop, and your HTTP handlers never run.
        """
        raise NotImplementedError

    async def generate(self, prompt, max_tokens, rid=None):
        """Async generator yielding text deltas.

        The critical part is the `finally:` block. If the consumer stops
        iterating -- client disconnected, request cancelled, timeout -- you
        must call engine.abort(rid) to release its KV blocks. Skip it and every
        abandoned request leaks memory until the process dies. This is the most
        common real-world serving leak.
        """
        raise NotImplementedError

    async def abort(self, rid):
        raise NotImplementedError


def create_app(engine, model_name="vllm-from-scratch"):
    """Build a FastAPI app exposing:

        GET  /health          -> {"status": "ok"}
        GET  /v1/models       -> {"object": "list", "data": [{"id": ...}]}
        POST /v1/completions  -> body: prompt, max_tokens, stream

    Non-streaming returns:
        {"id", "object": "text_completion", "created", "model",
         "choices": [{"index": 0, "text": ..., "finish_reason": "stop"}]}

    Streaming returns text/event-stream, one `data: {json}` line per delta,
    then a final chunk with finish_reason="stop", then literally:

        data: [DONE]

    Clients rely on that sentinel to know the stream ended cleanly rather than
    being cut off, so it is not optional.

    Start the engine loop from a lifespan handler, not at import time.
    """
    raise NotImplementedError("stage 15: implement create_app")
