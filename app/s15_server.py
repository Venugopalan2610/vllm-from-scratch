"""Stage 15 - async engine and an OpenAI-compatible API.

`./vc lore 15` for the insight. `./vc test 15` to check yourself.

Two rules, in opposite directions:

    the engine loop must never wait on HTTP
    HTTP must never wait on the GPU

So one background task runs step() for ever. Each HTTP request reads from a
queue that the loop fills.

Real vLLM V1 goes further and puts the engine in a separate PROCESS. The
Python overhead on the API side stopped the GPU between steps, by an amount
that they could measure.

The checks give a fake step engine, so this stage is about the plumbing, not
the model.
"""

import asyncio
import json
import time
import uuid


class AsyncLLMEngine:
    """Runs a synchronous step engine.

    The step engine that the caller gives provides:
        add_request(rid, prompt, max_tokens)
        step() -> list[(rid, delta_text, finished)]
        has_work() -> bool
        abort(rid)

    Required:
        .num_active                      the requests in progress
        async start() / stop()
        async generate(prompt, max_tokens, rid=None) -> async iterator of str
        async abort(rid)
    """

    def __init__(self, step_engine, idle_sleep=0.001):
        raise NotImplementedError("stage 15: implement AsyncLLMEngine")

    async def _loop(self):
        """The engine loop. One for each server, and it runs for ever.

            while running:
                if not engine.has_work():  await sleep(idle);  continue
                for rid, delta, finished in engine.step():
                    put (delta, finished) on the queue of that request
                await asyncio.sleep(0)      # yield, or the server stops

        The yield at the end is important. A tight loop with no await never
        gives control back to the event loop, and your HTTP handlers never
        run.
        """
        raise NotImplementedError

    async def generate(self, prompt, max_tokens, rid=None):
        """An async generator that yields text deltas.

        The `finally:` block is the critical part. If the consumer stops the
        iteration (a client disconnect, a cancel, a timeout), you must call
        engine.abort(rid) to free its KV blocks. If you do not, every
        abandoned request keeps its memory until the process stops. That is
        the most frequent memory leak of real servers.
        """
        raise NotImplementedError

    async def abort(self, rid):
        raise NotImplementedError


def create_app(engine, model_name="vllm-from-scratch"):
    """Make a FastAPI app with:

        GET  /health          -> {"status": "ok"}
        GET  /v1/models       -> {"object": "list", "data": [{"id": ...}]}
        POST /v1/completions  -> body: prompt, max_tokens, stream

    A response with no stream is:
        {"id", "object": "text_completion", "created", "model",
         "choices": [{"index": 0, "text": ..., "finish_reason": "stop"}]}

    A stream is text/event-stream: one `data: {json}` line for each delta,
    then a last chunk with finish_reason="stop", then exactly:

        data: [DONE]

    Clients use that line to know that the stream ended correctly and was
    not cut. So it is necessary.

    Start the engine loop from a lifespan handler, not at import time.
    """
    raise NotImplementedError("stage 15: implement create_app")
