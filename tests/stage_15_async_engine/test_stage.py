"""Stage 15 - Async engine and OpenAI-compatible API.

Spec in app/s15_server.py.

No GPU and no model here: a fake step-engine is injected, so these tests check
the PLUMBING -- that HTTP never blocks the engine loop, that streaming works,
and that a disconnected client's KV actually gets released.
"""

import asyncio
import json

import pytest

from app.s15_server import AsyncLLMEngine, create_app

pytestmark = pytest.mark.asyncio


class FakeStepEngine:
    """Emits one character of the prompt per step. Deterministic and instant."""

    def __init__(self):
        self.pending = {}
        self.aborted = []
        self.steps = 0

    def add_request(self, rid, prompt, max_tokens):
        text = (prompt or "x")[:max_tokens] or "x"
        self.pending[rid] = list(text)

    def has_work(self):
        return bool(self.pending)

    def abort(self, rid):
        if rid in self.pending:
            del self.pending[rid]
        self.aborted.append(rid)

    def step(self):
        self.steps += 1
        out = []
        for rid in list(self.pending):
            chars = self.pending[rid]
            ch = chars.pop(0)
            finished = not chars
            if finished:
                del self.pending[rid]
            out.append((rid, ch, finished))
        return out


@pytest.fixture
async def engine():
    e = AsyncLLMEngine(FakeStepEngine())
    await e.start()
    yield e
    await e.stop()


@pytest.fixture
async def client(engine):
    import httpx
    app = create_app(engine)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://test") as c:
        await c.get("/health")     # triggers startup
        yield c


# ---- engine ---------------------------------------------------------

async def test_generate_streams_deltas(engine):
    out = "".join([d async for d in engine.generate("hello", 5)])
    assert out == "hello"


async def test_concurrent_requests_interleave(engine):
    """Two generations must make progress together, not one after the other."""
    async def collect(p):
        return "".join([d async for d in engine.generate(p, 10)])

    a, b = await asyncio.gather(collect("abcde"), collect("12345"))
    assert a == "abcde"
    assert b == "12345"


async def test_many_concurrent_requests(engine):
    async def collect(i):
        return "".join([d async for d in engine.generate(f"req{i:03d}", 10)])

    results = await asyncio.gather(*[collect(i) for i in range(25)])
    assert results == [f"req{i:03d}" for i in range(25)]


async def test_active_count_returns_to_zero(engine):
    async for _ in engine.generate("abc", 3):
        pass
    await asyncio.sleep(0.05)
    assert engine.num_active == 0, "finished requests were not cleaned up"


async def test_abort_releases_the_request(engine):
    """A client that hangs up must free its KV, not leak it until timeout."""
    gen = engine.generate("abcdefghij", 10, rid="doomed")
    await gen.__anext__()             # consume one token, then walk away
    await gen.aclose()
    await asyncio.sleep(0.05)

    assert "doomed" in engine.engine.aborted, (
        "abandoning the stream did not abort the request -- in a real server "
        "this leaks KV blocks for every disconnected client"
    )
    assert engine.num_active == 0


async def test_engine_loop_survives_an_aborted_request(engine):
    gen = engine.generate("abcdefghij", 10, rid="gone")
    await gen.__anext__()
    await gen.aclose()
    out = "".join([d async for d in engine.generate("still works", 11)])
    assert out == "still works"


# ---- HTTP -----------------------------------------------------------

async def test_health_and_models(client):
    r = await client.get("/health")
    assert r.status_code == 200
    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"]


async def test_non_streaming_completion(client):
    r = await client.post("/v1/completions",
                          json={"prompt": "hello", "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["text"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["object"] == "text_completion"


async def test_streaming_completion_sse_format(client):
    chunks = []
    async with client.stream("POST", "/v1/completions",
                             json={"prompt": "abcde", "max_tokens": 5,
                                   "stream": True}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line[6:])

    assert chunks[-1] == "[DONE]", (
        "an OpenAI-compatible stream must terminate with 'data: [DONE]'"
    )
    text = "".join(
        json.loads(c)["choices"][0]["text"] for c in chunks[:-1]
    )
    assert text == "abcde"


async def test_streaming_and_non_streaming_agree(client):
    r = await client.post("/v1/completions",
                          json={"prompt": "consistent", "max_tokens": 10})
    batch = r.json()["choices"][0]["text"]

    parts = []
    async with client.stream("POST", "/v1/completions",
                             json={"prompt": "consistent", "max_tokens": 10,
                                   "stream": True}) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                parts.append(json.loads(line[6:])["choices"][0]["text"])

    assert "".join(parts) == batch, (
        "streaming and non-streaming responses must be identical -- this is "
        "exactly what stage 14's detokenizer invariant protects"
    )


async def test_concurrent_http_requests(client):
    async def one(i):
        r = await client.post("/v1/completions",
                              json={"prompt": f"p{i:02d}", "max_tokens": 5})
        return r.json()["choices"][0]["text"]

    got = await asyncio.gather(*[one(i) for i in range(10)])
    assert got == [f"p{i:02d}" for i in range(10)]


async def test_http_does_not_block_the_engine_loop(client, engine):
    """The engine must keep stepping while HTTP work is in flight.

    In vLLM V1 this is why EngineCore moved to its own process: Python overhead
    on the API side was measurably stalling the GPU between steps.
    """
    before = engine.engine.steps
    await asyncio.gather(*[
        client.post("/v1/completions",
                    json={"prompt": "abcdefgh", "max_tokens": 8})
        for _ in range(5)
    ])
    assert engine.engine.steps > before + 5
