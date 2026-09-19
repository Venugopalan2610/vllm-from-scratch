"""Stage 15 - an async engine and an OpenAI-compatible API.

The spec is in app/s15_server.py.

There is no GPU and no model here. A fake step engine replaces the real one,
so these checks examine the PLUMBING:

  - HTTP never stops the engine loop,
  - the stream works,
  - a client that disconnects frees its KV blocks.
"""

import asyncio
import json

import pytest

from app.s15_server import AsyncLLMEngine, create_app

pytestmark = pytest.mark.asyncio


class FakeStepEngine:
    """Emits one character of the prompt in each step. Deterministic and
    instant."""

    def __init__(self):
        self.pending_characters = {}
        self.aborted = []
        self.steps = 0

    def add_request(self, rid, prompt, max_tokens):
        text = (prompt or "x")[:max_tokens] or "x"
        self.pending_characters[rid] = list(text)

    def has_work(self):
        return bool(self.pending_characters)

    def abort(self, rid):
        self.pending_characters.pop(rid, None)
        self.aborted.append(rid)

    def step(self):
        self.steps += 1
        outputs = []
        for rid in list(self.pending_characters):
            characters = self.pending_characters[rid]
            character = characters.pop(0)
            finished = not characters
            if finished:
                del self.pending_characters[rid]
            outputs.append((rid, character, finished))
        return outputs


async def collect(engine, prompt, max_tokens):
    return "".join([delta async for delta in engine.generate(prompt,
                                                             max_tokens)])


def completion_text(response_body):
    return response_body["choices"][0]["text"]


async def stream_chunks(client, prompt, max_tokens):
    """-> the data lines of a streamed completion, without "data: "."""
    chunks = []
    async with client.stream("POST", "/v1/completions",
                             json={"prompt": prompt, "max_tokens": max_tokens,
                                   "stream": True}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line[len("data: "):])
    return chunks


@pytest.fixture
async def engine():
    async_engine = AsyncLLMEngine(FakeStepEngine())
    await async_engine.start()
    yield async_engine
    await async_engine.stop()


@pytest.fixture
async def client(engine):
    import httpx

    transport = httpx.ASGITransport(app=create_app(engine))
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://test") as http_client:
        await http_client.get("/health")     # starts the app
        yield http_client


# ---- engine ---------------------------------------------------------

async def test_generate_streams_deltas(engine):
    assert await collect(engine, "hello", 5) == "hello"


async def test_concurrent_requests_interleave(engine):
    """Two generations must progress together, not one after the other."""
    first, second = await asyncio.gather(collect(engine, "abcde", 10),
                                         collect(engine, "12345", 10))
    assert first == "abcde"
    assert second == "12345"


async def test_many_concurrent_requests(engine):
    results = await asyncio.gather(*[collect(engine, f"req{index:03d}", 10)
                                     for index in range(25)])
    assert results == [f"req{index:03d}" for index in range(25)]


async def test_active_count_returns_to_zero(engine):
    await collect(engine, "abc", 3)
    await asyncio.sleep(0.05)
    assert engine.num_active == 0, "the finished requests stay in the engine"


async def test_abort_releases_the_request(engine):
    """A client that disconnects must free its KV, not keep it until a
    timeout."""
    stream = engine.generate("abcdefghij", 10, rid="doomed")
    await stream.__anext__()             # read one token, then leave
    await stream.aclose()
    await asyncio.sleep(0.05)
    assert "doomed" in engine.engine.aborted, (
        "the abandoned stream did not abort the request. In a real server, "
        "every client that disconnects then keeps its KV blocks.")
    assert engine.num_active == 0


async def test_engine_loop_survives_an_aborted_request(engine):
    stream = engine.generate("abcdefghij", 10, rid="gone")
    await stream.__anext__()
    await stream.aclose()
    assert await collect(engine, "still works", 11) == "still works"


# ---- HTTP -----------------------------------------------------------

async def test_health_and_models(client):
    assert (await client.get("/health")).status_code == 200
    models = await client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"]


async def test_non_streaming_completion(client):
    response = await client.post("/v1/completions",
                                 json={"prompt": "hello", "max_tokens": 5})
    assert response.status_code == 200
    body = response.json()
    assert completion_text(body) == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["object"] == "text_completion"


async def test_streaming_completion_sse_format(client):
    chunks = await stream_chunks(client, "abcde", 5)
    assert chunks[-1] == "[DONE]", (
        "an OpenAI-compatible stream must end with 'data: [DONE]'")
    assert "".join(completion_text(json.loads(chunk))
                   for chunk in chunks[:-1]) == "abcde"


async def test_streaming_and_non_streaming_agree(client):
    response = await client.post("/v1/completions",
                                 json={"prompt": "consistent",
                                       "max_tokens": 10})
    chunks = await stream_chunks(client, "consistent", 10)
    streamed = "".join(completion_text(json.loads(chunk))
                       for chunk in chunks if chunk != "[DONE]")
    assert streamed == completion_text(response.json()), (
        "the streamed and whole responses must be the same. The detokenizer "
        "invariant of stage 14 protects exactly this.")


async def test_concurrent_http_requests(client):
    async def complete(index):
        response = await client.post("/v1/completions",
                                     json={"prompt": f"p{index:02d}",
                                           "max_tokens": 5})
        return completion_text(response.json())

    results = await asyncio.gather(*[complete(index) for index in range(10)])
    assert results == [f"p{index:02d}" for index in range(10)]


async def test_http_does_not_block_the_engine_loop(client, engine):
    """The engine must keep its steps while HTTP work is in progress.

    In vLLM V1, that is why EngineCore moved to its own process. The Python
    overhead on the API side stopped the GPU between steps, by an amount
    that they could measure.
    """
    steps_before = engine.engine.steps
    await asyncio.gather(*[
        client.post("/v1/completions",
                    json={"prompt": "abcdefgh", "max_tokens": 8})
        for _ in range(5)])
    assert engine.engine.steps > steps_before + 5
