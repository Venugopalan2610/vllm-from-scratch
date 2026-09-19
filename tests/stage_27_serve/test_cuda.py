"""Stage 27 - the server, for real.

The spec is in app/s27_serve.py. These checks send HTTP requests to your
server, with your stage 26 engine and the real model behind it.
"""

import asyncio
import json
import time

import pytest

from app.s15_server import AsyncLLMEngine
from app.s16_metrics import MetricsCollector
from app.s22_engine import LLMEngine
from app.s26_guided import GuidedEngine
from app.s27_serve import (
    EngineAdapter,
    build_app,
    prometheus_text,
    request_options,
    run_load,
)
from tests.helpers import CAPSTONE_PROMPTS

pytestmark = pytest.mark.asyncio

DATA_PREFIX = "data: "
DONE_LINE = "data: [DONE]"


def _release(engine):
    """FastAPI keeps every route function in a module-level lru_cache, so an
    app never dies, and its engine and KV pool never die either. One server
    in one process does not care. Twenty checks in one process do."""
    engine.runner.kv_caches.clear()


class Server:
    """The app, the engine and an HTTP client, for one check."""

    def __init__(self, model, num_blocks=256, **engine_options):
        self.metrics = MetricsCollector()
        self.engine = GuidedEngine(model, num_blocks, metrics=self.metrics,
                                   prefix_cache_blocks=0, **engine_options)
        self.app = build_app(self.engine, self.metrics)

    async def __aenter__(self):
        import httpx

        await self.app.state.serving.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test", timeout=120)
        return self

    async def __aexit__(self, *exc_info):
        await self.client.aclose()
        await self.app.state.serving.stop()
        _release(self.engine)


async def _event_payloads(client, path, body):
    """The JSON payloads of a stream, without the [DONE] line."""
    payloads = []
    async with client.stream("POST", path, json=body) as response:
        async for line in response.aiter_lines():
            if line.startswith(DATA_PREFIX) and line != DONE_LINE:
                payloads.append(json.loads(line[len(DATA_PREFIX):]))
    return payloads


async def _complete(client, prompt, max_tokens, stream=False):
    # temperature 0: the OpenAI default is 1, and these checks compare text.
    body = {"prompt": prompt, "max_tokens": max_tokens, "stream": stream,
            "temperature": 0}
    if not stream:
        response = await client.post("/v1/completions", json=body)
        return response.json()["choices"][0]["text"]
    payloads = await _event_payloads(client, "/v1/completions", body)
    return "".join(payload["choices"][0]["text"] for payload in payloads)


async def _chat(client, content, stream=False, **fields):
    body = {"messages": [{"role": "user", "content": content}],
            "stream": stream, **fields}
    if stream:
        return await _event_payloads(client, "/v1/chat/completions", body)
    return (await client.post("/v1/chat/completions", json=body)).json()


def _message_text(chat_response):
    return chat_response["choices"][0]["message"]["content"]


async def test_http_text_equals_the_engine_text(nvcc, tmodel_exact):
    direct = LLMEngine(tmodel_exact, 64)
    direct.add_request("direct", CAPSTONE_PROMPTS[0], 16)
    direct.run_to_completion()
    expected = tmodel_exact.tokenizer.decode(direct.output("direct"))
    async with Server(tmodel_exact) as server:
        assert await _complete(server.client, CAPSTONE_PROMPTS[0], 16) == expected
        assert await _complete(server.client, CAPSTONE_PROMPTS[0], 16,
                               stream=True) == expected


async def test_concurrent_requests_all_finish(nvcc, tmodel_exact):
    async with Server(tmodel_exact) as server:
        texts = await asyncio.gather(*[_complete(server.client, prompt, 12)
                                       for prompt in CAPSTONE_PROMPTS])
        assert all(texts)
        assert server.metrics.snapshot()["num_finished"] == len(CAPSTONE_PROMPTS)


async def test_a_disconnect_frees_real_blocks(nvcc, tmodel_exact):
    async with Server(tmodel_exact) as server:
        body = {"prompt": "Tell me a long story.", "max_tokens": 400,
                "stream": True}
        async with server.client.stream("POST", "/v1/completions",
                                        json=body) as response:
            num_events = 0
            async for line in response.aiter_lines():
                num_events += line.startswith(DATA_PREFIX)
                if num_events == 5:
                    break                         # the client leaves
        for _ in range(100):
            await asyncio.sleep(0.02)
            if not server.engine.has_work():
                break
        allocator = server.engine.allocator
        assert allocator.num_free == allocator.num_blocks


async def test_metrics_endpoint_speaks_prometheus(nvcc, tmodel_exact):
    async with Server(tmodel_exact) as server:
        await _complete(server.client, CAPSTONE_PROMPTS[1], 8)
        text = (await server.client.get("/metrics")).text
    declared = set()
    for line in text.strip().splitlines():
        if line.startswith("# TYPE"):
            _, _, name, kind = line.split()
            assert kind in ("gauge", "counter")
            declared.add(name)
        else:
            name, value = line.split()
            assert name in declared, f"{name} has no # TYPE line before it"
            float(value)
    for metric in ("ttft_p50", "itl_p99", "num_finished", "kv_utilization_max"):
        assert any(name.endswith(metric) for name in declared), metric


@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
def test_prometheus_text_writes_nan():
    text = prometheus_text({"ttft_p50": float("nan"), "num_requests": 0})
    assert "NaN" in text and "num_requests 0" in text


async def test_ttft_includes_the_wait_for_the_engine(nvcc, tmodel_exact):
    """The adapter records the arrival when the request reaches it, not when
    the engine takes the command one step later."""
    async with Server(tmodel_exact) as server:
        await server.client.post("/v1/completions", json={
            "prompt": "Tell me a story.", "max_tokens": 30})
        await asyncio.gather(*[_complete(server.client, prompt, 4)
                               for prompt in CAPSTONE_PROMPTS[:3]])
        for record in server.metrics.records.values():
            assert record.arrival <= record.scheduled <= record.token_times[0]


async def _loop_lag_ms(seconds=0.5):
    """The median time that the event loop takes to come back to a coroutine
    that only yields. Every request that the server handles waits this long
    before it moves."""
    gaps, end = [], time.perf_counter() + seconds
    while time.perf_counter() < end:
        start = time.perf_counter()
        await asyncio.sleep(0)
        gaps.append(time.perf_counter() - start)
    gaps.sort()
    return gaps[len(gaps) // 2] * 1e3


async def _lag_under_load(serving, engine):
    """Keep the engine busy, start the serving loop, and measure the lag."""
    # Eight requests that cannot stop early keep the engine busy. They go
    # directly to the engine, before the loop starts, so no thread races.
    for index in range(8):
        engine.add_request(f"background{index}", CAPSTONE_PROMPTS[index % 6],
                           150, ignore_eos=True)
    await serving.start()
    await asyncio.sleep(0.2)
    lag_ms = await _loop_lag_ms()
    assert engine.has_work(), "the load ended before the measurement"
    await serving.stop()
    _release(engine)
    return lag_ms


async def test_the_event_loop_stays_free(nvcc, tmodel):
    """While the engine works, the event loop must stay free. The stage 15
    loop, with step() inline, holds the loop for one whole step at a time.
    The check measures both loops, one after the other, on the same load."""
    inline_engine = LLMEngine(tmodel, 256, prefix_cache_blocks=0)
    inline_ms = await _lag_under_load(
        AsyncLLMEngine(EngineAdapter(inline_engine)), inline_engine)
    threaded_engine = LLMEngine(tmodel, 256, prefix_cache_blocks=0)
    threaded_ms = await _lag_under_load(
        build_app(threaded_engine, MetricsCollector()).state.serving,
        threaded_engine)
    print(f"\n  event loop lag under load: inline {inline_ms:.2f} ms, "
          f"threaded {threaded_ms:.2f} ms")
    assert threaded_ms * 5 < inline_ms


async def test_run_load_sends_everything(nvcc, tmodel):
    async with Server(tmodel, max_num_seqs=16) as server:
        prompts = [f"Write a short note about topic {index}."
                   for index in range(24)]
        seconds = await run_load(server.client, prompts, rate_per_s=40,
                                 max_tokens=16)
        snapshot = server.metrics.snapshot()
    assert snapshot["num_finished"] == 24
    assert seconds >= 23 / 40 * 0.5     # the arrivals really were spread out
    print(f"\n  24 requests in {seconds:.2f} s, TTFT p50 "
          f"{snapshot['ttft_p50'] * 1e3:.0f} ms, ITL p99 "
          f"{snapshot['itl_p99'] * 1e3:.1f} ms")


async def test_chat_answers_with_usage(nvcc, tmodel):
    async with Server(tmodel) as server:
        response = await _chat(server.client,
                               "What is the capital of France? One word.",
                               max_tokens=20, temperature=0)
    message = response["choices"][0]["message"]
    assert response["object"] == "chat.completion"
    assert message["role"] == "assistant"
    assert "Paris" in message["content"]
    assert "<|" not in message["content"], "a stop token got into the text"
    usage = response["usage"]
    assert (usage["total_tokens"]
            == usage["prompt_tokens"] + usage["completion_tokens"] > 0)
    assert response["choices"][0]["finish_reason"] == "stop"


async def test_length_is_reported(nvcc, tmodel):
    async with Server(tmodel) as server:
        response = await _chat(server.client, "Tell me a long story.",
                               max_tokens=5, temperature=0)
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 5


async def test_chat_stream_matches_non_stream(nvcc, tmodel):
    async with Server(tmodel) as server:
        whole = await _chat(server.client, "Say hello.", max_tokens=16,
                            temperature=0)
        payloads = await _chat(server.client, "Say hello.", stream=True,
                               max_tokens=16, temperature=0)
    streamed = "".join(payload["choices"][0]["delta"].get("content", "")
                       for payload in payloads)
    assert streamed == _message_text(whole)
    assert payloads[-1]["choices"][0]["finish_reason"] in ("stop", "length")


async def test_json_mode_returns_json(nvcc, tmodel):
    async with Server(tmodel) as server:
        responses = await asyncio.gather(*[
            _chat(server.client, f"Describe item {index} as JSON.",
                  max_tokens=max_tokens, temperature=0.8, seed=index,
                  response_format={"type": "json_object"})
            for index, max_tokens in enumerate([20, 40, 80])])
    for response in responses:
        json.loads(_message_text(response))


async def test_stop_string_ends_the_text(nvcc, tmodel):
    async with Server(tmodel) as server:
        response = (await server.client.post("/v1/completions", json={
            "prompt": "Count: 1, 2, 3,", "max_tokens": 30, "stop": [" 7"],
            "temperature": 0})).json()
    assert " 7" not in response["choices"][0]["text"]
    assert response["choices"][0]["finish_reason"] == "stop"


async def test_seed_makes_sampling_repeatable(nvcc, tmodel):
    async with Server(tmodel) as server:
        first, second, other = [
            _message_text(await _chat(server.client,
                                      "Write one sentence about the sea.",
                                      max_tokens=20, temperature=1.0,
                                      seed=seed))
            for seed in (5, 5, 6)]
    assert first == second and first != other


@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
def test_request_options_read_the_openai_fields():
    options = request_options({"temperature": 0.5, "top_p": 0.9, "seed": 3,
                               "stop": "x",
                               "response_format": {"type": "json_object"}})
    params = options["params"]
    assert (params.temperature, params.top_p, params.seed) == (0.5, 0.9, 3)
    assert tuple(options["stop"]) == ("x",) and options["json_mode"] is True
    assert "json_mode" not in request_options({}), (
        "an engine without stage 26 must still serve a plain request")
