# Build Your Own vLLM

Twenty stages, from a naive greedy loop to a paged, continuously-batched,
CUDA-graphed, speculatively-decoding inference server. Every stage is gated by
tests, and most stages are gated by a **measurement** — you don't advance
because your code runs, you advance because it got faster in the way the stage
predicted it would.

Runs entirely on one consumer GPU.

## Setup

Already done. Verify:

```bash
cd ~/vllm-from-scratch
./vc info      # your GPU's roofline: bandwidth, TFLOP/s, implied token ceilings
./vc list      # the ladder
```

## The loop

```bash
./vc lore      # what the current stage is teaching, and why
./vc test      # run its tests
./vc pass      # advance when green
```

You write code in `app/`. You never edit `tests/` — the tests are the spec.
Stage 01 and 02 are wired up and passing against a reference implementation, so
the harness is proven; `app/s01_naive.py` and `app/s02_cache.py` are stubbed for
you to fill in.

`.solutions/` holds reference implementations for the wired stages. It exists so
you can unstick yourself, not so you can start there.

## What makes this different from a tutorial

**The oracle is real.** Correctness tests compare your output against
HuggingFace transformers, token for token. There is no partial credit.

**The measurements are real.** `./vc info` measures *your* card: ~380 GB/s of
HBM bandwidth and ~50 TFLOP/s sustained bf16, so its roofline ridge point is
~125-130 FLOP per byte. A batch-1 decode step runs at 1 FLOP/byte — under 1% of
peak compute. Every optimization from stage 4 onward is a different way of
spending that waste, and the tests print the numbers as you claw it back.

The same command also reports burst vs sustained TFLOP/s, because this is a
150 W laptop GPU that boosts and then throttles. Benchmarks you run cold will
lie to you by ~25%, which is a lesson in itself.

**The failures are real.** Stage 02 has a test that demonstrates the same model
and the same prompt producing *different sentences* in bf16 depending on whether
you used a cache. Not a bug in your code — a fact about low-precision serving
that you need to have seen before you meet it in production.

## Where you are going

| Arc | Stages | What you build |
|---|---|---|
| A0 | 01-03 | Naive loop, KV cache, and the roofline that explains everything |
| A1 | 04-05 | Static batching, then continuous batching (the Orca idea) |
| A2 | 06-09 | **PagedAttention**: block allocator, paged kernel in Triton, prefix caching |
| A3 | 10-11 | Scheduler: admission, preemption, chunked prefill |
| A4 | 12-14 | CUDA graphs, batched sampler, streaming detokenization |
| A5 | 15-16 | Async engine, OpenAI-compatible API, the metrics that matter |
| A6 | 17-20 | Speculative decoding, quantization, guided decoding, tensor parallel |

Read [LORE.md](LORE.md) first. It's the conceptual spine: one physical fact
about memory bandwidth, and the twenty forced moves that follow from it.

## Notes on this machine

- RTX 4080 Laptop, 12 GB. Model is Qwen3-0.6B (28 layers, GQA 16:8) — small
  enough to iterate in seconds, real enough to have every structural feature
  that matters. Override with `VC_MODEL=...`.
- sm_89 means you have native FP8, which stage 18 uses.
- Stage 20 (tensor parallelism) runs 2 NCCL ranks on your single GPU. You get
  the sharding and collective logic right; you don't get the speedup.
- No system CUDA toolkit is installed and none is needed — Triton ships its own
  compiler and PyTorch bundles its runtime.
