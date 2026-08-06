# Build Your Own vLLM

Twenty stages, from a naive greedy loop to a paged, continuously-batched,
CUDA-graphed, speculatively-decoding inference server. Every stage is gated by
checks, and most are gated by a **measurement** — you don't advance because your
code runs, you advance because it got faster in the way the stage predicted.

Runs entirely on one consumer GPU.

The derivations behind these stages are at
[derivingsystems.com](https://derivingsystems.com), and
[The Course](https://derivingsystems.com/course.html) is the whole ladder on one
page if you want to read it before you build it.

## Start without installing anything

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Venugopalan2610/vllm-from-scratch/blob/master/colab.ipynb)

A free Colab T4, no local setup, about three minutes to stage 1. It also puts
the GPU stages (8, 12, 18) within reach of a machine that can't run them.

The catch is that a Colab runtime is temporary, so your work dies with the
session. The notebook's last two cells save it, either as a download or pushed
to your fork. Past about stage 5, do this locally instead.

## Or start locally

**Fork this repo**, then clone your fork. You want your own copy: `./vc submit`
commits your work, and you'll want somewhere to push it.

```bash
gh repo fork Venugopalan2610/vllm-from-scratch --clone
cd vllm-from-scratch
./setup.sh        # python 3.12 venv + torch + the model. one time, ~5 min.
```

No GPU? Setup still works, and about half the stages still run — the allocator,
scheduler, prefix cache, metrics, speculative sampling, guided decoding and
tensor-parallel stages are pure logic.

## The loop

```bash
./vc              # where am I?
./vc guide        # what to build for this stage, and why
                  # ... now go edit the file it names ...
./vc test         # run the checks. as often as you like.
./vc submit       # all green? banked, committed, next stage opens automatically
```

`./vc submit` refuses to advance while anything is red, so you cannot skip a
stage by accident. When it passes it commits **only `app/`** — your work — and
prints the next stage's guide.

That's the whole thing. Everything below is reference.

## What a guide looks like

```
==========================================================================
  Stage 07/20   Attention that reads through the page table   [****.]
  A2 - PagedAttention
==========================================================================

WHY THIS STAGE EXISTS
  The kernel must gather K/V from scattered blocks instead of striding a
  contiguous tensor. Do it in PyTorch first to get it CORRECT, then keep
  that as the reference oracle forever.

WHAT YOU'RE BUILDING
  paged_attn() in pure PyTorch, bit-comparable to stage 2's output.

  app/s07_paged_attn.py   <- edit this; the full spec is in its docstrings

HOW YOU'LL KNOW IT WORKED
  Correctness vs the contiguous implementation, then the slowdown you ate.

THE CHECKS (9)
  [ ] write kv scatters to the right slots
  [ ] matches dense reference
  [ ] gqa
      num_heads != num_kv_heads. Query head h reads KV head h // group.
  ...
```

The short "why" is in the guide. The **full spec** — signatures, the sketch, and
the specific traps — lives in the docstrings of the file you're editing. Open it.

## Other commands

```bash
./vc list             the whole ladder, with your progress
./vc guide 12         any stage's guide, not just the current one
./vc test 7           run any stage's checks
./vc peek             reveal the reference solution for this stage
./vc reset 5          rewind to stage 5 (your code is untouched)

./vc info             your GPU's measured roofline
./vc math 7 128       read-vs-compute timings, every division written out
./vc cliff            the L2-vs-VRAM bandwidth cliff
```

## If you get stuck

Reference solutions exist for all 20 stages, but they are **not on this branch**.
They live on `solutions`, so a fresh clone shows you stubs and nothing else.

```bash
./vc peek             # print this stage's reference solution
./vc peek 7           # any stage
./vc peek 7 --apply   # write it straight into the file
```

It fetches the branch on first use. Reach for it when you're stuck rather than
stalling — a stage you read the answer to beats a repo you abandoned. You can
also browse them directly:

```bash
git show solutions:.solutions/s07_paged_attn.py
```

## Rules

- You edit `app/`. You never edit `tests/` — the checks are the spec.
- Progress lives in `.progress.json` (gitignored). Delete it to start over.

**All 229 checks pass against the reference solutions.** Nothing here is
aspirational: if a check fails, it is your code, not the harness. Verify that
claim yourself any time — it pulls the solutions branch, runs everything, and
puts your stubs back:

```bash
dev/verify.sh          # whole suite
dev/verify.sh 7 8      # or just some stages
```

## Read this first

This course is the build half of a book. **[Deriving Systems](https://derivingsystems.com)**
is the derivation half: twelve chapters working out, from arithmetic you
can do on a napkin, why an inference engine has to look like this at all.
Chapters 7 through 12 lead directly into these stages.

- [The Ridge](https://derivingsystems.com/07-the-ridge.html) sets up stages 01-03
- [The Cache That Ate the Batch](https://derivingsystems.com/08-kv-cache.html) sets up stage 02
- [The Slot That Waited](https://derivingsystems.com/09-the-slot-that-waited.html) sets up stages 04-05
- [A Page Table for Tokens](https://derivingsystems.com/10-a-page-table-for-tokens.html) sets up stages 06-09
- [Below the Floor](https://derivingsystems.com/11-below-the-floor.html) sets up stages 03 and 12
- [Spending the Idle](https://derivingsystems.com/12-spending-the-idle.html) sets up stage 17

[LORE.md](LORE.md) is the in-repo conceptual spine: one physical fact about memory
bandwidth, and the twenty forced moves that follow from it. Section 1 answers
"is this IO-bound or CPU-bound?" with no jargon and every division written out.

## The ladder

| Arc | Stages | What you build |
|---|---|---|
| A0 | 01-03 | Naive loop, KV cache, and the roofline that explains everything |
| A1 | 04-05 | Static batching, then continuous batching (the Orca idea) |
| A2 | 06-09 | **PagedAttention**: block allocator, Triton kernel, prefix caching |
| A3 | 10-11 | Scheduler: admission, preemption, chunked prefill |
| A4 | 12-14 | CUDA graphs, batched sampler, streaming detokenization |
| A5 | 15-16 | Async engine, OpenAI-compatible API, the metrics that matter |
| A6 | 17-20 | Speculative decoding, quantization, guided decoding, tensor parallel |

Roughly half need no GPU at all — the allocator, scheduler, prefix cache,
sampler, detokenizer, metrics, guided decoding and speculative sampling are pure
logic, and they are tested hardest, because their failure modes (leaks,
starvation, livelock, distribution skew) are the ones that look like "the server
just got slow" in production.

## This machine

- RTX 4080 Laptop, 12 GB. Model is Qwen3-0.6B (28 layers, GQA 16:8) — small
  enough to iterate in seconds, real enough to have every structural feature
  that matters. Override with `VC_MODEL=...`.
- sm_89 means native FP8, which stage 18 uses.
- Stage 20 runs 2 gloo ranks on CPU. You get the sharding and collective logic
  right; on one GPU there is no speedup to be had.
- No system CUDA toolkit needed — Triton ships its own compiler and PyTorch
  bundles its runtime.
