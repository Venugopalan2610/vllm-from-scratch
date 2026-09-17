# Build Your Own vLLM

Twenty-odd stages, from a naive greedy loop to a paged, continuously-batched,
CUDA-graphed, speculatively-decoding inference server. Every stage is gated by
checks, and most are gated by a **measurement** — you don't advance because your
code runs, you advance because it got faster in the way the stage predicted.

Four of those stages are **CUDA you write yourself**: a paged attention kernel,
then the same kernel made to coalesce, then warp shuffles and split-K, then a
quantized GEMV with a fused epilogue. Not Triton. Real `.cu` files, real
`nvcc`, real register counts.

Runs entirely on one consumer GPU. **In PyTorch or in JAX** — the same ladder,
two backends, and about half the stages are shared between them because they
are pure logic and no framework appears in them at all.

The derivations behind these stages are at
[derivingsystems.com](https://derivingsystems.com), and
[The Course](https://derivingsystems.com/course.html) is the whole ladder on one
page if you want to read it before you build it.

## Start without installing anything

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Venugopalan2610/vllm-from-scratch/blob/master/colab.ipynb)

A free Colab T4, no local setup, about three minutes to stage 1. It also puts
the GPU stages (8, 8b, 8c, 12, 18, 18b) within reach of a machine that can't
run them — Colab already has a CUDA toolkit, which the kernel stages need.

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
./setup.sh --jax  # ...and JAX, if you want the second track. +~400MB.
```

No GPU? Setup still works, and about half the stages still run — the allocator,
scheduler, prefix cache, metrics, speculative sampling, guided decoding and
tensor-parallel stages are pure logic. A GPU but no `nvcc`? Everything runs
except the four CUDA stages, which skip rather than fail.

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

## Two tracks

```bash
./vc backend          which track you're on, and how far you got on each
./vc backend jax      switch. progress is banked per track.
./vc test 8 --jax     one command on the other track, without switching
```

The JAX track walks the original twenty stages with the same insights and the
same measurements. Eleven of them get a JAX twin — you edit `app/j08_paged_pallas.py`
instead of `app/cuda/s08_paged_attn.cu`, and `./vc guide` points you at the right
one. The other nine are **framework-free**: the block allocator, prefix cache,
scheduler, chunked prefill, detokenizer, async server, metrics, speculative
decoding and guided decoding contain no tensors worth porting, so both tracks
build and test the identical file.

The torch track has three stages the JAX track does not: **08b, 08c and 18b**
are CUDA, and there is no honest JAX equivalent of a warp shuffle. So the
ladders are 23 stages and 20 stages, and the numbering does not shift, because
the extra ones carry letters.

| | torch track | jax track |
|---|---|---|
| model | HuggingFace `transformers` | `jvllm/model.py`, ~250 lines of jnp |
| KV cache | grows by concatenation | preallocated, written into |
| stage 08 kernel | CUDA, and then 08b and 08c tune it | Pallas |
| stage 12 | CUDA graphs, to delete launch overhead | shape buckets, to delete recompiles |
| stage 20 | 2 gloo ranks on CPU | `shard_map` over a CPU device mesh |

`jvllm/` is provided, like `tests/helpers.py` — the JAX track needs a Qwen3 to
build on and there is no Flax one, so there is one here. Read
`jvllm/model.py` before stage 01; the cache API it hands you is why the two
ladders diverge where they do. [LORE.md §10](LORE.md) is the full argument.

Why bother, if the ideas are the same? Because the ideas being the same is the
finding. Bucketing shapes is not a CUDA trick — you rediscover it on the JAX
track for a completely unrelated reason. And the nine shared stages are a
direct measurement of how much of an inference engine is actually framework
code: not much.

## What a guide looks like

```
==========================================================================
  Stage 08c of 23   Warps, occupancy and split-K   [*****]
  A2 - PagedAttention   [torch]
==========================================================================

WHY THIS STAGE EXISTS
  At batch 1 the grid is (1, num_heads) and most of the SMs have nothing
  to do, so a kernel that is perfect on bandwidth still runs at a few
  percent of the card. Two fixes: __shfl_xor_sync to reduce inside a warp
  with no shared memory and no barrier, then split-K, which cuts the
  CONTEXT into chunks to manufacture blocks and merges the partial softmax
  states exactly. Batch 1 is every interactive request.

WHAT YOU'RE BUILDING
  Warp-shuffle reductions, then a split-K kernel and an exact merge pass.

  app/cuda/s08c_paged_attn_split.cu   <- edit this; the spec is in its header
  app/s08c_cuda_warps.py   <- and this

HOW YOU'LL KNOW IT WORKED
  At least 1.5x over stage 08b at 1, 2 and 4 sequences, and no more than a
  few percent given back at 64.

THE CHECKS (13)
  [ ] agrees with stage 07
      The oracle still has not moved.
  [ ] the merge is exact for any split count
  [ ] the reductions do not race
  [ ] no shared memory hazards
  ...
```

A CUDA stage names two files: the `.cu` you live in and the `.py` that builds
and calls it. `./vc peek` prints both.

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

**All 453 torch checks and all 360 JAX checks pass against the reference
solutions.** Nothing here is aspirational: if a check fails, it is your code,
not the harness. Verify that claim yourself any time — it pulls the solutions
branch, runs everything, and puts your stubs back:

```bash
dev/verify.sh          # whole suite, torch track
dev/verify.sh --jax    # whole suite, jax track
dev/verify.sh 7 8      # or just some stages
```

Run the two tracks as separate invocations rather than `--both`: together they
want four models resident on one card. The torch run is about three minutes
once the kernels are built, the JAX run about eight.

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
bandwidth, and the forced moves that follow from it. Section 1 answers
"is this IO-bound or CPU-bound?" with no jargon and every division written out.
Section 3b is the CUDA argument: why the same kernel is worth writing three
times, and which constraint each version is actually fighting. Section 9 is the
vocabulary, warps and coalescing and occupancy included.

## The ladder

| Arc | Stages | What you build |
|---|---|---|
| A0 | 01-03 | Naive loop, KV cache, and the roofline that explains everything |
| A1 | 04-05 | Static batching, then continuous batching (the Orca idea) |
| A2 | 06-09 | **PagedAttention**: block allocator, **CUDA kernel across 08, 08b, 08c**, prefix caching |
| A3 | 10-11 | Scheduler: admission, preemption, chunked prefill |
| A4 | 12-14 | CUDA graphs, batched sampler, streaming detokenization |
| A5 | 15-16 | Async engine, OpenAI-compatible API, the metrics that matter |
| A6 | 17-20 | Speculative decoding, quantization (**+ 18b, a CUDA int8 GEMV**), guided decoding, tensor parallel |

Roughly half need no GPU at all — the allocator, scheduler, prefix cache,
sampler, detokenizer, metrics, guided decoding and speculative sampling are pure
logic, and they are tested hardest, because their failure modes (leaks,
starvation, livelock, distribution skew) are the ones that look like "the server
just got slow" in production.

The four with a letter — 08b, 08c and 18b, plus 08 itself — are the CUDA ones.
They need `nvcc`, they are the only stages that do, and they are where the
course stops being about what to compute and starts being about which thread
touches which byte.

## This machine

- RTX 4080 Laptop, 12 GB. Model is Qwen3-0.6B (28 layers, GQA 16:8) — small
  enough to iterate in seconds, real enough to have every structural feature
  that matters. Override with `VC_MODEL=...`.
- sm_89 means native FP8, which stage 18 uses.
- Stage 20 runs 2 gloo ranks on CPU. You get the sharding and collective logic
  right; on one GPU there is no speedup to be had.
- **A CUDA toolkit IS needed** for stages 08, 08b, 08c and 18b, because you
  are writing `.cu` files and `nvcc` has to compile them. `./setup.sh` says so
  if it cannot find one. The other 19 stages need only the PyTorch wheel.
- The first run of a kernel stage spends 20-40 seconds in `nvcc`. After that
  the build is cached in `.cudacache/` and only a source change rebuilds it.
  `VC_CUDA_VERBOSE=1 ./vc test 8` shows the compiler command and the register
  counts.
- On the JAX track, Pallas's default Mosaic GPU backend needs sm_90+, so stage
  08 goes through the older Triton backend — which JAX gates on a hardcoded
  allowlist of device kinds that no laptop GPU is on. `jvllm/compat.py` reads
  your card's compute capability and registers it. If neither backend can
  compile, stage 08's checks skip rather than fail.
