# Build Your Own vLLM

32 stages on the torch track. You start with a naive greedy loop. You finish
with one inference server that you built from your own parts:

- a paged KV cache, with your block allocator and your CUDA kernels,
- continuous batching with a token budget, chunked prefill and preemption,
- prefix caching,
- CUDA graphs on the real decode step,
- int8 weights on your own GEMV kernel, and an FP8 KV cache on your own
  attention kernel,
- speculative decoding, lossless, inside the engine,
- a JSON mode that cannot produce invalid JSON,
- an OpenAI-compatible HTTP API: chat, sampling fields, stop strings,
  streaming, usage, and Prometheus metrics.

Tensor parallelism is the one part that stays a separate stage. It needs two
GPUs to be real, and the course runs on one.

Checks gate every stage. A **measurement** gates most of them. You do not
advance because your code runs. You advance because it became faster in the way
that the stage predicted.

Five of those stages are **CUDA that you write yourself**:

- a paged attention kernel,
- the same kernel, made to coalesce,
- warp shuffles and split-K,
- a quantized GEMV with a fused epilogue,
- the attention kernel again, on an FP8 cache.

This is not Triton. These are real `.cu` files, real `nvcc`, and real register
counts.

The course runs on one consumer GPU. You can use **PyTorch or JAX**. It is the
same ladder on two backends. Approximately half of the stages serve both
tracks, because they are pure logic and no framework appears in them.

**New to machine learning or to GPUs?** Start with Part 0 of the notebooks,
`course/Part0_FromAProgramToAModel/`. It explains a model, attention and the
GPU to a software engineer, and it needs no ML background. Each term of the
course is in [`course/GLOSSARY.md`](course/GLOSSARY.md), with an analogy from
software, and `./vc guide` prints the new terms of each stage.

The derivations behind these stages are at
[derivingsystems.com](https://derivingsystems.com).
[The Course](https://derivingsystems.com/course.html) puts the whole ladder on
one page. Read it before you build, if you want to.

## What you have at the end, measured

The last arc, the capstone (stages 21 to 28), connects your parts into one
engine. The checks then measure it against physics, on your card, in the same
minute. All of these are ratios, so they hold on any card:

| Measurement | Reference solution | The gate |
|---|---|---|
| tokens against HuggingFace in fp32 | identical | identical |
| batch-1 decode step, against the weight-read floor | 57% to 71% | 50% |
| CUDA graphs against eager, at batch 1 | 1.3x to 1.5x | 1.2x |
| int8 weights against bf16, batch-1 step | 1.47x to 1.55x | 1.3x |
| FP8 KV against bf16 KV, 16 x 2048 tokens | 1.33x to 1.46x | 1.2x |
| perplexity with int8 weights, with an FP8 KV cache | within 1% | within 3% |
| speculative decoding: greedy tokens against no speculation | identical | identical |
| speculative decoding on a copy task, batch 1 | 1.5x to 3.1x | 1.5x |
| JSON mode: answers that parse | all | all |
| the whole engine, against the roofline floor of its own steps | 39% to 59% | 30% |
| the whole engine, against your stage 05 engine, same requests | 3.2x to 4.6x | 2x |

The ranges come from repeated runs on one laptop GPU. A laptop throttles, so
the same code moves by 20% from run to run. That is why every gate compares
two things that the check measures one after the other.

To see the absolute numbers for your card, finish the capstone and run:

```bash
./vc bench      # tok/s, the roofline floor, and the ceiling of your card
./vc serve      # your server on localhost:8000, for curl
```

**Which models.** The capstone model (`tvllm/model.py`) runs the Qwen3 family
and the Llama family. Set `VC_MODEL`. What fits depends on two numbers of your
card: VRAM, and read bandwidth.

- **The weights must fit, with room for the KV pool.** In bf16 a model needs 2
  bytes for each parameter. On a 12 GB card that means approximately 3B
  parameters at most.
- **Batch 1 cannot go faster than bandwidth / weight bytes.** A 0.6B model
  reads 1.2 GB for each token. A 3B model reads 6 GB.
- **With a full KV pool, tok/s is close to bandwidth / (context × KV bytes for
  each token).** Qwen3-0.6B needs 112 KiB of KV for each token. Llama-3.2-1B
  needs 32 KiB. So at long context the larger Llama model serves more tokens.

## Start with no installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Venugopalan2610/vllm-from-scratch/blob/master/colab.ipynb)

Use a free Colab T4. There is no local setup, and stage 1 is approximately three
minutes away. Colab also puts the GPU stages (8, 8b, 8c, 12, 18, 18b, 21-28) into reach
of a machine that cannot run them. Colab has a CUDA toolkit, and the kernel
stages need one.

A Colab runtime is temporary, so your work dies with the session. The last two
cells of the notebook save it. They download it, or they push it to your fork.
After approximately stage 5, work locally instead.

## Or start locally

**Fork this repo**, then clone your fork. You want your own copy. `./vc submit`
commits your work, and you need a place to push it to.

```bash
gh repo fork Venugopalan2610/vllm-from-scratch --clone
cd vllm-from-scratch
./setup.sh        # python 3.12 venv + torch + the model. one time, ~5 min.
./setup.sh --jax  # ...and JAX, if you want the second track. +~400MB.
```

Do you have no GPU? Setup still works, and approximately half of the stages
still run. The allocator, the scheduler, the prefix cache, the metrics, the
speculative sampling, the guided decoding and the tensor-parallel stages are
pure logic.

Do you have a GPU but no `nvcc`? Everything runs except the four CUDA stages.
Those stages skip. They do not fail.

## Two ways in

The notebooks in [`course/`](course/) build the intuition. The ladder in `app/`
makes you build the engine. They have an order. Read the notebooks for a stage,
then build the stage.

```bash
.venv/bin/jupyter lab course/     # the lecture half
./vc guide                        # the project half
```

A notebook does the arithmetic by hand, measures the machine, plots the two
against each other, and then explains the gap. No check gates the notebooks in `course/`.
Nothing in `app/` is optional.

## The loop

```bash
./vc              # where am I?
./vc guide        # what to build for this stage, and why
                  # ... now go edit the file it names ...
./vc test         # run the checks. as often as you like.
./vc submit       # all green? banked, committed, next stage opens automatically
```

`./vc submit` refuses to advance while one check still fails, so you cannot
skip a stage by accident. When the checks pass, it commits **only `app/`**, which is
your work. It then prints the guide for the next stage.

That is the whole thing. Everything below is reference.

## Two tracks

```bash
./vc backend          which track you're on, and how far you got on each
./vc backend jax      switch. progress is banked per track.
./vc test 8 --jax     one command on the other track, without switching
```

The JAX track walks the original twenty stages, with the same insights and the
same measurements. Eleven stages have a JAX twin. You edit
`app/j08_paged_pallas.py` in the place of `app/cuda/s08_paged_attn.cu`, and
`./vc guide` points you at the correct file.

The other nine stages are **framework-free**. These nine hold no tensors that
are worth a port:

- the block allocator and the prefix cache,
- the scheduler and the chunked prefill,
- the detokenizer, the async server and the metrics,
- the speculative decoding and the guided decoding.

Both tracks build and test the same file.

The torch track has twelve stages that the JAX track does not have. **08b,
08c and 18b** are CUDA, and there is no honest JAX equivalent of a warp
shuffle. The capstone, **21 to 28**, connects your CUDA kernels into one
engine. So the ladders are 32 stages and 20 stages. The numbers of the shared stages do not
shift, because the extra stages carry letters or come at the end.

| | torch track | jax track |
|---|---|---|
| model | HuggingFace `transformers` | `jvllm/model.py`, ~250 lines of jnp |
| KV cache | grows by concatenation | preallocated, written into |
| stage 08 kernel | CUDA, and then 08b and 08c tune it | Pallas |
| stage 12 | CUDA graphs, to delete launch overhead | shape buckets, to delete recompiles |
| stage 20 | 2 gloo ranks on CPU | `shard_map` over a CPU device mesh |

`jvllm/` is code that the repo gives you, like `tests/helpers.py`. The JAX track needs a Qwen3
to build on, and there is no Flax one, so this repo holds one. Read
`jvllm/model.py` before stage 01. The cache API that it gives you is the reason
that the two ladders separate where they do. [LORE.md §10](LORE.md) holds the
full argument.

Why do this, if the ideas are the same? Because the sameness of the ideas is the
finding. A bucket for each shape is not a CUDA trick. You find the same idea again on
the JAX track, for a completely different reason. And the nine shared stages measure how
much of an inference engine is framework code. The answer is: not much.

## What a guide looks like

```
==========================================================================
  Stage 08c of 32   Warps, occupancy and split-K   [*****]
  A2 - PagedAttention   [torch]
==========================================================================

WHY THIS STAGE EXISTS
  At batch 1 the grid is (1, num_heads) and most of the SMs have nothing
  to do, so a kernel that is perfect on bandwidth still runs at a few
  percent of the card. Two fixes: __shfl_xor_sync to reduce inside a warp
  with no shared memory and no barrier, then split-K, which cuts the
  CONTEXT into chunks to manufacture blocks and merges the partial softmax
  states exactly. Batch 1 is every interactive request.

WHAT YOU ARE BUILDING
  Warp-shuffle reductions, then a split-K kernel and an exact merge pass.

  app/cuda/s08c_paged_attn_split.cu   <- edit this; the spec is in its header
  app/s08c_cuda_warps.py   <- and this

HOW YOU WILL KNOW IT WORKED
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

A CUDA stage names two files: the `.cu` file that you live in, and the `.py`
file that builds and calls it. `./vc peek` prints both.

The guide holds the short "why". The **full spec** is in the docstrings of the
file that you edit. It gives the signatures, the sketch, and the specific traps.
Open it.

## Other commands

```bash
./vc list             the whole ladder, with your progress
./vc guide 12         any stage's guide, not just the current one
./vc test 7           run any stage's checks
./vc peek             reveal the reference solution for this stage
./vc reset 5          rewind to stage 5 (your code is untouched)

./vc info             your GPU's measured roofline
./vc bench            your capstone engine against the roof of your card
./vc serve            your capstone server, on localhost:8000
./vc math 7 128       read-vs-compute timings, every division written out
./vc cliff            the L2-vs-VRAM bandwidth cliff
```

## If you get stuck

Reference solutions exist for all of the stages, but they are **not on this
branch**. They live on the `solutions` branch, so a fresh clone shows you stubs
and nothing else.

```bash
./vc peek             # print this stage's reference solution
./vc peek 7           # any stage
./vc peek 7 --apply   # write it straight into the file
```

The command fetches the branch on first use. Use it when you are stuck. Do not
stall. A stage that you read the answer to is better than a repo that you
abandoned. You can also browse the solutions directly:

```bash
git show solutions:.solutions/s07_paged_attn.py
```

## Rules

- You edit `app/`. You never edit `tests/`. The checks are the spec.
- Progress lives in `.progress.json`, which git ignores. Delete it to start
  again.

**All 600 torch checks and all 414 JAX checks pass against the reference
solutions.** Nothing here is aspirational. If a check fails, the cause is your
code and not the harness.

Verify that claim yourself at any time. The script pulls the solutions branch,
runs everything, and then puts your stubs back:

```bash
dev/verify.sh          # whole suite, torch track
dev/verify.sh --jax    # whole suite, jax track
dev/verify.sh 7 8      # or just some stages
```

Run the two tracks as separate commands. Do not use `--both`. Together they want
four models resident on one card. The torch run needs approximately four
minutes after the kernels build. The JAX run needs approximately seven minutes.

## Read this first

This course is the build half of a book.
**[Deriving Systems](https://derivingsystems.com)** is the derivation half. Its
twelve chapters start from arithmetic that you can do on a napkin. They work out
why an inference engine must look like this. Chapters 7 through 12 lead directly
into these stages.

- [The Ridge](https://derivingsystems.com/07-the-ridge.html) sets up stages 01-03
- [The Cache That Ate the Batch](https://derivingsystems.com/08-kv-cache.html) sets up stage 02
- [The Slot That Waited](https://derivingsystems.com/09-the-slot-that-waited.html) sets up stages 04-05
- [A Page Table for Tokens](https://derivingsystems.com/10-a-page-table-for-tokens.html) sets up stages 06-09
- [Below the Floor](https://derivingsystems.com/11-below-the-floor.html) sets up stages 03 and 12
- [Spending the Idle](https://derivingsystems.com/12-spending-the-idle.html) sets up stage 17

[LORE.md](LORE.md) is the conceptual spine inside this repo. It holds one
physical fact about memory bandwidth, and the moves that the fact forces.

Section 1 answers the question "is this IO-bound or CPU-bound?" It uses no
jargon and writes out every division. Section 3b is the CUDA argument: why the
same kernel is worth three versions, and which constraint each version fights.
Section 9 is the vocabulary. It includes warps, coalescing and occupancy.

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
| A7 | 21-28 | **The capstone**: your parts in one engine and one server: paged model, scheduler, graphs, int8, FP8 KV (24b), speculation, JSON mode, the OpenAI API, measured against the roof |

Approximately half of the stages need no GPU. The allocator, the scheduler, the
prefix cache, the sampler, the detokenizer, the metrics, the guided decoding and
the speculative sampling are pure logic.

These stages get the hardest tests. Their failure modes are leaks, starvation,
livelock and distribution skew. In production, all four look like "the server
became slow".

Five stages hold CUDA that you write: 08, 08b, 08c, 18b and 24b. They need
`nvcc`, and so does the capstone that runs them. They are
where the course stops to ask what to compute, and starts to ask which thread
touches which byte.

## This machine

- RTX 4080 Laptop, 12 GB. The model is Qwen3-0.6B (28 layers, GQA 16:8). It is
  small enough to iterate in seconds, and real enough to have every structural
  feature that matters. Override it with `VC_MODEL=...`.
- sm_89 gives native FP8, which stage 18 uses.
- Stage 20 runs 2 gloo ranks on the CPU. You make the sharding and the
  collective logic correct. On one GPU there is no speedup to get.
- **You need a CUDA toolkit** for stages 08, 08b, 08c, 18b and 24b, and for the
  capstone, which runs those kernels. You write `.cu` files, and `nvcc` must
  compile them. `./setup.sh` tells you if it cannot find one. The other 19
  stages need only the PyTorch wheel.
- The first run of a kernel stage spends 20 to 40 seconds in `nvcc`. The build
  then stays in `.cudacache/`, and only a source change rebuilds it.
  `VC_CUDA_VERBOSE=1 ./vc test 8` shows the compiler command and the register
  counts.
- On the JAX track, the default Mosaic GPU backend of Pallas needs sm_90 or
  more. So stage 08 goes through the older Triton backend. JAX gates that
  backend on a hardcoded allowlist of device kinds, and no laptop GPU is on the
  list. `jvllm/compat.py` reads the compute capability of your card and
  registers it. If neither backend compiles, the checks for stage 08 skip. They
  do not fail.
