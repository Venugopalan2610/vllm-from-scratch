# Build Your Own vLLM

> **Disclaimer.** This course is not affiliated with the
> [vLLM project](https://github.com/vllm-project/vllm) (Apache-2.0). It
> teaches the ideas behind vLLM V1's single-GPU architecture through a
> clean-room implementation. The result is a teaching engine, not a
> production system. See [What you did not build](#what-you-did-not-build)
> below.

> **Do not stop. Continue. Be better than before.**

Stage 01 is the slowest inference engine that you will ever write. It is slow
on purpose. For each new token, it computes all the earlier tokens again.

31 stages later, the same GPU runs a server that you built from your own
parts. Each stage between the two must be better than the stage before it:
faster, or more correct. A measurement on your own GPU proves it. Nobody
tells you that your code is good. The machine tells you.

## The questions that you will answer

- Your GPU can do tens of trillions of operations each second. Why does it
  spend most of a chat reply in a wait?
- Why do 32 users cost almost the same as 1 user?
- Why did vLLM take its central idea from the virtual memory of an operating
  system?
- How can a guess make a model faster, with no change to one word of its
  output?
- Why can a faster kernel give you no speedup at all?

You will not read these answers. You will measure them.

## What you build

One inference server, from your own parts:

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

Five stages are **CUDA that you write yourself**:

- a paged attention kernel,
- the same kernel, made to coalesce,
- warp shuffles and split-K,
- a quantized GEMV with a fused epilogue,
- the attention kernel again, on an FP8 cache.

These are real `.cu` files, real `nvcc`, and real register counts. This is not
Triton.

Tensor parallelism is the one part that stays a separate stage. It needs two
GPUs to be real, and the course runs on one consumer GPU. You can use
**PyTorch or JAX**: the same ladder on two backends.

The derivations behind these stages are at
[derivingsystems.com](https://derivingsystems.com).
[The Course](https://derivingsystems.com/course.html) puts the whole ladder on
one page.

## Start here

1. **Read this page to the end of "The philosophy".** Then stop reading and
   start.
2. **Set up.** Run `./setup.sh`, then `./vc info` to see what your GPU can do.
   If you have no GPU, use [Colab](#start-with-no-installation).
3. **Is ML or GPU work new to you?** Then do Part 0 first:
   `course/Part0_FromAProgramToAModel/`. It explains a model, attention and a
   GPU to a software engineer. [`course/GLOSSARY.md`](course/GLOSSARY.md)
   defines each term of the course, with an analogy from software.
4. **Run `./vc`.** It shows where you are. `./vc guide` shows the new words of
   the stage, why the stage exists, and what to build.
5. **Do the loop.** Edit the file in `app/`, run `./vc test`, and run
   `./vc submit` when all the checks pass. Read the notebooks of a stage before
   you build it.

## The philosophy

**Do not stop. Continue. Be better than before.** Each of the rules below comes
from that one sentence.

- **Be better than before.** Each stage must beat the stage before it, and a
  measurement decides. `./vc test` compares each run with your best run, so
  you always know if you went forward.
- **Build the slow version first, and measure it.** Then each improvement is
  a number, not an opinion.
- **Predict, then measure, then explain the difference.** A prediction that is
  wrong teaches you the most. The difference is where the hardware lesson is.
- **The checks are the spec.** You edit `app/`. You never edit `tests/`.
- **Do not stop when you are stuck. Change the approach.** After 5 runs with
  no new pass, `./vc` gives you three ways forward. One of them is to read the
  answer. A stage that you read the answer to is better than a repo that you
  stopped.

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
| int8 weights: KL from bf16 at the generated tokens (ISL 256, OSL 64) | 0.004 nats | 0.015 nats |
| FP8 KV cache: KL from bf16, through the decode kernel | 0.009 nats | 0.03 nats |
| speculative decoding: greedy tokens against no speculation | identical | identical |
| speculative decoding on a copy task, batch 1 | 1.5x to 3.1x | 1.5x |
| JSON mode: answers that parse | all | all |
| the whole engine, against the roofline floor of its own steps | 39% to 59% | 30% |
| the whole engine, against your stage 05 engine, same requests | 3.2x to 4.6x | 2x |


> [!NOTE]
> These measurements run at 128 to 2048 tokens. At 8k and above, the KV read
> dominates the weight read, block tables grow, flash-decoding matters, the
> prefix cache hit rate decides capacity, and the FP8 KV gain appears. A 0.6B
> model at 16k is cheap. Try `VC_MODEL=Qwen/Qwen3-0.6B ./vc bench --seqs 4`
> with long prompts to see the regime where your roofline argument flips.

The ranges come from repeated runs on one laptop GPU. A laptop throttles, so
the same code moves by 20% from run to run. That is why every gate compares
two things that the check measures one after the other.

To see the absolute numbers for your card, finish the capstone and run:

```bash
./vc bench      # tok/s, the roofline floor, and the ceiling of your card
./vc serve      # your server on localhost:8000, for curl
```

### What you did not build

| Production vLLM feature | Status in this course |
|---|---|
| Multi-process engine with ZMQ | ❌ Engine runs in-process |
| FlashAttention / FlashInfer backends | ❌ Your own CUDA kernel (slower) |
| Dozens of model architectures | Qwen3 and Llama families only |
| LoRA, multi-LoRA serving | ❌ Not covered |
| Mixture-of-Experts (MoE), expert parallelism | ❌ Not covered |
| Pipeline parallelism, multi-node | ❌ TP algebra only, simulated on one GPU |
| Cache-aware scheduling | Partial — prefix cache, no LRU-guided admission |
| Automatic memory profiling for pool sizing | ❌ Pool size is a manual constant |
| Beam search, `n > 1` sampling | ❌ Copy-on-write built but not used in engine |
| Logprobs, `min_p`, `bad_words`, tool calls | ❌ Not in the API |
| EAGLE / MTP draft models | ❌ N-gram drafting only |
| Sliding-window / MLA attention | ❌ Global attention only |
| Robust error handling (OOM, timeouts, watchdog) | ❌ Not covered |
| Batch-invariant decoding | ❌ Known to differ (see Criticism #10) |
| Goodput measurement at concurrency | ❌ Benchmark uses batch-at-once |

The course teaches the *ideas*. Production vLLM has hundreds of thousands of
lines that make those ideas survive real traffic. If you say "I built vLLM" in
an interview, name these gaps first.

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

**Why two families matter.** Qwen3-0.6B has GQA 16:8, head_dim 128 and tied
embeddings. Llama-3.2-1B has GQA 32:8, head_dim 64 and an untied lm_head.
Running `VC_MODEL=meta-llama/Llama-3.2-1B ./vc test` on your capstone finds
bugs that one config hides: hardcoded head_dim, wrong GQA ratio in the kernel,
and a cache layout that only one shape proves. If your code passes on both,
the shapes are parametric.

## Start with no installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Venugopalan2610/vllm-from-scratch/blob/master/colab.ipynb)

Colab gives you a CUDA toolkit and a GPU, so the kernel stages compile.
A **free T4** (sm_75) runs the pure-logic and CUDA stages but has **no native
bf16 tensor cores and no FP8 hardware**, so the capstone speed gates are
unreliable and stage 24b (FP8 KV) will not run. For the full capstone, use
a **Colab L4** (sm_89) or **A100** (sm_80) runtime. Stage 1 is approximately
three minutes away on any tier.

A Colab runtime is temporary, so your work dies with the session. The last two
cells of the notebook save it. They download it, or they push it to your fork.
After approximately stage 5, work locally instead.

## Or start locally

**Budget.** The course needs approximately **40 to 60 hours** of focused work,
a GPU with `nvcc` (CUDA toolkit), **≥ 10 GB of VRAM** for the capstone, and
**≈ 5 GB of disk** for the venv, model weights and build cache. A free Colab
GPU is enough for the first half. The capstone speed gates need at least an L4.

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

Do you have a GPU but no `nvcc`? Everything runs except the five CUDA stages.
Those stages skip. They do not fail.

## Two ways in

The notebooks in [`course/`](course/) build the intuition. The ladder in `app/`
makes you build the engine. They have an order. Read the notebooks for a stage,
then build the stage. The numbers in the names give the order of the
notebooks, from Part 0 to Part 8. [`course/README.md`](course/README.md#the-order)
explains it.

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


> [!NOTE]
> The JAX track is valuable for the "shapes are frozen" insight, but vLLM
> is torch. Maintaining two tracks (92 notebooks, two test ladders) is
> expensive. If you have limited time, do the torch track and the nine
> shared stages. The JAX track adds understanding, not coverage.

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
stop. A stage that you read the answer to is better than a repo that you
abandoned.

Read the answer, close it, and write the code yourself. Then continue.

You can also browse the solutions directly:

```bash
git show solutions:.solutions/s07_paged_attn.py
```

## Rules

- You edit `app/`. You never edit `tests/`. The checks are the spec.
- Progress lives in `.progress.json`, which git ignores. Delete it to start
  again.

**All 776 torch checks and all 589 JAX checks pass against the reference
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
The vocabulary is in [`course/GLOSSARY.md`](course/GLOSSARY.md).

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
