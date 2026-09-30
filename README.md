# Build Your Own vLLM

> **Disclaimer.** This course is not affiliated with the [vLLM project](https://github.com/vllm-project/vllm) (Apache-2.0). It teaches the ideas behind vLLM V1's single-GPU architecture through a clean-room implementation. See [what you build against upstream vLLM](docs/OVERVIEW.md#production-vllm-parity-what-you-built-vs-upstream).

> **Do not stop. Continue. Be better than before.**

Stage 01 is the slowest inference engine that you will ever write. It is slow on purpose. For each new token, it computes all the earlier tokens again.

At stage 28, the capstone, the same GPU runs an OpenAI-compatible inference server that you built from your own parts. Each stage between the two must be better than the stage before it: faster, or more correct. A measurement on your own GPU proves it. Nobody tells you that your code is good. The machine tells you.

You do not need to see the whole course to start. You need the next step, and `./vc` always shows it.

---

## The questions that you will answer

- Your GPU can do tens of trillions of operations each second. Why does it spend most of a chat reply in a wait?
- Why do 32 concurrent users cost almost the same as 1 user?
- Why did vLLM take its central idea from the virtual memory subsystem of an operating system?
- How can a speculative guess make a model faster with zero change to its output distribution?
- Why can a faster CUDA kernel give you no end-to-end speedup at all?

You will not read these answers. You will measure them.

---

## Quickstart

### Option A: Colab (no local GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Venugopalan2610/vllm-from-scratch/blob/master/colab.ipynb)

A free Google Colab T4 GPU (sm_75) runs all pure-logic stages and CUDA kernel stages. For the full Capstone speed gates and native FP8 hardware, use an L4 (sm_89) or A100 (sm_80) instance.

### Option B: Linux, with your own GPU

**Requirements:** Linux, an NVIDIA GPU with $\ge 10\text{ GB}$ VRAM, the CUDA Toolkit (`nvcc`), Python 3.12, and $\approx 9\text{ GB}$ of disk space.

```bash
git clone https://github.com/Venugopalan2610/vllm-from-scratch.git
cd vllm-from-scratch
./setup.sh      # ~3-5 minutes: creates .venv and prepares the models
./vc            # where you are, and what to do next
```

The notebooks use Qwen3-1.7B. The graded checks use Qwen3-0.6B, because many checks hold the model in bf16 and in fp32 at the same time, and 1.7B does not fit twice on a 12 GB card. With 24 GB or more, run the checks on 1.7B with `VC_MODEL=Qwen/Qwen3-1.7B`.

---

## Where to start

Read [`docs/MAP.md`](docs/MAP.md) first. It is one page: the whole engine in
the words of a web service (router, service, repo, database), and the box
that each stage builds. Then open Part 0 of the notebooks, which starts with
the same map.

---

## The loop

Each stage is the same four commands:

```bash
./vc guide        # what to build, why it matters, and the notebooks to read first
                  # ... edit the file in app/ that the guide names ...
./vc test         # run the checks against your code
./vc submit       # all green? commit your work and open the next stage
./vc run          # from stage 06: your parts inside one live engine
./vc note "..."   # a prediction that was wrong, or a sentence that confused you
```

Three rules:

1. **You edit `app/`. You never edit `tests/`.** The checks are the specification.
2. **Build the slow version first, then measure.** Every improvement is a number, not an opinion.
3. **If you are stuck after repeated attempts, `./vc peek` shows the reference solution.** A stage where you study the answer and understand the mechanism is better than an abandoned repository.

---

## When you want more

Open these when you need them, not before:

| | |
| :--- | :--- |
| [`docs/METHOD.md`](docs/METHOD.md) | The method: what to predict when your mind is empty, and six moves. The engine stays here. The method goes with you. |
| [`course/`](course/README.md) | The notebooks. Read the notebooks of a Part before you build its stages. `./vc guide` names them. |
| [`course/GLOSSARY.md`](course/GLOSSARY.md) | Every systems and ML term, with an analogy from software engineering. |
| [`LORE.md`](LORE.md) | The derivations behind the stages. `./vc lore` prints the one-paragraph insight of your stage. |
| [Deriving Systems](https://derivingsystems.com) | The book. It derives, on paper, why the engine must have this shape. |
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | The full map: all 35 stages, the measured gates, the JAX track, the parity with upstream vLLM, and every `./vc` command. |
