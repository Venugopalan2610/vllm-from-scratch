# Build Your Own vLLM

> **Disclaimer.** This course is not affiliated with the [vLLM project](https://github.com/vllm-project/vllm) (Apache-2.0). It teaches the ideas behind vLLM V1's single-GPU architecture through a clean-room implementation. See [Production vLLM Parity](#production-vllm-parity-what-you-built-vs-upstream) below.

> **Do not stop. Continue. Be better than before.**

Stage 01 is the slowest inference engine that you will ever write. It is slow on purpose. For each new token, it computes all the earlier tokens again.

36 stages later, the same GPU runs a production-style inference server that you built from your own parts. Each stage between the two must be better than the stage before it: faster, or more correct. A measurement on your own GPU proves it. Nobody tells you that your code is good. The machine tells you.

---

## The questions that you will answer

- Your GPU can do tens of trillions of operations each second. Why does it spend most of a chat reply in a wait?
- Why do 32 concurrent users cost almost the same as 1 user?
- Why did vLLM take its central idea from the virtual memory subsystem of an operating system?
- How can a speculative guess make a model faster with zero change to its output distribution?
- Why can a faster CUDA kernel give you no end-to-end speedup at all?

You will not read these answers. You will measure them.

---

## The Book & The Build: Deriving Systems

This repository is the **build half** of an end-to-end curriculum.  
**[Deriving Systems](https://derivingsystems.com)** is the **derivation half**.

Its twelve chapters start from arithmetic that you can do on a napkin and work out mathematically why an inference engine must have this exact shape. Reading the chapter and building the stage are complementary halves of the same lesson:

- **[The Ridge](https://derivingsystems.com/07-the-ridge.html)** — Sets up Stages 01–03 (the memory wall, arithmetic intensity, and rooflines).
- **[The Cache That Ate the Batch](https://derivingsystems.com/08-kv-cache.html)** — Sets up Stage 02 (KV cache memoization).
- **[The Slot That Waited](https://derivingsystems.com/09-the-slot-that-waited.html)** — Sets up Stages 04–05 (continuous batching & iteration scheduling).
- **[A Page Table for Tokens](https://derivingsystems.com/10-a-page-table-for-tokens.html)** — Sets up Stages 06–09 (PagedAttention & virtual block allocation).
- **[Below the Floor](https://derivingsystems.com/11-below-the-floor.html)** — Sets up Stages 03 and 12 (CUDA Graphs and kernel launch bubbles).
- **[Spending the Idle](https://derivingsystems.com/12-spending-the-idle.html)** — Sets up Stage 17 (lossless speculative decoding).

Inside this repository, two key companion documents guide your implementation:
- **[LORE.md](LORE.md)** — The conceptual spine. 25 sections of mathematical proofs, memory wall derivations, hardware profiling guides, and systems architecture.
- **[`course/GLOSSARY.md`](course/GLOSSARY.md)** — Defines every systems and ML term with an analogy from traditional software engineering.

---

## The Ladder: What You Build

You build one complete, high-performance LLM serving system from first principles across 9 architectural arcs:

| Arc | Stages | What you build |
| :--- | :--- | :--- |
| **A0: The Memory Wall** | 01–03 | Naive quadratic loop, KV cache memoization, and the hardware roofline model |
| **A1: The Batching Shift** | 04–05 | Static batching padding waste, then continuous batching (the Orca iteration schedule) |
| **A2: PagedAttention & CUDA** | 06–09 | **PagedAttention**: virtual memory allocator (`cuMemMap`), **3 custom CUDA kernels (08, 08b, 08c)**, and radix-tree prefix caching |
| **A3: The Scheduler** | 10–11 | Priority admission, recompute/swap preemption, and chunked prefill with SM budgeting |
| **A4: Latency & Output** | 12–14 | CUDA Graphs decode replay (2 µs dispatch), batched rejection sampler, and streaming UTF-8 detokenizer |
| **A5: Production Serving** | 15–16 | Async worker loop, OpenAI-compatible HTTP server (`/v1/chat/completions`), and Prometheus telemetry (TTFT, ITL) |
| **A6: Advanced Acceleration**| 17–20 | Tree-attention speculative decoding, quantization (**+ 18b, a custom Int8 GEMV CUDA kernel**), JSON grammar masking, and Megatron-style Tensor Parallelism |
| **A7: The Capstone Engine** | 21–28 | **The integrated server**: paged model, scheduler, graphs with pinned staging buffers, int8 weights, FP8 KV cache (24b), tree speculation, zero-copy POSIX shared-memory IPC, and client disconnect abort |
| **A8: The Frontier Extension** | 29–32 | **Advanced single-GPU serving**: LRU prefix cache eviction & cache-aware admission (29), Multi-LoRA serving with Batched GEMV (30), DeepSeek Multi-Head Latent Attention (MLA) with decode weight absorption (31), and TypeSafe Jev System 1 typed decision models (32) |
| **Capstone Practicums** | I, J, K | **Whole-system timeline profiling (`./vc nsys`)**, **hardware counters & bank conflict analysis (`./vc ncu`)**, and **TensorRT-LLM architecture synthesis** |

### Bare-Metal CUDA (Not Triton)
Five stages hold **CUDA that you write yourself** in real `.cu` files, compiled directly with `nvcc`:
- **Stage 08:** Paged attention decode kernel (`s08_paged_attn.cu`).
- **Stage 08b:** The same kernel, optimized for 128-byte DRAM transaction coalescing (`s08b_paged_attn_vec.cu`).
- **Stage 08c:** Intra-warp register shuffles (`__shfl_xor_sync`) and context split-K parallel reduction (`s08c_paged_attn_split.cu`).
- **Stage 18b:** Quantized Int8 matrix-vector multiply (GEMV) with fused bias epilogue (`s18b_gemv_int8.cu`).
- **Stage 24b:** Paged attention operating over an FP8 KV cache (`s24b_kv_fp8.cu`).

---

## Measured Proof: Grounded Against Physics

The Capstone (Stages 21–28) connects your parts into one engine and gates your code against physical hardware limits on your exact GPU. Every gate is measured on your card:

| Measurement | Reference Solution | The Gate |
| :--- | :--- | :--- |
| Tokens against HuggingFace in fp32 | Identical | Identical |
| Batch-1 decode step, against weight-read floor | 57% to 71% | 50% |
| CUDA graphs against eager, at batch 1 | 1.3x to 1.5x | 1.2x |
| Int8 weights against bf16, batch-1 step | 1.47x to 1.55x | 1.3x |
| FP8 KV against bf16 KV, 16 x 2048 tokens | 1.33x to 1.46x | 1.2x |
| Int8 weights: KL from bf16 at generated tokens | 0.004 nats | 0.015 nats |
| FP8 KV cache: KL from bf16 through decode kernel | 0.009 nats | 0.03 nats |
| Speculative decoding: greedy tokens vs target | Identical | Identical |
| Speculative decoding on copy task, batch 1 | 1.5x to 3.1x | 1.5x |
| JSON mode: generated outputs that parse | 100% | 100% |
| The whole engine, against roofline floor | 39% to 59% | 30% |
| The whole engine, against Stage 05 naive batching | 3.2x to 4.6x | 2.0x |

To benchmark, inspect, and serve your engine on your machine:
```bash
./vc tui        # launch the interactive terminal workbench (TUI)
./vc bench      # tok/s, the roofline floor, and hardware ceiling of your card
./vc serve      # launch your server on localhost:8000
./vc nsys       # profile end-to-end continuous batching under Nsight Systems
```

### Production vLLM Parity: What You Built vs. Upstream

You built the complete single-GPU core of vLLM. Here is how your engine maps to upstream production vLLM:

| Architectural Component | Built in this course | Upstream Production Scope |
| :--- | :--- | :--- |
| **Paged KV Cache** | Custom CUDA kernels (coalesced, split-K, FP8) | FlashAttention-3 / FlashInfer backends |
| **Continuous Batching** | Dynamic iteration-level prefill & decode scheduler | Same architecture (Orca scheduler) |
| **Prefix Caching** | Radix cache + LRU eviction under memory pressure (Stage 29) | Upstream v1 BlockManager LRU cache |
| **Multi-LoRA Serving** | Multi-LoRA dispatch + Batched GEMV (Stage 30) | Upstream Punica / BGMV CUDA kernels |
| **MLA Attention** | DeepSeek MLA + Decode Weight Absorption (Stage 31) | Upstream DeepSeek-V2/V3 decode kernels |
| **Graph Dispatch** | CUDA Graphs with pinned staging buffers (Stage 12, 23) | Same architecture (`CUDAGraphRunner`) |
| **Speculative Decoding** | Multi-branch Tree-Attention + n-gram drafts (Stage 17, 25) | Speculative decoding framework |
| **Structured Output** | Automaton JSON grammar mask compilation (Stage 19, 26) | Outlines / XGrammar integration |
| **Server & IPC** | `/v1/chat/completions` + POSIX shared memory ring (Stage 27) | AsyncLLM engine & multi-process IPC |
| **System 1 Decision Models** | Clean-room TypeSafe Jev architecture (Choice, Score, Noul) with zero KV-cache overhead (Stage 32) | Front-door admission, fast guardrails, and dynamic routing |

> *Note on Jev (Stage 32): TypeSafe AI's commercial Jev weights are closed-source/hosted. Stage 32 implements a clean-room architecture of the published Jev System 1 specification (Choice, Score, Noul), which can be mounted onto open encoder backbones (e.g. `microsoft/deberta-v3-large` or community reproductions) for local deployment.*

#### What Requires Multi-GPU / Multi-Node Hardware
Features intentionally beyond the single-GPU scope of this course:
- **Multi-Node Cluster Distributed Systems:** Multi-node Ray clusters and NCCL collective fabrics (Stage 20 provides the single-node Tensor Parallelism algebra).
- **Mixture of Experts (MoE):** Fused MoE kernels across dozens of routed experts (e.g. DeepSeek/Mixtral 8x22B).
- **Vendor-Specific Assembly Tuning:** Hand-tuned SASS micro-optimizations found in vendor libraries (FlashAttention-3, FlashInfer).

---

## Quickstart

### Option A: Launch with 1-Click (No Local GPU Required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Venugopalan2610/vllm-from-scratch/blob/master/colab.ipynb)

A free Google Colab T4 GPU (sm_75) runs all pure-logic stages and CUDA kernel stages. For the full Capstone speed gates and native FP8 hardware, use an L4 (sm_89) or A100 (sm_80) instance.

### Option B: Run Locally on Linux

**Requirements:** Linux, an NVIDIA GPU with $\ge 10\text{ GB}$ VRAM, the CUDA Toolkit (`nvcc`), Python 3.12, and $\approx 9\text{ GB}$ of disk space.

The notebooks use Qwen3-1.7B. The graded checks use Qwen3-0.6B, because many checks hold the model in bf16 and in fp32 at the same time, and 1.7B does not fit twice on a 12 GB card. With 24 GB or more, run the checks on 1.7B with `VC_MODEL=Qwen/Qwen3-1.7B`.

```bash
# 1. Clone your fork
git clone https://github.com/Venugopalan2610/vllm-from-scratch.git
cd vllm-from-scratch

# 2. Run setup (~3-5 minutes, creates .venv and prepares the models)
./setup.sh

# 3. Inspect your GPU's physical roofline limits
./vc info
```

---

## The Development Loop

The course uses a unified CLI (`./vc`) to guide your implementation and verify correctness:

```bash
./vc              # Where am I on the ladder?
./vc guide        # What to build for this stage, why it matters, and the spec
                  # ... edit the file named in app/ ...
./vc test         # Run the checks against your code
./vc submit       # All green? Commits your work and unlocks the next stage
```

### The Rules
1. **You edit `app/`. You never edit `tests/`.** The checks are the specification.
2. **Build the slow version first, then measure.** Every improvement is a number, not an opinion.
3. **Reference solutions:** If you are stuck after repeated attempts, reveal the reference solution:
   ```bash
   ./vc peek             # View reference solution for your current stage
   ./vc peek 8           # View reference solution for stage 8
   ./vc peek 8 --apply   # Apply the reference solution to your workspace
   ```
   *A stage where you study the answer and understand the mechanism is better than an abandoned repository.*

---

## Dual-Track Architecture: PyTorch & JAX

```bash
./vc backend          # Check active track and stage progress
./vc backend jax      # Switch to the JAX track (progress banked per track)
./vc test 8 --jax     # Test on the JAX track without switching
```

- **PyTorch Track (32 stages):** Compiles raw `.cu` files with `nvcc` and captures CUDA Graphs.
- **JAX Track (20 stages):** Uses `jvllm/model.py`, Pallas GPU kernels, and static shape bucketing to eliminate XLA recompilation.
- **Framework-Free Stages:** 9 stages (the block allocator, prefix cache, scheduler, detokenizer, metrics, and speculative verification) contain zero tensor code—proving that the core architecture of an inference engine is pure systems logic.

---

## CLI Reference

```bash
./vc list             # The full 32-stage ladder with your progress
./vc guide 12         # Read the guide for any specific stage
./vc test 7           # Run checks for any specific stage
./vc info             # Print measured GPU bandwidth, TFLOPs, and roofline ridge
./vc cliff            # Measure and plot the L2-cache vs VRAM bandwidth cliff
./vc math 7 128       # Print read-vs-compute timings with every division written out
./vc nsys             # Trace continuous batching execution with NVIDIA Nsight Systems
./vc ncu 8 64         # Profile kernel hardware counters with NVIDIA Nsight Compute
./vc bench            # Benchmark your capstone against physical roofline limits
./vc serve            # Launch your production OpenAI-compatible API server
```

---

## Verification

Every stage in this repository is mathematically grounded and tested against reference solutions:
```bash
dev/verify.sh          # Verify the complete test suite (torch track)
dev/verify.sh --jax    # Verify the JAX track
dev/verify.sh 6 8 28   # Verify specific stages
```
