# Overview

The full map of the course. You do not need it to start. The
[README](../README.md) and `./vc` show you the next step, and that is all
that you need on the first day. For the shape of the engine, read
[MAP.md](MAP.md). For the method, read [METHOD.md](METHOD.md).

---

## The Ladder: What You Build

**35 stages:** 33 core stages and 2 optional extensions (30 and 31). The core
stages are 01 to 29, plus 08b, 08c, 18b and 24b. The capstone is stage 21 to
28. `./vc` counts the same way: "3/33 stages, 0/2 extensions".

The stages are in 9 arcs. Arc A1 to A8 match Part 1 to Part 8 of the notebooks:

| Arc | Stages | What you build |
| :--- | :--- | :--- |
| **A1: The Naive Loop** | 01–03 | Naive quadratic loop, KV cache memoization, and the hardware roofline model |
| **A2: Batching** | 04–05 | Static batching padding waste, then continuous batching (the Orca iteration schedule) |
| **A3: PagedAttention** | 06–09 | **PagedAttention**: a block allocator and page tables (plus a model of the CUDA virtual-memory API), **3 custom CUDA kernels (08, 08b, 08c)**, and hash-chained prefix caching with copy-on-write |
| **A4: The Scheduler** | 10–11 | Admission against a KV budget, recompute/swap preemption, and chunked prefill with a token budget |
| **A5: How To Make It Fast** | 12–14 | CUDA Graphs decode replay with shape buckets, a batched sampler (temperature, top-k, top-p, penalties, seeds), and a streaming UTF-8 detokenizer |
| **A6: The Server** | 15–16 | Async worker loop, OpenAI-compatible HTTP server (`/v1/chat/completions`), and Prometheus telemetry (TTFT, ITL) |
| **A7: Modern vLLM**| 17–20 | Tree-attention speculative decoding, quantization (**+ 18b, a custom Int8 GEMV CUDA kernel**), JSON grammar masking, and Megatron-style Tensor Parallelism |
| **A8: The Capstone** | 21–28 | **The integrated server**: paged model, scheduler, graphs with pinned staging buffers, int8 weights, FP8 KV cache (24b), n-gram speculative decoding, guided JSON, a POSIX shared-memory event ring, and client disconnect abort |
| **A9: Beyond the Capstone** | 29–31 | LRU prefix cache eviction & cache-aware admission (29). **Optional extensions, with thin checks:** Multi-LoRA serving with batched GEMV (30), and DeepSeek Multi-Head Latent Attention (MLA) with decode weight absorption (31) |
| **Capstone Practicums** (notebooks, not stages) | I, J, K | **I:** a timeline with Nsight Systems, a notebook with checks. **J:** hardware counters with Nsight Compute, a notebook with checks. **K (optional reading):** map your engine onto TensorRT-LLM |

### Bare-Metal CUDA (Not Triton)
Five stages hold **CUDA that you write yourself** in real `.cu` files, compiled directly with `nvcc`:
- **Stage 08:** Paged attention decode kernel (`s08_paged_attn.cu`).
- **Stage 08b:** The same kernel, optimized for 128-byte DRAM transaction coalescing (`s08b_paged_attn_vec.cu`).
- **Stage 08c:** Intra-warp register shuffles (`__shfl_xor_sync`) and context split-K parallel reduction (`s08c_paged_attn_split.cu`).
- **Stage 18b:** Quantized Int8 matrix-vector multiply (GEMV) with the dequantization fused into the epilogue (`s18b_gemv_int8.cu`).
- **Stage 24b:** Paged attention operating over an FP8 KV cache (`s24b_kv_fp8.cu`).

---

## Measured Proof

The Capstone (Stages 21–28) connects your parts into one engine and gates your code against the hardware limits of your GPU. Every gate is measured on your card. [`docs/proof/`](proof/README.md) has one full run of every check, on one card, and `dev/proof.sh` makes it again on yours:

| Measurement | Reference Solution | The Gate |
| :--- | :--- | :--- |
| Tokens against HuggingFace in fp32 | Identical | Identical |
| Batch-1 decode step, against weight-read floor | 57% to 71% | 50% |
| CUDA graphs against eager, at batch 1 | 1.3x to 1.5x | 1.2x |
| Int8 weights against bf16, batch-1 step | 1.27x (a hot laptop card) to 1.55x | 1.2x |
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

You built the core ideas of single-GPU vLLM, in a smaller and slower form. Here is how your engine maps to upstream vLLM. Upstream changes fast: check the current source before you rely on a name in the right column.

| Architectural Component | Built in this course | Upstream Production Scope |
| :--- | :--- | :--- |
| **Paged KV Cache** | Custom CUDA kernels (coalesced, split-K, FP8) | FlashAttention-3 / FlashInfer backends |
| **Continuous Batching** | Dynamic iteration-level prefill & decode scheduler | Same architecture (Orca scheduler) |
| **Prefix Caching** | Hash-chained block cache (Stage 09) + LRU eviction under memory pressure (Stage 29) | Hash-based prefix caching of full blocks, with an LRU queue of free blocks (vLLM V1). SGLang uses a radix tree instead. |
| **Multi-LoRA Serving** | Multi-LoRA dispatch + Batched GEMV (Stage 30, optional) | Upstream Punica / BGMV CUDA kernels |
| **MLA Attention** | DeepSeek MLA + Decode Weight Absorption (Stage 31, optional) | Upstream DeepSeek-V2/V3 decode kernels |
| **Graph Dispatch** | CUDA Graphs with pinned staging buffers (Stage 12, 23) | CUDA graphs captured for a set of batch sizes. The class names change between versions. |
| **Speculative Decoding** | N-gram drafts, rejection sampling and a tree-attention mask (Stage 17); n-gram drafts in the engine (Stage 25) | N-gram, draft-model and EAGLE-style speculation |
| **Structured Output** | Automaton JSON grammar mask compilation (Stage 19, 26) | Outlines / XGrammar integration |
| **Server & IPC** | `/v1/chat/completions` + POSIX shared memory ring (Stage 27) | AsyncLLM engine & multi-process IPC |

#### What Requires Multi-GPU / Multi-Node Hardware
Features intentionally beyond the single-GPU scope of this course:
- **Multi-Node Cluster Distributed Systems:** Multi-node Ray clusters and NCCL collective fabrics (Stage 20 provides the single-node Tensor Parallelism algebra).
- **Mixture of Experts (MoE):** Fused MoE kernels across dozens of routed experts (e.g. DeepSeek/Mixtral 8x22B).
- **Vendor-Specific Assembly Tuning:** Hand-tuned SASS micro-optimizations found in vendor libraries (FlashAttention-3, FlashInfer).

---

## Dual-Track Architecture: PyTorch & JAX

```bash
./vc backend          # Check active track and stage progress
./vc backend jax      # Switch to the JAX track (progress banked per track)
./vc test 8 --jax     # Test on the JAX track without switching
```

- **PyTorch Track (33 core stages, and 2 optional extensions):** Compiles raw `.cu` files with `nvcc` and captures CUDA Graphs.
- **JAX Track (20 stages):** 11 stages in JAX (01–05, 07, 08, 12, 13, 18 and 20), plus the 9 framework-free stages that both tracks share. The capstone is torch only. Uses `jvllm/model.py`, Pallas GPU kernels, and static shape bucketing to eliminate XLA recompilation.
- **Framework-Free Stages:** 9 stages (the block allocator, prefix cache, scheduler, detokenizer, metrics, and speculative verification) contain zero tensor code—proving that the core architecture of an inference engine is pure systems logic.

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
- **[LORE.md](../LORE.md)** — The conceptual spine. Derivations of the memory wall, hardware profiling guides, and systems architecture.
- **[`course/GLOSSARY.md`](../course/GLOSSARY.md)** — Defines every systems and ML term with an analogy from traditional software engineering.

---

## CLI Reference

```bash
./vc list             # The full ladder with your progress
./vc guide 12         # Read the guide for any specific stage
./vc test 7           # Run checks for any specific stage
./vc info             # Print measured GPU bandwidth, TFLOPs, and roofline ridge
./vc cliff            # Measure and plot the L2-cache vs VRAM bandwidth cliff
./vc math 7 128       # Print read-vs-compute timings with every division written out
./vc nsys             # Trace continuous batching execution with NVIDIA Nsight Systems
./vc ncu 8 64         # Profile kernel hardware counters with NVIDIA Nsight Compute
./vc bench            # Benchmark your capstone against physical roofline limits
./vc serve            # Launch your OpenAI-compatible API server
./vc run              # Your parts in the live engine, with and without your newest part
./vc note "..."       # Write down a wrong prediction or a confusion; ./vc note reads them
```

---

## Verification

`dev/verify.sh` runs the checks of each stage against its reference solution:
```bash
dev/verify.sh          # Verify the complete test suite (torch track)
dev/verify.sh --jax    # Verify the JAX track
dev/verify.sh 6 8 28   # Verify specific stages
```
