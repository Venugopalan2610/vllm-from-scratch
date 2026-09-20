# Part 8, The Capstone

Stages: stages 21-28, 24b.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._helper`). Its solution is in `solutions/`.

Stages 24 to 27 put quantization, speculative decoding and guided decoding
into the engine. The Part 7 notebooks teach those features.

| | |
|---|---|
| [`1_engine/`](1_engine/) | Why the parts never met, and the flat batch that lets them. One scheduler rule for every case. Then build that scheduler. **The challenge needs no GPU.** |
| [`2_graphs/`](2_graphs/) | A thousand small kernels in each step, against the weight-read floor. The padding row that overwrites block 0. |
| [`3_roof/`](3_roof/) | The floor of a step, in tokens for each GB/s, and why a larger model can serve more tokens. Then measure a real step against it. |
| [`4_nsys/`](4_nsys/) | **Capstone Practicum I: Timeline Profiling with Nsight Systems.** Trace `./vc nsys`, measure CPU dispatch bubbles, and verify stream concurrency and NVTX step markers. |
| [`5_ncu/`](5_ncu/) | **Capstone Practicum J: Kernel Micro-Architecture with Nsight Compute.** Run `./vc ncu 8 64`, extract the 5 golden hardware counters, measure DRAM throughput vs theoretical peak, and eliminate bank conflicts. |
| [`6_trtllm/`](6_trtllm/) | **Capstone Practicum K: Dual-Engine Architecture Synthesis.** Map your `Scheduler`, `KVBlockManager`, and custom CUDA kernels into NVIDIA TensorRT-LLM's C++ `GptManager` and `KvCacheManager`. |

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.

## Capstone Practicums: Profiling, Hardware Counters & Production Synthesis

The capstone is not complete when the Python tests pass. An industrial inference engine
must be verified against physical silicon, CPU launch overheads, and production C++ architectures.
These three practicums are integral requirements of Part 8:

1. **Practicum I (`4_nsys/`): Whole-System Timeline Profiling (`./vc nsys`)**
   Trace continuous batching execution with NVTX annotations. Quantify CPU dispatch bubbles
   between decode iterations and verify pinned memory staging buffer concurrency.
2. **Practicum J (`5_ncu/`): Kernel Micro-Architecture Profiling (`./vc ncu 8 64`)**
   Profile your Stage 08 paged-attention kernel under Nsight Compute. Measure achieved DRAM
   throughput against your card's roofline ceiling and verify bank-conflict-free shared memory access.
3. **Practicum K (`6_trtllm/`): TensorRT-LLM Architecture Synthesis**
   Map your scheduler, block allocator, and attention kernels directly into NVIDIA TensorRT-LLM's
   C++ `GptManager` and `KvCacheManager`.
4. **Multi-turn serving & Prefix Caching:** Send multi-turn requests to your
   server and observe the prefix cache hit rate on consecutive turns.
5. **Batch Invariance:** Measure token differences between batch 1 and batch 32
   due to non-associative floating-point addition in GEMM reductions (see LORE.md §12).
6. **Long Context Roofline:** Measure your engine at 16k context where KV reads
   overtake weight reads, flipping the roofline constraint.
7. **Upstream Source Study:** Read `vllm/v1/core/sched/scheduler.py` in the
   upstream repository and map the five major production capabilities
   (LoRA allocation, multi-step scheduling, priority queues, guided grammar
   state, and memory profiling).

---

## Practicum I Deep Dive: Profiling with Nsight Systems

Production systems engineers do not guess where latency comes from; they profile.
This repo equips you with direct CLI commands for NVIDIA's core profilers:

### Whole-System Timeline Profiling: `./vc nsys` (Nsight Systems)
Nsight Systems profiles the interaction between CPU application code, the CUDA runtime/driver,
and GPU execution queues across all CUDA streams:

```bash
# Run the continuous batching engine under Nsight Systems
./vc nsys
```

This generates `.cudacache/engine_profile.nsys-rep`. Inspect it with `nsys-ui`:
1. **The Execution Bubble (CPU Dispatch Overhead):**
   - Inspect the gap between the end of step $N$'s attention kernel and the launch of step $N+1$'s GEMM.
   - If there is a >100 µs gap where the GPU is idle, Python's GIL, memory allocations, or detokenization
     are starving the GPU. That is why Stage 27 built `ProcessEngineWorker` and zero-copy shared memory IPC.
2. **CUDA Stream Concurrency & HtoD Transfers:**
   - Verify whether Host-to-Device memory copies (`slot_mapping`, `block_tables`) overlap with kernel
     execution or block the compute stream. Stage 23 solved blocking transfers with pinned `StagingBuffers`.
3. **NVTX Annotations:**
   - Steps in `./vc nsys` are wrapped with `torch.cuda.nvtx.range_push("engine_step_X")`. In the timeline,
     you can see exactly which phase (scheduling vs model forward vs sampling) took how many microseconds.

---

## TensorRT-LLM vs. vLLM: The Dual-Engine Mapping

Everything you built in `vllm-from-scratch` maps directly to NVIDIA's production inference framework, **TensorRT-LLM**:

| Concept | `vllm-from-scratch` | Upstream vLLM | NVIDIA TensorRT-LLM |
| :--- | :--- | :--- | :--- |
| **Continuous Batching Scheduler** | `Scheduler` in `app/s22_engine.py` | `Scheduler` in `vllm/v1/core/sched/` | `GptManager` / `BatchingManager` |
| **KV Block Allocator** | `KVBlockManager` in `app/s22_engine.py` | `BlockPool` in `vllm/v1/core/` | `KvCacheManager` (in `src/kvCacheManager.cpp`) |
| **Attention Kernel Backend** | `paged_attention_cuda` in Stage 08 | FlashInfer / FlashAttention | `PagedKvCache` + `DecoderMaskedMultiheadAttentionPlugin` |
| **Execution Acceleration** | `GraphedModelRunner` in Stage 23 | CUDA Graph Runner | TensorRT Engine Plan (`.engine`) |
| **Speculative Decoding** | `rejection_sample` + `build_tree_mask` | `vllm/spec_decode/` | Medusa / Eagle Decoding Plugin |
| **Multi-GPU Parallelism** | `ColumnParallelLinear` / `RowParallelLinear` | `vllm/model_executor/` | `NcclPlugin` (Tensor Parallelism via NCCL) |

---

## Ten Systems Engineering Interview Scenarios (And How You Defend Them)

These 10 technical defense scenarios test whether you truly understand the physical hardware
and systems engineering behind high-performance LLM serving:

### Q1: "Why is LLM decode memory-bandwidth bound, and what does that imply for GPU hardware utilization?"
**Your Defense:** In autoregressive decode, the sequence generates one token per step ($M=1$). The GPU must read all model weights (e.g. 14 GB for a 7B bf16 model) from HBM to registers to perform a single GEMV operation. The arithmetic intensity is:
$$\text{Arithmetic Intensity} = \frac{2 \times 7\times 10^9 \text{ FLOPs}}{14\times 10^9 \text{ Bytes}} = 1.0\text{ FLOP/byte}$$
On an H100 with 3.35 TB/s HBM3 bandwidth and 2,000 TFLOPs FP16 compute, the hardware roofline ceiling at batch 1 is:
$$\text{Max Throughput} = 3.35\times 10^{12}\text{ B/s} \times 1.0\text{ FLOP/B} = 3.35\text{ TFLOP/s}$$
That is less than **0.2% of the GPU's compute capability**. The GPU compute units sit idle waiting for memory. To increase compute utilization, you must batch requests together so weights are read once and reused across $B$ tokens.

### Q2: "What causes a shared memory bank conflict, and how do you diagnose and resolve it?"
**Your Defense:** Shared memory is organized into 32 banks, each 4 bytes (32 bits) wide. Successive 32-bit words map to successive banks. A bank conflict occurs when two or more threads in the same warp access different memory addresses within the same bank simultaneously. When this happens, the hardware serializes the memory accesses, multiplying latency.
In Nsight Compute, track `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`.
To resolve it:
1. **Padding:** Add a dummy column (e.g. `__shared__ half smem[WARP_SIZE][HEAD_DIM + 1]`) to shift bank assignments.
2. **Swizzling:** XOR the column index with the row index so that threads in a warp access orthogonal banks.

### Q3: "Explain how CUDA Graphs accelerate decoding, and what hazards occur when integrating them with dynamic KV caching."
**Your Defense:** At small batch sizes, a single decode step consists of dozens of small kernel launches (GEMV, layer norm, RoPE, attention). Launching them from the host costs 3–5 µs per kernel in CPU driver overhead, creating gaps on the GPU timeline. A CUDA Graph captures the entire sequence into a static hardware command buffer, allowing the host to replay the entire graph with a single `cudaGraphLaunch` call (<5 µs total).
The hazard with prefix caching:
CUDA Graphs fix memory addresses at capture time. When prefix caching dynamically allocates or shares physical blocks:
1. Padded rows must write to slot `-1` and read context `0`. If a padding row uses slot `0`, it will overwrite block 0, which belongs to a live request.
2. The host must update static `block_tables` and `slot_mapping` tensors via pre-allocated pinned staging buffers (`StagingBuffers`), and synchronize with the CUDA stream before calling `graph.replay()` so kernels never read stale pointers.

### Q4: "What is Warp Divergence, and how do you minimize it in reduction kernels?"
**Your Defense:** A warp of 32 threads executes in lockstep (SIMT). If threads within a warp take different branches of an `if/else` condition, both paths are executed sequentially with inactive threads masked out. In reduction trees (like softmax in attention), branch conditions like `if (tid < stride)` can cause divergence. We eliminate divergence by using warp-shuffle intrinsics (`__shfl_xor_sync` / `__shfl_down_sync`) to pass values directly between registers across warp lanes without shared memory or branch divergence.

### Q5: "How does PagedAttention eliminate memory fragmentation compared to contiguous allocation?"
**Your Defense:** In naive serving, each request pre-allocates a contiguous tensor sized for `max_model_len` (e.g. 4096 tokens). Because most requests terminate early (e.g. at 200 tokens), up to 80% of VRAM is wasted in **internal fragmentation**. Furthermore, memory cannot be shared across sequences.
PagedAttention mimics OS virtual memory:
- Physical KV memory is partitioned into fixed-size blocks (e.g. 16 tokens).
- A sequence maintains a logical block table.
- Blocks are allocated on-demand as tokens are generated.
- Internal waste is bounded by `block_size - 1` tokens.
- Sequences sharing common prefixes point to the same physical blocks via reference counting, enabling zero-copy block reuse.

### Q6: "Why does Python multiprocessing matter for an LLM serving engine, and what is the bottleneck of `mp.Queue`?"
**Your Defense:** In Python, the Global Interpreter Lock (GIL) prevents multiple native threads from executing Python bytecodes simultaneously. If FastAPI request deserialization, detokenization, and the GPU step loop run in the same process, detokenizing a large batch blocks the thread from launching the next GPU step, creating execution bubbles on the GPU timeline.
Separating the engine into a child OS process (`ProcessEngineWorker`) isolates the GPU step loop.
However, standard Python `mp.Queue` serializes every object using `pickle`. At 5,000 tokens/sec, `pickle.dumps` and `loads` eat 30-50% CPU. In production, we eliminate `pickle` by writing fixed-size binary structs directly into POSIX `SharedMemory` ring buffers (`SharedMemoryEventRing`) or using ZeroMQ IPC.

### Q7: "What is the difference between Time-To-First-Token (TTFT) and Inter-Token-Latency (ITL), and how does Chunked Prefill balance them?"
**Your Defense:** TTFT is the latency from request arrival until the first token is emitted (dominated by the compute-bound prefill pass over the entire prompt). ITL is the latency between consecutive emitted tokens during decode (memory-bandwidth bound).
In naive serving, a new 4,000-token prompt runs as one gigantic GEMM that occupies 100% of SMs for 100 ms, causing an unacceptable 100 ms latency spike for all concurrent decode streams.
**Chunked Prefill** bounds prompt execution to a token budget (e.g. 512 tokens per step). The prefill is evaluated over multiple steps alongside decode tokens, keeping ITL flat while maintaining high overall throughput.

### Q8: "How does Tree-Attention speculative decoding differ from standard n-gram speculation?"
**Your Defense:** Standard linear speculation proposes $K$ tokens in a single chain. The acceptance probability decays exponentially ($a^K$), so beyond $K=3$ or $4$, compute is wasted on rejected tokens.
In **Tree-Attention Speculative Decoding** (Medusa/EAGLE):
- The draft model produces a tree of multiple candidate branches.
- All tree nodes are verified in a **single forward pass** of the target model.
- A custom 2D attention mask ensures candidate tokens only attend to their ancestors in the tree and the prompt history, isolating sibling branches from each other.
- The engine identifies the longest valid path in the tree and emits those tokens plus a bonus token from the winning branch.

### Q9: "What is the difference between Pinned Host Memory and Pageable Host Memory during CUDA transfers?"
**Your Defense:** Pageable memory is managed by the OS virtual memory subsystem and can be swapped to disk. The GPU DMA engine cannot access pageable memory directly because the physical page addresses could change mid-transfer. When `cudaMemcpy` is called on pageable memory, the CUDA driver first copies the data into an internal pinned staging buffer, then initiates DMA to the GPU, making the transfer synchronous and doubling CPU work.
**Pinned (Page-Locked) Memory** (`cudaMallocHost` or `torch.empty(..., pin_memory=True)`) locks the physical pages in RAM, allowing the GPU DMA engine to transfer data asynchronously via PCIe without CPU intervention (`cudaMemcpyAsync`), overlapping data transfer with GPU computation.

### Q10: "If an Nsight Systems profile shows high GPU utilization but low Goodput, what are the primary root causes?"
**Your Defense:** High GPU utilization with low goodput means the hardware is executing operations that do not produce SLA-compliant output tokens. Common root causes:
1. **Preemption Thrashing:** The block pool is undersized for the admitted concurrency. The scheduler repeatedly preempts sequences, evicts their KV blocks, and later recomputes them from scratch. Compute is burned recomputing tokens rather than emitting new ones.
2. **Prefill/Decode Interference:** Unbounded prefill passes starve decode passes, causing ITL to violate user SLOs (>50 ms), so all generated tokens count as waste.
3. **Speculative Decoding Rejection Storms:** The draft model's acceptance rate has collapsed, so the target model spends FLOPs verifying draft candidates that are 100% rejected.
