# Capstone Practicum K: Dual-Engine Architecture Synthesis (TensorRT-LLM)

> *"The principles of continuous batching, paged memory, and graph execution are identical across runtimes. Only the language and execution engine change."*

## Objective
Synthesize what you built across all 32 stages and map your Python/CUDA architecture directly into NVIDIA's production C++ LLM serving runtime, **TensorRT-LLM**.

## Source Reference
Open the open-source TensorRT-LLM repository:
[`NVIDIA/TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM)

## The Dual-Engine Mapping

| Component | `vllm-from-scratch` Capstone | Upstream vLLM V1 | NVIDIA TensorRT-LLM |
| :--- | :--- | :--- | :--- |
| **Batching Scheduler** | `Scheduler` in `app/s22_engine.py` | `Scheduler` in `vllm/v1/core/sched/` | `GptManager` / `BatchingManager` |
| **KV Memory Manager** | `KVBlockManager` in `app/s22_engine.py` | `BlockPool` in `vllm/v1/core/` | `KvCacheManager` (in `src/kvCacheManager.cpp`) |
| **Attention Backend** | `paged_attention_cuda` (Stages 08, 08b, 08c) | FlashInfer / FlashAttention-3 | `PagedKvCache` + `DecoderMaskedMultiheadAttentionPlugin` |
| **Graph Execution** | `GraphedModelRunner` (Stage 23) | CUDA Graph Runner | TensorRT Engine Plan (`.engine`) |
| **Speculative Decoding** | `rejection_sample` + `build_tree_mask` (Stage 17) | `vllm/spec_decode/` | Medusa / EAGLE Decoding Plugin |
| **Multi-GPU Parallelism**| `ColumnParallelLinear` / `RowParallelLinear` (Stage 20)| `vllm/model_executor/` | `NcclPlugin` (Tensor Parallelism via NCCL) |

## What You Must Deliver
1. **Trace `src/kvCacheManager.cpp` in TensorRT-LLM:**
   - Locate the block allocation logic and reference counting. Compare it line-by-line with your `BlockAllocator` in `app/s06_blocks.py`.
2. **Trace `GptManager` in TensorRT-LLM:**
   - Identify where in-flight batching decides between admitting a new prefill request vs scheduling ongoing decodes.
   - Compare its token budget rule with your `Scheduler.schedule()` implementation in `app/s22_engine.py`.
3. **C++ Systems Architecture Review:**
   - Explain how TensorRT-LLM avoids the Python GIL, how it uses C++ RAII stream wrappers, and how its execution graph differs from eager PyTorch.
