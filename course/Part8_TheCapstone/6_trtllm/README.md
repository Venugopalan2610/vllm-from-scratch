# Capstone Practicum K (optional reading): your engine and TensorRT-LLM

> *"The principles of continuous batching, paged memory, and graph execution are identical across runtimes. Only the language and execution engine change."*

## Objective
Map what you built onto NVIDIA's C++ LLM serving runtime, **TensorRT-LLM**. This practicum is reading. It has no check.

> **The names change.** The table below uses names from TensorRT-LLM and vLLM at the time of writing. Older TensorRT-LLM releases schedule in `GptManager`, and newer ones have an Executor API; vLLM moved its speculative decoding code into `vllm/v1/`. Find the current names in the source before you rely on them. That search is part of the exercise.

## Source Reference
Open the open-source TensorRT-LLM repository:
[`NVIDIA/TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM)

## The Dual-Engine Mapping

| Component | `vllm-from-scratch` Capstone | Upstream vLLM V1 | NVIDIA TensorRT-LLM |
| :--- | :--- | :--- | :--- |
| **Batching Scheduler** | `Scheduler` in `app/s22_engine.py` | `Scheduler` in `vllm/v1/core/sched/` | `GptManager` / `BatchingManager` |
| **KV Memory Manager** | `KVBlockManager` in `app/s22_engine.py` | `BlockPool` in `vllm/v1/core/` | `KVCacheManager` (the batch manager of the C++ runtime) |
| **Attention Backend** | `paged_attention_cuda` (Stages 08, 08b, 08c) | FlashInfer / FlashAttention-3 | `PagedKvCache` + `DecoderMaskedMultiheadAttentionPlugin` |
| **Graph Execution** | `GraphedModelRunner` (Stage 23) | CUDA Graph Runner | TensorRT Engine Plan (`.engine`) |
| **Speculative Decoding** | `rejection_sample` + `build_tree_mask` (Stage 17) | `vllm/spec_decode/` | Medusa / EAGLE Decoding Plugin |
| **Multi-GPU Parallelism**| `ColumnParallelLinear` / `RowParallelLinear` (Stage 20)| `vllm/model_executor/` | `NcclPlugin` (Tensor Parallelism via NCCL) |

## Questions
1. **Find the KV cache manager of TensorRT-LLM** (search the source for `KVCacheManager`):
   - Locate the block allocation logic and reference counting. Compare it line-by-line with your `BlockAllocator` in `app/s06_blocks.py`.
2. **Find the in-flight batching scheduler of TensorRT-LLM:**
   - Identify where in-flight batching decides between admitting a new prefill request vs scheduling ongoing decodes.
   - Compare its token budget rule with your `Scheduler.schedule()` implementation in `app/s22_engine.py`.
3. **C++ Systems Architecture Review:**
   - Explain how TensorRT-LLM avoids the Python GIL, how it uses C++ RAII stream wrappers, and how its execution graph differs from eager PyTorch.
