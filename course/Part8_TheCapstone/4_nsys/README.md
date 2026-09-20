# Capstone Practicum I: Whole-System Timeline Profiling with Nsight Systems

> *"You do not guess where latency comes from. You measure the machine."*

## Objective
Profile your complete Capstone continuous-batching engine under real load using NVIDIA Nsight Systems (`nsys`). Measure CPU dispatch bubbles, CUDA stream concurrency, Host-to-Device memory copy overheads, and NVTX execution ranges.

## Command
Run the built-in timeline profiler:
```bash
./vc nsys
```
This generates `.cudacache/engine_profile.nsys-rep`. Open it in the Nsight Systems GUI:
```bash
nsys-ui .cudacache/engine_profile.nsys-rep
```

## What You Must Inspect & Verify

### 1. The CPU Execution Bubble (Dispatch Overhead)
- Locate two consecutive decode steps on the GPU timeline.
- Measure the time delta between the end of step $N$'s attention kernel and the launch of step $N+1$'s GEMM.
- **Analysis:** If the GPU timeline shows an idle gap >100 µs between iterations, CPU work (Python GIL contention, FastAPI deserialization, or detokenization) is starving the GPU.
- **Verification:** Explain how Stage 27's `ProcessEngineWorker` and zero-copy shared-memory IPC (`SharedMemoryEventRing`) eliminate this gap.

### 2. Stream Concurrency & Memory Staging
- Zoom into the CUDA compute stream and transfer streams.
- Check if Host-to-Device copies (e.g. `slot_mapping`, `block_tables`) block kernel execution.
- **Verification:** Confirm that pinned host staging buffers (`StagingBuffers` in `app/s23_graphs.py`) allow PCIe transfers to execute asynchronously without stalling the CPU thread.

### 3. NVTX Range Spans
- Expand the NVTX row in `nsys-ui`.
- Inspect the annotations: `engine_step_X`, `schedule`, `model_forward`, `sample`.
- Calculate the percentage of iteration time spent in the PyTorch model forward pass versus Python scheduling logic.
