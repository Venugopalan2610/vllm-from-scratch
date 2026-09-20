# Part 5, Making It Fast

Stages: stages 12-14.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._helper`). Its solution is in `solutions/`.

| | |
|---|---|
| [`1_cudaGraphs/`](1_cudaGraphs/) | The CPU as the bottleneck. A graph refuses to change shape, and a server changes shape every step. Then capture one. |
| [`2_sampling/`](2_sampling/) | Temperature, top-k, top-p and penalties on eight words. Then the sampler as the bottleneck, and one you can prove is correct. |
| [`3_detokenize/`](3_detokenize/) | Why you cannot decode one token at a time. Then write the streaming detokenizer. **No GPU.** |

## Systems Foundations: Launch Latency & CUDA Graphs

### 1. Eliminating CPU Launch Bubbles (<15 µs to ~2 µs)
At batch 1 or small batch sizes, each decode step executes dozens of small kernels
(RMSNorm, QKV projection, RoPE, attention, MLP, sampler). Launching each kernel from
the host incurs 3–5 µs of CPU driver overhead. Across 30 layers, the CPU spends
>100 µs queuing work while the GPU starves in idle bubbles.
- **CUDA Graphs (Stage 12 & Stage 23):** Captures the static execution topology into an
  on-device hardware command list. Replaying the entire decode step takes a single
  launch call (<2 µs), completely eliminating host-side dispatch bubbles.

### 2. Pinned vs. Pageable Memory in CUDA Transfers
- **Pageable Host Memory:** Managed by the OS virtual memory subsystem. The GPU DMA
  engine cannot access it directly because physical pages might swap or migrate.
  `cudaMemcpy` must synchronously copy into an internal staging buffer first,
  doubling CPU work and stalling the CPU stream.
- **Pinned (Page-Locked) Memory (`torch.empty(..., pin_memory=True)` / `cudaMallocHost`):**
  Locks physical pages in RAM, enabling true asynchronous DMA (`cudaMemcpyAsync`)
  that overlaps with GPU kernel execution.

### 3. The Prefix-Caching Replay Hazard & Staging Buffers
CUDA Graphs require fixed memory addresses at capture time. When integrated with
dynamic prefix caching (Stage 23):
- A padding row in a batch cannot map to slot `0` or block `0`. If it does, graph
  replay will silently overwrite Block 0—corrupting the shared prompt cache for all
  subsequent requests.
- Padded rows must route to scratch slot `-1`.
- Dynamic block tables and input tokens must be transferred into pinned host-to-device
  staging buffers (`StagingBuffers`) and synchronized with the stream before
  `graph.replay()` is invoked.

**When you finish this Part:** [Part 6, The Server](../Part6_TheServer/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
