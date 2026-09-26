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
| [`4_incidents/`](4_incidents/) | Nine tickets: a graph that reads a stale buffer, padding that writes into block 0, an empty top-p set, and half a character. Then break working code on purpose, and find three hidden faults from their behaviour. Do this section after stage 14. **The tickets need no GPU. The second notebook needs one.** |

## Systems Foundations: Launch Latency & CUDA Graphs

### 1. Removing the CPU cost of the launches
At batch 1 or a small batch, one decode step runs about a thousand small kernels
(RMSNorm, the projections, RoPE, attention, the MLP) across all the layers. In eager
PyTorch, the CPU spends several microseconds to launch each one, so the CPU can need
more time than the GPU. On an RTX 4080 Laptop GPU, one eager decode step of Qwen3-0.6B at
batch 1 took about 11 ms, and the same step replayed as a graph took 5.4 ms (see the
incidents of this Part).
- **CUDA Graphs (Stage 12 & Stage 23):** capture the kernels of a step once, and replay
  them with one launch. The CPU cost of the launches goes away. The GPU time stays. So a
  graph helps only while the CPU is the slower side: at a small batch, or with a small
  model.

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
