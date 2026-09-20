# Part 3, PagedAttention

Stages: stages 06-09, 08b, 08c.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._helper`). Its solution is in `solutions/`.

| | |
|---|---|
| [`1_blocks/`](1_blocks/) | Where the KV memory goes, and why most of it is empty. A page table for tokens. Then build the allocator. **No GPU.** |
| [`2_gather/`](2_gather/) | Decode attention on numbers you can check by hand. The same arithmetic with the keys scattered. Then write the oracle. |
| [`3_kernels/`](3_kernels/) | Sectors, strides, and a cliff you can predict. Why blocks must outnumber SMs. Then make a gather coalesce. |
| [`4_sharing/`](4_sharing/) | Refcounts and copy-on-write. Prefix caching, measured. Then build it, and find the hashing bug that gives fluent wrong output. |

## Kernel Micro-Architecture & Hardware Profiling

Stages 08, 08b, and 08c move from Python logic into bare-metal CUDA. You do not
guess why a kernel is slow; you measure the hardware pipeline:

```bash
./vc ncu 8 64    # Profile stage 08 with Nsight Compute at batch 64
```

### The 5 Golden Hardware Counters
1. **DRAM Throughput (`dram__bytes.sum.per_second`):** Achieved memory bandwidth.
   Compare against `./vc info` peak. If low during decode, the kernel is
   latency-stalled on memory requests, not bandwidth-bound.
2. **Coalescing Ratio (`smsp__average_data_bytes_per_sector_mem_global_op_ld.pct`):**
   Percentage of each 32-byte DRAM sector used by the warp. 100% = coalesced (Stage 08b).
   <50% = scattered memory loads wasting bus transactions.
3. **Warp Occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`):**
   Ratio of active warps to theoretical maximum per SM, bounded by register
   spills and shared memory allocation.
4. **Shared Memory Bank Conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`):**
   Shared memory has 32 banks (4 bytes wide). If two threads in a warp access
   different addresses in the same bank, accesses serialize. Solved via padding
   (`[WARP_SIZE][HEAD_DIM + 1]`) or XOR swizzling.
5. **Warp Stall Reasons (`smsp__warp_issue_stalled_...`):**
   Identifies why the warp scheduler stalled: `stall_memory_throttle` (waiting on DRAM/L2),
   `stall_barrier` (waiting on `__syncthreads()`), or `stall_not_selected`.

### Systems Foundations Baked into Part 3
- **Warp Divergence Elimination:** Stage 08c replaces branched reduction trees with
  register-level warp-shuffle intrinsics (`__shfl_xor_sync`), exchanging data directly
  between registers across warp lanes with zero shared memory and zero divergence.
- **Low-Level Virtual Memory Management (`cuMemMap`):** Stage 06 implements dynamic
  virtual address reservation and physical block mapping (`cuMemCreate`, `cuMemMap`,
  `cuMemSetAccess`), showing how modern runtimes avoid VRAM fragmentation without
  reallocating giant contiguous buffers.
- **PagedAttention vs. Contiguous Memory:** Eliminates up to 80% internal fragmentation
  by bounding waste to `block_size - 1` tokens and enabling zero-copy prompt sharing.

**When you finish this Part:** [Part 4, The Scheduler](../Part4_TheScheduler/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
