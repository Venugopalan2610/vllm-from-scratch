# Capstone Practicum J: Kernel Micro-Architecture with Nsight Compute

> *"An inference engine is only as fast as its attention and GEMV kernels."*

## Objective
Profile the custom CUDA kernels running inside your Capstone engine using NVIDIA Nsight Compute (`ncu`). Extract the 5 golden hardware counters, compare achieved memory bandwidth against theoretical peak, and diagnose shared-memory bank conflicts.

## Command
Profile your Stage 08 PagedAttention kernel:
```bash
./vc ncu 8 64
```

## The 5 Golden Counters & Verification

| Counter | Metric | What to Inspect |
| :--- | :--- | :--- |
| **DRAM Throughput** | `dram__bytes.sum.per_second` | Compare achieved memory bandwidth against `./vc info` peak. If low during decode, the kernel is latency-stalled on memory requests rather than bandwidth-saturated. |
| **Coalescing Ratio** | `smsp__average_data_bytes_per_sector_mem_global_op_ld.pct` | Ratio of useful bytes to transferred bytes per 32-byte DRAM sector. Compare Stage 08 (uncoalesced) vs Stage 08b (coalesced). |
| **Warp Occupancy** | `sm__warps_active.avg.pct_of_peak_sustained_active` | Active warps vs theoretical peak per SM. Check if high register counts per thread or shared-memory usage restrict occupancy. |
| **Bank Conflicts** | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory has 32 banks (4 bytes wide). If 2 threads in a warp access different words in the same bank, accesses serialize. |
| **Warp Stall Reasons** | `smsp__warp_issue_stalled_...` | Identify why the SM warp scheduler stalled: `stall_memory_throttle`, `stall_barrier`, or `stall_not_selected`. |

## What You Must Calculate
1. **DRAM Bandwidth Efficiency:** Calculate:
   $$\text{Efficiency} = \frac{\text{Achieved DRAM Bandwidth}}{\text{Theoretical Peak Bandwidth from } \texttt{./vc info}} \times 100\%$$
2. **Bank Conflict Latency Penalty:** If bank conflicts $> 0$, calculate the serialized cycle cost and explain the two standard mitigations: padding (`[WARP_SIZE][HEAD_DIM + 1]`) and XOR index swizzling.
