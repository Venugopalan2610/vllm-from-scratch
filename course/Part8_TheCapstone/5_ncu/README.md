# Capstone Practicum J: read the counters

> *A clock tells you that a kernel is slow. The counters tell you why.*

## The notebook

Open [`part8_ncu_1_CCreadTheCounters_helper.ipynb`](part8_ncu_1_CCreadTheCounters_helper.ipynb).
It compiles three small kernels: an uncoalesced read, a coalesced read, and a
shared-memory tile with and without bank conflicts. You measure them with a
clock, then with the counters of Nsight Compute (`ncu`):

| Counter | Metric |
| :--- | :--- |
| DRAM throughput | `dram__bytes.sum.per_second` |
| Sector use | `smsp__average_data_bytes_per_sector_mem_global_op_ld.pct` |
| Bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` |
| Shared loads | `smsp__inst_executed_op_shared_ld.sum` |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |

## The permission

On many Linux machines, only an administrator can read the counters. `ncu` then
stops with `ERR_NVGPUCTRPERM`. The notebook detects it, prints the fix (a driver
option and a reboot), and still runs its clock checks. The counter checks run
when the permission is there.

## Then your kernels (after stage 08b)

    ./vc ncu 8 64
    ./vc ncu 8b 64

Compare the sector use of the two kernels with the gain that stage 08b
measured, and check how near stage 08b is to the bandwidth of the card.
