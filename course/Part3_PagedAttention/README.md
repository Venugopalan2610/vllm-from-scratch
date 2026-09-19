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

**When you finish this Part:** [Part 4, The Scheduler](../Part4_TheScheduler/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
