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

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
