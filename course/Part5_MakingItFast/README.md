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

**When you finish this Part:** [Part 6, The Server](../Part6_TheServer/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
