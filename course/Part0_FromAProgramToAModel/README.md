# Part 0, From a Program to a Model

Before stage 01.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._helper`). Its solution is in `solutions/`.

| | |
|---|---|
| [`1_model/`](1_model/) | A model is a function: tokens in, logits out, and a large read-only table of weights. Then the generation loop, and what it costs. Then write the loop. |
| [`2_attention/`](2_attention/) | Attention as a lookup that returns a mix. Heads, and the fact that makes a KV cache possible. Then memoize the lookup. **This section needs no GPU.** |
| [`3_gpu/`](3_gpu/) | A machine built for one kind of work: bandwidth, FLOPs and the ridge point. A kernel is the body of a loop. Then predict the time of four workloads, and measure them. |
| [`4_incidents/`](4_incidents/) | Nine tickets from the outside: a tokenizer that does not match, an overflow, a test that cannot fail, and a kernel that is already fast. Find the cause, and prove it with a number. Then break working code on purpose, and find three hidden faults from their behaviour. **The tickets need no GPU. The second notebook needs one.** |

**When you finish this Part:** [Part 1, The Naive Loop](../Part1_TheNaiveLoop/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
