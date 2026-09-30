# Part 2, Batching

Stages: stages 04-05.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._challenge`). Its solution is in `solutions/`.

## Start here: a ticket that you cannot solve yet

**Ticket 2.4: the job that took nine times longer.** Reported by the data team:

> We summarize 10,000 support tickets each night with a static batch of
> 32. The plan said 3.5 hours. It took 31 hours. The GPU was at 100%
> utilization the whole time.

You do not have the tools for this yet. Keep it in mind while you work. Each
section of this Part gives you a piece of the answer. At the end,
[`3_incidents/`](3_incidents/) asks you to find the cause and prove it with a
number.

First, one guess: which layer of the engine would you look at?
([the map](../../docs/MAP.md))

## The sections

| | |
|---|---|
| [`1_padding/`](1_padding/) | Extra users are almost free, until they are not. What a static batch throws away. Then find the batch size that is fastest. |
| [`2_continuous/`](2_continuous/) | Schedule once per iteration. Twenty lines, and the largest win in the course. Then write the scheduler. **No GPU.** |
| [`3_incidents/`](3_incidents/) | Eight tickets about many users at once: padding, masks, queues and the wrong batch size. Then break working code on purpose, and find three hidden faults from their behaviour. Do this section after stage 05. **The tickets need no GPU. The second notebook needs one.** |

**When you finish this Part:** [Part 3, PagedAttention](../Part3_PagedAttention/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
