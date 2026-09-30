# Part 4, The Scheduler

Stages: stages 10-11.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._challenge`). Its solution is in `solutions/`.

## Start here: a ticket that you cannot solve yet

**Ticket 4.1: the preemption storm.** Reported by the SRE team:

> At the peak, 38% of the requests are preempted at least once, and the
> throughput falls by half. We raised `max_num_seqs` from 128 to 256 to
> admit more requests, and it got worse.

You do not have the tools for this yet. Keep it in mind while you work. Each
section of this Part gives you a piece of the answer. At the end,
[`3_incidents/`](3_incidents/) asks you to find the cause and prove it with a
number.

First, one guess: which layer of the engine would you look at?
([the map](../../docs/MAP.md))

## The sections

| | |
|---|---|
| [`1_admission/`](1_admission/) | A sequence grows while it runs, so admission promises memory you do not have. Swap against recompute, as a ratio. Then build the scheduler. |
| [`2_chunked/`](2_chunked/) | One long prompt stops the whole server. Then the token budget that fixes it, and a p99 that hides the damage. |
| [`3_incidents/`](3_incidents/) | Eight tickets about time: preemption storms, stalls, starvation, and a p99 that sees nothing. Then break working code on purpose, and find three hidden faults from their behaviour. Do this section after stage 11. **The tickets need no GPU. The second notebook needs one.** |

**When you finish this Part:** [Part 5, Making It Fast](../Part5_MakingItFast/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
