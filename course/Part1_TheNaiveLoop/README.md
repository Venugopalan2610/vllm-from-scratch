# Part 1, The Naive Loop

Stages: stages 01-03.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._challenge`). Its solution is in `solutions/`.

## Start here: a ticket that you cannot solve yet

**Ticket 1.7: the chatbot gets slower as it talks.** Reported by users:

> The first words come fast. A long answer slows down until it crawls.

You do not have the tools for this yet. Keep it in mind while you work. Each
section of this Part gives you a piece of the answer. At the end,
[`4_incidents/`](4_incidents/) asks you to find the cause and prove it with a
number.

First, one guess: which layer of the engine would you look at?
([the map](../../docs/MAP.md))

## The sections

| | |
|---|---|
| [`1_arithmetic/`](1_arithmetic/) | Read against compute, and the ridge point. The KV cache in bytes per token. **This section needs no GPU.** |
| [`2_roofline/`](2_roofline/) | Measure your own card. Watch a matmul cross the ridge. Then put the real model on the plot. Prefill and decode sit on opposite sides. |
| [`3_kvCache/`](3_kvCache/) | What the quadratic cost really is. What the cache costs in bytes, and a 2x error the config invites. Then write both loops and make them agree. |
| [`4_incidents/`](4_incidents/) | Educational incident scenarios, from the outside. Eleven invented tickets: find the cause from the symptoms, and prove it with a number. Then break your own loop on purpose, and find three hidden faults from their behaviour. Do this section after stage 03. |

**When you finish this Part:** [Part 2, Batching](../Part2_Batching/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
