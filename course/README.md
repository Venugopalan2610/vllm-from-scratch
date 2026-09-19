# The notebooks

The ladder in `app/` makes you build an inference engine. These notebooks show
you why the engine must have this shape.

The two halves ask different questions:

| | `course/` | `app/` + `./vc` |
|---|---|---|
| unit | one idea, one notebook | one stage, one file |
| you | run it, change a number, watch the plot move | write the code from a spec |
| gate | none. You pass nothing. | measured. The code must get faster. |
| result | a number you can explain | a number you produced |

Read the notebooks for a stage first. Then build the stage. The notebook is
where you write a prediction down. The stage is where you pay for it.

## The rule about numbers

**No notebook ends with a number that is true only on one machine.**

An absolute measurement is an input. It is never a conclusion. A notebook that
ends with "1.5x" teaches you one fact about one laptop. A notebook that ends
with a ratio teaches you the mechanism. The ratio also tells you which way the
answer moves on different hardware.

Each notebook ends with one of these three things:

- **A dimensionless number.** The ridge point is FLOP per byte. Coalescing
  efficiency is bytes used per byte fetched. Occupancy is warps resident per
  warp possible. These numbers have no units. They keep their meaning on every
  card.
- **A cost in units of something you measured.** A step time is a multiple of
  one plain decode step. The scheduler simulations count steps, not seconds.
  Multiply by your own step time to get seconds.
- **A rule that connects a software knob to a hardware number.** The best
  static batch is `B_ridge`. The best token budget is `KNEE`. Collectives
  overtake compute at batch `6d / (C(R-1) x HBM/interconnect)`.

A notebook that needs a hardware constant obeys three rules:

1. It names the constant in capitals at the top of the notebook.
2. It tells you how to measure your own value.
3. It states the conclusion as a function of that constant.

The demos then sweep the constant. You watch the answer move. You do not trust
one row of the author's table.

A comparison across named cards is the one exception. It exists to show you
how the ratio responds to different hardware.

## Structure

Each section holds three kinds of file. The name tells you which kind:

```
Part3_PagedAttention/kernels/
  part3_kern_memoryTransactions.ipynb            demo: run it, read it
  part3_kern_fillingTheMachine.ipynb             demo
  part3_kern_CCcoalesceTheGather_helper.ipynb    challenge: fill the blanks
  solutions/
    part3_kern_CCcoalesceTheGather.ipynb         the same notebook, complete
```

- **A demo is small.** It teaches one idea. It does the arithmetic by hand
  first. Then it measures the machine. Then it plots the two against each
  other. A demo keeps its output, so you can read it without a GPU.
- **A challenge** gives you the structure and removes the code. The comments
  stay. You write the lines.
- **A solution** is the same notebook, complete. Its output is empty, so your
  own run shows you the answer.

## How to run them

```bash
./setup.sh                 # this installs the notebook packages too
.venv/bin/jupyter lab      # or open the folder in VS Code
```

Each notebook finds the repo root by itself. The directory you start from does
not matter. A notebook that compiles CUDA needs `nvcc`. The other notebooks
need a GPU. Some of them need no GPU at all.

## What exists

The ladder has seven arcs. The notebooks follow them.

**Part 1, The Naive Loop** (stages 01-03)

| | |
|---|---|
| `arithmetic/` | Read against compute, and the ridge point. The KV cache in bytes per token. **This section needs no GPU.** |
| `roofline/` | Measure your own card. Watch a matmul cross the ridge. Then put the real model on the plot. Prefill and decode sit on opposite sides. |
| `kvCache/` | What the quadratic cost really is. What the cache costs in bytes, and a 2x error the config invites. Then write both loops and make them agree. |

**Part 2, Batching** (stages 04-05)

| | |
|---|---|
| `padding/` | Extra users are almost free, until they are not. What a static batch throws away. Then find the batch size that is fastest. |
| `continuous/` | Schedule once per iteration. Twenty lines, and the largest win in the course. Then write the scheduler. **No GPU.** |

**Part 3, PagedAttention** (stages 06-09, 08b, 08c)

| | |
|---|---|
| `blocks/` | Where the KV memory goes, and why most of it is empty. A page table for tokens. Then build the allocator. **No GPU.** |
| `gather/` | Decode attention on numbers you can check by hand. The same arithmetic with the keys scattered. Then write the oracle. |
| `kernels/` | Sectors, strides, and a cliff you can predict. Why blocks must outnumber SMs. Then make a gather coalesce. |
| `sharing/` | Refcounts and copy-on-write. Prefix caching, measured. Then build it, and find the hashing bug that gives fluent wrong output. |

**Part 4, The Scheduler** (stages 10-11)

| | |
|---|---|
| `admission/` | A sequence grows while it runs, so admission promises memory you do not have. Swap against recompute, as a ratio. Then build the scheduler. |
| `chunked/` | One long prompt stops the whole server. Then the token budget that fixes it, and a p99 that hides the damage. |

**Part 5, Making It Fast** (stages 12-14)

| | |
|---|---|
| `cudaGraphs/` | The CPU as the bottleneck. A graph refuses to change shape, and a server changes shape every step. Then capture one. |
| `sampling/` | Temperature, top-k, top-p and penalties on eight words. Then the sampler as the bottleneck, and one you can prove is correct. |
| `detokenize/` | Why you cannot decode one token at a time. Then write the streaming detokenizer. **No GPU.** |

**Part 6, The Server** (stages 15-16)

| | |
|---|---|
| `async/` | One loop, two clocks, and what happens when a client disconnects. Then build the engine loop. **No GPU.** |
| `metrics/` | Six numbers, a Pareto frontier, and why throughput is not the answer. Then find which of three projects your p99 asks for. **No GPU.** |

**Part 7, Modern vLLM** (stages 17-20, 18b)

| | |
|---|---|
| `speculative/` | Thirty-two guesses for the price of one. Then verify, reject, and prove the distribution did not move. |
| `quantization/` | Fewer bytes per weight, and the trap that cancels the win. Then quantize, and find the speed you lost. |
| `guided/` | Mask the logits that cannot come next. Then make invalid output impossible, and find what the mask does not promise. **No GPU.** |
| `tensorParallel/` | Where to cut, and how few times the ranks must talk. Then count the collectives. **No GPU.** |

## The method

Each notebook obeys five steps:

1. **Do the arithmetic first.** Count sectors, bytes, SMs, blocks. Use paper
   or four lines of Python. Measure nothing yet.
2. **Write the prediction down.** Give a number or a bound. This step makes
   the next step informative.
3. **Measure with care.** Hold the other variables still. Use the same
   footprint, the same thermal state, and no store traffic in a read test.
   Most wrong conclusions about GPUs come from the experiment, not the
   hardware.
4. **Plot the prediction against the measurement.** Look at the shape first.
   Look at the magnitude second. A model with the correct shape and a constant
   offset found the mechanism. A model that hits one point and misses the
   shape found a coincidence.
5. **Explain the difference.** People skip this step. The hardware lives here.
   Every notebook has one.

You will be wrong in public a few times. This is deliberate. In
`part3_kern_CCcoalesceTheGather` the sector model predicts 8x and the
measurement gives 3x. The gap is the lesson. That kernel wastes requests, not
bytes. No amount of code reading tells you which.

## The language

These notebooks use ASD-STE100 Simplified Technical English. Sentences are
short. The voice is active. One word has one meaning.
