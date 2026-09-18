# The notebooks

The ladder in `app/` makes you build an inference engine. These notebooks make
you understand why it has to look like that.

They are a different shape of work and they answer a different question:

| | `course/` | `app/` + `./vc` |
|---|---|---|
| unit | one idea, one notebook | one stage, one file |
| you | run it, change a number, watch the plot move | write the thing from a spec |
| gate | none. Nothing to pass. | measured. It has to get faster. |
| ends with | a number you can explain | a number you produced |

Read the notebooks for a stage first, then build the stage. The notebooks are
where a prediction gets written down; the stage is where it gets paid for.

## Structure

Each section holds three kinds of file, and the name says which:

```
Part3_PagedAttention/kernels/
  part3_kern_memoryTransactions.ipynb            demo: run it, read it
  part3_kern_fillingTheMachine.ipynb             demo
  part3_kern_CCcoalesceTheGather_helper.ipynb    code challenge: blanks to fill
  solutions/
    part3_kern_CCcoalesceTheGather.ipynb         the same notebook, filled in
```

- **Demos** are small and self-contained. One idea each. They do the arithmetic
  by hand first, then measure the machine, then plot the two against each other.
  They ship with their output, so you can read them without a GPU.
- **Code challenges** (`CC`) hand you the scaffolding with the implementation
  lines removed. The comments stay; the code goes. Fill them in.
- **Solutions** are the same notebook, complete, with the output cleared so your
  own run is the reveal.

## What exists so far

The ladder has seven arcs and the notebooks follow them. Two arcs are written.

**Part 1, The Naive Loop** (stages 01-03)

| | |
|---|---|
| `arithmetic/` | Read against compute, and the ridge point. The KV cache in bytes per token. **Needs no GPU.** |
| `roofline/` | Measure your own card. Watch a matmul cross the ridge. Then put the real model on the plot and find prefill and decode on opposite sides of it. |
| `kvCache/` | What the quadratic actually costs. What the cache costs in bytes, including a 2x error the config invites you to make. Then write both loops and make them agree. |

**Part 2, Batching** (stages 04-05)

| | |
|---|---|
| `padding/` | Extra users are nearly free, until they are not. What a static batch throws away. Then find the batch size that is actually fastest. **`padding` needs no GPU except the first demo.** |
| `continuous/` | Schedule per iteration: twenty lines, the largest win in the course. What it does to the tail. Then write the scheduler. **No GPU.** |

**Part 3, PagedAttention** (stages 06-09, 08b, 08c)

| | |
|---|---|
| `blocks/` | Where the KV memory goes and why most of it is empty. A page table for tokens. Then build the allocator. **No GPU.** |
| `gather/` | Decode attention on numbers you can check by hand. The same arithmetic with the keys scattered. Then write the oracle every later kernel is checked against. |
| `kernels/` | Sectors, strides and a cliff you can predict. Why blocks must outnumber SMs. Then coalesce a block-table gather. |
| `sharing/` | Refcounts and copy-on-write. Automatic prefix caching, measured. Then build it, including the hashing bug that produces fluent wrong output. |

**Part 4, The Scheduler** (stages 10-11)

| | |
|---|---|
| `admission/` | A sequence's cache grows while it runs, so admission is a promise about memory you do not have yet. Swap against recompute, as an inequality rather than a slogan. Then build the scheduler that does not fall over. |
| `chunked/` | One long prompt stops the whole server. Then the token budget that fixes it, and a p99 that hides the damage. |

**Part 5, Making It Fast** (stages 12-14)

| | |
|---|---|
| `cudaGraphs/` | The CPU as the bottleneck. A graph refuses to change shape and a server does nothing else. Then capture one and survive a changing batch. |
| `sampling/` | Temperature, top-k, top-p and penalties on eight words. Then the sampler as the bottleneck, and a batched one you can prove is correct. |
| `detokenize/` | Why you cannot decode tokens one at a time. Then write the streaming detokenizer. **No GPU.** |

**Part 6, The Server** (stages 15-16)

| | |
|---|---|
| `async/` | One loop, two clocks, and what happens when a client hangs up. Then build the engine loop. **No GPU.** |
| `metrics/` | Six numbers, a Pareto frontier, and why throughput is not the answer. Then work out which of three projects your p99 is asking for. **No GPU.** |

**Part 7, Modern vLLM** (stages 17-20, 18b)

| | |
|---|---|
| `speculative/` | Thirty-two guesses for the price of one. Then verify, reject, and prove the distribution did not move. |
| `quantization/` | Fewer bytes per weight, and the trap that undoes it. Then quantize it and find the speed you lost. |
| `guided/` | Masking the logits that cannot legally come next. Then make invalid output unreachable, and find what the mask still does not promise. **No GPU.** |
| `tensorParallel/` | Where to cut, and how few times the ranks have to talk. Then count the collectives. **No GPU.** |

All seven parts are written.

## How to run them

```bash
./setup.sh                 # the notebook deps come with it
.venv/bin/jupyter lab      # or open the folder in VS Code
```

Every notebook finds the repo root on its own, so it does not matter which
directory you launch from. Anything that compiles CUDA needs `nvcc`; the rest
needs only a GPU, and some of it does not need that either.

## The rule about numbers

**No notebook concludes with a number that is only true on the machine it was
written on.**

Absolute measurements are fine and often necessary, but they are inputs, never
conclusions. A notebook that ends in "1.5x" has taught you a fact about an
RTX 4080 Laptop. A notebook that ends in a ratio has taught you the mechanism,
and the ratio tells you which way it moves when the hardware changes.

So every notebook here ends in one of three things:

1. **A dimensionless quantity.** The ridge point is FLOP per byte. Coalescing
   efficiency is bytes used per byte fetched. Occupancy is warps resident over
   warps possible. Tensor-parallel cost is collective time over compute time.
   None of them carry a unit, and none of them change meaning on another card.
2. **A cost in units of something you measured yourself.** Step times are
   expressed as multiples of one plain decode step; the scheduler simulations
   run on a clock whose tick *is* a step. Multiply by your own step time when
   you want seconds.
3. **A rule relating a workload knob to a hardware parameter.** "The best
   static batch is `B_ridge`." "The best token budget is `KNEE`." "Talking
   overtakes computing at batch `6d / (C(R-1) x HBM/interconnect)`."

Where a notebook does need a hardware constant, it is named in capitals at the
top, it says how to measure your own, and the conclusion is stated as a
function of it. The demos then sweep it, so you can see the answer move rather
than take the author's row on trust.

The one exception is a spec-sheet comparison across named cards, which exists
precisely to show how the ratio responds to hardware.

## The method, and why every notebook follows it

1. **Do the arithmetic first.** Sectors, bytes, SMs, blocks. On paper, or in
   four lines of Python, before anything is measured.
2. **Write the prediction down.** A number, or a bound. Committing to it is
   what makes the next step informative rather than decorative.
3. **Measure, carefully.** Hold the confounds still: same footprint, same
   thermal state, no store traffic in a read benchmark. Most wrong conclusions
   about GPUs are experiment design, not hardware.
4. **Plot the two together.** Shape first, magnitude second. A model with the
   right shape and a constant offset has found the mechanism. A model that
   nails one point and misses the shape has found a coincidence.
5. **Explain the residual.** This is the step people skip, and it is where the
   hardware actually lives. Every notebook here has one.

You will be wrong in public a few times doing this, on purpose. In
`part3_kern_CCcoalesceTheGather` the sector model predicts up to 8x and the
measurement returns about 3x, and the gap is the entire lesson: that kernel was
wasting requests, not bytes, and no amount of staring at the code would have
told you which.
