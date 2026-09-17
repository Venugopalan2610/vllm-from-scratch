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

**Part 3, PagedAttention** (stages 06-09, 08b, 08c)

| | |
|---|---|
| `kernels/` | Sectors, strides and a cliff you can predict. Why blocks must outnumber SMs. Then coalesce a block-table gather and measure it. |

Parts 2, 4, 5, 6 and 7 are not written yet.

## How to run them

```bash
./setup.sh                 # the notebook deps come with it
.venv/bin/jupyter lab      # or open the folder in VS Code
```

Every notebook finds the repo root on its own, so it does not matter which
directory you launch from. Anything that compiles CUDA needs `nvcc`; the rest
needs only a GPU, and some of it does not need that either.

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
