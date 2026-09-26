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

## The order

The order is in the names. Open the Parts from `Part0` to `Part8`. In each
Part, open the sections in the order of their numbers. In each section, open
the notebooks in the order of their numbers:

```
Part0_FromAProgramToAModel/          Part 0, first
  1_model/                           section 1
    part0_mdl_1_aModelIsAFunction    notebook 1: a demo
    part0_mdl_2_theGenerationLoop    notebook 2: a demo
    part0_mdl_3_CCwriteTheLoop_helper  notebook 3: the challenge, always last
  2_attention/                       section 2
```

Read the notebooks of a Part before you build its stages.

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
Part3_PagedAttention/3_kernels/
  part3_kern_1_memoryTransactions.ipynb            demo: run it, read it
  part3_kern_2_fillingTheMachine.ipynb             demo
  part3_kern_3_CCcoalesceTheGather_helper.ipynb    challenge: fill the blanks
  solutions/
    part3_kern_3_CCcoalesceTheGather.ipynb         the same notebook, complete
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

## The words

This course uses the words of two fields that are not software engineering:
machine learning and GPU programming. Three things help you with them:

- **Part 0** teaches the basic words, for a software engineer with no ML
  background. Start there if a word in Part 1 is new to you.
- **[GLOSSARY.md](GLOSSARY.md)** defines each term, gives an analogy from
  software, and names the notebook and the stage that teach it.
- **The first use of a term** in a notebook is in bold with a definition, or
  it is a link to the glossary. `dev/jargon.py` checks this, in the reading
  order.

## When it gets hard

**Do not stop. Continue. Be better than before.**

A notebook that confuses you is not a sign that you are slow. It is the part
of the work where you learn. Try these, in this order:

1. **Find the word.** Most confusion is one term that you do not know yet.
   Look for it in [GLOSSARY.md](GLOSSARY.md).
2. **Go back one notebook.** Each notebook uses the result of the one before
   it. The number in its name shows which one that is.
3. **Change one number, and run the cell again.** Watch what moves. A
   prediction that is wrong teaches you more than a correct one.
4. **Open the solution of the challenge.** Read it, close it, and write the
   code yourself. Then continue.

## What exists

Part 0 comes before the ladder. Then the ladder has eight arcs, and the
notebooks follow them. The README of each Part lists its sections, with one
line on each.

- [Part 0, From a Program to a Model](Part0_FromAProgramToAModel/): before stage 01
- [Part 1, The Naive Loop](Part1_TheNaiveLoop/): stages 01-03
- [Part 2, Batching](Part2_Batching/): stages 04-05
- [Part 3, PagedAttention](Part3_PagedAttention/): stages 06-09, 08b, 08c
- [Part 4, The Scheduler](Part4_TheScheduler/): stages 10-11
- [Part 5, Making It Fast](Part5_MakingItFast/): stages 12-14
- [Part 6, The Server](Part6_TheServer/): stages 15-16
- [Part 7, Modern vLLM](Part7_ModernVLLM/): stages 17-20, 18b
- [Part 8, The Capstone](Part8_TheCapstone/): stages 21-28, 24b, and Capstone Practicums I and J (timeline profiling and kernel counters, with checks) and K (optional reading on TensorRT-LLM)
- Beyond the capstone: stage 29 (LRU prefix eviction), and the optional extensions 30 (multi-LoRA) and 31 (MLA). They have no notebooks.

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
`part3_kern_3_CCcoalesceTheGather` the sector model predicts 8x and the
measurement gives 3x. The gap is the lesson. That kernel wastes requests, not
bytes. No amount of code reading tells you which.

## The language

These notebooks use ASD-STE100 Simplified Technical English. Sentences are
short. The voice is active. One word has one meaning.

## Exercises beyond the stages

These exercises go past the stages. They are not gated. Do them after the
capstone, or skip them. Each one closes a gap between this engine and a
production system.

### Exercise A: Multi-turn conversation

Send three turns of chat to your server. Each turn includes the full history.
Measure the prefix cache hit rate on the second and third turn. If the hit
rate is zero, your cache is not keying on the prompt correctly. If it is high,
you just saw why prefix caching exists.

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 20}'
# Copy the assistant reply into the next request as a message.
curl localhost:8000/metrics | grep hit
```

### Exercise B: Read the real vLLM scheduler

Open the upstream vLLM scheduler:
[`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py).

Map it to your `Scheduler` class in `app/s22_engine.py`. Answer these
questions:

1. What data structure does upstream use for waiting vs running requests?
2. How does upstream decide which request to preempt?
3. How does upstream handle requests that need more blocks than the pool?
4. What does upstream do with LoRA slots that your scheduler does not?
5. Where does upstream enforce the token budget?

List the five things upstream does that you do not. Explain why each one
matters for a system that serves 1000 users.

### Exercise C: Batch invariance

Run the same prompt with the same seed at batch 1 and batch 32. Compare
the tokens. If they differ, the cause is floating-point non-associativity:
a different batch size changes the matmul reduction tree, which changes
rounding. Count the positions that differ. This is a current, open problem.
See LORE.md §12.

### Exercise D: Failure modes

Break your server on purpose:

1. Send a prompt longer than `MAX_PROMPT_TOKENS`. Does the server reject it
   or crash?
2. Send 1000 concurrent requests. Does the engine OOM or does it preempt?
3. Kill the client in the middle of a stream. Do the blocks leak?
4. Feed NaN weights to the model. Does `check_logits` catch it?

Each failure that crashes the server is a bug worth fixing.

### Exercise E: Process Isolation & Goodput Under Load

Benchmark the server in single-process mode vs multi-process mode:

```bash
# Run server with the engine loop isolated in its own child process
./vc serve --multiprocess
```

Compare the p99 ITL (Inter-Token Latency) jitter between the two modes.
Notice how separating Python GC and HTTP handling from the CUDA
step loop stabilizes latency and eliminates execution bubbles.

### Exercise F: Zero-Copy Shared-Memory IPC

Inspect `SharedMemoryEventRing` in `app/s27_serve.py`.
Standard Python `multiprocessing.Queue` runs `pickle.dumps` and `pickle.loads`
on every event, and sends it through a pipe. Measure the cost for each event, on your
machine, for a small event and for a large one (with log-probabilities). Then measure
packing a fixed binary struct directly into POSIX shared memory. At what rate of events,
and at what size, does the difference matter? See LORE.md §13.

### Exercise G: Tree-Attention Speculative Decoding

Linear speculative drafting (stage 17) decays rapidly beyond 4 tokens.
Open `build_tree_mask` in `app/s17_speculative.py` and trace how EAGLE and Medusa
evaluate 16 draft hypotheses simultaneously in ONE forward pass.
Prove that candidates in sibling branches never see each other's tokens,
and evaluate the speedup on structured JSON generation. See LORE.md §17.

### Exercise H: Virtual Memory Management with cuMemMap

Read LORE.md §18. `VirtualMemoryBlockManager` in `app/s06_blocks.py` is a model of
the CUDA virtual-memory API in Python. It calls no driver function. Explain what
`cuMemAddressReserve` and `cuMemMap` would let a KV cache do that a preallocated
tensor cannot, and what it would cost. Then read the current vLLM source, and find
where it uses the real API, and where it does not.

