# The Lore

Everything in vLLM descends from one physical fact. Learn the fact, and the rest of
the system stops being a pile of tricks and becomes a series of forced moves.

---

## 1. The fact: is this IO-bound or CPU-bound?

That is the whole question. It is the same question that you ask about any
service.
Everything else in this document falls out of the answer.

**No number below appears without its arithmetic.** You should never have to
reconstruct where one came from.

### The setup

A 7B model is a pile of numbers ("weights") sitting in GPU RAM:

```
7,000,000,000 weights  x  2 bytes each (bf16)  =  14 GB
```

To generate **one token**, the GPU must read all 14 GB and do arithmetic on it.
Two steps. Time them separately.

### Step 1: read the weights

```
    14 GB          GB cancels, leaving seconds
------------  =  0.037 s  =  37 ms
  380 GB/s          ^-- an example card. ./vc info measures yours
```

### Step 2: do the math

The model uses each weight in exactly one multiply-and-add. That is 2
operations for each weight:

```
7e9 weights  x  2 ops  =  14e9 operations  =  14 GFLOP
```

("FLOP" = one arithmetic operation on a decimal number. Not energy. Not
physics. Just an op, the way a request is a request.)

```
    14 GFLOP          GFLOP cancels, leaving seconds
---------------  =  0.00025 s  =  0.25 ms
  57,000 GFLOP/s        ^-- the same example card
```

### Compare the two

| Step | The division | Time |
|---|---|---|
| Read 14 GB of weights | `14 GB / 380 GB/s` | **37 ms** |
| Compute on them | `14 GFLOP / 57000 GFLOP/s` | **0.25 ms** |

```
37 ms reading  /  0.25 ms computing  =  150x more time spent waiting
```

**The GPU is idle over 99% of the time, waiting for memory.** It is not
short of arithmetic. It is short of *bytes arriving*.

This is the N+1 query problem. To make one token at a time, you do a full table
scan to answer one question.

### The fix, and where the batch size comes from

Read the weights once, answer many questions with them. If 128 requests are
waiting, run them in the *same* pass:

| | 1 request | 128 requests |
|---|---|---|
| Bytes read | 14 GB | **14 GB** — same weights, read once |
| Read time | `14 / 380` = 37 ms | `14 / 380` = **37 ms** |
| Operations | 14 GFLOP | `14 x 128` = 1792 GFLOP |
| Compute time | `14 / 57000` = 0.25 ms | `1792 / 57000` = **31 ms** |
| Tokens out | 1 | **128** |
| Wall clock | ~37 ms | ~40 ms |

**128 times the output in the same wall-clock time.** All 128 requests share the
expensive part, which is the 14 GB that must cross the memory bus.

Say it the other way. The GPU reads one weight from VRAM **one time**. That
weight then sits in a register. The GPU multiplies it by the activation of
request 1, then request 2, then request 3, up to B. One byte, B multiply-adds.

The bytes are the expensive part. So the game is to get more work out of every
byte that you already paid to bring in.

In linear-algebra terms, batching changes *which operation you are running*:

```
batch 1:   y = W @ x     matrix x vector  (GEMV)  -> each weight used once
batch B:   Y = W @ X     matrix x matrix  (GEMM)  -> each weight used B times
```

A batch turns a GEMV into a GEMM. A GPU exists to run a GEMM. A GEMV wastes it.

This is the same instinct as cache blocking or loop tiling on a CPU. Load the
line one time. Take all of the work out of it before the cache drops it.

Look at the two time columns. They are now close: 37 ms to read, 31 ms to
compute. They become equal at one batch size, and that is where the batch size
comes from:

```
   37 ms of reading
--------------------  =  150 requests before compute becomes the bottleneck
 0.25 ms per request
```

> **At batch ~150, decode stops waiting on memory on this example card.**
> Below it, an extra request costs almost nothing. Above it, you pay for
> arithmetic.

That is the entire argument for continuous batching, and why production servers
set `max_num_seqs` (the batch size knob) in the hundreds.

You do not quite reach that batch size. Each concurrent request also needs its
own KV cache in VRAM. You run out of memory before you run out of arithmetic.
PagedAttention exists to relax *that* constraint.

### "Why not batch to infinity, then?"

To be compute-bound is the *goal*. The silicon that you paid for is finally
busy, and it no longer waits on memory. So why stop at approximately 150?

Because past the ridge you gain nothing. 7B on this card, read = 37 ms,
compute = 0.25 ms per request:

| Batch | Read | Compute (`0.25 x B`) | Step time | Throughput | Latency/user |
|---|---|---|---|---|---|
| 1 | 37 ms | 0.25 ms | 37 ms | `1/0.037` = 27 tok/s | 37 ms |
| 50 | 37 ms | 12 ms | 37 ms | `50/0.037` = 1,351 tok/s | 37 ms |
| **150** | 37 ms | 37 ms | **37 ms** | `150/0.037` = **4,054 tok/s** | **37 ms** |
| 300 | 37 ms | 75 ms | 75 ms | `300/0.075` = 4,000 tok/s | 75 ms |
| 600 | 37 ms | 150 ms | 150 ms | `600/0.150` = 4,000 tok/s | 150 ms |

Look at the last two rows. **The throughput is flat.** Double the batch, and
you double the step time. You make two times the tokens in two times the time.
The gain is zero. At the same time, the inter-token latency of every user
doubled.

The reason is mechanical. Past the ridge, step time is `0.25ms x B`, so:

```
      B tokens              1
------------------  =  ----------  =  constant, the B cancels
   0.25ms x B sec         0.25ms
```

You have hit the compute roofline and it does not move. Hence "ridge point": a
peak you sit on, not a wall you push through.

### "But the weights never change. Why can they not stay in the cache?"

That is true. Only a retraining run changes them. But read-only does not mean
free to read, and the weights are much too large for the cache:

```
L2 cache on this GPU       50 MB
7B model in bf16       14,000 MB   ->  278x too big
Qwen3-0.6B in bf16      1,200 MB   ->   24x too big
```

So the weights live in **VRAM**, and "read 14 GB from VRAM" *is* the 37 ms.
Residency in VRAM is not a saving. It is the cost. Every forward pass pulls the
full 14 GB across the bus again, token after token, forever.

The engine does use the fact that the weights never change. It does not use a
cache to do it:

- **A batch.** The same unchanging weights serve all B requests (stages 04-05).
- **Quantization.** Make them smaller offline, to fp8 or int4. 14 GB becomes 7
  or 3.5, and the read time falls by the same factor (stage 18).
- **Tensor parallelism.** Shard them, so each GPU reads only its slice
  (stage 20).

All three attack the same denominator. None of them makes the read free,
because 14 GB does not fit in 50 MB.

### The compressed form that you see everywhere else

A paper or a blog post does not write out both timings. It divides them into one
number and compares that number against the hardware. This is the "roofline
model". It confuses everybody, so here it is, slowly.

**Two different quantities that happen to share the same units.**

|  | What it is | Changes when... |
|---|---|---|
| **~150 FLOP/byte** | a property of your **hardware** | you buy a different GPU |
| **B FLOP/byte** | a property of your **workload** | you change the batch size |

You cannot observe either number with a probe on a running GPU. You compute both
on paper, in advance, and then you compare them. That comparison is the whole
model.

**Why they share units.** The hardware number is a rate divided by a rate, so
the seconds cancel out:

```
  57,000 GFLOP/s          GFLOP     /s
------------------  =  ---------- x ---  =  150 FLOP per byte
     380 GB/s               GB       /s
```

A plain count ratio remains. It is FLOP for each byte, and it holds no time. It
has exactly the same shape as the ratio of the workload. That is the *only*
reason to do this division: to put the hardware and the workload on the same
footing.

On a laptop GPU this number moves by 20% or more from run to run, because the
clocks throttle. Do not chase the exact value.

**Why the workload's ratio is exactly B.** Let `N` = number of weights, bf16:

```
Bytes moved  = 2 bytes/weight x N             = 2N     (read ONCE, whatever B is)
Operations   = 2 ops/weight x N x B requests  = 2NB

   2NB
--------  =  B FLOP per byte        <- the 2 and the N cancel out
    2N
```

At batch 8 you do 8 FLOP for each byte that you fetch. At batch 128 you do 128.
This is not a measurement. It describes the work that you asked for.

That clean `= B` is an accident of bf16. There, 2 bytes for each weight cancels
2 ops for each weight. In fp8, at 1 byte for each weight, it becomes
`2NB / N` = **2B**. That is two times the intensity at the same batch size.

This is a second reason that quantization helps. It is separate from the read
time, which also becomes half as large.

**It is your two timings, in a different order.** It says nothing new:

```
memory-bound  means   time_reading   >   time_computing

                        Bytes             FLOPs
                       -------      >    -------
                          BW               Rate

  multiply both sides by Rate, divide both sides by Bytes:

                         Rate             FLOPs
                        ------      >    -------
                          BW              Bytes

                    150 (hardware)   >   B (workload)
```

It is the same inequality with the terms in a different order. The intensity
form is better for one reason only. The model size drops out, so one number
covers every model.

**What a profiler actually shows you.** Neither ratio. It shows achieved *rates*.
Batch 1 on a 7B model:

```
memory:   14 GB     / 0.037 s  =    378 GB/s      vs    380 peak  ->  ~100% busy
compute:  14 GFLOP  / 0.037 s  =    378 GFLOP/s   vs 57,000 peak  ->   ~0.7% busy
```

The 150 never appears in the output of a profiler. You compute it in advance, to
predict *which of those two lines reaches its limit*.

### Everything else follows

One fact makes the whole system: *decode waits on memory, so an extra request
costs almost nothing*. It gives you all of this:

- Batch as hard as possible → **continuous batching** (stage 5)
- What limits the batch? KV cache memory → **PagedAttention** (stages 6-9)
- Who gets memory when it runs out? → **the scheduler** (stage 10)
- Prefill is the opposite (compute-bound) and disrupts decode → **chunked prefill** (11)
- 1 ms of GPU work must not cost 1 ms of Python → **CUDA graphs** (12)
- A check on K tokens costs the same as one new token → **speculative decoding** (17)
- Fewer weight bytes = proportionally faster decode → **quantization** (18)

Every stage in this repo is a consequence of the roofline. Stage 3 makes you measure
it on your own GPU so it stops being a slogan.

---

## 2. Why the KV cache is the real product

The KV cache is the state that makes decoding incremental. Its size:

```
bytes = 2 (K and V)
      * num_layers
      * num_kv_heads * head_dim     <- note: KV heads, not query heads
      * seq_len
      * dtype_bytes
```

Three properties make it uniquely difficult to manage. They are the reason that
a whole paper exists about it:

1. **It grows one token at a time**, unpredictably. You cannot know the final size
   at admission, because you do not know when the model will emit EOS.
2. **It is enormous.** At long context it rivals or exceeds the weights. It is the
   binding constraint on batch size, and batch size is throughput.
3. **It is per-sequence**, so it fragments.

Pre-vLLM systems allocated one contiguous buffer per sequence, sized to `max_len`.
The waste came in three flavors:

- **Reserved**: allocated for tokens not generated yet
- **Internal fragmentation**: the tail of the buffer if the sequence stops early
- **External fragmentation**: gaps between buffers that fit nothing

Measured waste in the vLLM paper: **60-80%** of KV memory. Meaning ~4x fewer
concurrent sequences than the hardware could hold. Meaning ~4x less throughput.

**Grouped-Query Attention (GQA)** is the other half of this story. Llama-3-8B has
32 query heads and only **8** KV heads. That cuts the KV bytes by 4. The
designers of a modern model think about KV cache pressure.

When you compute the cache size in stage 2, use `num_kv_heads`. If you do not,
your answer is wrong by the GQA factor.

---

## 3. PagedAttention: the OS analogy, in full

The insight is that this is **virtual memory**, a solved problem from 1961.

| Operating system | vLLM |
|---|---|
| Process | Sequence / request |
| Page (4 KB) | Block (16 tokens of KV) |
| Page table | Block table |
| Physical frame | Physical KV block in the pool |
| Virtual address space | Logical token positions 0..n |
| Internal fragmentation | ≤ 1 partial block per sequence |
| Copy-on-write after fork | CoW when parallel samples diverge |
| Page cache | Automatic prefix caching |
| Swapping | Swap blocks to CPU under pressure |

There is a cost. Attention can no longer stride through a contiguous tensor. The
kernel must **gather K and V through the block table**. That is why
PagedAttention needed a custom kernel, and why it could not call
FlashAttention.

Stage 7 makes you feel the slowdown in PyTorch. Stages 8, 8b and 8c make you win
it back in CUDA.

The payoff beyond memory: **sharing becomes a pointer operation**.

- `n=4` parallel samples share the blocks of the prompt. The engine counts the
  references and copies a block when the samples diverge.
- Beam search shares the common prefix across all beams.
- **Automatic prefix caching**: hash the contents of a block. One prefill of a
  shared 2000-token system prompt then serves *all* users, forever. The warm
  TTFT collapses.

The last one looks like cheating in production. It is a page cache.

---

## 3b. Three stages inside one kernel

Stages 08, 08b and 08c write the same function three times. The arithmetic never
changes. Only the constraint changes: a different part of the GPU limits each
version. The three stages have an order, so that a fix to one exposes the next.

**08 is about the decomposition.** The first decision in a kernel is what one
block owns. Everything else follows from it. Here a block owns one (sequence,
query head) pair, and it makes one `head_dim` output vector.

No block needs the result of another block. So the kernel has no global
synchronisation. That fact makes the decomposition correct, and not the
arithmetic.

Inside the block, stage 08 does the obvious thing. It gives one thread to each
context position, and each thread walks `head_dim` alone. It is correct. It is
approximately 5 times faster than the PyTorch loop, because a Python loop over
sequences was never real competition.

**08b is about the memory system.** The obvious mapping has one specific defect.
At one instruction, the 32 threads of a warp read 32 addresses that sit
`head_dim` elements apart.

The memory system serves a warp in 128-byte transactions. So it issues 32
transactions where 2 are sufficient. Every byte that you asked for arrives. You
spent ten times too many requests to get them.

Turn the mapping sideways. The threads then cooperate *along* `head_dim`, at 16
bytes each, and the requests merge. The arithmetic is the same and the answers
are the same. The kernel gains approximately **2x**. It moves from half of the
streaming bandwidth of the card to almost all of it.

Sixteen bytes is not an arbitrary number. It is the widest load that one thread
can issue. And 8 lanes at 16 bytes is exactly one 128-byte transaction.

Coalescing is the most valuable habit in CUDA. This is the cheapest place to
learn it.

**08c is about the execution resources.** Now the kernel is perfect on
bandwidth and still terrible at batch 1:

```
grid = (num_seqs, num_heads) = (1, 16) = 16 blocks
this laptop GPU has 58 SMs
```

Three quarters of the machine is idle. No memory optimisation helps, because
memory was never the problem. There are two fixes:

- **Warp shuffles.** The lanes that share a K row are in one warp, so
  `__shfl_xor_sync` can reduce their partial dot products by exchanging
  registers directly. It needs no shared memory and no `__syncthreads`. It is
  worth approximately 1.6x alone. It also trades one deadlock trap for another.
  A `_sync` primitive never returns if its mask names a lane that never
  arrives.

- **Split-K**, which is also called flash-decoding. If the (sequence, head)
  pairs cannot fill the GPU, make more blocks: cut the *context* into chunks.
  Each block computes a partial `(m, l, acc)`. A second kernel merges them with
  `exp(m_j - M)`. That is the same rescale that the online softmax already
  does, one level up, and it is exact. At one sequence this is worth **10x**.

Split-K is a trade. You see the other side of it at 64 sequences. There the grid
was already full, and there was no idle time to sell.

When each row group gets its own softmax, every lane computes its own maximum
and its own rescale. Before, one thread did that for the whole block.

Redundant arithmetic is free until the kernel stops to wait on memory. At a full
grid it is not free. You give back a few percent. A real engine dispatches on
the batch size for this exact reason.

Hold the four numbers together:

- 2x from how you read memory,
- 1.6x from how you reduce,
- 10x from enough blocks to fill the machine,
- and a few percent back, where the last one buys nothing.

These are not the same kind of win. A profiler separates them. A stopwatch does
not. That is what `./vc ncu` is for.

And one bug pays for the whole stage. A block reduction that returns its answer
through shared memory needs a barrier after every thread READS it. A barrier
before the read is not sufficient.

The next reduction uses the same scratch memory. A warp that runs ahead
overwrites a result that a slower warp did not collect yet. A few percent of the
output is then wrong on some inputs and correct on others.

`compute-sanitizer --tool racecheck` finds this bug in one run, and it names
both source lines. A stopwatch, a print statement and an afternoon do not.

### And the same three questions about a GEMV

Stage 18b is the shortest version of the whole argument. Weight-only int8 makes
the bytes that decode must read half as many. The decode time must then also
become almost half as large.

That is true on one condition. The dequantize must happen on a value that is
already in a register. If you materialise the bf16 weight first, you add a
full-size write and a full-size read. The kernel is then slower than no
quantization at all.

Then measure against cuBLAS, and watch the win disappear between one row and
four rows. A GEMV is not a small GEMM. That crossover is the ridge from stage 3
again, in milliseconds that you measured yourself.

---

## 4. Continuous batching: the other 4x

From **Orca (OSDI '22)**, and arguably as important as paging.

**A static batch**: collect 8 requests, and run until all 8 finish. A request
that needs 10 tokens sits padded and idle while a request that needs 500 tokens
finishes. In real traffic the output lengths differ by a factor of 10 to 100. So
most of your batch slots hold nothing.

**Continuous batching** means that the scheduler works at **iteration
granularity**. After *every* forward pass it decides the batch again. A finished
sequence leaves, and a waiting sequence joins. There is no padding to a common
length, and no wait for the slowest sequence.

One detail makes this work. Sequences of different lengths can share a batch,
because attention is per-sequence in any case. So flatten all of the tokens into
one ragged tensor, and give the kernel a length array (`cu_seqlens`). There is
no padding, at any time.

---

## 5. Prefill vs decode: one engine, two workloads

|  | Prefill | Decode |
|---|---|---|
| Input | Whole prompt at once | 1 token per sequence |
| Bottleneck | **Compute** (big GEMMs) | **Memory bandwidth** |
| Arithmetic intensity | High | ~1 op/byte |
| Parallelism | Within a sequence | Across sequences |
| Latency metric | **TTFT** | **TPOT / ITL** |
| Wants | Few big chunks | Max batch size |

The two workloads fight. One 8000-token prefill fills a whole step. Every
sequence in the middle of generation stalls, and the user sees the token stream
stop.

**Chunked prefill** (Sarathi-Serve) cuts the prefill into fixed token budgets.
It then schedules the chunks together with the decodes, in one mixed batch. The
decode tokens, which wait on bandwidth, travel with the prefill GEMM, which
needs compute. That fills both sides of the roofline.

This is the main dial between throughput and latency in a modern server.

**Disaggregated prefill** goes further. It gives prefill and decode separate GPU
pools, and it sends the KV blocks between them across the network. That isolates
TTFT from TPOT completely. Large deployments move in this direction.

---

## 6. Timeline

| When | Idea | Why it mattered |
|---|---|---|
| 2022 | **Orca** — iteration-level scheduling | Continuous batching. Stop waiting on stragglers. |
| 2022 | **FlashAttention** | Tiles and an online softmax. The N x N score matrix never exists. |
| 2023 | **vLLM / PagedAttention** (SOSP) | KV cache as virtual memory. Kills fragmentation. |
| 2023 | **Speculative decoding** | A check on K tokens costs about as much as one new token. It loses nothing. |
| 2023 | **S-LoRA** | Thousands of LoRA adapters on one base model, paged like KV. |
| 2023 | **GQA** goes mainstream | Model architecture bends to KV cache pressure. |
| 2024 | **Sarathi-Serve** — chunked prefill | Stop prefill from stalling decode. |
| 2024 | **Automatic prefix caching** | Hash the contents of a block. Requests then share a prefix. |
| 2024 | **FP8 KV cache** | Halve the other big memory reader. |
| 2025 | **vLLM V1 rewrite** | Isolated EngineCore process, unified scheduler, persistent batch. |
| 2025+ | **Disaggregated P/D** | Separate fleets for prefill and decode. TTFT and TPOT stop interfering. |

---

## 7. Anatomy of the real vLLM

This is approximately what you rebuild. Read it, and you can then read the real
source:

```
AsyncLLM  (HTTP / OpenAI-compatible surface)
   |  IPC
EngineCore  ── own process, so Python on the API side can't stall the GPU
   |
   ├── Scheduler ──── waiting / running queues, token budget per step,
   |                  preemption, chunked prefill; emits a SchedulerOutput
   |
   ├── KVCacheManager ── BlockPool, free list, refcounts, prefix-cache hash table
   |
   └── ModelRunner  ── builds the flat input tensors + block tables,
        |              owns CUDA graphs, runs the sampler
        └── Model ── attention backend (FlashAttention / FlashInfer / CUDA)
```

**Key V0 → V1 changes** (2025), and you find all of them again yourself:

- The engine loop moved into its **own process**. The Python overhead on the API
  side stalled the GPU between steps by a measurable amount.
- **The scheduler no longer separates prefill from decode.** There is one token
  budget for each step. A request contributes as many tokens as it needs.
  Chunked prefill is now the default, and not a mode.
- **A persistent batch.** The engine changes the input tensors in place across
  steps. It does not build them again. So the CUDA graphs stay valid, and the
  Python work in each step falls to approximately zero.
- Prefix caching is now on by default. The hash lookup became cheap enough to be
  free.

---

## 8. Numbers worth memorizing

- KV cache per token, Llama-3-8B (32 layers, 8 KV heads, head_dim 128, bf16):
  `2 * 32 * 8 * 128 * 2 = 128 KB/token`. A 4k-token conversation = **512 MB**.
  On your 12 GB card, after ~5 GB for an 8B model in fp8, that is approximately a dozen
  such conversations. **That number is your throughput.**
- Block size 16 is the standard default. It is large enough to spread the cost
  of a block-table lookup. It is small enough that the average waste, 8 tokens
  for each sequence, is noise.
- A decode step at batch 1 on a small model is approximately 1 ms of GPU work.
  Python and the kernel launches can cost the same, if you do not optimise them.
  That is why CUDA graphs exist.
- A good draft gives a speculative decoding acceptance rate of 60% to 80%.
  Expect 1.5x to 2.5x. Expect almost nothing on creative text, which has a high
  entropy.

And for the kernels, four hardware constants and what they cost you:

- A **warp is 32 threads** and they issue one instruction together. Every
  reduction, divergence and shuffle question starts here.
- The memory system serves a warp in **128-byte transactions**, and the widest
  load one thread can issue is **16 bytes**. 8 lanes x 16 bytes is exactly one
  transaction, which is why stage 08b targets that width and gets ~2x.
- **Registers for each SM**, which is 65536 on most modern cards. Divide that
  number by the registers for each thread. The result limits how many warps an
  SM holds, and that is how much memory latency it hides. `ptxas` prints the
  numerator. `./vc info` prints the denominator.
- **The blocks must be more than the SMs**, by a large margin. If they are not,
  the machine idles. At batch 1, a (num_seqs, num_heads) grid gives 16 blocks
  against approximately 58 SMs. That one fact is worth 10x in stage 08c.

---

## 9. Vocabulary

`course/GLOSSARY.md` defines each term of the course. Each entry has an
analogy from software engineering, and the notebook and the stage that teach
the term. `./vc guide` prints the terms of each stage.

---

## 10. The second track: what changes when shapes are frozen

Everything above is about one physical fact, memory bandwidth, and the moves
that it forces. None of it depends on PyTorch.

But the *shape* of the code that makes those moves depends on one thing very
strongly. Does your framework dispatch kernels at runtime, or does it compile
programs in advance?

Run `./vc backend jax`, and you build the ladder again on XLA. Eleven of the
twenty stages have a JAX twin. The other nine are pure logic, and they are the
same file on both tracks:

- the allocator and the prefix cache,
- the scheduler and the chunked prefill,
- the detokenizer, the server and the metrics,
- the speculative decoding and the guided decoding.

That split is the lesson. **Most of an inference engine is not framework code.**
The bookkeeping is the product.

Where the twins diverge, they diverge from one root cause.

### XLA compiles for exact shapes

PyTorch dispatches one kernel for each op, at runtime, from the shape that the
tensor has at that moment.

XLA compiles a whole program for one specific set of shapes. A different set of
shapes is a different program. XLA traces and compiles it again, and that costs
seconds of wall clock on the critical path.

So the tax you are trying to remove is not the same tax:

| | torch | jax |
|---|---|---|
| stage 12's enemy | thousands of kernel launches per step | one compile per unseen shape |
| the fix | CUDA graphs | shape buckets + AOT compile |
| what makes it work | static pointers and shapes | static shapes |

The same fix comes from a completely different reason. Think about that. It is
the strongest evidence that a bucket is not a CUDA trick. You use a bucket
whenever the cost to *prepare* the work is large next to the work.

### The KV cache stops being a thing you grow

torch gives you a cache that is one token longer at each step. In JAX that means
one recompile for each token. So you **preallocate the cache to max_len and
write into it** with a dynamic slice. The shapes never change, and one compile
serves the whole decode.

Which drags four stages sideways:

- **02.** You own the cache. Call `init_cache(batch, max_len)`, and carry
  `cache_len` yourself. Round max_len up to a bucket, and a prompt of 200 tokens
  and a prompt of 250 tokens then share one compiled decode step. That is the
  idea from stage 12, and you find it on stage 2.
- **04.** Pad on the right, and not on the left. Read the logits from an index
  for each row. The waste from the padding is the same. Only the layout moved.
- **05.** Eviction cannot make the batch smaller, because the batch dimension
  *is* the compilation. So the batch is a fixed **slot table**. A sequence that
  finishes frees a slot, a new sequence fills one, and the engine copies
  nothing. That is a page table with one page for each sequence, two stages
  before you build the real one.
- **05, again.** A step costs the same with one busy slot or with eight. So the
  win from continuous batching appears only as **useful tokens for each forward
  pass**. It does not appear as a faster step. On this track, occupancy is the
  whole game.

### Two places the JAX version is simply better

- **Stage 08** is Pallas and not CUDA. The online softmax is the same. One thing
  changes: the block-table lookup must happen inside the kernel, with a computed
  index. No BlockSpec can say "which page do I need? ask the page table."
- **Stage 20** stops to be a simulation. `shard_map` over a mesh of CPU devices
  gives you real shardings and a real `psum`. So "column-parallel, then
  row-parallel, with exactly one all-reduce" becomes two PartitionSpecs that you
  can read. The test then counts the collectives in the compiled HLO, and proves
  that there is one.

### And one place it is honestly worse

JAX writes the preallocated cache again at every step, for each layer, inside
the scan. So the decode time grows with the ceiling that you *chose*, and not
with the context that you *hold*.

That is approximately 1.3x from a 128-slot cache to a 1024-slot cache. The torch
track is almost flat over the same range. Buffer donation does not fix this. The
copy is inside the scan.

The fix is to make the buffer granular, so that a step touches only the blocks
that it needs. That is PagedAttention. The JAX track gives you a reason to want
stages 06 to 09 before you reach them.

---

## 11. The capstone: the parts meet

Stages 06 to 20 each build one part, and each part passes its checks alone.
That proves the parts. It does not prove the engine. These facts only appear
when the parts run together, and the capstone (stages 21 to 28) is where you
meet them.

**The host decides the shape of the engine.** The HuggingFace model grows its
cache by concatenation, and it hides the attention call. No paged kernel can
live inside it. So the capstone model, `tvllm/model.py`, takes a FLAT batch
and calls an attention backend that you write. That one interface change lets
a single step hold decodes and prefill chunks from many sequences.

**The scheduler has only one rule.** Stages 10 and 11 were two separate
simulations. In a real engine, every sequence has pending tokens: one for a
decode, many for a prefill, all of them after a preemption. Each step gives
pending tokens to sequences until the budget runs out. vLLM V1 schedules like
this, and it is why V1 has no separate prefill step.

**Padding must go somewhere harmless.** A captured graph has a fixed batch, so
the engine pads. Stage 12 padded with zeros. In a paged engine, slot 0 is
block 0, and block 0 belongs to a real sequence. A padding row must write to
slot -1, which the stage 08 kernel skips.

**A real step blocks the event loop.** Stage 15 called step() inline, and the
fake step took no time. A real step holds the GPU for 5 to 50 ms, and no HTTP
request moves for that time. Put step() in a worker thread, and send it
commands through a queue. vLLM V1 separates the API server and the engine core
for the same reason.

**A kernel has a crossover, so the engine dispatches.** Your int8 GEMV wins at
a few rows and loses above them, because the rows then share the weight read.
So each matmul takes the int8 path below a crossover and bf16 above it. Measure
the crossover in the real graphed step. One matrix of a small model fits in L2,
and a benchmark of it gives the wrong answer.

**At long context, the KV cache is the bytes.** At 16 sequences of 2048 tokens
the KV read is three times the weight read. An FP8 cache halves it. The kernel
must then convert two values with one instruction, or it waits on arithmetic
and gives back most of the win.

**A verify is a decode batch.** K draft tokens are K+1 decode rows that share
one block table. Each row sees its own context, so the verify is causal, it
runs on the decode kernel, and it replays a graph. With greedy decoding the
output is the same token for token. With sampling, the rejection rule of stage
17 keeps the distribution exact.

**A grammar mask is a cache lookup.** A JSON automaton has a few dozen states
in one answer, and the mask depends only on the state. Build each mask one
time and keep it. Near max_tokens, admit only the tokens that bring the object
closer to complete, or the answer ends open.

### What the capstone measures, and why as ratios

The last stage divides the roofline floor of the steps that your engine ran
by the time that they took. For each step:

```
floor = max(weight_bytes / BW,  2 * params * tokens / FLOPS)
      + context_tokens * kv_bytes_per_token / BW
```

A throughput number changes from card to card. This ratio does not, because
the floor changes by the same factor. The reference engine reaches 39% to 59%
of the roof on a laptop. The gap is where to work next:

- **Prefill steps run eager.** Capture a mixed step, or write a paged prefill
  kernel, so that a prefill chunk does not gather its whole context first.
- **About a thousand small kernels in each step.** Fuse RoPE with the Q and K
  norms, and fuse the residual add with the next norm.
- **The host syncs one time in each step.** Overlap the host work of step n+1
  with the GPU work of step n.
- **Int8 capacity, not only latency.** Stage 24 keeps a bf16 copy of each
  weight for the large batches. A mixed-precision kernel that is fast at every
  batch size, like Marlin in vLLM, removes that copy.
- **Speculation for guided requests.** A JSON request does not speculate now.
  Check each draft token against the automaton, and it can.
- **Tensor parallelism in the engine.** Stage 20 shards on one machine with
  gloo. It needs two GPUs to be real.

---

## Appendix A: "What if a model fit entirely in cache?"

This is a reasonable question, after you accept that decode waits on memory.
What if the weights were small enough to live in the on-chip cache of the GPU,
and not in VRAM? Run `./vc cliff` to measure it on your own card.

### The cliff is real

```
working set   achieved bandwidth
    8 MB       1,531 GB/s   fits in 50MB L2
   32 MB       1,985 GB/s   fits in 50MB L2
   64 MB         412 GB/s   spills to VRAM     <- cliff
  256 MB         380 GB/s   spills to VRAM
```

That is approximately **5 times the bandwidth**, and the cliff falls exactly
where L2 runs out. A model that fits in the cache really does decode
approximately 5 times faster at batch 1. The argument "you must batch to
approximately 150" then becomes much weaker.

### But look what fits in 50 MB

| Precision | Params that fit |
|---|---|
| bf16 (2 bytes) | 25M |
| int8 (1 byte) | 50M |
| int4 (0.5 bytes) | 100M |
| 1.58-bit ternary | ~250M |

GPT-2 small was 124M. BERT-base was 110M. So "it fits in the cache" means
approximately the capability of 2019.

To put frontier intelligence into that space, you need approximately a 1000x
gain in intelligence for each parameter. The scaling laws move the other way.
Capability climbs with the logarithm of the parameter count.

### The catch that does not go away

Give yourself the magic 25M-parameter genius anyway. **The KV cache does not
become smaller.** The context length and the layer count set its size. The
parameter count does not. A long conversation still needs hundreds of MB.

So the bottleneck does not disappear. It *moves*. The weights stop to be the
largest read, and the KV cache becomes almost all of it.

Everything in stages 06 to 09 then becomes **more** important, and not less:
paging, prefix sharing and eviction. The memory-bound problem does not change
with scale.

### The industry already made this bet, in silicon and not in models

Nobody waited for the models to become smaller. Companies built chips with
enough SRAM to hold the model:

- **The Groq LPU**: a few hundred MB of on-chip SRAM, and *no external DRAM*.
  The whole sales argument is what the roofline predicts. The batch-1 latency is
  exceptional, because the chip never touches HBM.
- **Cerebras**: wafer-scale, with tens of GB of on-chip SRAM.

They scale up when they connect many chips together, so that one model spans the
combined SRAM. The theory holds, and the latency win is real.

The open question is the cost for each token. A GPU takes back its bandwidth
disadvantage with a batch. See the 4,000 tok/s row in section 1. An SRAM machine
wins on latency, and it needs much silicon for each model. Does that work at
scale? That is a live commercial argument, and not a settled one.

You cannot use this trick on a GPU. L2 is a *cache*, and not a scratchpad. You
cannot pin residency, and the hardware evicts what it wants.

Groq manages its SRAM explicitly, with a deterministic schedule. That is exactly
why the trick works there and is difficult here.

### Where the idea already pays off

- **Speculative decoding** (stage 17). The draft model is very small, so it
  effectively lives in the cache. You get the speed of the small model *and* the
  output distribution of the big model. That is as near as you get to both, and
  it is why the technique works.
- **MoE.** A 400B model with 15B active parameters for each token moves 15B of
  bytes, and not 400B. It is the same instinct: cut the bytes for each token,
  and do not cut the capability.
- **On-device.** A phone NPU that runs a model of a few hundred MB already lives
  in this regime.

**The verdict:** the mechanism is real. The bet is not that "models become small
enough for the cache". The bet is that "hardware grows enough SRAM". That bet
already has money behind it, and the chips already ship.

The KV cache kills the pure version. It scales with the context, and not with
the parameters. So it refuses to disappear.

---

## 11b. Reading the real vLLM source

The strongest exercise at the end of this course is not another benchmark. It
is: open the upstream vLLM scheduler, map it to the class you wrote, and list
what upstream does that you do not.

Start here:
[`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

Your `Scheduler` in `app/s22_engine.py` is about 100 lines. The upstream
scheduler is about 600. The difference is not complexity for its own sake.
Every extra line handles a case that a single-GPU teaching engine does not
see:

- **LoRA slot management.** A LoRA adapter uses GPU memory. The scheduler must
  know which adapters are loaded and swap them.
- **Priority scheduling.** Some requests pay more. The scheduler must admit
  them first.
- **Encoder-decoder models.** The encoder output has a different shape and
  different memory. The scheduler tracks both.
- **Multi-step scheduling.** The scheduler plans multiple decode steps ahead
  to amortize the Python overhead.
- **Structured output state.** The grammar state of a guided-decoding request
  travels with the request through preemption and resumption.

Send one small patch to the upstream project. A typo fix, a docstring, a test.
That is how a course about vLLM should end.

---

## 12. The batch-invariance problem

Two bf16 runs that differ only in the order of additions (because the batch
size changed) disagree at about 2% of token positions. Users report this as
"the same prompt gives different text when the server is busy."

This is a current, open problem. Floating-point addition is not associative:
`(a + b) + c ≠ a + (b + c)` in general. When the batch size changes, the
reduction tree inside a matmul changes, and the rounding changes with it.
The difference is tiny in logits (≈ 1e-6) but large enough to flip the
argmax at about 2% of positions.

**Why this matters for a serving engine.** A user who sends the same prompt
twice, with the same seed, expects the same text. If the server was busier on
the second call, a different batch size means a different reduction order,
which means different logits, which means different tokens. This breaks
reproducibility, and there is no simple fix.

**What you can do.** Measure it: run the same prompt at batch 1 and batch 32
with the same seed. Count the positions that differ. If the rate is above zero,
you now know why, and you can explain it in an interview.

---

## 13. The Engine Process Split (vLLM V1)

In early serving engines (including vLLM v0 and stage 15/27 in single-process mode),
the HTTP server (FastAPI/Uvicorn), detokenizer, and GPU execution loop share
a single Python interpreter. Even with worker threads, the Python Global Interpreter
Lock (GIL) and garbage collection create execution bubbles:

1. **GIL Contention:** Serializing a large JSON response or decoding token strings
   in the event loop blocks the worker thread from submitting the next GPU step.
2. **GC Pauses:** Allocation and cleanup of thousands of request objects pauses
   all Python threads, starving the GPU and dropping SM utilization.

vLLM V1 moved the engine loop into a completely isolated child process communicating
over non-blocking IPC queues. The main process handles HTTP connections, request
parsing, and token streaming, while the child process runs CUDA graph steps without
interruption. You can run this mode via `./vc serve --multiprocess` or inspect
`ProcessEngineWorker` in `app/s27_serve.py`.

---

## 14. Cache-Aware Scheduling and Goodput

Two concepts distinguish production engines from toys:

- **Cache-Aware Scheduling:** Standard Orca-style schedulers admit waiting requests
  strictly FIFO (First-Come, First-Served). But when a prefix cache is present,
  different waiting requests hit different amounts of precomputed KV blocks.
  Admitting a request with a 100-token cached prefix costs 1/10th the prefill time
  and requires fewer fresh blocks than a cold request. Cache-aware admission scores
  candidates in the queue and schedules prefix hits first, maximizing overall
  cluster capacity.
- **Goodput over Raw Throughput:** Benchmarking batch-at-once raw tokens/sec ignores
  real user service level objectives (SLOs). Goodput measures output tokens per second
  that actually arrive within target Time-To-First-Token (TTFT) and Inter-Token-Latency (ITL)
  deadlines. If an engine delivers 10,000 tok/s but every client experiences 5-second
  jitters, the goodput is 0. Stage 28 gates on goodput under continuous load.

---

## 15. CUDA Graphs vs. Prefix Caching: The Replay Hazard and Staging Buffers

CUDA Graphs (stage 23) eliminate CPU kernel launch overhead by baking an entire execution graph
of thousands of GPU operations into hardware command buffers. But combining CUDA Graphs with
dynamic prefix caching (stage 09/22) introduces a critical memory hazard:

### The Replay Hazard
At graph capture time, tensor memory addresses are permanently fixed. During decoding,
different requests share or acquire physical KV blocks dynamically via prefix caching.
If you pass the new block tables into the graph's static input tensors:
1. **The Block 0 Corruption Trap:** Padded batch rows MUST have `slot_mapping = -1` and
   `context_len = 0`. If a padding row accidentally writes to slot `0` or references block `0`,
   the graph kernel will write garbage KV entries into block 0, which belongs to a live sequence!
   This corrupts other users' cached prefixes silently.
2. **Race Conditions & Stale Pointers:** If host-to-device transfers are dispatched asynchronously
   without stream synchronization, `graph.replay()` will launch immediately and execute
   attention against stale, half-copied block table indices.

### Persistent Pinned Staging Buffers
In naive implementations:
```python
host = torch.tensor([token_ids, positions, slots, context_lens])
device_tensor.copy_(host, non_blocking=True)
```
Every step calls the CPU memory allocator to create a brand new unpinned host tensor.
Because the memory is unpinned (pageable), the CUDA driver cannot perform asynchronous DMA;
it falls back to a synchronous copy that stalls the CPU worker thread.
Production engines use **pre-allocated pinned host memory staging buffers** (`pin_memory=True`):
```python
staging.token_ids.copy_(token_ids)
device.token_ids.copy_(staging.token_ids, non_blocking=True)
torch.cuda.current_stream().synchronize()  # Guarantee visibility before graph.replay()
```
This guarantees zero CPU allocations during decode, true asynchronous DMA transfer,
and complete memory safety before graph execution.

---

## 16. Chunked Prefill, Decode Interference, and SM Partitioning

Mixed batching (running prefill and decode simultaneously in one step) solves the prefill bubble
problem, but introduces a severe hardware contention problem: **Streaming Multiprocessor (SM) Starvation**.

### The Arithmetic Asymmetry
- **Prefill:** Compute-bound matrix multiplication (GEMM: $M \times K \times N$ where $M$ is prompt length).
  High arithmetic intensity ($\approx 100\text{ FLOP/byte}$). Prefill kernels launch hundreds of
  thread blocks that saturate all SMs and Tensor Cores for tens of milliseconds.
- **Decode:** Memory-bandwidth-bound matrix-vector multiplication (GEMV: $M = 1$).
  Low arithmetic intensity ($\approx 1\text{ FLOP/byte}$). Decode requests are latency-critical:
  users expect Inter-Token Latency (ITL) under 20-30 ms.

### The Interference Trap
When a 2,048-token prefill shares a step with 16 decode streams:
The prefill GEMM waves occupy every SM on the GPU. The small decode thread blocks are queued
behind prefill waves in the hardware scheduler. Decode tokens that normally take 2 ms to compute
are delayed by 50-100 ms, causing massive ITL spikes (p99 latency collapse).

### How Production Engines Fix It
1. **Chunked Prefill (Stage 21/22):** Break long prefills into chunks capped by a strict `token_budget`
   (e.g., 512 tokens). No single prefill wave can monopolize the GPU indefinitely.
2. **SM Partitioning (CUDA Green Contexts / MPS):** Modern GPUs allow dividing hardware SMs
   into isolated partitions. For example, reserve 20% of SMs exclusively for decode kernels
   so decode ITL is completely immune to prefill compute storms.
3. **Disaggregated Prefill-Decode (Splitwise / Mooncake):** Separate physical GPU clusters for prefill
   and decode, transferring KV blocks across nodes over high-speed RDMA InfiniBand networks.

---

## 17. Tree-Attention Speculative Decoding (EAGLE & Medusa)

Stage 17 demonstrated linear speculative decoding: draft $K$ tokens in a sequence, verify them
in one target forward pass. But linear drafting suffers from exponential decay:
$$\text{Expected accepted tokens} = \sum_{i=1}^K a^i$$
If per-token acceptance rate $a = 0.6$, the probability of accepting 5 consecutive tokens is $0.6^5 \approx 7.7\%$.
Beyond $K=4$, linear drafting pays compute to generate draft tokens that are almost always discarded.

### Multi-Branch Tree Speculation
Modern speculative engines (Medusa, EAGLE, SpecInfer) propose a **tree of candidate hypotheses**
rather than a single chain. For example, propose 3 candidates for token 1, and 2 candidates for
each branch of token 2, creating a tree of 16 candidate tokens.
All 16 candidate tokens are verified simultaneously in **ONE target model forward pass**!

### The 2D Tree Attention Mask
To evaluate multiple branching hypotheses without cross-branch interference, the target model
uses a custom 2D attention mask:
- Candidate token $i$ can attend to candidate token $j$ if and only if $j$ is an ancestor of $i$
  in the draft tree (or $j == i$).
- Sibling branches cannot attend to each other ($M[i, j] = 0$).
- All candidate tokens attend to the full shared prompt prefix.
The target model verifies all candidate paths in parallel and accepts the longest valid path,
emitting the accepted tokens plus one bonus token from the winning leaf.

---

## 18. Virtual Memory Management with `cuMemMap` (vLLM V1)

In stage 06, we built a `BlockAllocator` managing logical blocks within a contiguous PyTorch tensor:
```python
kv_cache = torch.empty((num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim), device="cuda")
```
This is how vLLM v0 worked. In production, this static contiguous allocation has severe limitations:

### The Fragmentation Bottleneck
1. **Contiguous Allocation Failure:** Allocating a 30 GB contiguous tensor at startup requires a single
   unbroken slab of virtual and physical VRAM. If weight loading or CUDA runtime initialization
   fragments VRAM, the allocation fails with CUDA OOM even if 35 GB of VRAM is free!
2. **Static Pool Sizing:** You cannot dynamically expand the KV cache pool without reallocating
   the entire tensor and copying gigabytes of existing KV data.

### Low-Level Driver VMM Architecture
vLLM V1 rebuilt memory management using CUDA low-level Virtual Memory Management driver APIs:
1. **`cuMemAddressReserve`:** Reserve a huge contiguous range of *virtual address space*
   (e.g., 128 GB) at zero physical memory cost. (Virtual address space is 48-bit and practically infinite).
2. **`cuMemCreate`:** Allocate physical memory chunks in fixed 2MB pages (`CUmemGenericAllocationHandle`).
3. **`cuMemMap`:** Map physical pages into arbitrary virtual address ranges on demand.
4. **`cuMemSetAccess`:** Set GPU read/write permissions for the mapped ranges.
5. **`cuMemUnmap`:** Decouple physical pages from virtual address space instantly when blocks are freed.

This architecture completely decouples virtual addresses from physical GPU memory. The KV cache
can grow and shrink page by page, physical pages can be remapped without copying data, and contiguous
memory fragmentation is eliminated entirely.

---

## 19. GPU Profiling with Nsight Systems & Nsight Compute

In a production engine, you do not optimize based on intuition. You optimize based on hardware counters.

### Whole-System Timelines: `./vc nsys` (Nsight Systems)
Nsight Systems exposes the **interaction between CPU runtime and GPU hardware queues**:
1. **Launch Latency Bubbles:** If the CPU scheduler takes 2 ms between steps, the GPU timeline
   shows an empty gap. Even if your kernel is 100% optimal, the card is idle.
2. **Transfer Synchronization:** Asynchronous host-to-device transfers must happen on a stream.
   If unpinned memory is passed, the transfer becomes synchronous and halts the event loop.

### Kernel Deep-Dives: `./vc ncu` (Nsight Compute)
Nsight Compute inspects SM internal execution. Five counters define a kernel:
- `dram__bytes.sum.per_second`: Achieved memory bandwidth.
- `smsp__average_data_bytes_per_sector_mem_global_op_ld.pct`: Global memory coalescing efficiency.
- `sm__warps_active.avg.pct_of_peak_sustained_active`: Achieved occupancy.
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`: Shared memory bank serialization.
- `smsp__warp_issue_stalled_barrier_pct`: Warps waiting at synchronization points.

---

## 20. Modern C++ & Tensor Cores (CUTLASS vs. Hand-Written CUDA)

Stage 08 implemented paged attention using scalar arithmetic on CUDA cores. In production AI
infrastructure, almost all matrix multiplications run on **Tensor Cores**.

### SIMT vs. Cooperative Matrix Multiply
- **CUDA Cores:** Thread-independent scalar operations. Each thread issues its own arithmetic.
- **Ampere / Ada Tensor Cores (HMMA):** 32 threads in a warp collaborate to execute matrix multiplication
  via `mma.sync.aligned.m16n8k16` in hardware registers.
- **Hopper / Blackwell Tensor Cores (WGMMA):** 128 threads in a Warp Group execute asynchronous
  matrix multiplications (`wgmma.mma_async`), reading directly from shared memory without register staging.

### Why Systems Engineers Use CUTLASS
Writing raw PTX for Tensor Cores requires manual software pipelining and bank conflict resolution.
NVIDIA's **CUTLASS** library decouples these using C++ templates:
- **Tiling:** Divides GEMMs into Threadblock-level, Warp-level, and Instruction-level tiles.
- **Collective Pipelines:** Automates multi-stage shared memory buffering.
- **Epilogues:** Fuses activation functions, bias additions, and scaling directly into the output pass.

---

## 21. TensorRT-LLM vs. vLLM: The Dual-Engine Universe

When deploying models on NVIDIA silicon, two systems dominate the landscape:
**vLLM** (open-source Python/C++ engine) and **TensorRT-LLM** (NVIDIA's specialized C++ inference engine).

The architectures are twins expressed in different languages:
- `vLLM Scheduler` corresponds to TensorRT-LLM's `GptManager` and `BatchingManager`.
- `vLLM BlockManager` corresponds to TensorRT-LLM's `KvCacheManager`.
- `vLLM PagedAttention` corresponds to TensorRT-LLM's `PagedKvCache` and masked attention plugins.
- `vLLM CUDA Graphs` corresponds to TensorRT's compiled `.engine` execution plan.

Understanding `vllm-from-scratch` gives you the exact blueprint required to navigate, profile, and optimize both ecosystems.

---

## 22. LRU Prefix Eviction: Radix Trees Under Memory Pressure

Prefix caching (Stage 09/10) enables multiple requests sharing a prompt prefix to reuse physical KV blocks without redundant prefill computation. However, a naive prefix cache faces a dilemma:

1. **Immediate Freeing:** If blocks are freed when a sequence finishes, multi-turn chat sessions and popular system prompts lose their cached state immediately.
2. **Permanent Caching:** If completed sequence blocks remain pinned in memory indefinitely, new requests with unfamiliar prompts will trigger `OutOfBlocks` errors and crash or stall the engine.

The solution deployed in production vLLM is an **LRU Eviction Policy combined with Reference Counting**:

- **Active Blocks (`ref_count > 0`):** Blocks currently assigned to running sequences. These are strictly protected from eviction.
- **Cached Inactive Blocks (`ref_count == 0`):** Blocks whose requests have finished. Instead of being returned to the free list, they remain indexed in the prefix radix tree and enter a doubly-linked LRU queue.
- **Eviction on Demand:** When `BlockAllocator.allocate()` cannot satisfy an allocation request, it calls `LRUPrefixCache.evict(needed_blocks)`. The cache pops the least recently accessed blocks with zero references, clears their cache mapping, and recycles their physical IDs back to the allocator.
- **Cache-Aware Scheduling:** In high-concurrency environments, incoming requests in the waiting queue are sorted descending by their prefix cache hit length (`CacheAwareScheduler`). Prioritizing requests that hit warm cache blocks minimizes prefill time, drastically lowers Time-To-First-Token (TTFT), and maximizes system goodput.

---

## 23. Multi-LoRA Serving and Batched GEMV (BGMV)

Enterprise inference often requires serving hundreds of customer-specific LoRA adapters (Low-Rank Adaptation) simultaneously. Hosting hundreds of distinct base models in GPU memory is impossible, but maintaining the base model while applying distinct adapter deltas per sequence is tractable.

A LoRA adapter updates base weight $W_0$ with low-rank matrices $A$ and $B$:
$$W = W_0 + \frac{\alpha}{r} (A \cdot B), \quad A \in \mathbb{R}^{d \times r}, \quad B \in \mathbb{R}^{r \times k}$$

In an unbatched setting or naive grouping, requests for different adapters must run in separate forward passes, annihilating the throughput benefits of continuous batching.

vLLM solves this using **Batched GEMV (BGMV)**:
1. **Unified Base Matmul:** All sequences in the current batch (regardless of which adapter they use) are grouped into a single standard GEMM forward pass through the base model weights $W_0$:
   $$Y_{base} = X \cdot W_0$$
2. **Per-Sequence LoRA Dispatch:** For the adapter delta, decode tokens have a sequence length of 1 ($M=1$). Multiplying activation vector $x_i \in \mathbb{R}^{1 \times d}$ by low-rank matrices $A_i$ and $B_i$ is a Vector-Matrix product (GEMV).
3. **The BGMV Kernel:** A specialized CUDA kernel takes the batch activations $X \in \mathbb{R}^{B \times d}$, an integer index array `adapter_indices[B]`, and adapter weight tables. Each CUDA thread block is assigned a sequence $i$, indexes into its assigned adapter matrices $A_{\text{adapter}}$ and $B_{\text{adapter}}$, computes the low-rank delta in registers, scales by $\frac{\alpha}{r}$, and accumulates into the base output:
   $$Y_i = Y_{base, i} + \frac{\alpha_i}{r_i} (x_i \cdot A_i \cdot B_i)$$

This enables 50+ fine-tuned adapters to run concurrently within a single inference batch with virtually zero latency degradation compared to base-model inference.

---

## 24. DeepSeek Multi-Head Latent Attention (MLA) and Decode Weight Absorption

At long context lengths (e.g., 64k to 128k tokens), the KV cache becomes the primary memory bottleneck of modern LLMs. In standard Multi-Head Attention (MHA), each token stores $2 \cdot H_{kv} \cdot D$ elements. For a 70B parameter model with 128k context, the KV cache alone requires tens of gigabytes per sequence.

DeepSeek-V2 and DeepSeek-V3 introduced **Multi-Head Latent Attention (MLA)** to break this memory wall through low-rank compression and mathematical projection absorption:

### 1. KV Low-Rank Compression
During prefill, input token hidden state $x_t$ is projected into a compressed latent KV vector $c_t^{KV}$:
$$c_t^{KV} = x_t W_{DKV}^T \quad (d_c \ll H \cdot D)$$
Because rotary positional embeddings (RoPE) cannot be cleanly applied to compressed latent representations without losing relative position invariance, MLA extracts a separate small decoupled RoPE key:
$$k_t^R = x_t W_{KR}^T \quad (d_R \ll D)$$
The KV cache stores **only** $(c_t^{KV}, k_t^R)$. For typical dimensions ($H=8, D=32, d_c=64, d_R=16$), the stored state shrinks from 512 floats to 80 floats per token—a **>84% memory reduction**.

### 2. The Decode Weight Absorption Trick
Naively decompressing $K = c^{KV} W_{UK}^T$ and $V = c^{KV} W_{UV}^T$ into memory during decode would defeat the purpose of compression, incurring massive memory bandwidth penalties.

MLA exploits the associative property of matrix multiplication:
$$Q \cdot K^T = Q (c^{KV} W_{UK}^T)^T = Q W_{UK} (c^{KV})^T = (Q W_{UK}) (c^{KV})^T$$

During decode attention:
1. **Absorb $W_{UK}$ into Query:**
   $$Q_{\text{absorbed}} = Q_{\text{nope}} W_{UK}$$
   Attention scores are computed directly between $Q_{\text{absorbed}}$ and the compressed latent cache $c^{KV}$ without ever materializing uncompressed keys in VRAM.
2. **Absorb $W_{UV}$ into Output:**
   Attention weights are multiplied directly by the compressed latent cache $c^{KV}$ to produce a compressed context vector $c_{\text{context}}$. The uncompression matrix $W_{UV}$ is then applied once to the aggregated context:
   $$\text{Output} = c_{\text{context}} W_{UV}$$

Decode attention executes directly against compressed latents with zero intermediate decompression and zero mathematical approximation error.

---

## 25. System 1 Decision Models: Non-Autoregressive Typed Inference (TypeSafe Jev)

In classical cognitive science, Daniel Kahneman distinguishes between **System 1** (fast, instinctive, automated decisions) and **System 2** (slow, deliberate, step-by-step reasoning). 

Modern LLM serving engines are almost exclusively built for **System 2**:
- They generate output sequentially, token by token.
- Each generated token requires reading the entire KV cache and all model weights from HBM.
- A 100-token response requires 100 round-trips through GPU memory, bound by memory bandwidth.

However, many critical software automation tasks—such as **request classification, front-door guardrails, quality grading, and router dispatch**—do not require conversational prose. They require **typed, deterministic, probabilistic decisions**.

TypeSafe AI introduced **Jev**, a specialized non-autoregressive "System 1" model designed specifically for software decisions rather than free-form text:

### 1. The Three Typed Primitives
Rather than sampling over a 150k vocabulary of sub-word tokens, a System 1 model maps hidden representations directly to typed schema heads:
1. **`Choice`:** Categorical distribution over discrete targets (e.g. `["billing", "support", "technical"]` or adapter IDs) with normalized confidence probabilities:
   $$\mathbf{p} = \text{Softmax}(W_{\text{choice}} \cdot h_L)$$
2. **`Score`:** Continuous calibrated scalar regression bounded to an explicit range $[A, B]$ via scaled sigmoid projection:
   $$\text{Score} = A + (B - A) \cdot \sigma(w_{\text{score}}^T h_L + b)$$
3. **`Noul`:** A pure probability $p \in [0, 1]$ evaluating the belief/truth value of a boolean proposition (e.g. `is_jailbreak_prompt`):
   $$\text{Noul} = \sigma(w_{\text{noul}}^T h_L + b)$$

### 2. The Serving Implication: Zero KV Cache Overhead
Because Jev executes non-autoregressively in a single forward pass:
- **No KV Cache Allocation:** It never calls `BlockAllocator.allocate()` or occupies physical blocks in PagedAttention.
- **50x–200x Latency Advantage:** A single forward step completes in 1–2 milliseconds on modern GPUs, compared to 100–500 ms for an autoregressive LLM decode loop.
- **The Front-Door Pattern:** In high-performance serving architectures, a System 1 model sits at the entrance of the scheduler. It screens incoming requests for safety violations (`Noul`), scores priority (`Score`), and routes requests to specialized LoRA adapters (`Choice`) before a single byte of precious KV cache is allocated in the main engine.


