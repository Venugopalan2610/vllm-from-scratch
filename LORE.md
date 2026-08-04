# The Lore

Everything in vLLM descends from one physical fact. Learn the fact, and the rest of
the system stops being a pile of tricks and becomes a series of forced moves.

---

## 1. The fact: is this IO-bound or CPU-bound?

That's the whole question. It's the same question you'd ask about any service.
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
  380 GB/s          ^-- your card, measured by ./vc info
```

### Step 2: do the math

Each weight is used in exactly one multiply-and-add. That's 2 operations
per weight:

```
7e9 weights  x  2 ops  =  14e9 operations  =  14 GFLOP
```

("FLOP" = one arithmetic operation on a decimal number. Not energy. Not
physics. Just an op, the way a request is a request.)

```
    14 GFLOP          GFLOP cancels, leaving seconds
---------------  =  0.00028 s  =  0.28 ms
  50,000 GFLOP/s        ^-- your card, measured by ./vc info
```

### Compare the two

| Step | The division | Time |
|---|---|---|
| Read 14 GB of weights | `14 GB / 380 GB/s` | **37 ms** |
| Compute on them | `14 GFLOP / 50000 GFLOP/s` | **0.28 ms** |

```
37 ms reading  /  0.28 ms computing  =  132x more time spent waiting
```

**The GPU is idle over 99% of the time, waiting for memory.** It is not
short of arithmetic. It is short of *bytes arriving*.

This is the N+1 query problem. Generating one token at a time is doing a full
table scan to answer a single question.

### The fix, and where the batch size comes from

Read the weights once, answer many questions with them. If 128 requests are
waiting, run them in the *same* pass:

| | 1 request | 128 requests |
|---|---|---|
| Bytes read | 14 GB | **14 GB** — same weights, read once |
| Read time | `14 / 380` = 37 ms | `14 / 380` = **37 ms** |
| Operations | 14 GFLOP | `14 x 128` = 1792 GFLOP |
| Compute time | `14 / 50000` = 0.28 ms | `1792 / 50000` = **36 ms** |
| Tokens out | 1 | **128** |
| Wall clock | ~37 ms | ~40 ms |

**128x the output for the same wall-clock time**, because the expensive part
(dragging 14 GB across the memory bus) got shared by all 128.

Said the other way round: one weight is fetched from VRAM **once**, and while it
sits in registers it gets multiplied against request 1's activation, then
request 2's, then request 3's... up to B. Same byte, B multiply-adds. Bytes are
what's expensive, so the game is squeezing more work out of every byte you
already paid to drag in.

In linear-algebra terms, batching changes *which operation you are running*:

```
batch 1:   y = W @ x     matrix x vector  (GEMV)  -> each weight used once
batch B:   Y = W @ X     matrix x matrix  (GEMM)  -> each weight used B times
```

Batching turns a GEMV into a GEMM. GEMMs are what GPUs are built for; GEMVs
waste them. If you have ever done cache blocking or loop tiling on a CPU, it is
the identical instinct: load the line once, extract every bit of work from it
before it is evicted.

Notice the two time columns just became equal — 37 ms reading, 36 ms computing.
That's not a coincidence, it's where the batch size came from:

```
   37 ms of reading
--------------------  =  132 requests before compute becomes the bottleneck
 0.28 ms per request
```

> **Batch ~130 is where decode stops being memory-bound on this GPU.**
> Below it, extra requests are nearly free. Above it, you're paying for math.

That is the entire argument for continuous batching, and why production servers
set `max_num_seqs` (the batch size knob) in the hundreds.

You won't quite reach it, because each concurrent request also needs its own KV
cache in VRAM, and you run out of memory before you run out of arithmetic.
Relaxing *that* constraint is what PagedAttention is for.

### "Why not batch to infinity, then?"

Being compute-bound is the *goal* — it means the silicon you paid for is finally
busy instead of idling on memory. So why stop at ~150?

Because past the ridge you gain nothing. 7B on this card, read = 37 ms,
compute = 0.25 ms per request:

| Batch | Read | Compute (`0.25 x B`) | Step time | Throughput | Latency/user |
|---|---|---|---|---|---|
| 1 | 37 ms | 0.25 ms | 37 ms | `1/0.037` = 27 tok/s | 37 ms |
| 50 | 37 ms | 12 ms | 37 ms | `50/0.037` = 1,351 tok/s | 37 ms |
| **150** | 37 ms | 37 ms | **37 ms** | `150/0.037` = **4,054 tok/s** | **37 ms** |
| 300 | 37 ms | 75 ms | 75 ms | `300/0.075` = 4,000 tok/s | 75 ms |
| 600 | 37 ms | 150 ms | 150 ms | `600/0.150` = 4,000 tok/s | 150 ms |

Look at the last two rows. **Throughput is flat.** Double the batch, double the
step time, produce twice the tokens in twice the time — net zero. Meanwhile
every user's inter-token latency doubled.

The reason is mechanical. Past the ridge, step time is `0.25ms x B`, so:

```
      B tokens              1
------------------  =  ----------  =  constant, the B cancels
   0.25ms x B sec         0.25ms
```

You have hit the compute roofline and it does not move. Hence "ridge point": a
peak you sit on, not a wall you push through.

### "But the weights never change — can't they just stay in cache?"

They never change, true. The only thing that alters them is retraining. But
read-only does not mean free to read, and they are far too large to cache:

```
L2 cache on this GPU       50 MB
7B model in bf16       14,000 MB   ->  278x too big
Qwen3-0.6B in bf16      1,200 MB   ->   24x too big
```

So the weights live in **VRAM**, and "read 14 GB from VRAM" *is* the 37 ms.
Residency in VRAM is not a saving, it is the cost. Every forward pass drags the
full 14 GB across the bus again, token after token, forever.

Immutability *is* exploited — just never by caching:

- **Batching** — the same unchanging weights serve all B requests (stages 04-05)
- **Quantization** — shrink them offline to fp8/int4, so 14 GB becomes 7 or 3.5,
  and read time falls proportionally (stage 18)
- **Tensor parallelism** — shard them so each GPU reads only its slice (stage 20)

All three attack the same denominator. None makes the read free, because 14 GB
does not fit in 50 MB.

### The compressed form you'll see everywhere else

Papers and blog posts don't write out both timings. They divide them away into a
single number and compare it against the hardware. This is the "roofline model",
and it trips everyone up, so here it is slowly.

**Two different quantities that happen to share the same units.**

|  | What it is | Changes when... |
|---|---|---|
| **~138 FLOP/byte** | a property of your **hardware** | you buy a different GPU |
| **B FLOP/byte** | a property of your **workload** | you change the batch size |

Neither one is something you observe by probing a running GPU. Both are computed
on paper, in advance, and then compared. That comparison is the entire model.

**Why they share units.** The hardware number is a rate divided by a rate, so
the seconds cancel out:

```
  57,000 GFLOP/s          GFLOP     /s
------------------  =  ---------- x ---  =  150 FLOP per byte
     380 GB/s               GB       /s
```

That leaves a plain count ratio -- FLOP per byte, no time in it -- which is
exactly the same shape as the workload's ratio. That is the *only* reason anyone
performs this division: to get hardware and workload onto comparable footing.

(This number bounces between ~124 and ~150 run to run on a laptop GPU as clocks
throttle. Don't chase the exact value.)

**Why the workload's ratio is exactly B.** Let `N` = number of weights, bf16:

```
Bytes moved  = 2 bytes/weight x N             = 2N     (read ONCE, whatever B is)
Operations   = 2 ops/weight x N x B requests  = 2NB

   2NB
--------  =  B FLOP per byte        <- the 2 and the N cancel out
    2N
```

At batch 8 you do 8 FLOP per byte fetched; at batch 128, 128. It is not a
measurement, it is a description of the work you asked for.

That clean `= B` is an accident of bf16, where 2 bytes/weight cancels
2 ops/weight. In fp8 (1 byte/weight) it becomes `2NB / N` = **2B** -- twice the
intensity at the same batch size. That is a second, independent reason
quantization helps, separate from the halved read time.

**It is your two timings, rearranged.** Nothing new is being said:

```
memory-bound  means   time_reading   >   time_computing

                        Bytes             FLOPs
                       -------      >    -------
                          BW               Rate

  multiply both sides by Rate, divide both sides by Bytes:

                         Rate             FLOPs
                        ------      >    -------
                          BW              Bytes

                    138 (hardware)   >   B (workload)
```

Identical inequality, terms shuffled. The intensity form is preferred only
because the model size drops out, so one number covers every model.

**What a profiler actually shows you.** Neither ratio. It shows achieved *rates*.
Batch 1 on a 7B model:

```
memory:   14 GB     / 0.037 s  =    378 GB/s      vs    380 peak  ->  ~100% busy
compute:  14 GFLOP  / 0.037 s  =    378 GFLOP/s   vs 57,000 peak  ->   ~0.7% busy
```

The 138 never appears in any profiler output. It is what you compute beforehand
to predict *which of those two lines will be pegged*.

### Everything else follows

That single fact — *decode waits on memory, so extra requests are nearly free* —
generates the entire system:

- Batch as hard as possible → **continuous batching** (stage 5)
- What limits the batch? KV cache memory → **PagedAttention** (stages 6-9)
- Who gets memory when it runs out? → **the scheduler** (stage 10)
- Prefill is the opposite (compute-bound) and disrupts decode → **chunked prefill** (11)
- 1 ms of GPU work shouldn't cost 1 ms of Python → **CUDA graphs** (12)
- Verifying K tokens costs the same as making 1 → **speculative decoding** (17)
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

Three properties make it uniquely nasty to manage, and they're why a whole paper
exists about it:

1. **It grows one token at a time**, unpredictably. You cannot know the final size
   at admission, because you don't know when the model will emit EOS.
2. **It's enormous.** At long context it rivals or exceeds the weights. It is the
   binding constraint on batch size, and batch size is throughput.
3. **It's per-sequence**, so it fragments.

Pre-vLLM systems allocated one contiguous buffer per sequence, sized to `max_len`.
The waste came in three flavors:

- **Reserved**: allocated for tokens not generated yet
- **Internal fragmentation**: the tail of the buffer if the sequence stops early
- **External fragmentation**: gaps between buffers that fit nothing

Measured waste in the vLLM paper: **60-80%** of KV memory. Meaning ~4x fewer
concurrent sequences than the hardware could hold. Meaning ~4x less throughput.

**Grouped-Query Attention (GQA)** is the other half of this story. Llama-3-8B uses 32
query heads but only **8** KV heads, cutting KV bytes 4x. Modern models are designed
around KV cache pressure. When you compute cache size in stage 2, use `num_kv_heads`
or you'll be off by the GQA factor.

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

The cost: attention can no longer stride a contiguous tensor. The kernel must
**gather K/V through the block table**. That's why PagedAttention needed a custom
kernel and couldn't just call FlashAttention. Stage 7 makes you feel the slowdown in
PyTorch; stage 8 makes you win it back in Triton.

The payoff beyond memory: **sharing becomes a pointer operation**.

- `n=4` parallel samples share the prompt's blocks, refcounted, CoW on divergence
- Beam search shares the common prefix across all beams
- **Automatic prefix caching**: hash block contents, and a shared 2000-token system
  prompt is prefilled once for *all* users, forever. Warm TTFT collapses.

That last one is why APC feels like cheating in production. It's just a page cache.

---

## 4. Continuous batching: the other 4x

From **Orca (OSDI '22)**, and arguably as important as paging.

**Static batching**: gather 8 requests, run until all 8 finish. A request that needs
10 tokens sits padded and idle while one needing 500 tokens finishes. Output lengths
in real traffic vary by 10-100x, so most of your batch slots are dead air.

**Continuous batching** = scheduling at **iteration granularity**. After *every*
forward pass the scheduler re-decides the batch: finished sequences leave, waiting
ones join. No padding to a common length, no waiting for the straggler.

The subtlety that makes it work: sequences at different lengths can share a batch
because attention is per-sequence anyway. You flatten all tokens into one ragged
tensor and hand the kernel a length array (`cu_seqlens`). No padding, ever.

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

They fight. A single 8000-token prefill occupies a whole step, and every sequence
mid-generation stalls — users see a visible hitch in their token stream.

**Chunked prefill** (Sarathi-Serve) splits the prefill into fixed token budgets and
co-schedules chunks alongside decodes in one mixed batch. It piggybacks the
bandwidth-starved decode tokens onto the compute-heavy prefill GEMM, filling both
sides of the roofline. This is the main throughput-vs-latency dial in modern servers.

**Disaggregated prefill** takes it further: separate GPU pools for prefill and decode,
shipping KV blocks between them over the network. You isolate TTFT from TPOT
completely. This is where large deployments are heading.

---

## 6. Timeline

| When | Idea | Why it mattered |
|---|---|---|
| 2022 | **Orca** — iteration-level scheduling | Continuous batching. Stop waiting on stragglers. |
| 2022 | **FlashAttention** | Tiling + online softmax; never materialize the N×N score matrix. |
| 2023 | **vLLM / PagedAttention** (SOSP) | KV cache as virtual memory. Kills fragmentation. |
| 2023 | **Speculative decoding** | Exploit that verify-K ≈ cost of generate-1. Lossless. |
| 2023 | **S-LoRA** | Thousands of LoRA adapters on one base model, paged like KV. |
| 2023 | **GQA** goes mainstream | Model architecture bends to KV cache pressure. |
| 2024 | **Sarathi-Serve** — chunked prefill | Stop prefill from stalling decode. |
| 2024 | **Automatic prefix caching** | Content-hash blocks; share prefixes across requests. |
| 2024 | **FP8 KV cache** | Halve the other big memory reader. |
| 2025 | **vLLM V1 rewrite** | Isolated EngineCore process, unified scheduler, persistent batch. |
| 2025+ | **Disaggregated P/D** | Separate prefill and decode fleets; isolate TTFT from TPOT. |

---

## 7. Anatomy of the real vLLM

Roughly what you're rebuilding, so you can read the source afterward:

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
        └── Model ── attention backend (FlashAttention / FlashInfer / Triton)
```

**Key V0 → V1 changes** (2025), all of which you'll independently rediscover:

- Engine loop moved into its **own process** — the API server's Python overhead was
  measurably stalling the GPU between steps.
- **No prefill/decode distinction in the scheduler.** Just a per-step token budget;
  a request contributes however many tokens it needs. Chunked prefill becomes the
  default rather than a mode.
- **Persistent batch**: the input tensors are mutated in place across steps instead
  of being rebuilt, so CUDA graphs stay valid and Python work per step goes to ~0.
- Prefix caching on by default, because the hash lookup got cheap enough to be free.

---

## 8. Numbers worth memorizing

- KV cache per token, Llama-3-8B (32 layers, 8 KV heads, head_dim 128, bf16):
  `2 * 32 * 8 * 128 * 2 = 128 KB/token`. A 4k-token conversation = **512 MB**.
  On your 12 GB card, after ~5 GB for an 8B model in fp8, that's roughly a dozen
  such conversations. **That number is your throughput.**
- Block size 16 is the standard default: big enough to amortize block-table lookups,
  small enough that the average waste (8 tokens/sequence) is noise.
- A decode step at batch 1 on a small model is ~1 ms of GPU work. Unoptimized Python
  and kernel launches can cost about the same. Hence CUDA graphs.
- Speculative decoding acceptance rates run ~60-80% with a good draft; expect
  1.5-2.5x, and near zero on high-entropy creative text.

---

## 9. Vocabulary

- **TTFT** — time to first token; dominated by queue wait + prefill.
- **TPOT / ITL** — time per output token / inter-token latency; the streaming smoothness.
- **Goodput** — throughput that actually met its latency SLO. The metric that matters.
- **Ragged / flat batch** — all sequences' tokens concatenated, described by `cu_seqlens`.
- **Slot mapping** — for each token in the flat batch, the physical KV slot to write to.
- **Block table** — per-sequence array mapping logical block index → physical block id.
- **Preemption** — evicting a running sequence under memory pressure (swap or recompute).
- **Online softmax** — the running max/sum trick letting you softmax in tiles, from
  FlashAttention. You'll implement it in stage 8.
