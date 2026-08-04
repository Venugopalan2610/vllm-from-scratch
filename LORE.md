# The Lore

Everything in vLLM descends from one physical fact. Learn the fact, and the rest of
the system stops being a pile of tricks and becomes a series of forced moves.

---

## 1. The fact: decode is memory-bound

Generating one token with a batch of 1 requires reading **every weight in the model**
from HBM into the SMs, doing a tiny amount of math with each, and throwing it away.

For a 7B model in bf16 that is ~14 GB of reads to produce **one token**.

```
time_per_token  >=  model_bytes / memory_bandwidth
```

On a GPU with ~430 GB/s of bandwidth: `14 GB / 430 GB/s ~= 33 ms/token ~= 30 tok/s`.
That is a hard ceiling. No kernel is beating it, because the data has to move.

Now the punchline. If you decode **64 sequences at once**, you read those same 14 GB
*once* and produce **64 tokens**. The math per weight went up 64x; the bytes moved
stayed flat. Time per step barely changes.

> **Decode throughput is almost free in batch size, until you run out of memory to
> hold the batch.**

That single sentence generates the entire system:

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
