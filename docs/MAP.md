# The map

Read this before you build anything.

A web service has a shape that you already know: router, service, repo,
database. When something breaks, the shape tells you where to look. An
inference engine has a shape too. This page gives it to you first. Each stage
of the course then builds one box of it.

---

## The words

Most GPU words are new names for ideas that you already know. The idea
first, then the name:

| The idea, in software words | The name that GPU people use |
|---|---|
| A function. Text goes in, and a score for each possible next piece of text comes out. | the model |
| A large read-only table of numbers. The model reads all of it to make one token. | the weights |
| The piece of text that the model reads and writes, as an integer id. | a token |
| The requests that the engine serves in the same call. | the batch |
| The first call for a request. It reads the whole prompt at once. | prefill |
| Each call after that. It makes one new token for each request. | a decode step |
| A memo table for each request, so that the model never repeats its work on earlier tokens. | the KV cache |
| A fixed-size page of that memo table. | a block |
| A function that runs on the GPU. | a kernel |
| A call from the CPU that starts a kernel. It has a fixed cost, like an RPC. | a kernel launch |
| Code that waits for memory reads, not for its arithmetic: "I/O-bound". | memory-bound |
| Evict a request from memory, and run it again later. | preemption |
| A slow version that you trust, to compare the fast version with. | an oracle |

---

## The layers

```
  client ── POST /v1/chat/completions
     │
┌────▼─────────────────────────────────────────────────────────────┐
│ ROUTER        the API server                                     │  15, 27
│               parse, chat template, stream, abort on disconnect  │
├────┬─────────────────────────────────────────────────────────────┤
│    │  a command on a deque. The event loop never waits for the GPU.
├────▼─────────────────────────────────────────────────────────────┤
│ SERVICE       the engine loop: step() forever, in its own thread │  05, 22
│               ├── the scheduler: who runs in this step?          │  10, 11
│               ├── the sampler: logits → one token for each row   │  13
│               └── the detokenizer: tokens → text, stop strings   │  14
├──────────────────────────────────────────────────────────────────┤
│ REPO          the KV cache manager                               │  06, 09, 29
│               blocks, block tables, the prefix cache             │
│               It owns the state of every request.                │
├──────────────────────────────────────────────────────────────────┤
│ DB DRIVER     the model runner                                   │  21, 23
│               builds one flat batch, replays a CUDA graph        │
├──────────────────────────────────────────────────────────────────┤
│ DB ENGINE     the model and its attention backend                │  07, 08, 08b, 08c
│               reads K and V through the block table              │
└──────────────────────────────────────────────────────────────────┘
  SCHEMA        the model config: layers, KV heads, head_dim, dtype  02
                It sets the size of everything above.
  METRICS       TTFT, ITL, KV use, preemptions                       16
```

Stages 01 to 03 are the model call alone, before any of the boxes exist.
Stage 05 is the first engine loop: small, but it has the right shape. The
capstone (21 to 28) puts every box together, and you then have the whole
diagram in code.

---

## The life of one request

1. **Router.** The request arrives. The server applies the chat template and
   checks the limits. It puts an `add_request` command on a deque. *(27)*
2. **Service.** At the start of the next step, the engine loop takes the
   command. It tokenizes the prompt and makes a sequence in the WAITING
   queue. *(22)*
3. **Service → Repo.** The scheduler plans the step. Running sequences get
   their one decode token first. Waiting sequences join until the token budget
   is gone. *(10, 11, 22)* For a new sequence, the repo looks up the prefix
   cache, so only the uncached tokens need a prefill. *(09)* Then it allocates
   blocks for the new tokens. *(06)*
4. **Repo is full.** If no blocks are free, the engine drops the prefix cache
   first. Then it preempts the newest sequence: it frees the blocks and puts
   the sequence back at the front of the queue, to compute again later.
   *(10, 22)*
5. **DB driver.** The runner builds one flat batch: the decodes first, then the
   prefill chunks, with their block tables. A step of decodes only replays a
   captured graph. *(21, 23)*
6. **DB engine.** In each layer, attention writes the new K and V into their
   slots and reads the context through the block table. *(07, 08c)*
7. **Service.** The sampler turns the logits into one token for each row. The
   detokenizer turns the tokens into text, and it holds back text that can
   start a stop string. *(13, 14)*
8. **Router.** The new text goes to the queue of the request, and then to the
   client as a server-sent event. *(15, 27)*
9. **The end.** The request finishes, or the client disconnects. The repo
   frees its blocks. Its full blocks stay in the prefix cache, until the LRU
   order evicts them. *(09, 22, 29)*

Steps 3 to 8 are one call of `step()`. They repeat, for every running
request together, until nothing is left.

---

## The map, running: the live engine

From stage 06, this map runs as one engine: `tvllm/live.py`. Each box is a
slot. A slot holds your part when you finish its stage, and a given, naive
part before that:

| Box | Given, before your stage | Yours, from stage |
|---|---|---|
| Repo | reserve max_model_len for each sequence | 06, then 09 |
| DB engine | one growing K and V tensor for each sequence | 07, then 08, 08b, 08c |
| Service: scheduler | first come, first served, no preemption | 10, then 11 |
| DB driver | eager | 12 |
| Service: sampler | argmax only | 13 |
| Service: detokenizer | decode each token alone | 14 |
| Router | none | 15 |
| Metrics | none | 16 |

`./vc run` runs one workload on your engine so far, with your newest part and
without it. The difference is what your newest stage bought, and its bill.
The live engine is simple and slow on purpose. The capstone (21 to 28) asks
you to write the real one.

---

## Where to look when it breaks

| Layer | When it breaks, you see | Taught in |
|---|---|---|
| Router | The model writes its reasoning into the answer: a wrong chat template | 27 |
| Router | A client disconnects, and the blocks never come back | 15, 27 |
| Service | All the streams stall together while one step runs | 27 |
| Service (scheduler) | One long prompt arrives, and every other user's tokens slow down | 11 |
| Service (scheduler) | More slots, and less throughput: a preemption storm | 16, Part 6 |
| Service (sampler) | A retry with the same seed gives different text | 13 |
| Service (detokenizer) | Broken characters, double spaces, a stop string that leaks | 14 |
| Repo | The server holds fewer requests every hour: a block leak | 22 |
| Repo | Fluent, wrong answers after a prefix cache hit: a hash bug | 09 |
| DB driver | One sequence changes its output a few tokens later: a padding row wrote into block 0 | 23 |
| DB engine | Batch 1 is slow, but the kernel is "fast": most SMs have no work | 08c |

---

## The patterns, in words that you know

| Engine pattern | Stage | Nearest software pattern | Where the analogy stops |
|---|---|---|---|
| Continuous batching | 05 | A worker pool that takes the next job when a worker is free | All the workers move in the same step |
| Paged KV cache | 06, 07 | Virtual memory: a page table | Attention must read through the table on every step |
| Copy-on-write blocks | 09 | `fork()`, or a branch in Git | The same |
| Prefix caching | 09 | A content-addressed cache, like the Git object store | The key of a block is a chain of hashes, not one hash |
| Preemption by recompute | 10 | Evict a cached value, and compute it again from the source | The recompute usually costs more than a copy to CPU memory. The engine chooses it because it can cut the recompute into chunks and schedule them |
| Chunked prefill | 11 | A large migration in small batches, so that it does not lock the table | The "lock" is the GPU step that all the other users wait for |
| CUDA graphs | 12, 23 | A prepared statement: plan one time, run many times | A graph also fixes the shapes and the memory addresses |
| Speculative decoding | 17, 25 | Optimistic concurrency: guess, check, roll back | The check must not change the output distribution |
| Weight quantization | 18, 18b, 24 | Compression: store fewer bytes, decompress where you read | The decompression must happen inside the kernel, or it costs more than it saves |
| Guided decoding | 19, 26 | Input validation by a state machine | It runs on every token, so the masks must be cached |
| Tensor parallelism | 20 | A table sharded across nodes, with one aggregation step | The aggregation happens twice in every layer |
| LRU prefix eviction | 29 | An LRU cache | A block in use can never be evicted |

Each analogy gives you a place to start. The last column is where the engine
stops being the software that you know, and it is usually where the lesson of
the stage is.
