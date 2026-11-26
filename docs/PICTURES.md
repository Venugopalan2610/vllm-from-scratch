# The pictures

One ASCII picture for each stage. `./vc guide` shows the picture of the
stage under THE PICTURE, and `./vc picture [stage]` shows it alone. The TUI
shows it in its Picture tab.

The rules for a picture:

- It shows the mechanism of the stage, with real numbers, not a metaphor.
- Each symbol has its definition in the picture or in the line under it.
- Two different quantities never have the same example number.
- Plain ASCII only, at most 72 columns, so that every terminal shows it.

A heading is a stage id from stages.yaml. A heading with ` jax` after the id
is the picture of the JAX track, for a stage that is different on that
track. The first line under the heading is the caption. `tests/test_docs.py`
checks every rule that a program can check.

---

## map

Where each stage lives in the engine. The numbers are stages.

```text
  client -- POST /v1/chat/completions
     |
+----v-------------------------------------------------------+
| ROUTER     the API server                                  |  15 27
|            parse, chat template, stream, abort             |
+----|-------------------------------------------------------+
|    |  a command on a deque. The event loop never waits.    |
+----v-------------------------------------------------------+
| SERVICE    the engine loop: step() forever                 |  04 05 22
|            +-- the scheduler: who runs in this step?       |  10 11
|            +-- the sampler: logits -> one token per row    |  13
|            +-- the detokenizer: tokens -> text, stops      |  14
+------------------------------------------------------------+
| REPO       the KV cache manager                            |  06 09 29
|            blocks, block tables, the prefix cache          |
+------------------------------------------------------------+
| DB DRIVER  the model runner                                |  21 23
|            one flat batch, a replayed CUDA graph           |  12
+------------------------------------------------------------+
| DB ENGINE  the model and its attention backend             |  07 08
|            reads K and V through the block table           |  08b 08c
+------------------------------------------------------------+
  THE MODEL CALL alone, before any box exists                   01 02 03
  SCHEMA     layers, KV heads, head_dim, dtype                  02
  METRICS    TTFT, ITL, KV use, preemptions                     16
  FASTER     speculation 17 25, int8 18 18b 24, FP8 KV 24b
  MORE       guided 19 26, tensor parallel 20, LoRA 30, MLA 31
  THE END    one engine 22, served 27, measured 28
```

---

## 01-forward-pass

Each step calls the model on the WHOLE prefix again.

```text
  step 1   model([The, cat])             -> logits -> argmax -> sat
  step 2   model([The, cat, sat])        -> logits -> argmax -> on
  step 3   model([The, cat, sat, on])    -> logits -> argmax -> the
                  \__ the whole prefix, every step __/

  The rows of K and V that each step computes:
    step:     1      2       3        4         5
    rows:   [##]  [###]  [####]  [#####]  [######]
             P+0   P+1    P+2     P+3      P+4

    P = the prompt length (here P = 2). Step t computes P + t - 1 rows.
    N steps compute about N x N / 2 rows: the cost grows as O(N^2).
```

## 02-kv-cache

Compute the K and V of a token one time. Keep them.

```text
  WITHOUT a cache (stage 01)          WITH a cache (stage 02)
  step 1: K,V of [A B C]              prefill: K,V of [A B C]
                                               cache = [A B C]
  step 2: K,V of [A B C D]            step 2:  K,V of [D] only
                                               cache = [A B C D]
  step 3: K,V of [A B C D E]          step 3:  K,V of [E] only
                                               cache = [A B C D E]

  The query of the new token reads the whole cache:
    q(E) . [K(A) K(B) K(C) K(D) K(E)] -> weights -> weighted sum of V

  bytes of KV for one token
    = 2 (K and V) x layers x kv_heads x head_dim x bytes per value
    = 2 x 28 x 8 x 128 x 2  =  114,688 bytes  =  112 KiB   (example)
```

## 03-roofline

The roofline: decode waits on memory, prefill waits on arithmetic.

```text
  achieved FLOP/s
    ^
    |                      ridge
    |                        v
    |                        +======================== peak FLOP/s
    |                      /           * prefill of 2000 tokens:
    |                    /               2000 FLOPs per byte read
    |                  /
    |                /  <- slope = HBM bandwidth, in bytes/s
    |              /
    |            * <- decode at batch 1: 1 FLOP per byte read
    |          /      (2 FLOPs per weight, 2 bytes per bf16 weight)
    +--------/---------------------------------------------->
           arithmetic intensity = FLOPs / bytes read from HBM

  time of one step = max( bytes / bandwidth , FLOPs / peak FLOP/s )
  decode:  the first term is larger.   prefill: the second term is.
```

## 04-static-batching

A static batch runs until its slowest sequence ends.

```text
  Prompts of 3, 4 and 5 tokens, LEFT-padded to the same length:
    [pad  pad   How  are  you ] -> next token
    [pad  What  is   the  time] -> next token
    [Tell me    a    joke ?   ] -> next token
    attention_mask = 0 on a pad, 1 on a real token

  4 sequences, 8 decode steps.  # = a token  x = EOS  . = a wasted slot
             step:  1 2 3 4 5 6 7 8
    seq A           # # # # # # # #
    seq B           # # # x . . . .
    seq C           # # # # # x . .
    seq D           # x . . . . . .

    slots used   8 + 4 + 6 + 2  = 20
    slots paid   4 x 8          = 32
    waste        12 / 32        = 37.5 %
```

## 05-continuous-batching

The scheduler decides at every step, so no slot waits for padding.

```text
  The same 4 slots. Letter = the sequence in the slot. x = its EOS.
             step:  1 2 3 4 5 6 7 8
    slot 0          A A A A A A A A
    slot 1          B B B x F F F F     B ends at step 4, F joins at 5
    slot 2          C C C C C x G G     C ends at step 6, G joins at 7
    slot 3          D x H H H H H H     D ends at step 2, H joins at 3

    waiting queue          running set              finished
    [F G H I J ...]  -->  [A B C D] (4 slots)  -->  [D B C ...]
                   admit                       EOS

  Every slot does useful work at every step: 32 of 32 slots.
```

## 06-block-allocator

Cut the KV cache into fixed blocks, and give them out on demand.

```text
  CONTIGUOUS: reserve max_len = 40 slots for each sequence
    seq A |AAAAAAA.................................|   7 of 40
    seq B |BBBBBBBBBB..............................|  10 of 40
    reserved 80, used 17, waste 63 = 79 %

  PAGED: blocks of block_size = 4 slots (the real engine uses 16)
    physical pool | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
                  |AAAA|BBBB|BB..|AAA.|BBBB|    |    |    |
    block table A = [0, 3]        7 tokens -> ceil(7 / 4)  = 2 blocks
    block table B = [1, 4, 2]    10 tokens -> ceil(10 / 4) = 3 blocks
    free list     = [5, 6, 7]
    reserved 20, used 17, waste 3 = 15 %   (less than 1 block each)

  token i of a sequence lives at
    physical block = table[i // block_size],  offset = i % block_size
```

## 07-paged-attention

Gather K and V through the block table, then do ordinary attention.

```text
  One sequence, one head. block_size = 4. context length = 10 tokens.
  block table = [5, 2, 7]

  physical K cache   ... | blk 2 | ... | blk 5 | ... | blk 7 | ...
                             |             |             |
  gather, in table order:  [blk 5: tokens 0 to 3]
                           [blk 2: tokens 4 to 7]
                           [blk 7: tokens 8 to 9, 2 slots unused]
                                     |
                                     v
                 K = 10 rows x head_dim     V = 10 rows x head_dim
                                     |
  q (1 x head_dim) -> softmax(q K^T / sqrt(head_dim)) V -> out

  reference_attention(): the same math on a dense cache. The oracle.
  The check: paged out == reference out, within a tolerance.
```

## 08-paged-attention-cuda

One thread block owns one output vector, and streams K and V once.

```text
  grid = (num_seqs, num_heads). Block (s, h) owns out[s, h], alone.

  block (s, h):                              m   = running max  = -inf
    q = the query of (s, h)                  l   = running sum  = 0
    for each entry b of block_table[s]:      acc = running out  = 0
      load the K and V tiles of block b
      scores = q . K^T / sqrt(head_dim)
      m_new  = max(m, max(scores))
      p      = exp(scores - m_new)
      acc    = acc * exp(m - m_new) + p @ V
      l      = l   * exp(m - m_new) + sum(p)
      m      = m_new
    out[s, h] = acc / l

  HBM reads:  q once, K and V once.   HBM writes: out once.
  The score row never goes to HBM. No block talks to another block.
```

## 08-paged-attention-cuda jax

The same online softmax in Pallas. The block table lookup is inside.

```text
  grid = (num_seqs, num_heads). Program (s, h) owns out[s, h].

  The BlockSpec index_map chooses the tile of q and of out:
    q_spec   = BlockSpec((1, 1, head_dim), lambda s, h: (s, h, 0))
  It CANNOT choose the K tile: that tile depends on block_table[s].

  So the kernel reads the table, then reads the cache by index:
    for i in range(num_blocks_of_s):
      b = block_table[s, i]                 <- a computed index
      k = k_cache[b, :, h, :]               <- a dynamic read
      v = v_cache[b, :, h, :]
      (the online softmax of stage 08: m, l, acc)
    out[s, h] = acc / l
```

## 08b-cuda-memory

Turn the mapping sideways, so that one warp reads contiguous bytes.

```text
  head_dim = 128 values of bf16 = 256 bytes in one row of K.
  A warp = 32 threads. Their loads issue together.

  STAGE 08: thread t owns position t, and walks head_dim alone
    load j:  t0 -> K[pos 0][j]  t1 -> K[pos 1][j] .. t31 -> K[pos 31][j]
    the 32 addresses are 256 bytes apart
    -> 32 separate 32-byte sectors, 2 useful bytes in each: 6 % used

  STAGE 08b: the threads cooperate ALONG head_dim, 16 bytes each
    K[pos p]   = | t0  | t1  | t2  | ... | t15 |    256 bytes
    K[pos p+1] = | t16 | t17 | t18 | ... | t31 |    256 bytes
    -> 512 contiguous bytes = 4 whole 128-byte transactions: 100 % used

  The arithmetic and the answers do not change. The bandwidth does.
```

## 08c-cuda-warps

Split the context, so that every SM has work at batch 1.

```text
  Batch 1, 16 heads: grid = (1, 16) = 16 thread blocks.
  A card with 46 SMs:
    [##][##][##] ... [##]  [..][..][..] ... [..]
     16 SMs busy            30 SMs idle

  SPLIT-K: cut a context of 4096 tokens into 8 chunks of 512 tokens
    grid = (1, 16, 8) = 128 thread blocks: every SM has work
    chunk c returns a partial state (m_c, l_c, acc_c)

  THE MERGE (exact):
    M   = max over c of m_c
    L   = sum over c of l_c   x exp(m_c - M)
    ACC = sum over c of acc_c x exp(m_c - M)
    out = ACC / L

  WARP SHUFFLE: a sum of 32 values inside one warp, in 5 steps
    __shfl_xor_sync with offsets 16, 8, 4, 2, 1     (2^5 = 32)
    no shared memory, no __syncthreads()
```

## 09-prefix-sharing

Two block tables point at the same physical blocks.

```text
  Two requests share a system prompt of 8 tokens. block_size = 4.
    request 1: [s s s s][s s s s][How are ...]
    request 2: [s s s s][s s s s][Write a ...]     s = a system token

    block table 1 = [3, 6, 1]        refcount   block 3 = 2
    block table 2 = [3, 6, 4]                   block 6 = 2
                     \__/                       block 1 = 1
                    shared                      block 4 = 1

  PREFIX CACHE: key = hash(hash of the block before, the 4 token ids)
    h1 = hash(none, s s s s) -> block 3
    h2 = hash(h1,   s s s s) -> block 6
    request 2 finds h1 and h2: it skips the prefill of 8 tokens

  COPY-ON-WRITE: a write into a block with refcount 2
    -> copy it to a free block, write the copy, refcount 2 -> 1
```

## 10-admission-preemption

Admit when the blocks fit. Preempt when the next token does not fit.

```text
              admit, if enough blocks are free
  WAITING  -------------------------------->  RUNNING  -EOS->  FINISHED
     ^                                            |
     |   PREEMPT by recompute:                    | no free block
     +--- free all its blocks, keep its     <-----+ for the next token
          token ids, prefill again later

  Each step:
    need = the running sequences whose next token starts a new block
    while need > free blocks:
      preempt the NEWEST running sequence, and free its blocks

  SWAP is the other choice: copy the blocks to CPU memory, and copy
  them back later. Recompute needs no host memory, and it can start
  again from a cached prefix.
```

## 11-chunked-prefill

Cut a long prefill into chunks, and mix the chunks with decodes.

```text
  token budget for each step = 512.   d = one decode token

  WITHOUT chunks: a prompt of 8000 tokens runs in ONE step
    step 1: [d d d d]
    step 2: [d d d d][P P P P P P P P ... all 8000 ...]   <- very long
    step 3: [d d d d]
    the 4 decoding sequences wait for the whole prefill

  WITH chunks: 4 decodes + 508 prefill tokens in each step
    step 2: [d d d d][P: tokens    0 to  507]
    step 3: [d d d d][P: tokens  508 to 1015]
    step 4: [d d d d][P: tokens 1016 to 1523]
    ...     every step has about the same length

  The position of the first token of a chunk = the tokens prefilled.
```

## 12-cuda-graphs

Capture the step one time, then replay it with one launch.

```text
  EAGER: the CPU launches each kernel. The GPU waits between them.
    CPU  |L|L|L|L|L|L|L|L|L|L|L|L|         L = one kernel launch
    GPU    |k|  |k|  |k|  |k|  |k|  |k|     k = one small kernel
              ^    ^ gaps: the GPU is idle

  GRAPH: one replay starts all the kernels
    CPU  |R|                               R = one graph replay
    GPU    |k|k|k|k|k|k|k|k|k|k|k|k|       no gaps

  A graph has fixed shapes. Capture one graph for each bucket:
    buckets = [1, 2, 4, 8]
    batch 3 -> bucket 4, with 1 padding row
    batch 5 -> bucket 8, with 3 padding rows
    batch 9 -> no bucket -> the eager fallback
```

## 12-cuda-graphs jax

JAX compiles one program for each shape. Buckets stop the recompiles.

```text
  WITHOUT buckets: each new batch size is a new shape
    batch 3 -> trace + compile (seconds)
    batch 5 -> trace + compile (seconds)
    batch 6 -> trace + compile (seconds)     on the critical path

  WITH buckets = [1, 2, 4, 8]: pad up to the next bucket
    batch 3 -> shape 4 -> compile one time
    batch 5 -> shape 8 -> compile one time
    batch 6 -> shape 8 -> cache hit
    batch 7 -> shape 8 -> cache hit          4 compiles, at most

  donate_argnums: XLA writes the new cache into the old buffer,
  in place, and keeps no second copy of it.
```

## 13-sampling

One vectorized pass applies the parameters of every row.

```text
  logits: 3 rows x vocab.   The parameters, as tensors of 3 values:
    temperature = [0.7, 1.0, 0.0]
    top_k       = [ 50,   0,   1]        0 = no limit
    top_p       = [0.9, 1.0, 1.0]

  1. repetition penalty, on the token ids that each row saw
  2. logits / temperature                        all rows, one op
  3. top-k: a logit below the k-th largest -> -inf
  4. top-p: sort, cumsum of softmax; the tail after p -> -inf
  5. sample each row with its own seeded generator
     a row with temperature 0.0 -> argmax (greedy)

  No Python loop over the rows. 64 rows cost the same 5 steps.
```

## 14-detokenization

Decode a window of tokens, and emit only the new complete text.

```text
  The character e-acute is 2 bytes in UTF-8: C3 A9.
    tokens:    "Caf"     <C3>      <A9>     " bar"
  NAIVE: decode each token alone
    text:      "Caf"     "?"       "?"      " bar"   -> "Caf?? bar"
  INCREMENTAL: text(window + new) - text(window)
    text:      "Caf"     ""        "e-acute" " bar"  -> correct
                         ^ an incomplete character: hold it, emit ""

  A STOP STRING "END" can cross a token boundary:
    tokens:   "The"   " E"    "ND"
    after " E": hold back "E", because "E" can start "END"
    after "ND": "END" found -> stop, and emit only "The "
```

## 15-async-engine

HTTP never waits for the GPU, and the GPU never waits for HTTP.

```text
  client --POST--> [ asyncio event loop ]
                     | add_request(prompt) -> a queue for the request
                     v
                   [ engine loop: step() forever ]
                     | each step puts the new tokens on each queue
                     v
  client <--SSE--- data: {"text": " the"}
                   data: {"text": " cat"}
                   data: [DONE]

  client disconnects -> abort(request_id) -> the engine drops it
```

## 16-observability

The times of one request, and the gauges of the engine.

```text
  arrive       admit                first      2nd        3rd
    |---wait----|------prefill-------|----------|----------| ...
    |<----------- TTFT ------------->|<-- ITL ->|<-- ITL ->|
    TPOT = (time of the last token - time of the first token)
           / (output tokens - 1)

  gauges, at each step
    KV utilization = used blocks / total blocks
    preemptions    = the count of preempted sequences
    queue wait     = admit time - arrive time

  sweep max_num_seqs over 8, 16, 32, 64:
    p99 latency ^                    * 64
                |              * 32
                |        * 16
                |   * 8
                +------------------------> throughput
```

## 17-speculative-decoding

Draft K tokens cheaply. Check all of them in one pass of the model.

```text
  context: "... the cat sat on the"
  draft (n-gram lookup), K = 4:   mat   and   the   dog

  ONE pass of the target model, on 5 rows:
    row:          0      1      2      3      4
    input:       the    mat    and    the    dog
    target:      mat    and    then   ...    ...
    draft:       mat    and    the
                 ok     ok     NO  -> stop here
  accept "mat" "and", then take the target token "then"
  -> 3 new tokens for the cost of about 1 decode step

  SAMPLING: accept a draft token x with probability min(1, p(x) / q(x))
    p = the target probability, q = the draft probability
  On a reject, sample from max(0, p - q), normalized.
  The output distribution is EXACTLY the target distribution.
```

## 18-quantization

Store each weight in 1 byte, with one scale for each row.

```text
  bf16 row:   [ 0.12   -0.50    0.31    0.07 ]      2 bytes each
  scale:      s = max |w| / 127 = 0.50 / 127 = 0.003937
  int8 row:   q = round(w / s)
              [   30    -127      79      18 ]      1 byte each
  dequantize: q x s = [0.1181  -0.5000  0.3110  0.0709]

  decode reads every weight one time for each token:
    bf16: 14 GB / 504 GB/s = 27.8 ms for each token
    int8:  7 GB / 504 GB/s = 13.9 ms for each token

  FP8 KV cache: the same idea for the other large reader.
  The guard: top-1 agreement with bf16, and the KL from bf16, in nats.
```

## 18b-int8-gemv-cuda

Apply the scale at the end of the dot product, in a register.

```text
  DEQUANTIZE, THEN MATMUL (PyTorch):
    int8 W --read 1 B--> bf16 copy --write 2 B--> GEMM --read 2 B-->
    bytes moved for each weight: 1 + 2 + 2 = 5.   bf16 alone: 2.

  FUSED EPILOGUE (your GEMV):
    for each output row j:
      acc = 0                                  an fp32 register
      for each k:  acc += W_int8[j, k] x x[k]
      y[j] = acc x scale[j]                    the scale, one time
    bytes moved for each weight: 1

  THE CROSSOVER: y = x W^T, with R rows in x
    R = 1:  the GEMV wins, because it reads half the bytes
    R grows: cuBLAS bf16 shares one weight read across the R rows
    You measure the R where cuBLAS starts to win. Above it, dispatch.
```

## 19-guided-decoding

Mask the illegal tokens before the sampler sees the logits.

```text
  schema: {"ok": boolean}
  automaton:  S0 -{-> S1 -"ok"-> S2 -:-> S3 -true|false-> S4 -}-> END

  In state S3:
    tokens:   "true"   "}"    " false"   "7"    "t"
    logits:    2.1     0.3     4.0       1.2    0.8
    mask:      0      -inf     0        -inf    0
    sum:       2.1    -inf     4.0      -inf    0.8   -> sample

  mask cache: automaton state -> a boolean mask over the vocabulary
  Build each mask one time, and OFF the critical path of the step.
```

## 20-tensor-parallel

Two ranks, each with half of the heads and half of the MLP columns.

```text
                 x  (the same copy on both ranks)
          +------------------+------------------+
     rank 0: heads 0 to 7          rank 1: heads 8 to 15
     QKV columns, out rows         QKV columns, out rows
          |  y0                         |  y1
          +--- all-reduce (1):  y = y0 + y1 ----+
          |                             |
     rank 0: MLP up cols 0 to H/2  rank 1: MLP up cols H/2 to H
             down rows 0 to H/2            down rows H/2 to H
          |  z0                         |  z1
          +--- all-reduce (2):  z = z0 + z1 ----+

    H = the intermediate size of the MLP
  2 all-reduces in each layer. The payload of each, for one token:
    hidden_size x 2 bytes = 2048 x 2 = 4096 bytes  (example)
```

## 21-paged-model

One flat batch: the decodes first, then a prefill chunk.

```text
  sequences A, B, C decode. Sequence P prefills its first 5 tokens.
    tokens:     [ A  | B  | C  | P  P  P  P  P ]
    positions:  [ 7  | 12 | 3  | 0  1  2  3  4 ]
    slots:      [ the cache slot for the K and V of each token ]

  the model (tvllm/model.py), for each layer:
    q, k, v   = the projections of x
    write_kv(k, v, slots)             <- your stage 08 kernel
    out[0:3]  = paged decode(q[0:3])  <- your stage 08c kernel,
                                         ONE launch for all decodes
    out[3:8]  = prefill attention over the chunk and its prefix

  The check: the tokens are identical to HF, in fp32.
```

## 22-engine

One rule: give pending tokens to sequences, until the budget ends.

```text
  pending tokens:  a decode -> 1.   a new prompt -> all its tokens.
                   after a preemption -> prompt + generated tokens.

  one step, token budget = 256:
    seq A  decode      1 pending  -> takes 1      budget left 255
    seq B  decode      1 pending  -> takes 1      budget left 254
    seq C  prompt    600 pending  -> takes 254    budget left   0
    seq D  waiting    40 pending  -> waits for the next step

  step():  schedule -> model runner -> sampler -> detokenizer
           -> stop checks -> free the blocks of finished sequences
  A new prompt starts after its blocks in the prefix cache.
```

## 23-graphs

Replay the decode step. A padding row writes to slot -1.

```text
  capture one graph for each bucket: [1, 2, 4, 8, 16, 32]

  3 real rows in bucket 4: one PADDING row
    row:     0      1      2      3 (padding)
    slot:    41     97     13     -1
                                   ^ -1: the write kernel skips it
  With slot 0, the padding row writes into block 0, slot 0.
  Block 0 belongs to a real sequence: a silent corruption.

  eager step at batch 1:   about 1000 small kernels, about 2x the floor
  graphed step:            1 replay, near the floor
```

## 24-quantized

Each matmul dispatches on its row count.

```text
  time of one matmul
    ^                                          .  int8 GEMV
    |                                     .
    |                                .
    |   bf16 cuBLAS  ============X=====================
    |                       .    ^
    |                  .         the crossover
    |   int8 GEMV .
    +------------------------------------------------> rows R
         1     2     4     8     16    32

  R <= crossover -> the int8 GEMV (1 byte for each weight)
  R >  crossover -> cuBLAS bf16 (the rows share one weight read)
  Measure the crossover in the REAL graphed step: one matrix alone
  of a small model fits in L2, and gives a wrong answer.
```

## 24b-fp8-kv-cache

At long context, the KV cache is most of the bytes. Halve it.

```text
  16 sequences x 2048 tokens x 112 KiB for each token (stage 02):
    bf16 KV: 3.5 GiB to read in each decode step
    FP8  KV: 1.75 GiB, and twice the tokens fit in the pool

  in the kernel:
    one 16-byte load = 16 FP8 values      (8 bf16 values before)
    one instruction converts 2 FP8 values
    scores = (q x k_scale) . k_fp8        the key scale, in q
    out    = (p @ v_fp8) x v_scale        the value scale, in out
```

## 25-speculative

A verify is K + 1 causal decode rows on one block table.

```text
  context = 20 tokens, t0 to t19.   draft K = 3:  d1 d2 d3
  verify = K + 1 = 4 decode rows, one block table:
    row 0:  query t19   sees t0 .. t19
    row 1:  query d1    sees t0 .. t19 d1
    row 2:  query d2    sees t0 .. t19 d1 d2
    row 3:  query d3    sees t0 .. t19 d1 d2 d3
  -> causal, on your stage 08c kernel, in a captured graph

  GREEDY: accept while the draft == the argmax of the row before.
  SAMPLING: the stage 17 rule, with q = one-hot on the draft token.
  The slots of a rejected draft take the next write.
```

## 26-guided

Walk a character automaton, and cache the masks by state.

```text
  stage 19: validate the whole text again, for each token
    -> a new key for each token: the cache never hits

  stage 26: one automaton state after each character
    {"a": 1}  and  {"b": 2}  pass through the SAME few dozen states
    mask cache: key = state      -> the second answer builds almost
                                    no new mask

  NEAR max_tokens: allow only the tokens that close the object
    3 tokens left, text so far:  {"a": [1, 2
    allowed: "]"  "]}"   -> the answer still parses
```

## 27-serve

The event loop never runs step(). A worker thread does.

```text
  event loop thread                     engine worker thread
  POST /v1/chat/completions             while True:
    apply the chat template               take the commands on the deque
    deque.append(("add", req))  ----->    step()        5 to 50 ms
    await the queue of the req  <-----    call_soon_threadsafe(put)

  the client disconnects
    deque.append(("abort", id)) ----->    abort: free the real blocks

  GET /metrics answers at once, because no step blocks the loop.
```

## 28-benchmark

Divide by the floor of the steps that your engine ran.

```text
  floor of one step = the bytes it must read / peak bandwidth
                      (the weights + the KV of every sequence in it)

  your engine ran S steps:
    floor_seconds     = the sum, over the S steps, of each floor
    measured_seconds  = the wall time of the S steps
    floor_over_measured = floor_seconds / measured_seconds

  example: 2.1 s / 5.6 s = 0.375          the target: at least 0.30

  tokens/s changes from card to card. floor_over_measured does not.
```

## 29-lru-cache

A block at refcount 0 stays cached, in LRU order, until memory is short.

```text
  in use (refcount > 0):   [3] [6] [1]
  cached (refcount 0):     oldest -> [9] [4] [12] -> newest
  free (empty):            none

  a block is necessary, and no block is free
    -> evict the OLDEST cached block: 9
  a new request hits block 4
    -> refcount 0 -> 1, and block 4 leaves the LRU list

  cache-aware admission: order WAITING by its cached prefix depth
    req X: 6 blocks cached   -> admit first
    req Y: 2 blocks cached   -> second
    req Z: 0 blocks cached   -> last
```

## 30-multi-lora

One launch applies a different adapter to each row.

```text
  y = x W  +  x A_i B_i          i = the adapter of the row
    A_i: hidden x r,  B_i: r x hidden,  r = the rank, e.g. 16

  batch:  row 0 -> adapter 2     row 1 -> adapter 0
          row 2 -> adapter 2     row 3 -> adapter 7
          idx = [2, 0, 2, 7]

  base:   Y = X W              one GEMM, one shared weight
  BGMV:   one launch. Row j reads A[idx[j]] and B[idx[j]].
          not 3 launches, one for each adapter
```

## 31-mla

Cache a small latent, and absorb the up-projections at decode.

```text
  STANDARD: K and V for each head, for each token
    2 x heads x head_dim = 2 x 32 x 128 = 8192 values
  MLA: one latent c, plus a small key for RoPE, for each token
    d_c + d_rope         = 512 + 64     = 576 values    (93 % less)

  K = c W_uk,  V = c W_uv          never stored
  ABSORB at decode:
    score = q . K^T = q (c W_uk)^T = (q W_uk^T) . c^T
            -> W_uk goes into the query: attend against c directly
    out   = p V W_o = p c (W_uv W_o)
            -> W_uv goes into the output projection
```
