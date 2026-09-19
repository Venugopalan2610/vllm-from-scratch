# Glossary

This course comes from two fields that are not software engineering: machine
learning and GPU programming. Each field has its own words. This file defines
each word that the course uses.

Each entry has three parts:

- **The definition.** One to three sentences.
- **Like.** The nearest idea from software engineering, if a good one exists.
  An analogy is not a definition. It gives you a place to start.
- **Taught in.** The notebook section and the stage that teach the term. Read
  those for the full idea.

`./vc guide` prints the definition and the analogy of each term that a stage
teaches. It finds them from the stage numbers in *Taught in*.

Part 0 teaches the basic words of all three fields. Start there if a word in
Part 1 is new to you.

`dev/jargon.py` reads this file. It makes sure that each notebook defines a
term, or links to this file, before the course uses the term. *Also:* gives
the other names of a term. **Check:** no marks a word that is common in
software, so the checker does not report it.

---

## 1. The model

### Token

*Also:* tokens, token id, token ids

A piece of text, often a word or a part of a word. The model reads and writes
integers, and each integer is the id of one token.

**Like:** an interned string. The id is the index into the table.

**Taught in:** Part 0 `1_model/`. Stage 01.

### Tokenizer

*Also:* tokenizers, tokenize

The code that turns text into token ids, and token ids back into text. Each
model has its own tokenizer, and the two must match.

**Like:** a codec with a fixed dictionary.

**Taught in:** Part 0 `1_model/`. Part 5 `3_detokenize/`.

### Vocabulary

*Also:* vocab, vocabulary size

The full list of tokens that a tokenizer knows. Qwen3 knows about 152,000
tokens. The model gives one score to each of them at each step.

**Like:** the symbol table of the codec.

**Taught in:** Part 0 `1_model/`.

### Model

*Also:* forward pass, language model, LLM

A function that takes token ids and returns a score for each possible next
token. One call of the function is a forward pass.

**Like:** a pure function with a very large constant table: the weights.

**Taught in:** Part 0 `1_model/`. Stage 01.

### Weights

*Also:* parameters, params, parameter count

The numbers that the model learned in training. At inference nothing changes
them: they are read-only data. A 0.6B model has 0.6 billion of them.

**Like:** a large read-only lookup table that each call reads from start to end.

**Taught in:** Part 0 `1_model/`. Part 1 `1_arithmetic/`.

### Embedding

*Also:* embeddings, embedding table

The first step of the model. It replaces each token id with a vector of
numbers from a table: one row for each token in the vocabulary.

**Like:** an array lookup, `table[token_id]`.

**Taught in:** Part 0 `1_model/`.

### Hidden state

*Also:* hidden states, hidden size, d_model, residual stream, activations

The vector that represents one token while it moves through the layers. The
hidden size is its length, for example 1024.

**Like:** the state object that each stage of a pipeline reads and updates.

**Taught in:** Part 0 `1_model/`.

### Layer

*Also:* layers, transformer layer, decoder layer

One stage of the model: attention, then an MLP. A model repeats the same kind
of layer many times, for example 28.

**Like:** one stage of a pipeline. The model is the same stage, repeated with
different weights.

**Taught in:** Part 0 `1_model/`.

### Transformer

*Also:* transformers

The design of almost all current language models: a stack of layers, each with
attention and an MLP.

**Taught in:** Part 0 `1_model/`.

### Attention

*Also:* self-attention, attend

The step in each layer where a token reads information from earlier tokens. It
is a lookup that returns a weighted mix of all entries, not one entry.

**Like:** a dictionary lookup with fuzzy matching. Each key matches a little,
and the result mixes the values by how well each key matched.

**Taught in:** Part 0 `2_attention/`. Part 3 `2_gather/`.

### Query, key and value

*Also:* query, queries, Q, K, V, QKV

The three vectors that each token makes for attention. The query asks. Each
key answers how well it matches. Each value is what the token returns.

**Like:** the search term, the index entries, and the stored records.

**Taught in:** Part 0 `2_attention/`. Stage 02.

### Attention head

*Also:* head, heads, num_heads, head_dim

One independent attention lookup. A layer runs many heads side by side, each
with its own queries, keys and values. head_dim is the vector length of one head.

**Like:** several indexes on the same data, each for a different question.

**Taught in:** Part 0 `2_attention/`.

### Grouped-query attention

*Also:* GQA, KV heads, num_kv_heads

A design where several query heads share one key head and one value head. It
makes the KV cache smaller by the size of the group.

**Like:** many readers that share one index, not one index for each reader.

**Taught in:** Part 0 `2_attention/`. Part 1 `3_kvCache/`.

### MLP

*Also:* feed-forward, FFN

The second step of each layer. Two or three large matmuls that transform each
token alone, with no information from other tokens.

**Like:** a `map` over the tokens.

**Taught in:** Part 0 `1_model/`.

### RoPE

*Also:* rotary position embedding, rotary embedding, position embedding

The way the model knows the order of tokens. It rotates each query and key by
an angle that depends on the position of the token.

**Taught in:** Part 0 `2_attention/`.

### Causal mask

*Also:* causal

The rule that a token can attend only to itself and to earlier tokens, never
to later tokens.

**Like:** a log that you can read only up to your own entry.

**Taught in:** Part 0 `2_attention/`.

### Context

*Also:* context length, context window, context_len, context_lens

All the tokens that a sequence holds at one time: the prompt and the tokens
that the model generated so far.

**Taught in:** Part 0 `2_attention/`. Stage 02.

### Prompt

*Also:* prompts, system prompt

The text that the user sends. The model continues it.

**Check:** no

**Taught in:** Part 0 `1_model/`.

### Logits

*Also:* logit

The raw scores that the model gives to each token of the vocabulary for the
next position. A higher score means a more likely next token.

**Taught in:** Part 0 `1_model/`.

### Softmax

*Also:* softmaxed

The function that turns logits into probabilities. It makes each score
positive with `exp`, then divides by the sum, so the results add up to 1.

**Taught in:** Part 0 `1_model/`.

### Autoregressive generation

*Also:* autoregressive, generation, generate

The loop that makes text: predict one token, append it to the input, and call
the model again. Each new token depends on all earlier tokens.

**Like:** a loop where each iteration reads the output of the one before.

**Taught in:** Part 0 `1_model/`. Stage 01.

### Greedy decoding

*Also:* greedy, argmax

Always choose the token with the highest logit. The output is deterministic.

**Taught in:** Part 0 `1_model/`. Stage 01.

### Sampling

*Also:* sample, sampler

Choose the next token at random, with the probabilities that softmax gives.

**Like:** a weighted random choice, `random.choices(tokens, weights=probs)`.

**Taught in:** Part 0 `1_model/`. Part 5 `2_sampling/`. Stage 13.

### Temperature

*Also:* temperatures

A number that divides the logits before softmax. Below 1 it makes the choice
safer. Above 1 it makes the choice more random.

**Taught in:** Part 0 `1_model/`. Part 5 `2_sampling/`. Stage 13.

### Top-k

*Also:* top_k

Keep only the k tokens with the highest logits, and sample from them.

**Taught in:** Part 5 `2_sampling/`. Stage 13.

### Top-p

*Also:* top_p, nucleus, nucleus sampling

Keep the smallest set of top tokens whose probabilities add up to p, and
sample from them. The set is small when the model is sure, and large when it
is not.

**Taught in:** Part 5 `2_sampling/`. Stage 13.

### Repetition penalty

*Also:* penalty, penalties

A change to the logits of tokens that already appeared, so the model repeats
itself less.

**Taught in:** Part 5 `2_sampling/`. Stage 13.

### EOS

*Also:* end of sequence, stop token, eos_ids

A special token that means "the reply is complete". Generation stops when the
model makes it.

**Like:** a null terminator.

**Taught in:** Part 0 `1_model/`.

### Detokenization

*Also:* detokenize, detokenizer, incremental detokenization

The conversion of token ids back into text while the model streams them. One character
can need two tokens, so you cannot always decode one token alone.

**Taught in:** Part 5 `3_detokenize/`. Stage 14.

### Stop string

*Also:* stop strings, stop condition

Text that ends the reply when it appears, for example `"\n\n"`. It can start in
one token and end in the next.

**Taught in:** Part 5 `3_detokenize/`. Stage 14.

### dtype

*Also:* data type, bf16, bfloat16, fp16, fp32, float32

The number format of a tensor. bf16 uses 2 bytes for each number, and fp32
uses 4. Fewer bytes means less memory traffic.

**Taught in:** Part 0 `1_model/`. Part 1 `1_arithmetic/`.

### Tensor

*Also:* tensors

A multi-dimensional array of numbers, all with the same dtype. torch and JAX
use tensors for all data.

**Check:** no

**Like:** a NumPy array that can live in GPU memory.

**Taught in:** Part 0 `1_model/`.

---

## 2. The inference engine

### Inference

*Also:* inference engine, serving

Use of a trained model to make outputs. This course builds an inference
engine: the server that runs the model for many users.

**Taught in:** Part 0 `1_model/`.

### Prefill

*Also:* prefills, prefilling

The first forward pass of a request. It processes all the prompt tokens in one
call. Its speed depends on arithmetic.

**Taught in:** Part 0 `1_model/`. Part 1 `2_roofline/`. Stage 03.

### Decode

*Also:* decode step, decoding

Each forward pass after prefill. It processes one new token for each sequence.
Its speed depends on memory bandwidth, because it reads all the weights for
little arithmetic.

**Taught in:** Part 0 `1_model/`. Part 1 `2_roofline/`. Stage 03.

### Step

*Also:* steps, engine step

One forward pass of the engine, for all the sequences that it runs at that
time. The engine is a loop of steps.

**Check:** no

**Taught in:** Part 2 `2_continuous/`. Stage 05.

### KV cache

*Also:* KV, key-value cache, kv_cache

The keys and values of all earlier tokens, kept in GPU memory. Each new token
then computes only its own key and value.

**Like:** memoization of the attention inputs.

**Taught in:** Part 0 `2_attention/`. Part 1 `3_kvCache/`. Stage 02.

### Sequence

*Also:* sequences, seq, num_seqs

One reply that the engine generates: a prompt and its output tokens.

**Like:** one in-flight request, with its state.

**Taught in:** Part 2 `1_padding/`. Stage 05.

### Request

*Also:* requests

A call from a client that asks for one reply.

**Check:** no

**Taught in:** Part 2 `2_continuous/`. Part 6 `1_async/`. Stage 15.

### Batch

*Also:* batches, batch size, batching

The sequences that one forward pass processes together. The pass reads the
weights once, and all sequences in the batch share that read.

**Like:** a bulk operation: one round trip for many items.

**Taught in:** Part 1 `1_arithmetic/`. Part 2 `1_padding/`. Stage 04.

### Static batching

*Also:* static batch

Start a batch, and run it until the last sequence finishes. A finished
sequence keeps its slot and does no useful work.

**Like:** a barrier after each batch.

**Taught in:** Part 2 `1_padding/`. Stage 04.

### Continuous batching

*Also:* iteration-level scheduling

Decide the batch again at each step. A finished sequence leaves at once, and a
waiting sequence takes its slot.

**Like:** a work queue with a thread pool, not a barrier.

**Taught in:** Part 2 `2_continuous/`. Stage 05.

### Padding

*Also:* padded, pad

Extra dummy tokens that make all rows of a batch the same length. The GPU does
real work on them and throws the results away.

**Taught in:** Part 2 `1_padding/`. Part 8 `1_engine/`. Stage 04. Stage 21.

### Flat batch

*Also:* ragged batch, ragged, cu_seqlens

All the tokens of all sequences in one list, with no padding. A second list,
often named `cu_seqlens`, tells where each sequence starts.

**Like:** a packed array with an offsets array, as in a CSR matrix.

**Taught in:** Part 8 `1_engine/`. Stage 21.

### Scheduler

*Also:* scheduling, schedule

The code that decides, at each step, which sequences run and how many tokens
each one processes.

**Check:** no

**Taught in:** Part 2 `2_continuous/`. Part 4. Stage 10. Stage 22.

### Admission

*Also:* admit, admitted

The step that moves a waiting request into the running set. The engine admits only when it
has memory for the request.

**Like:** admission control in a server.

**Taught in:** Part 4 `1_admission/`. Stage 10.

### Preemption

*Also:* preempt, preempted, preemptions

The engine removes a running sequence to free its memory, when the KV cache is
full. The
sequence restarts later.

**Like:** the OS evicting a process under memory pressure.

**Taught in:** Part 4 `1_admission/`. Stage 10. Stage 22.

### Recompute

*Also:* recomputation

A way to restart a preempted sequence: drop its KV cache, and run prefill
again later.

**Taught in:** Part 4 `1_admission/`. Stage 10.

### Swap

*Also:* swap out, swap in, swapping

A way to preempt a sequence: copy its KV cache to CPU memory, and copy it back
later.

**Like:** OS swap to disk.

**Check:** no

**Taught in:** Part 4 `1_admission/`. Stage 10.

### Chunked prefill

*Also:* chunked, chunking

Divide a long prompt into chunks, and prefill one chunk in each step, next to
the decodes of other sequences. One long prompt then cannot stop the server.

**Like:** cooperative multitasking: yield after each slice of work.

**Taught in:** Part 4 `2_chunked/`. Stage 11. Stage 22.

### Token budget

*Also:* token budgets

The maximum number of tokens that one step processes. Decodes and prompt
chunks share it.

**Taught in:** Part 4 `2_chunked/`. Stage 11. Stage 22.

### Watermark

A fraction of the KV blocks that the engine keeps free when it admits new
requests. The running sequences then have space to grow.

**Taught in:** Part 4 `1_admission/`. Stage 10.

### KV block

*Also:* KV blocks, block_size, num_blocks

A fixed-size piece of the KV cache that holds the keys and values of a small
number of tokens, for example 16.

**Like:** a memory page.

**Taught in:** Part 3 `1_blocks/`. Stage 06.

### Block table

*Also:* block tables, block_tables, page table

For each sequence, the list of the physical KV blocks that hold its tokens, in
order.

**Like:** the page table of a process.

**Taught in:** Part 3 `1_blocks/`. Stage 06.

### Slot mapping

*Also:* slot_mapping

For each token of a step, the physical place in the KV cache where the engine
writes its key and value.

**Taught in:** Part 3 `2_gather/`. Stage 07.

### Block allocator

*Also:* allocator, free list

The code that gives out free KV blocks and takes them back.

**Like:** a fixed-size memory allocator with a free list.

**Taught in:** Part 3 `1_blocks/`. Stage 06.

### Fragmentation

*Also:* fragmented

Memory that the engine reserved but does not use. A contiguous cache for each
sequence wastes most of its reservation.

**Like:** internal fragmentation in a memory allocator.

**Taught in:** Part 3 `1_blocks/`. Stage 06.

### PagedAttention

*Also:* paged attention, paged

Attention that reads the KV cache through a block table, so the blocks of a
sequence can be anywhere in memory. vLLM introduced it.

**Like:** virtual memory for the KV cache.

**Taught in:** Part 3. Stages 06 to 08c.

### Prefix cache

*Also:* prefix caching, prefix sharing, cache hit, hit rate

Reuse of the KV blocks of a prompt start that an earlier request already
computed, for example a shared system prompt.

**Like:** a content-addressed cache: the hash of the tokens is the key.

**Taught in:** Part 3 `4_sharing/`. Stage 09.

### Reference count

*Also:* refcount, refcounts, ref_counts

The number of sequences that use one KV block. The allocator frees the block
when the count reaches 0.

**Like:** reference counting in a garbage collector.

**Taught in:** Part 3 `4_sharing/`. Stage 09.

### Copy-on-write

*Also:* copy on write, CoW

Shared blocks stay shared until one sequence must write. Then that sequence
gets its own copy.

**Like:** `fork()` and copy-on-write pages.

**Taught in:** Part 3 `4_sharing/`. Stage 09.

### ISL and OSL

*Also:* ISL, OSL, input sequence length, output sequence length

The input sequence length is the tokens of the prompt. The output sequence
length is the tokens that the engine generates. A serving benchmark states
both, because prefill cost follows ISL and decode cost follows OSL.

**Like:** the request size and the response size of an API benchmark.

**Taught in:** Stage 18. Stage 28.

### KL divergence

*Also:* KL, KLD, mean KL, nats

A measure of how far one probability distribution is from another. It is 0
when the two are the same. The course uses it to compare a quantized model
with bf16 at each generated token.

**Like:** a diff score between two outputs, where 0 means no difference.

**Taught in:** Part 7 `2_quantization/`. Stage 18. Stage 24. Stage 24b.

### Top-1 agreement

*Also:* fidelity

The fraction of positions where two models choose the same most likely token.
Two bf16 runs with a different order of additions already disagree at about
2% of positions, so it cannot be 100%.

**Like:** the match rate of two implementations on the same inputs.

**Taught in:** Stage 18. Stage 24.

### Speculative decoding

*Also:* speculation, speculative

Guess several next tokens with a cheap method, then check all the guesses in
one forward pass. The output is exactly the same as without the guesses.

**Like:** branch prediction.

**Taught in:** Part 7 `1_speculative/`. Stage 17. Stage 25.

### Draft

*Also:* drafts, draft model, draft tokens, drafted

The guessed tokens in speculative decoding, and the cheap method that makes
them.

**Taught in:** Part 7 `1_speculative/`. Stage 17. Stage 25.

### Acceptance rate

*Also:* acceptance

The fraction of draft tokens that the check keeps.

**Like:** the hit rate of a branch predictor.

**Taught in:** Part 7 `1_speculative/`. Stage 17. Stage 25.

### Rejection sampling

*Also:* modified rejection sampling

The rule that accepts or rejects each draft token so that the output has the
exact distribution of the large model.

**Taught in:** Part 7 `1_speculative/`. Stage 17. Stage 25.

### Quantization

*Also:* quantize, quantized, quantizing, dequantize, dequantized

Store the weights, or the KV cache, with fewer bits for each number, plus a
scale to get the value back. Fewer bytes to read makes decode faster.

**Like:** lossy compression.

**Taught in:** Part 7 `2_quantization/`. Stage 18. Stage 24.

### int8

*Also:* INT8

An 8-bit integer format. With one scale for each row, it stores weights in 1
byte, not 2.

**Taught in:** Part 7 `2_quantization/`. Stage 18. Stage 24.

### fp8

*Also:* FP8, e4m3

An 8-bit floating-point format. It has an exponent, so it keeps small and
large values in the same tensor better than int8.

**Taught in:** Part 7 `2_quantization/`. Stage 24b.

### Per-channel scale

*Also:* per-channel, scales, scale

One scale number for each row of a weight matrix. The real value is the stored
integer times the scale of its row.

**Check:** no

**Taught in:** Part 7 `2_quantization/`. Stage 18. Stage 24.

### Guided decoding

*Also:* structured output, constrained decoding, constrained output

Force the output to follow a grammar, for example valid JSON. The engine
removes the tokens that break the grammar before it samples.

**Like:** input validation, but on each token before the model commits it.

**Taught in:** Part 7 `3_guided/`. Stage 19. Stage 26.

### Logit mask

*Also:* logit masking, mask the logits

Set the logits of forbidden tokens to minus infinity, so softmax gives them
probability 0.

**Taught in:** Part 7 `3_guided/`. Stage 19. Stage 26.

### FSM

*Also:* finite state machine, state machine

A set of states and rules for moving between them. Guided decoding uses one to
know which tokens are legal next.

**Like:** the state machine in a parser or a regex engine.

**Taught in:** Part 7 `3_guided/`. Stage 19. Stage 26.

### Tensor parallelism

*Also:* tensor parallel, TP

Divide each weight matrix across several GPUs, so each GPU holds and reads
only a part of the model.

**Like:** sharding a table across servers.

**Taught in:** Part 7 `4_tensorParallel/`. Stage 20.

### Rank

*Also:* ranks

One GPU, or one process, in a group of GPUs that work together.

**Like:** one node of a cluster, with a numeric id.

**Taught in:** Part 7 `4_tensorParallel/`. Stage 20.

### All-reduce

*Also:* all_reduce, allreduce

A collective where each rank gives a tensor, and each rank gets back the sum
of all of them.

**Like:** a distributed sum, where each node gets the result.

**Taught in:** Part 7 `4_tensorParallel/`. Stage 20.

### Collective

*Also:* collectives

A communication step that all ranks do together, for example an all-reduce.

**Taught in:** Part 7 `4_tensorParallel/`. Stage 20.

### Shard

*Also:* shards, sharding, sharded

The part of a weight matrix that one rank holds.

**Taught in:** Part 7 `4_tensorParallel/`. Stage 20.

### CUDA graph

*Also:* CUDA graphs, graph capture, capture, replay, graphed

A recording of a fixed sequence of GPU kernels. The CPU then replays all of
them with one call, not one launch for each kernel.

**Like:** a prepared statement: plan one time, run many times.

**Taught in:** Part 5 `1_cudaGraphs/`. Stage 12. Stage 23.

### Bucket

*Also:* buckets, shape bucket

One of a small set of batch sizes for which the engine captured a CUDA graph.
A step runs the smallest bucket that holds its batch, with padding.

**Like:** size classes in a memory allocator.

**Taught in:** Part 5 `1_cudaGraphs/`. Stage 12. Stage 23.

---

## 3. The GPU

### GPU

*Also:* GPUs, graphics card, card

A processor with thousands of simple threads and very fast memory. It runs one
operation on a large amount of data at the same time.

**Like:** a very wide SIMD machine.

**Taught in:** Part 0 `3_gpu/`.

### Host and device

*Also:* host, device

The CPU and its memory are the host. The GPU and its memory are the device.
Data must move from host to device before the GPU can use it.

**Check:** no

**Taught in:** Part 0 `3_gpu/`.

### HBM

*Also:* VRAM, DRAM, GPU memory, device memory, global memory

The main memory of the GPU. It is large and fast, but it is far from the
arithmetic units, so each read costs time.

**Like:** RAM, compared with the CPU caches.

**Taught in:** Part 0 `3_gpu/`. Part 1 `2_roofline/`. Stage 03.

### Bandwidth

*Also:* memory bandwidth, GB/s, read bandwidth

The bytes for each second that the GPU can move between its memory and its
arithmetic units. Decode speed depends on it.

**Like:** the throughput of a network link.

**Taught in:** Part 0 `3_gpu/`. Part 1 `2_roofline/`. Stage 03.

### FLOP

*Also:* FLOPs, FLOPS, FLOP/s, TFLOP/s, flops

One floating-point operation, for example a multiply or an add. FLOP/s is the
number of them for each second.

**Taught in:** Part 0 `3_gpu/`. Part 1 `1_arithmetic/`.

### Arithmetic intensity

*Also:* intensity, FLOP per byte, FLOP/byte

The FLOPs that a piece of work does for each byte that it reads.

**Like:** the ratio of CPU work to I/O in a job.

**Taught in:** Part 0 `3_gpu/`. Part 1 `1_arithmetic/`. Stage 03.

### Ridge point

*Also:* ridge, B_ridge

FLOP/s divided by bytes/s for one GPU. Work with a lower arithmetic intensity
waits for memory. Work with a higher intensity waits for arithmetic.

**Taught in:** Part 0 `3_gpu/`. Part 1 `1_arithmetic/`. Stage 03.

### Roofline

*Also:* roofline model, rooflines

A plot of the maximum speed of a GPU against arithmetic intensity. It shows if
a piece of work waits for memory or for arithmetic.

**Taught in:** Part 1 `2_roofline/`. Stage 03. Stage 28.

### Memory-bound

*Also:* memory bound, bandwidth-bound

Work that waits for bytes. A faster memory makes it faster. More arithmetic
units do not.

**Like:** an I/O-bound job.

**Taught in:** Part 0 `3_gpu/`. Part 1 `1_arithmetic/`. Stage 03.

### Compute-bound

*Also:* compute bound

Work that waits for arithmetic. More FLOP/s makes it faster. More bandwidth
does not.

**Like:** a CPU-bound job.

**Taught in:** Part 0 `3_gpu/`. Part 1 `1_arithmetic/`. Stage 03.

### Matmul

*Also:* matmuls, matrix multiplication, GEMM, GEMMs

A matrix multiplication. Most of the arithmetic in a model is matmuls.

**Taught in:** Part 0 `3_gpu/`. Part 1 `2_roofline/`.

### GEMV

*Also:* matrix-vector

A matrix times a vector. Decode at batch 1 is a chain of GEMVs, and a GEMV has
a very low arithmetic intensity.

**Taught in:** Part 0 `3_gpu/`. Stage 18b.

### cuBLAS

The matmul library from NVIDIA. torch calls it for most matmuls.

**Like:** a standard library that is very hard to beat.

**Taught in:** Stage 18b.

### L2 cache

*Also:* L2

A cache on the GPU chip between the SMs and HBM. It is much faster than HBM,
and much smaller.

**Like:** the CPU last-level cache.

**Taught in:** Part 3 `3_kernels/`.

### SM

*Also:* SMs, streaming multiprocessor, num_sms

One of the processor cores of a GPU. A GPU has tens to hundreds of them. Each
SM runs many threads at the same time.

**Like:** a CPU core with very wide SIMD and many hardware threads.

**Taught in:** Part 0 `3_gpu/`. Part 3 `3_kernels/`.

### CUDA

The programming model and the toolkit that NVIDIA gives to write code for its
GPUs.

**Taught in:** Part 0 `3_gpu/`. Stage 08.

### nvcc

The CUDA compiler.

**Like:** gcc for GPU code.

**Taught in:** Part 0 `3_gpu/`. Stage 08.

### Kernel

*Also:* kernels, CUDA kernel

A function that runs on the GPU. One launch runs the same function in many
threads at the same time, each with its own index.

**Like:** the body of a parallel for loop, where the loop index is the thread id.

**Taught in:** Part 0 `3_gpu/`. Stage 08.

### Launch

*Also:* launches, kernel launch, launch overhead

The CPU call that starts a kernel. Each launch costs a few microseconds of CPU
time, even when the kernel does almost nothing.

**Like:** the fixed cost of a system call or an RPC.

**Taught in:** Part 0 `3_gpu/`. Part 5 `1_cudaGraphs/`. Stage 12.

### Thread

*Also:* threads, threadIdx

One instance of a kernel, with its own index. A GPU thread is much smaller and
cheaper than a CPU thread.

**Check:** no

**Taught in:** Part 0 `3_gpu/`.

### Warp

*Also:* warps

A group of 32 GPU threads that run the same instruction at the same time.

**Like:** one SIMD instruction with 32 lanes.

**Taught in:** Part 0 `3_gpu/`. Part 3 `3_kernels/`. Stage 08c.

### Lane

*Also:* lanes

The index of one thread in its warp, from 0 to 31.

**Like:** a SIMD lane.

**Taught in:** Part 3 `3_kernels/`.

### Thread block

*Also:* thread blocks, CUDA block, blockIdx, blockDim

A group of threads that run on the same SM and can share memory. A launch runs
many blocks.

**Taught in:** Part 0 `3_gpu/`. Stage 08.

### Grid

*Also:* gridDim

All the thread blocks of one launch.

**Taught in:** Part 0 `3_gpu/`. Stage 08.

### Shared memory

*Also:* smem, __shared__

A small, fast memory on each SM that the threads of one block share.

**Like:** a scratchpad that the programmer manages, not a cache.

**Taught in:** Stage 08.

### Registers

*Also:* register

The fastest storage on the GPU, private to each thread. Each SM has a fixed
number, and the threads divide them.

**Taught in:** Stage 08c.

### Occupancy

The warps that an SM holds at one time, divided by the maximum that it can
hold. More resident warps hide more memory latency. Occupancy tells you how
much latency the SM can hide. It does not tell you how busy the SM is.

**Like:** the number of requests in flight that keep a server busy while each
one waits on I/O.

**Taught in:** Part 3 `3_kernels/`. Stage 08c.

### Coalescing

*Also:* coalesce, coalesced, uncoalesced

When the threads of a warp read neighbor addresses, the GPU merges their reads
into a small number of large memory transactions.

**Like:** reads that stay in the same cache lines.

**Taught in:** Part 3 `3_kernels/`. Stage 08b.

### Sector

*Also:* sectors

The 32-byte unit that the GPU memory system fetches. A read of 4 bytes still
fetches the whole sector.

**Like:** a cache line, but 32 bytes.

**Taught in:** Part 3 `3_kernels/`.

### Spill

*Also:* spills, register spill

The compiler ran out of registers and put values in slow memory. It always
makes the kernel slower, and the build log of ptxas always shows it.

**Taught in:** Stage 08c.

### Warp shuffle

*Also:* shuffle, __shfl_xor_sync

An instruction that moves values between the registers of threads in one warp,
with no shared memory and no barrier. Each lane in the mask must reach the
instruction. If one lane does not, the instruction never returns.

**Taught in:** Part 3 `3_kernels/`. Stage 08c.

### Split-K

*Also:* flash-decoding

Divide the context of one sequence across several thread blocks, when there
are too few sequences to fill the GPU. Each block writes a partial result. A
second pass rescales and merges the parts, and the result is exact.

**Taught in:** Stage 08c.

### Online softmax

*Also:* running max

A softmax that reads its inputs in pieces. It keeps a running maximum and a
running sum, so it never needs all the scores at one time.

**Like:** a streaming aggregate that makes one pass over the data.

**Taught in:** Stage 08.

### Fused kernel

*Also:* fusion, fuse, fused, fused epilogue, epilogue

One kernel that does the work of two or more kernels, so the data in between
stays in registers and never goes to memory.

**Like:** a query plan that pipelines two operators, not one that writes a
temporary table.

**Taught in:** Part 7 `2_quantization/`. Stage 18b.

### Interconnect

*Also:* PCIe, NVLink

The link between the GPU and the CPU, or between GPUs. It is much slower than
HBM.

**Like:** the network between servers.

**Taught in:** Part 4 `1_admission/`. Part 7 `4_tensorParallel/`.

### Nsight Compute

*Also:* ncu

The NVIDIA profiler that reads the hardware counters of a kernel.

**Like:** `perf` for a GPU kernel.

**Taught in:** Stage 08b.

### XLA

The compiler that JAX uses. It compiles a whole function for exact tensor
shapes.

**Taught in:** The JAX track. LORE.md section 10.

### jit

*Also:* JIT, compile, recompile

Just-in-time compilation. JAX compiles a function the first time that it sees
a new input shape.

**Check:** no

**Taught in:** The JAX track.

### Pallas

The JAX language for custom GPU kernels.

**Taught in:** The JAX track, `app/j08_paged_pallas.py`.

---

## 4. Serving and metrics

### Latency

*Also:* latencies

The time from the request to the complete reply.

**Check:** no

**Taught in:** Part 2 `2_continuous/`. Part 6 `2_metrics/`. Stage 16.

### Throughput

The work that the engine completes for each second, in tokens or requests.

**Check:** no

**Taught in:** Part 6 `2_metrics/`. Stage 16. Stage 28.

### TTFT

*Also:* time to first token

The time from the request to the first output token. The queue wait and the
prefill set it.

**Taught in:** Part 6 `2_metrics/`. Stage 16.

### TPOT

*Also:* ITL, inter-token latency, time per output token

The time between two output tokens of one reply. It decides how smooth the
stream looks.

**Taught in:** Part 6 `2_metrics/`. Stage 16.

### Percentile

*Also:* p50, p90, p99, median

The value that a given fraction of the samples stay under. p99 is the latency
that 99% of requests beat.

**Like:** the same percentiles that you use for a web service.

**Taught in:** Part 6 `2_metrics/`. Stage 16. Stage 28.

### SLO

*Also:* SLA, SLAs, SLOs

The latency promise to the users, for example "TTFT under 1 second".

**Taught in:** Part 6 `2_metrics/`. Stage 16.

### Goodput

The throughput of the requests that met their SLO. It is the metric that
matters.

**Like:** throughput that counts only the successful responses.

**Taught in:** Part 6 `2_metrics/`. Stage 16. Stage 28.

### Pareto frontier

*Also:* Pareto

The set of settings where you cannot improve one metric without a loss in
another.

**Taught in:** Part 6 `2_metrics/`.

### OpenAI API

*Also:* OpenAI-compatible, chat completions

The HTTP API that OpenAI defined for chat and completion requests. Most
inference servers accept the same requests, so one client works with all of
them.

**Like:** a de facto standard REST API.

**Taught in:** Stage 15. Stage 27.

### Streaming

*Also:* stream, streamed

Send each token to the client when the engine makes it, not the full reply at
the end.

**Check:** no

**Taught in:** Part 5 `3_detokenize/`. Part 6 `1_async/`. Stage 15. Stage 27.
