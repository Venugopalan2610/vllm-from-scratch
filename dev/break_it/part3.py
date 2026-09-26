"""Part 3, break it on purpose: blocks, the page table, kernels and sharing."""

PART = 3
NAME = 'PagedAttention'
FOLDER = 'course/Part3_PagedAttention/5_incidents'
IMPORTS = '''import random
import cudalib
from transformers import AutoModelForCausalLM, AutoTokenizer'''

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into a working allocator, page table, kernel or prefix
cache, and you watch what it does.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** Which sequence, which length, which size?
- **What?** What does the wrong result look like?
- **Which guard?** Which check would catch it?

`lab.py` has a correct `Allocator`, a paged cache on plain tensors
(`new_cache`, `write`, `gather`), and `check(pool)`, which lists the broken
invariants of a pool. Most exercises need no model. Exercise 3 uses Qwen3-1.7B,
and Exercises 5 and 6 compile two small CUDA kernels.

This notebook needs a GPU with about 6 GB free.
'''

LOAD = '''
### run this cell

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'
BLOCK = lab.BLOCK
print('a correct pool, 200 random sequences with forks:',
      lab.workload(lab.Allocator(4000), forks=True, lengths=(1, 600)) or 'no violation')
'''

DRILLS = [
    ('d1', '''
# Exercise 1: abort forgets to free

The pool has an `abort()` that removes the sequence and does not release its
blocks. Simulate five days. Each day 2,000 requests arrive, and about 6% of
them are aborted after 100 to 400 tokens. At the end of each day no request
is active.

This is Ticket 1 of the incident file. Predict the blocks lost each day, and
the loss for each abort.
''', '''
class LeakyPool(lab.Allocator):
  def abort(self, seq):
    del self.tables[seq]                                               # THE FAULT: no release
    del self.lengths[seq]

pool = LeakyPool(20_000)
rng = random.Random(0)
seq = 0
print(f'day 0: {pool.num_free():6,} free')
for day in range(1, 6):
  aborts = blocks_of_aborts = 0
  for _ in range(2000):
    length = rng.randint(100, 400)
    pool.allocate(seq, length)
    if rng.random() < 0.06:
      aborts += 1
      blocks_of_aborts += len(pool.tables[seq])
      pool.abort(seq)
    else:
      pool.free(seq)
    seq += 1
  print(f'day {day}: {pool.num_free():6,} free   {aborts} aborts holding {blocks_of_aborts} blocks')
print('the invariants at the end:', lab.check(pool))
'''),

    ('d2', '''
# Exercise 2: the block that comes one token late

Six sequences decode 20 tokens each through a paged cache. The append
allocates a new block when the sequence **fills** a block, after it counts
the new token. The block tables go to the attention as one tensor, padded with
0. The reference is a contiguous copy of the same K and V.

This is Ticket 2. Predict which sequences go wrong. The pool hands out block 0
first.
''', '''
torch.manual_seed(0)
pool_list = list(range(63, -1, -1))                                    # pops 0, 1, 2, ...
cache = lab.new_cache(64)
prompts = [30, 32, 45, 48, 16, 17]
tables, lengths, truth = [], [], []
for n in prompts:
  tables.append([pool_list.pop() for _ in range(math.ceil(n / BLOCK))])
  lengths.append(n)
  truth.append((torch.randn(n, 8, 128, device=device), torch.randn(n, 8, 128, device=device)))
width = 8
def padded(row):                                                       # the tensor the kernel sees
  return tables[row] + [0] * (width - len(tables[row]))
for row, (k, v) in enumerate(truth):
  for position in range(lengths[row]):
    lab.write(cache, padded(row), position, k[position], v[position])

for step in range(20):
  for row in range(len(prompts)):
    lengths[row] += 1                                                  # THE FAULT: count first,
    if lengths[row] % BLOCK == 0:                                      # then allocate one step early
      tables[row].append(pool_list.pop())
    k_new, v_new = torch.randn(8, 128, device=device), torch.randn(8, 128, device=device)
    truth[row] = (torch.cat([truth[row][0], k_new[None]]), torch.cat([truth[row][1], v_new[None]]))
    lab.write(cache, padded(row), lengths[row] - 1, k_new, v_new)

q = torch.randn(8, 128, device=device)
for row, n in enumerate(prompts):
  k, v = lab.gather(cache, padded(row), lengths[row])
  error = (lab.attend(q, k, v) - lab.attend(q, *truth[row])).abs().max().item()
  print(f'prompt {n:2d} (n mod 16 = {n % 16:2d}), blocks {tables[row]}: error {error:.3f}')
'''),

    ('d3', '''
# Exercise 3: a prefix cache that hashes only the block

Two products share a server: LegalBot and ChefBot. Their system prompts have
the same length, 48 tokens, and both are followed by the same safety text of
128 tokens (8 blocks). LegalBot ran first, so its blocks are in the cache. The
hash of a block is `hash(tuple(tokens))`, with no parent.

Build the cache of a ChefBot request as that prefix cache would: blocks 0 to
2 are misses (ChefBot's own K and V), blocks 3 to 10 are hits (LegalBot's K
and V), and the question is computed on top.

This is Ticket 3. Predict three things: how different the K of the hit blocks
is in the first layer and in a deep layer, the [KL](../../GLOSSARY.md#kl-divergence) between the two next-token
distributions, and the answer of ChefBot.
''', '''
def fixed(text, n):
  ids = tokenizer(text).input_ids
  assert len(ids) >= n, len(ids)
  return ids[:n]

legal = fixed('You are LegalBot, the contract assistant of a law firm. You answer in formal legal '
              'language, cite the clauses of the agreement between the parties, and never give '
              'advice outside the terms and conditions of the contract that the client signed with the '
              'firm, and you always remind the client that a lawyer must review every document.', 48)
chef = fixed('You are ChefBot, the friendly cooking assistant of a recipe website. You answer with '
             'simple recipes, list the ingredients first, then the steps, and you always suggest '
             'a vegetarian option and a dessert that goes well with the dish of the user.', 48)
safety = fixed('Stay polite and helpful. Do not share private data. If the request is unsafe, refuse '
               'it briefly and explain why. Keep the answer short and clear for the user. ' * 8, 128)
question = tokenizer('\\nUser: How do I make a quick tomato soup?\\nAssistant:').input_ids
prompt_a, prompt_b = legal + safety, chef + safety + question
shared = slice(48, 48 + 128)

hashes_a = {hash(tuple(prompt_a[i:i + BLOCK])) for i in range(0, len(prompt_a), BLOCK)}
lookup = ['hit' if hash(tuple(prompt_b[i:i + BLOCK])) in hashes_a else 'miss'
          for i in range(0, 48 + 128, BLOCK)]                          # THE FAULT: no parent hash
print('lookup of the ChefBot blocks 0 to 10:', lookup)

ids = lambda t: torch.tensor([t], device=device)
stop = set(model.generation_config.eos_token_id)
kv_a = lab.model_kv(model, ids(prompt_a))
kv_b = lab.model_kv(model, ids(prompt_b[:-1]))
for layer in (0, 14, 27):
  k_a, k_b = kv_a[layer][0][:, :, shared], kv_b[layer][0][:, :, shared]
  print(f'layer {layer:2d}: the K of the hit blocks differs by {((k_a - k_b).norm() / k_b.norm()).item():.1%}')

def spliced():
  cache = lab.cache_from([lab.slice_kv(kv_b, 0, 48), lab.slice_kv(kv_a, 48, 48 + 128)])
  with torch.inference_mode():
    model(ids(question[:-1]), past_key_values=cache, use_cache=True)
  return cache

with torch.inference_mode():
  good = model(ids([prompt_b[-1]]), past_key_values=lab.cache_from([kv_b])).logits[0, -1].float().log_softmax(-1)
  bad = model(ids([question[-1]]), past_key_values=spliced()).logits[0, -1].float().log_softmax(-1)
print(f'KL of the next-token distribution: {(good.exp() * (good - bad)).sum().item():.3f} nats')

correct = lab.continue_greedy(model, lab.cache_from([kv_b]), prompt_b[-1], 60, stop)
wrong = lab.continue_greedy(model, spliced(), question[-1], 60, stop)
print('first difference at:', next((i for i, (a, b) in enumerate(zip(correct, wrong)) if a != b), None))
print('correct:', repr(tokenizer.decode(correct)))
print('hashed :', repr(tokenizer.decode(wrong)))
'''),

    ('d4', '''
# Exercise 4: a session id in the first block

A correct prefix cache, with chained hashes. 500 requests share a system
prompt of about 1,200 tokens. The template puts a new session id in the first
line. Then move the session id to the end of the system prompt, and run the
same 500 requests again.

This is Ticket 4. Predict both hit rates.
''', '''
body = ' '.join(['Follow the house style, answer in short paragraphs, and cite the sources.'] * 80)
def request(session, question, id_first):
  head = f'You are the assistant of Acme. Session: {session}.' if id_first else 'You are the assistant of Acme.'
  tail = '' if id_first else f' Session: {session}.'
  return tokenizer(head + ' ' + body + tail + '\\nUser: ' + question).input_ids

def hit_rate(requests):
  cache, hit_tokens, total = set(), 0, 0
  for ids in requests:
    parent, missed = None, False
    for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK):
      h = hash((parent, tuple(ids[i:i + BLOCK])))
      if not missed and h in cache:
        hit_tokens += BLOCK
      else:
        missed = True
        cache.add(h)
      parent = h
    total += len(ids)
  return hit_tokens / total

rng = random.Random(0)
sessions = [f'{rng.getrandbits(32):08x}' for _ in range(500)]
questions = [f'What is the status of order {rng.randint(1000, 9999)}?' for _ in range(500)]
for id_first in (True, False):                                         # THE FAULT when True
  requests = [request(s, q, id_first) for s, q in zip(sessions, questions)]
  print(f'session id {"first" if id_first else "last "}: {len(requests[0])} tokens per request, '
        f'hit rate {hit_rate(requests):.1%}')
'''),

    ('d5', '''
# Exercise 5: one thread for each row

Two kernels read the same 268 MB of keys: 1,048,576 rows of 128 bf16 values.
Kernel A gives one thread to one row, and each thread walks its 128 values.
Kernel B gives 16 threads to one row, and each thread reads 16 bytes at once.

This is Ticket 5. Predict the GB/s of each, against `./vc info`. The first run
compiles, which takes 20 to 40 seconds.
''', '''
KERNELS = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

// A: one thread for each row                                   THE FAULT
__global__ void row_per_thread(const __nv_bfloat16* keys, float* out, long rows) {
  long row = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.f;
  for (int d = 0; d < 128; ++d) sum += __bfloat162float(keys[row * 128 + d]);
  out[row] = sum;
}

// B: 16 threads for each row, 16 bytes each, and a grid-stride loop
__global__ void row_per_16(const __nv_bfloat16* keys, float* out, long rows) {
  long lane = threadIdx.x % 16;
  for (long row = (blockIdx.x * (long)blockDim.x + threadIdx.x) / 16; row < rows;
       row += (long)gridDim.x * blockDim.x / 16) {
    uint4 chunk = reinterpret_cast<const uint4*>(keys + row * 128)[lane];
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&chunk);
    float sum = 0.f;
    for (int i = 0; i < 8; ++i) sum += __bfloat162float(v[i]);
    for (int offset = 8; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset, 16);
    if (lane == 0) out[row] = sum;
  }
}

void run_a(torch::Tensor keys, torch::Tensor out) {
  long rows = keys.size(0);
  row_per_thread<<<(rows + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)keys.data_ptr(), out.data_ptr<float>(), rows);
}
void run_b(torch::Tensor keys, torch::Tensor out, int64_t blocks) {
  row_per_16<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)keys.data_ptr(), out.data_ptr<float>(), keys.size(0));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_a", &run_a); m.def("run_b", &run_b); }
"""
kernels = cudalib.build_source('inc3_rows', KERNELS)

keys = torch.randn(1 << 20, 128, device=device, dtype=torch.bfloat16)
out = torch.empty(1 << 20, device=device)

def ms_per_call(fn, iters=20):
  fn()
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - start) / iters * 1000

for name, fn in [('A, one thread for each row', lambda: kernels.run_a(keys, out)),
                 ('B, 16 threads for each row', lambda: kernels.run_b(keys, out, 1 << 16))]:
  ms = ms_per_call(fn)
  print(f'{name}: {ms:.2f} ms, {keys.numel() * 2 / ms / 1e6:4.0f} GB/s')
print('the same answer:', torch.allclose(out, keys.float().sum(1), atol=1e-2))
'''),

    ('d6', '''
# Exercise 6: too few blocks for the machine

Kernel B of Exercise 5 has a grid-stride loop, so any number of thread
blocks reads the whole tensor. Launch it with 16 blocks, as a decode kernel
at batch 1 does with one block for each of 16 heads. Then with more blocks.

This is Ticket 6. Your card has the number of SMs that `./vc info` prints.
Predict the GB/s for each grid.
''', '''
SMS = torch.cuda.get_device_properties(0).multi_processor_count
print('SMs on this card:', SMS)
for blocks in (16, SMS, 4 * SMS, 1 << 16):
  ms = ms_per_call(lambda: kernels.run_b(keys, out, blocks))
  print(f'{blocks:6d} blocks: {ms:6.2f} ms, {keys.numel() * 2 / ms / 1e6:4.0f} GB/s')
del keys, out
'''),

    ('d7', '''
# Exercise 7: blocks of 256 tokens

A pool of 400,000 token slots. The sequences have random lengths from 1 to
800 tokens. Fill the pool with sequences until the next one does not fit, for
three block sizes.

This is Ticket 7. Predict the sequences that fit with blocks of 16 and of 256.
''', '''
rng = random.Random(0)
seqs = [rng.randint(1, 800) for _ in range(5000)]
for block in (16, 32, 256):                                            # 256: THE FAULT
  used = count = tokens = 0
  for n in seqs:
    need = math.ceil(n / block) * block
    if used + need > 400_000:
      break
    used, count, tokens = used + need, count + 1, tokens + n
  print(f'block {block:3d}: {count} sequences fit, {tokens / used:.1%} of the allocated slots hold a token')
'''),

    ('d8', '''
# Exercise 8: four samples and no copy-on-write

A prompt, then 4 samples that share its blocks (`fork`). Each sample writes 5
new tokens of its own. The write does not check the reference count, so it
writes into the shared block. Do it for a prompt of 37 tokens and one of 48.

This is Ticket 8. Predict which samples are correct for each prompt.
''', '''
torch.manual_seed(1)
for prompt_len in (37, 48):
  pool = lab.Allocator(64)
  cache = lab.new_cache(64)
  pool.allocate('prompt', prompt_len)
  k0, v0 = torch.randn(prompt_len, 8, 128, device=device), torch.randn(prompt_len, 8, 128, device=device)
  for p in range(prompt_len):
    lab.write(cache, pool.tables['prompt'], p, k0[p], v0[p])
  samples = [f's{i}' for i in range(4)]
  truth = {}
  for s in samples:
    pool.fork('prompt', s)
    truth[s] = (k0.clone(), v0.clone())
  for step in range(5):
    for s in samples:
      position = pool.lengths[s]
      pool.append(s)                                                   # THE FAULT: no copy-on-write
      k, v = torch.randn(8, 128, device=device), torch.randn(8, 128, device=device)
      lab.write(cache, pool.tables[s], position, k, v)
      truth[s] = (torch.cat([truth[s][0], k[None]]), torch.cat([truth[s][1], v[None]]))
  q = torch.randn(8, 128, device=device)
  errors = []
  for s in samples:
    k, v = lab.gather(cache, pool.tables[s], pool.lengths[s])
    errors.append((lab.attend(q, k, v) - lab.attend(q, *truth[s])).abs().max().item())
  print(f'prompt {prompt_len}: errors of the 4 samples {[round(e, 3) for e in errors]}')
'''),
]

MYSTERY = '''
# Exercise 9: three mystery allocators

The module `mystery.py` holds three allocators: `PoolA`, `PoolB` and
`PoolC`. They have the same methods as `lab.Allocator`. Each one has one
fault. **Do not open the file.**

`lab.workload(pool, forks=..., lengths=..., seed=...)` runs random sequences
through a pool and returns every broken invariant that `lab.check` saw. The
default workload is a start, not an answer.

For each allocator:

1. Run the default workload.
2. Design a second experiment that makes the fault visible. Which variable do
   you change? Forks or no forks? The lengths? The order of the calls?
3. Write your diagnosis: the fault, and the experiment that proved it.
4. Only then, open `mystery.py` and check.

A hint about the method: the default workload has no forks, and its
sequences stay short. One allocator fails it at once, and your work
is to explain how. The other two pass it.
'''

MYSTERY_CODE = '''
from mystery import PoolA, PoolB, PoolC

for name, Pool in [('A', PoolA), ('B', PoolB), ('C', PoolC)]:
  problems = lab.workload(Pool(4000))
  print(f'Pool{name}: {len(problems)} violations', problems[:3])
'''

MYSTERY_NAMES = ['PoolA', 'PoolB', 'PoolC']

FINGERPRINT_ROWS = ['abort forgets to free', 'the block that comes one token late',
                    'a hash with no parent', 'a session id in the first block',
                    'one thread for each row', 'too few blocks for the machine',
                    'blocks of 256 tokens', 'no copy-on-write']

SOLUTIONS = {
    'd1': '''
### What happens

| day | free blocks | aborts | blocks those aborts held |
|---|---|---|---|
| 0 | 20,000 | | |
| 1 | 18,129 | 117 | 1,871 |
| 2 | 16,578 | 104 | 1,551 |
| 3 | 14,712 | 113 | 1,866 |
| 4 | 12,486 | 138 | 2,226 |
| 5 | 10,865 | 98 | 1,621 |

- **Crash?** No. Not until the pool is empty, in about 11 days here.
- **What?** Each day the pool loses **exactly** the blocks that the aborted
  requests held: 20,000 - 18,129 = 1,871. `check` finds 9,135 blocks that
  are neither in a table nor on the free list.

**The fingerprint.** A loss at idle that equals the blocks of one kind of
event. **The guard.** The invariant `used + free = total` at every idle
moment.
''',
    'd2': '''
### What happens

| prompt | n mod 16 | error |
|---|---|---|
| 30 | 14 | 0.712 |
| 32 | 0 | 0.795 |
| 45 | 13 | 0.000 |
| 48 | 0 | 0.350 |
| 16 | 0 | 0.846 |
| 17 | 1 | 0.000 |

- **Crash?** No. The padded table gives block 0 to every missing entry.
- **When?** The three prompts whose length is a multiple of 16 are wrong.
  Their first decode token needs a new block at once, and the block comes one
  step late.
- **The victim.** The prompt of 30 tokens has no fault of its own, and it is
  wrong too. It owns **block 0**, and the three faulty sequences wrote their
  late tokens into it.
- 45 and 17 are exact.

**The fingerprint.** Faults at the multiples of 16, and one innocent victim
that owns block 0. **The guard.** Assert `position // 16 < len(table)` before
each write. Pad the tables with -1, never with a real block id.
''',
    'd3': '''
### What happens

- **The lookup.** Blocks 0 to 2 miss, and blocks 3 to 10 **hit**. A hit after
  a miss is impossible for a correct prefix cache.
- **The K of the hit blocks.** In layer 0 it differs by **0.0%**. Layer 0 sees
  only the token and its position, and those are the same. In layer 14 it
  differs by 32.8%, and in layer 27 by 11.4%. Every layer after the first
  mixes in the prefix, and the prefix was LegalBot.
- **The KL.** 0.645 nats at the first token.
- **The text.** It differs at token 2. The correct answer starts `"Hello!
  I'm ChefBot, your friendly cooking assistant."`. The hashed one starts
  `"Hello! Making a quick tomato soup is ..."`. ChefBot forgot its own name.
  The recipe is still a recipe.

**The surprise.** The damage is quiet. With 2 shared blocks instead of 8, my
first version of this exercise gave the same 40 tokens, and a KL of only
0.03 nats. A test that reads the text passes. The ticket's customer saw legal
language, and that needs a larger share of wrong blocks. A small share gives
a quietly different answer. That is worse, because nobody reports it.

**The fingerprint.** A hit after a miss. The first layer is identical, and
the deep layers differ. **The guard.** Assert that the hits of a lookup form a
prefix. Compare the logits of a cached request with the same request on an
empty cache, and alert on the KL.
''',
    'd4': '''
### What happens

- Session id in the first line: **0.0%** hit rate, for requests of 1,232
  tokens that share about 1,200 tokens of text.
- Session id at the end: **97.1%**.

With a chained hash, block 0 decides every later hash. A new id in block 0
makes every request unique from its first block. The same 1,200 tokens are
computed 500 times.

**The fingerprint.** Exactly zero, and no change when the cache grows.
**The guard.** Predict the hit rate from the template, and alert when the
measurement is far below it.
''',
    'd5': '''
### What happens

On my card (RTX 4080 Laptop GPU, 294 GB/s in `./vc info`):

| kernel | ms | GB/s |
|---|---|---|
| A, one thread for each row | 1.29 | 209 |
| B, 16 threads for each row | 0.85 | 317 |

Both give the same answer. A reaches 66% of B. That is less bad than the 6.25%
sector efficiency of Ticket 5 suggests. The L1 cache saves A: a thread reads
2 bytes of a sector, and its next 15 loads find the rest of the sector in L1.
The cache hides part of the cost, and the loads still wait for it.

B reads at 317 GB/s, more than the 294 of `./vc info`. The streaming copy of
`./vc info` reads and writes. B only reads, and a pure read is a little
faster.

**The fingerprint.** A kernel below the bandwidth of the card, with a high L1
hit rate. The cache hit rate is high **because** the pattern is bad.
''',
    'd6': '''
### What happens

The card has 58 SMs.

| blocks | ms | GB/s |
|---|---|---|
| 16 | 2.02 | 133 |
| 58 | 0.93 | 289 |
| 232 | 0.85 | 314 |
| 65,536 | 0.85 | 318 |

With 16 blocks, 42 of the 58 SMs have nothing to do, and the kernel reaches
42% of its best. The same code reaches its best at 4 blocks for each SM. The
work is the same. Only the shape of the grid changed.

The ratio is better than 16 / 58 = 28%, because one SM can pull more than its
share of the bandwidth. But it cannot pull all of it.

**The fingerprint.** A good kernel that is slow only at a small batch.
**The guard.** Count the blocks of the grid against the SMs, at batch 1.
''',
    'd7': '''
### What happens

| block size | sequences that fit | slots that hold a token |
|---|---|---|
| 16 | 1,005 | 98.2% |
| 32 | 988 | 96.2% |
| 256 | 766 | 74.6% |

Blocks of 256 hold 24% fewer sequences than blocks of 16. The mean length is
about 400, and the formula `(256 - 1) / 2 = 127.5` empty slots for each
sequence predicts 400 / 527.5 = 75.8% of the slots in use. The simulation
gives 74.6%.

**The fingerprint.** Fewer sequences in the same memory, and the waste that
the formula predicts.
''',
    'd8': '''
### What happens

| prompt | errors of the 4 samples |
|---|---|
| 37 tokens | 0.425, 0.380, 0.672, **0.000** |
| 48 tokens | 0.000, 0.000, 0.000, 0.000 |

- With 37 tokens, the last block of the prompt holds 5 tokens and is shared
  by 4 samples. Each sample writes its new tokens into that block, into the
  same slots. The last writer, sample 4, wins. Samples 1 to 3 then attend to
  the tokens of sample 4.
- With 48 tokens, the last block is full. Each sample gets a new block of its
  own, and nothing shared is written. All four are exact.

**The fingerprint.** The last sample is correct, the others are not, and the
bug disappears at the multiples of 16. **The guard.** Assert that the
refcount is 1 before each write.
''',
}

MYSTERY_SOL = '''
### The three mysteries

**`PoolC` fails at once: `append` does not remove the block that it takes.**
It reads the next free block, puts it in the table, and removes it from the
free list only at the next `allocate`. Two appends before an allocate get the
**same block**. A minimal experiment shows it: two sequences of 16 tokens,
then one append each. Both tables end in block 2, with a refcount of 1, and
block 2 is still on the free list. In an engine, two sequences write their
tokens into one block: Ticket 2 of Part 1, from the allocator.

**`PoolA` frees every block of a sequence, whatever its refcount.** Without
forks, a refcount is always 1, so it is correct. The experiment:
`lab.workload(PoolA(4000), forks=True)`. On my run it gave 13,145 violations:
blocks that are in a table **and** on the free list. When a child ends, the
parent loses its shared blocks, and the next allocation gives them to a
stranger.

**`PoolB` never grows a table beyond 32 blocks in `append`.** Sequences below
512 tokens never see it. The experiment: longer sequences,
`lab.workload(PoolB(8000), lengths=(400, 700))`. On my run: `seq 0: 623
tokens in 38 blocks`. The table stopped growing, and every later token has no
block. The limit came from the width of a kernel's table, and it silently
became a limit on the context.

### The method

| Mystery | The variable to change |
|---|---|
| A | forks: sharing makes the refcount matter |
| B | the length: past the limit that nobody wrote down |
| C | the order of the calls: two appends before an allocate |

The invariants of `lab.check` found all three faults. The work was to find a
workload that breaks them.
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| abort forgets to free | no, until the pool is empty | every day | a loss equal to the blocks of the aborts | used + free = total at idle |
| the block that comes one token late | no | lengths that are multiples of 16 | wrong attention, and a victim that owns block 0 | position // 16 < len(table); pad with -1 |
| a hash with no parent | no | a shared middle text | a hit after a miss; a quietly different answer | hits must form a prefix; KL against an empty cache |
| a session id in the first block | no | always | a hit rate of exactly 0 | the rate that the template predicts |
| one thread for each row | no | always | 66% of a coalesced read here | GB/s against the card |
| too few blocks for the machine | no | small grids, batch 1 | 42% of the bandwidth with 16 blocks | blocks against SMs |
| blocks of 256 tokens | no | always | 24% fewer sequences | the fraction of slots in use |
| no copy-on-write | no | prompts that end inside a block | the last writer is right, the others are not | refcount 1 before a write |

No fault in this table crashed. The two faults with a victim, block 0 and the
shared prompt block, damaged a sequence that had done nothing wrong. When an
innocent request breaks, look for a block that it shares with someone.
'''
