"""Part 8, break it on purpose: the whole engine against the roof."""

PART = 8
NAME = 'The Capstone'
FOLDER = 'course/Part8_TheCapstone/7_incidents'
IMPORTS = '''import copy, random
import cudalib
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache'''

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into a working benchmark, step loop, pool, staging
buffer, kernel or budget, and you watch what it does.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** Which step, which load, which context?
- **What?** What does the wrong result look like: a number above the roof, an
  idle GPU, a lost block, a wrong value?
- **Which guard?** Which check would catch it?

`lab.floor_ms` is the floor of one step from stage 28. The card numbers come
from `./vc info`: replace them with yours.

This notebook needs a GPU with about 9 GB free.
'''

LOAD = '''
### run this cell

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'

BANDWIDTH, FLOPS = 294e9, 49e12          # ./vc info: the streaming bandwidth and the sustained bf16 rate
WEIGHTS = sum(p.numel() * p.element_size() for p in model.parameters())
PARAMS = sum(p.numel() for p in model.parameters())
KV = 2 * 28 * 8 * 128 * 2
def floor(computed, context):
  return lab.floor_ms(computed, context, WEIGHTS, PARAMS, KV, BANDWIDTH, FLOPS)
print(f'weights {WEIGHTS / 1e9:.2f} GB, one decode step at batch 1 >= {floor(1, 0):.1f} ms')
'''

DRILLS = [
    ('d1', '''
# Exercise 1: a floor that counts the prefix cache hits

16 requests share a system prompt of about 1,500 tokens, and each adds a
question of about 20 tokens. Prefill them two ways, and time each: with no
cache, and with the system prompt computed once and reused (a prefix cache).
Then compute the ratio floor / time two ways: with the floor of every prompt
token of every request, as the benchmark of Ticket 1 does, and with the floor
of the tokens that the engine really computed.

Predict the four ratios.
''', '''
SYSTEM = ('You are the support assistant of a bank. Answer politely, in short paragraphs, and never share '
          'account data. ' * 60)
questions = [f'Question {i}: how do I change the address on my account number {1000 + i}?' for i in range(16)]
system_ids = tokenizer(SYSTEM, return_tensors='pt').input_ids.to(device)
question_ids = [tokenizer(q, return_tensors='pt').input_ids.to(device) for q in questions]
S = system_ids.shape[1]
print(f'system prompt {S} tokens, questions {[q.shape[1] for q in question_ids][:4]}... tokens')

def run(prefix_cache):
  with torch.inference_mode():
    shared = DynamicCache()
    if prefix_cache:
      model(system_ids, past_key_values=shared, use_cache=True, logits_to_keep=1)
    for q in question_ids:
      if prefix_cache:
        cache = copy.deepcopy(shared)
        model(q, past_key_values=cache, use_cache=True, logits_to_keep=1)
      else:
        model(torch.cat([system_ids, q], dim=1), use_cache=True, logits_to_keep=1)

run(True)                                                              # warm up
for prefix_cache in (False, True):
  torch.cuda.synchronize()
  start = time.perf_counter()
  run(prefix_cache)
  torch.cuda.synchronize()
  ms = (time.perf_counter() - start) * 1000
  every_token = sum(floor(S + q.shape[1], 0) for q in question_ids)                  # THE FAULT
  computed = ((floor(S, 0) if prefix_cache else 0)
              + sum(floor(q.shape[1], S) if prefix_cache else floor(S + q.shape[1], 0) for q in question_ids))
  print(f'prefix cache {str(prefix_cache):5s}: {ms:6.0f} ms   floor of every prompt token / time = '
        f'{every_token / ms:5.0%}   floor of the computed tokens / time = {computed / ms:5.0%}')
'''),

    ('d2', '''
# Exercise 2: the bytes of a long-context step

Batch 8, and a static cache that holds 256 tokens, then 4,096 tokens of
context for each sequence. Time one decode step at each context. The step at
256 is almost all weights. The difference is the KV cache. Then predict what
int8 weights (half the weight bytes) and an FP8 KV cache (half the KV bytes)
would each give at 4,096.

This is Ticket 2. Predict both speedups at 4,096.
''', '''
@torch.inference_mode()
def step_ms(batch, context):
  cache = StaticCache(config=model.config, max_cache_len=context + 64)
  model(torch.randint(0, 1000, (batch, context), device=device), past_key_values=cache,
        use_cache=True, logits_to_keep=1)
  token = torch.randint(0, 1000, (batch, 1), device=device)
  ms = lab.ms_per_call(lambda: model(token, past_key_values=cache, use_cache=True), iters=20)
  del cache
  torch.cuda.empty_cache()
  return ms

short, long = step_ms(8, 256), step_ms(8, 4096)
weight_part, kv_part = short, long - short
print(f'batch 8: {short:.1f} ms at 256 tokens, {long:.1f} ms at 4,096 tokens')
print(f'at 4,096: the weights about {weight_part:.1f} ms ({weight_part / long:.0%}), the KV about {kv_part:.1f} ms ({kv_part / long:.0%})')
print(f'int8 weights -> {long / (weight_part / 2 + kv_part):.2f}x    FP8 KV cache -> '
      f'{long / (weight_part + kv_part / 2):.2f}x    both -> {long / (weight_part / 2 + kv_part / 2):.2f}x')
'''),

    ('d3', '''
# Exercise 3: CPU work between the steps

A greedy loop at batch 1, 60 tokens. After each step the engine does 4 ms of
CPU work: the scheduler, the detokenizer. Time three loops: no CPU work, the
CPU work after each step (in series, as in Ticket 3), and the CPU work for
token n while the GPU computes token n + 1.

Predict the ms of each step.
''', '''
def cpu_work(ms=4.0):                                                  # busy, like Python work
  end = time.perf_counter() + ms / 1000
  while time.perf_counter() < end:
    pass

@torch.inference_mode()
def loop(mode, tokens=60):
  ids = tokenizer('The three largest cities in Japan are', return_tensors='pt').input_ids.to(device)
  out = model(ids, use_cache=True)
  cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
  host = torch.empty(1, 1, dtype=torch.long, pin_memory=True)
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(tokens):
    out = model(token, past_key_values=cache, use_cache=True)          # the GPU starts at once
    token = out.logits[:, -1:].argmax(-1)
    if mode == 'series':
      int(token)                                                       # wait for the GPU ...
      cpu_work()                                                       # THE FAULT: ... then work
    elif mode == 'overlap':
      cpu_work()                                                       # work while the GPU runs
      host.copy_(token, non_blocking=True)
  torch.cuda.synchronize()
  return (time.perf_counter() - start) / tokens * 1000

for mode in ('none', 'series', 'overlap'):
  print(f'CPU work {mode:8s}: {loop(mode):5.1f} ms per step')
'''),

    ('d4', '''
# Exercise 4: preemption keeps the last block

A pool of 8,000 blocks. 300 sequences grow to random lengths, and 1,250 times
one of them is preempted and later resumed. The preemption frees every block
except the last one, as the code of Ticket 4 does. At the end every sequence
finishes.

Predict the free blocks at the end.
''', '''
pool = lab.Pool(8000)
rng = random.Random(0)
lengths = {seq: rng.randint(20, 300) for seq in range(300)}
for seq, n in lengths.items():
  pool.grow(seq, n)

def preempt(seq):
  table = pool.tables.pop(seq)
  pool.free_list += table[:-1]                                         # THE FAULT: the last block stays

for _ in range(1250):
  seq = rng.randrange(300)
  preempt(seq)
  pool.grow(seq, lengths[seq])                                         # resume: allocate again
for seq in list(pool.tables):
  pool.release(seq)
print(f'free at the end: {pool.num_free()} of {pool.total}; lost: {pool.total - pool.num_free()}')
'''),

    ('d5', '''
# Exercise 5: one pinned staging buffer, reused at once

The engine copies the block table of each step from a pinned CPU buffer to
the GPU with `non_blocking=True`, and then at once writes the table of the
next step into the same buffer. Make the GPU busy first, as a heavy step
does. Then check what the GPU received. Then do the same with an event that
the CPU waits on before it reuses the buffer.

This is Ticket 5. Predict what the GPU received in each case.
''', '''
busy = torch.randn(4096, 4096, device=device)
host = torch.empty(8, dtype=torch.long, pin_memory=True)
table = torch.empty(8, dtype=torch.long, device=device)
done = torch.cuda.Event()

for guard in (False, True):
  for _ in range(20):
    busy @ busy                                                        # the GPU is behind the CPU
  host[:] = torch.arange(8)                                            # the table of step n
  table.copy_(host, non_blocking=True)
  done.record()
  if guard:
    done.synchronize()                                                 # wait until the copy has read the buffer
  host[:] = 100 + torch.arange(8)                                      # THE FAULT: step n + 1 writes at once
  torch.cuda.synchronize()
  print(f'{"with the event" if guard else "no guard      "}: the GPU received {table.tolist()} (step n sent 0 to 7)')
'''),

    ('d6', '''
# Exercise 6: a shared tile of 128 floats per row

A kernel keeps a tile of 32 x 128 floats in shared memory, and each thread of
a warp reads one **column**, many times. Then the same kernel with rows of 129
floats. The first run compiles the kernel.

This is Ticket 6. Predict the ratio of the two times.
''', '''
KERNEL = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <int WIDTH>
__global__ void column_reads(float* out, int repeats) {
  __shared__ float tile[32][WIDTH];
  int lane = threadIdx.x;
  for (int c = 0; c < 128; ++c) tile[lane][c] = lane * 0.5f + c;
  __syncwarp();
  float sum = 0.f;
  for (int r = 0; r < repeats; ++r)
    for (int c = 0; c < 128; ++c) sum += tile[lane][(c + r) & 127];     // 32 lanes, 32 rows, one column
  out[blockIdx.x * 32 + lane] = sum;
}

void run(torch::Tensor out, int64_t width, int64_t repeats) {
  int blocks = out.numel() / 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (width == 128) column_reads<128><<<blocks, 32, 0, stream>>>(out.data_ptr<float>(), repeats);
  else column_reads<129><<<blocks, 32, 0, stream>>>(out.data_ptr<float>(), repeats);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""
kernel = cudalib.build_source('inc8_banks', KERNEL)
out = torch.empty(4096 * 32, device=device)
times = {}
for width in (128, 129):                                               # 128: THE FAULT
  times[width] = lab.ms_per_call(lambda: kernel.run(out, width, 64), iters=10)
  print(f'rows of {width} floats: {times[width]:6.2f} ms, the bank of column 0 for the 32 lanes: '
        f'{len({(lane * width) % 32 for lane in range(32)})} different bank(s)')
print(f'ratio: {times[128] / times[129]:.1f}x')
'''),

    ('d7', '''
# Exercise 7: more running sequences than the token budget

A simulated scheduler: `max_num_seqs` is 256, the token budget is 192, and
decodes go first. At the peak 230 sequences are running, and a new request
arrives every step. Run 200 steps, then do the same with a budget of 768.

This is Ticket 7. Predict the tokens that each running sequence gets, and
what happens to the new requests.
''', '''
def simulate(budget, max_seqs=256, steps=200, seed=0):
  rng = random.Random(seed)
  running = [rng.randint(1, 400) for _ in range(230)]                   # the tokens that each still decodes
  waiting, served, got, slots = [], 0, 0, 0
  for step in range(steps):
    waiting.append(300)                                                 # a new prompt of 300 tokens
    rng.shuffle(running)
    decodes = min(len(running), budget)                                 # decodes first
    got, slots = got + decodes, slots + len(running)
    running = [left - 1 if i < decodes else left for i, left in enumerate(running)]
    running = [left for left in running if left > 0]
    room = budget - decodes                                             # then the prefills
    while waiting and room > 0 and len(running) < max_seqs:
      take = min(waiting[0], room)
      waiting[0] -= take
      room -= take
      if waiting[0] == 0:
        waiting.pop(0)
        running.append(200)
        served += 1
  return got / slots, served, len(waiting)

for budget in (192, 768):                                              # 192: THE FAULT
  rate, served, waiting = simulate(budget)
  print(f'budget {budget}: a running sequence got a token in {rate:.0%} of the steps; '
        f'{served} new requests got a first token, {waiting} still wait')
'''),
]

MYSTERY = '''
# Exercise 8: three mystery benchmarks

The module `mystery.py` holds three functions: `ratio_a`, `ratio_b` and
`ratio_c`. Each takes a step log and the card, and returns the ratio floor /
time of stage 28. A step in the log is a dict: `computed` (tokens run through
the model), `cached` (prompt tokens that the prefix cache served), `context`
(tokens of KV that the step read) and `ms`. Each one has one fault. **Do not
open the file.**

`floor(computed, context)` from the load cell is the correct floor of one
step, so you can compute the true ratio of any log you build.

The cell below runs them on a simple log: identical decode steps, short
context, no prefix cache. All three agree within 1%.

For each function:

1. Build a step log that reaches the fault, and compute the true ratio.
2. Write your diagnosis: the fault, and the log that proved it.
3. Only then, open `mystery.py` and check.

A hint about the method: one fault needs the prefix cache. One fault needs a
long context. One fault needs steps of very different sizes.
'''

MYSTERY_CODE = '''
from mystery import ratio_a, ratio_b, ratio_c

card = dict(weight_bytes=WEIGHTS, params=PARAMS, kv_per_token=KV, bandwidth=BANDWIDTH, flops=FLOPS)
simple = [dict(computed=8, cached=0, context=8 * 16, ms=13.0) for _ in range(100)]
true = sum(floor(s['computed'], s['context']) for s in simple) / sum(s['ms'] for s in simple)
print(f'true {true:.3f}   a {ratio_a(simple, **card):.3f}   b {ratio_b(simple, **card):.3f}   c {ratio_c(simple, **card):.3f}')
'''

MYSTERY_NAMES = ['ratio_a', 'ratio_b', 'ratio_c']

FINGERPRINT_ROWS = ['a floor that counts the prefix cache hits', 'int8 weights at long context',
                    'CPU work in series with the GPU', 'preemption keeps the last block',
                    'one pinned staging buffer, reused at once', 'a stride of 128 floats in shared memory',
                    'more running sequences than the budget']

SOLUTIONS = {
    'd1': '''
### What happens

| | time | floor of every prompt token / time | floor of the computed tokens / time |
|---|---|---|---|
| no prefix cache | 1,852 ms | 81% | 81% |
| prefix cache | 432 ms | **349%** | 67% |

Without the cache the two floors are the same, because the engine computes
every prompt token. With the cache, the engine computes the system prompt
once, and the benchmark still charges for it 16 times. The result is 349% of
the roof: impossible, so the measurement is broken.

The honest ratio **falls** to 67% with the cache. The engine does 4.3x less
work in total, but each step now has only 20 tokens, and a step of 20 tokens
reads all the weights for very little compute. The cache is a large win in
time, and a smaller efficiency of the steps that remain. Both are true.
''',
    'd2': '''
### What happens

On my card, batch 8:

- 17.4 ms at 256 tokens of context, 69.5 ms at 4,096.
- At 4,096, the weights are about 17.4 ms (25%) and the KV cache about 52.0 ms
  (75%).
- int8 weights would give **1.14x**. An FP8 KV cache would give **1.60x**.
  Both would give 2.00x.

This is Ticket 2 on a smaller card: at long context, the KV cache is most of
the bytes, and the optimization of the weights targets the smaller term.
''',
    'd3': '''
### What happens

| CPU work of 4 ms | ms per step |
|---|---|
| none | 13.7 |
| in series | 20.3 |
| overlapped | 14.9 |

In series, each step waits for the GPU (`int(token)`), and then the GPU waits
for the CPU. The step grows by the whole CPU work, plus the cost of the
synchronization. Overlapped, the CPU works while the GPU computes the next
step, and the GPU never waits: the cost falls from 6.6 ms to 1.2 ms. This is
the gap of the Nsight Systems timeline of Ticket 3, and the reason that vLLM V1
schedules step n + 1 while step n runs.

The overlap works only because the next input stays on the GPU. The loop
never reads the token on the CPU before it launches the next step.
''',
    'd4': '''
### What happens

- Free at the end: **6,750** of 8,000. Lost: **1,250**, exactly the number of
  preemptions.

Each preemption loses the last block of the sequence. The resume allocates a
new table, and nobody frees the old block. A leak of one block for each
event is the fingerprint: count the event, and compare.
''',
    'd5': '''
### What happens

- No guard: the GPU received `[100, ..., 107]`, the table of the **next**
  step, not `[0, ..., 7]`.
- With the event: `[0, ..., 7]`, correct.

The copy with `non_blocking=True` from pinned memory returns at once. It runs
when the GPU reaches it in the stream, after the 20 matmuls. By then the CPU
has written step n + 1 into the same buffer. In an engine, the attention of
step n reads the block table of step n + 1: wrong tokens, only under load,
and never with `CUDA_LAUNCH_BLOCKING=1`, which makes every launch wait.
''',
    'd6': '''
### What happens

- Rows of 128 floats: **8.29 ms**. The 32 lanes read column 0 from 1 bank.
- Rows of 129 floats: **0.53 ms**. The 32 lanes read from 32 banks.
- The ratio is **15.6x**.

With a row of 128 floats, `(lane x 128 + column) mod 32` is the same for every
lane, so the 32 reads of a warp go through one bank, one after the other. One
extra float for each row moves each lane to its own bank. The arithmetic and
the result do not change.
''',
    'd7': '''
### What happens

| budget | a running sequence got a token in | new requests with a first token | still waiting |
|---|---|---|---|
| 192 | 96% of the steps | 16 | 184 |
| 768 | 100% of the steps | 146 | 54 |

With a budget of 192 and 230 running sequences, the decodes take the whole
budget. The running sequences stutter (4% of their steps are skipped), and
the new requests starve: 16 in 200 steps, and only when enough old sequences
finished. The GPU is not the limit. The configuration is. With 768 the
running set fills to `max_num_seqs`, and then the queue waits for a free
slot, which is the correct kind of wait.
''',
}

MYSTERY_SOL = '''
### The three mysteries

On the simple log the true ratio is 0.904, and all three functions give 0.900
to 0.904. The logs that break them:

| log | true | the function |
|---|---|---|
| 16 prefill steps of 20 tokens, 1,320 of them from the prefix cache | 0.452 | `ratio_a`: **3.503** |
| 100 decode steps of batch 8 at 4,096 tokens of context, 69 ms each | 0.355 | `ratio_b`: 0.170 |
| 99 decode steps of 13 ms, and one prefill of 4,096 tokens in 600 ms | 0.769 | `ratio_c`: 0.900 |

**`ratio_a` counts the cached tokens as computed.** It is Exercise 1: with
prefix hits, the ratio passes the roof. A ratio above 1 is the proof.

**`ratio_b` has no KV term.** It is correct at short context. At long
context the KV bytes are most of the floor, and a floor without them is half
the real one, so the engine looks two times worse than it is. This fault is
pessimistic, and it sends the team to optimize a problem that does not exist.

**`ratio_c` averages the ratio of each step.** 99 small steps at 90% and one
large prefill at 48% average to 90%. But the large step is 32% of the time.
The right ratio is the sum of the floors over the sum of the times: 77%. An
average of ratios gives each step the same weight, and a mix of small and
large steps then hides the expensive ones.

### The method

A benchmark is code. Build the step log for which you know the answer: with
the cache, with long context, with steps of different sizes. The simple log
that the benchmark came with was the one case where the three faults are
invisible.
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| a floor that counts the prefix cache hits | no | with prefix hits | 349% of the roof | the ratio must stay below 100% |
| int8 weights at long context | no | long context | 1.14x where 2x was hoped | the byte split of the real step |
| CPU work in series with the GPU | no | always | 20.3 ms against 13.7 | the GPU idle time between steps |
| preemption keeps the last block | no | after preemptions | a loss equal to the preemptions | used + free = total at idle |
| one pinned staging buffer, reused at once | no | under load only | the next step's table | an event before each reuse |
| a stride of 128 floats in shared memory | no | always | 15.6x slower | the bank-conflict counter |
| more running sequences than the budget | no | at the peak | starved new requests | budget > max_num_seqs, asserted at startup |

The whole course in one table: every fault gives a number, and no fault
crashes. The engine runs, the text flows, and only a measurement that you
trust finds the fault.
'''
