"""Part 4, break it on purpose: the scheduler."""

PART = 4
NAME = 'The Scheduler'
FOLDER = 'course/Part4_TheScheduler/3_incidents'
IMPORTS = '''import random, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache'''

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into a working scheduler, a working prefill or a
working copy, and you watch what it does.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** Which request, which step, which load?
- **What?** What does the wrong result look like: a stall, a storm, a
  starved request, wrong text?
- **Which guard?** Which check would catch it?

`lab.Scheduler` is a small scheduler with one rule: decodes first, then
prefill chunks, then admission, all inside a token budget. It preempts by
recompute. `lab.run` drives it one step at a time, with a step time of about
13 ms plus 0.07 ms for each token (a model of Qwen3-1.7B on a laptop GPU).
Three exercises time the real model instead.

This notebook needs a GPU with about 8 GB free.
'''

LOAD = '''
### run this cell

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'

def summary(name, requests, log, scheduler):
  computed = sum(tokens for _, tokens, _ in log)
  useful = sum(r.prompt + r.output - 1 for r in requests)
  ttft = sorted(r.first_token - r.arrival for r in requests)
  gaps = [g for r in requests for g in r.gaps]
  print(f'{name}: done at {log[-1][0]:5.0f} s, TTFT median {ttft[len(ttft) // 2]:5.1f} s, '
        f'preempted {sum(r.preemptions > 0 for r in requests) / len(requests):4.0%} of the requests '
        f'(one of them {max(r.preemptions for r in requests)} times), '
        f'recomputed {1 - useful / computed:4.0%} of the work, longest gap {max(gaps):5.1f} s')
'''

DRILLS = [
    ('d1', '''
# Exercise 1: admit on the prompt

A pool of 12,000 blocks. 190 requests arrive at once, each with a prompt of
1,000 tokens and an answer of 1,500. Run two admission rules: `'prompt'`
admits when the pool has room for the prompt, and `'final'` admits only when
the pool has room for the final length of every request that runs.

This is Ticket 1 of the incident file. Predict the preemptions, the wasted
work, and the median time to the first token of each rule.
''', '''
for admission in ('prompt', 'final'):                                   # 'prompt': THE FAULT
  requests = [lab.Request(i, 0.0, 1000, 1500) for i in range(190)]
  scheduler = lab.Scheduler(12_000, budget=2048, admission=admission)
  requests, log = lab.run(scheduler, requests)
  summary(f'admit on {admission:6s}', requests, log, scheduler)
print('the pool holds', 12_000 // math.ceil(2500 / 16), 'requests of full length')
'''),

    ('d2', '''
# Exercise 2: a recompute in one piece

A request of 12,000 tokens was preempted, and it resumes by recompute. Time
the real prefill of 12,000 tokens in one pass, as the resume of Ticket 2 does.
Then time the same prefill in chunks of 512.

Predict the freeze of every other stream in each case.
''', '''
whole = lab.prefill_ms(model, 12_000)                                   # THE FAULT: one pass
chunks = lab.prefill_ms(model, 12_000, chunk=512)
print(f'one pass: {whole:6.0f} ms   -> every stream waits this long')
print(f'chunks of 512: {len(chunks)} steps, the longest {max(chunks):4.0f} ms, the total {sum(chunks):6.0f} ms')
flop = 2 * 1.72e9 * 12_000
print(f'the matmul FLOP alone: {flop / 1e12:.1f} TFLOP -> {flop / 49e12 * 1000:.0f} ms at 49 TFLOP/s')
kv = 12_000 * 114_688
print(f'a swap of the same request: {kv / 1e9:.2f} GB')
'''),

    ('d3', '''
# Exercise 3: the percentile that sees nothing

64 streams run for 30 minutes. Each step takes 22 ms. Every 3 minutes a long
document arrives, and its prefill freezes every stream for the time that you
measured for one pass in Exercise 2 (the prefill is not chunked).

This is Ticket 3. Predict the p50, the p99 and the p99.9 of the time between
tokens, and the fraction of the streams that saw a gap above 1 s.
''', '''
STEP, FREEZE = 0.022, whole / 1000
gaps, worst = [], []
for stream in range(64):
  clock, next_document, stream_gaps = random.Random(stream).uniform(0, 0.022), 180.0, []
  while clock < 30 * 60:
    gap = STEP
    if clock >= next_document:
      gap += FREEZE                                                     # THE FAULT: one-pass prefill
      next_document += 180.0
    stream_gaps.append(gap)
    clock += gap
  gaps += stream_gaps
  worst.append(max(stream_gaps))
gaps.sort()
at = lambda q: gaps[int(q * (len(gaps) - 1))] * 1000
print(f'{len(gaps):,} gaps: p50 {at(0.5):.0f} ms, p99 {at(0.99):.0f} ms, p99.9 {at(0.999):.0f} ms, max {gaps[-1] * 1000:.0f} ms')
print(f'the frozen gaps are {sum(g > 1 for g in gaps) / len(gaps):.3%} of all gaps')
print(f'streams with a gap above 1 s: {sum(w > 1 for w in worst)} of 64')
'''),

    ('d4', '''
# Exercise 4: a budget of 128 tokens

The token budget decides how many prefill tokens of a long document share a
step with 64 decodes. Time the real chunks of a document at a context of 8,000
tokens, for several chunk sizes. Then estimate, for each budget, the time
between tokens of the chat users (one step) and the time to the first token of
a document of 16,000 tokens.

This is Ticket 4. Predict the time to the first token with a budget of 128.
''', '''
DECODES = 64
rows = []
for budget in (128, 256, 512, 1024, 2048):                              # 128: THE FAULT
  chunk = budget - DECODES
  chunk_ms = statistics.median(lab.prefill_ms(model, chunk * 4, chunk=chunk, context=8000))
  step = chunk_ms + DECODES * 0.07                                      # the decodes ride along
  steps = math.ceil(16_000 / chunk)
  rows.append((budget, chunk, step, steps))
  print(f'budget {budget:5d}: {chunk:5d} prefill tokens per step, one step {step:6.1f} ms, '
        f'document TTFT {steps} steps x {step:.0f} ms = {steps * step / 1000:5.1f} s')
'''),

    ('d5', '''
# Exercise 5: where the preempted request waits

A pool of 3,000 blocks under a steady overload: a request every 0.3 s for 10
minutes, with prompts of 300 to 2,000 tokens and answers of 200 to 800. The
scheduler preempts the newest request. Run three policies:

- admission on the prompt, and the victim goes to the **back** of the queue
  (Ticket 5);
- admission on the prompt, and the victim goes to the **front** (the fix that
  the engineer of Ticket 5 proposes);
- admission on the final length, and the victim goes to the front.

Predict the most preemptions of one request, and the slowest request, for
each policy.
''', '''
rng = random.Random(3)
def workload():
  rng.seed(3)
  return [lab.Request(i, i * 0.3, rng.randint(300, 2000), rng.randint(200, 800)) for i in range(2000)]
for admission, front in [('prompt', False), ('prompt', True), ('final', True)]:
  requests = workload()
  scheduler = lab.Scheduler(3000, budget=1024, admission=admission, preempted_to_front=front)
  requests, log = lab.run(scheduler, requests)
  slowest = max(requests, key=lambda r: r.finish - r.arrival)
  print(f'admit on {admission:6s}, victim to the {"front" if front else "back ":5s}: '
        f'{scheduler.preemptions:5d} preemptions, the most for one request {max(r.preemptions for r in requests):3d}, '
        f'the slowest request {slowest.finish - slowest.arrival:5.0f} s, all done at {log[-1][0]:5.0f} s')
'''),

    ('d6', '''
# Exercise 6: every chunk starts at position 0

A chunked prefill in chunks of 64 tokens. The fault: each chunk gets the
positions 0 to 63, not the positions after the chunks before it. The prompt
puts an instruction at its start, then about 150 tokens of text, then a
question.

This is Ticket 6. Predict what the model answers, and whether a prompt of 50
tokens is affected.
''', '''
stop = set(model.generation_config.eos_token_id)
def chat(text):
  text = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
  return tokenizer(text).input_ids
long_prompt = chat('Reply in French only. ' + 'The museum opens at nine, the cafe at ten, and the garden '
                   'closes at dusk in winter. ' * 7 + 'When does the cafe open?')
short_prompt = chat('Reply in French only. When does the cafe open if it opens at ten?')

@torch.inference_mode()
def chunked_greedy(ids, chunk, reset_positions, max_tokens=30):
  cache = DynamicCache()
  for start in range(0, len(ids) - 1, chunk):
    part = torch.tensor([ids[start:min(start + chunk, len(ids) - 1)]], device=device)
    first = 0 if reset_positions else start                             # THE FAULT when True
    positions = torch.arange(first, first + part.shape[1], device=device)[None]
    model(part, past_key_values=cache, position_ids=positions, use_cache=True, logits_to_keep=1)
  token, out = ids[-1], []
  for step in range(max_tokens):
    position = torch.tensor([[len(ids) - 1 + step]], device=device)
    logits = model(torch.tensor([[token]], device=device), past_key_values=cache,
                   position_ids=position, use_cache=True).logits
    token = int(logits[0, -1].argmax())
    if token in stop:
      break
    out.append(token)
  return out

for name, ids in [('long prompt', long_prompt), ('short prompt', short_prompt)]:
  good = chunked_greedy(ids, 64, reset_positions=False)
  bad = chunked_greedy(ids, 64, reset_positions=True)
  print(f'{name}: {len(ids)} tokens = {math.ceil((len(ids) - 1) / 64)} chunks')
  print('   correct positions:', repr(tokenizer.decode(good)))
  print('   positions reset  :', repr(tokenizer.decode(bad)))
'''),

    ('d7', '''
# Exercise 7: a swap into pageable memory

Copy the KV cache of a request of 12,000 tokens (1.38 GB) from the GPU to the
CPU, into ordinary memory and into pinned memory. Time the copy, and time how
long the call takes to return with `non_blocking=True`.

This is Ticket 8. Predict the GB/s of each copy, and whether the CPU waits.
''', '''
kv = torch.empty(12_000 * 114_688 // 2, dtype=torch.bfloat16, device=device)
for pinned in (False, True):                                           # False: THE FAULT
  host = torch.empty(kv.shape, dtype=kv.dtype, pin_memory=pinned)
  host.copy_(kv)                                                        # touch the pages once
  torch.cuda.synchronize()
  start = time.perf_counter()
  host.copy_(kv, non_blocking=True)
  returned = time.perf_counter() - start
  torch.cuda.synchronize()
  total = time.perf_counter() - start
  print(f'{"pinned  " if pinned else "pageable"}: the call returned after {returned * 1000:6.1f} ms, '
        f'the copy took {total * 1000:6.1f} ms = {kv.numel() * 2 / total / 1e9:5.1f} GB/s')
  del host
del kv
torch.cuda.empty_cache()
'''),
]

MYSTERY = '''
# Exercise 8: three mystery schedulers

The module `mystery.py` holds three schedulers: `SchedulerA`, `SchedulerB` and
`SchedulerC`. They take the same arguments as `lab.Scheduler`. Each one has
one fault. **Do not open the file.**

The cell below runs each one on a light workload and prints the summary. All
three look healthy there.

For each scheduler:

1. Design an experiment that makes the fault visible. Which variable do you
   change? The load? The mix of sizes? The number of decodes? Look at
   `requests` and at the step log too, not only at the summary.
2. Write your diagnosis: the fault, and the experiment that proved it.
3. Only then, open `mystery.py` and check.

A hint about the method: one fault needs preemption. One fault needs a mix of
short and long prompts under load, and it makes the median time to the first
token better. One fault needs many running decodes, and it is only visible in
the step log.
'''

MYSTERY_CODE = '''
from mystery import SchedulerA, SchedulerB, SchedulerC

for name, Scheduler in [('A', SchedulerA), ('B', SchedulerB), ('C', SchedulerC)]:
  requests = [lab.Request(i, i * 0.5, 400, 100) for i in range(100)]
  scheduler = Scheduler(8000, budget=512)
  requests, log = lab.run(scheduler, requests)
  summary(f'Scheduler{name}', requests, log, scheduler)
'''

MYSTERY_NAMES = ['SchedulerA', 'SchedulerB', 'SchedulerC']

FINGERPRINT_ROWS = ['admit on the prompt', 'a recompute in one piece', 'a percentile over all the steps',
                    'a budget of 128 tokens', 'admission that loops preemptions',
                    'every chunk starts at position 0', 'a swap into pageable memory']

SOLUTIONS = {
    'd1': '''
### What happens

| admission | done at | TTFT median | preempted | recomputed | longest gap |
|---|---|---|---|---|---|
| on the prompt | 92 s | 7.5 s | 56% (one of them 8 times) | 35% of the work | 49.2 s |
| on the final length | 92 s | 34.4 s | 0% | 0% | 0.2 s |

The pool holds only 76 requests of full length, and admission on the prompt
lets all 190 in. Then 35% of all the computation is recompute, and one answer
stops for 49 seconds in the middle.

**The surprise.** Both rules finish the batch at the same time, and the
faulty rule has a **better** median time to the first token. It starts
everyone at once, and then it pays with stalls in the middle of the answers.
A dashboard that shows only TTFT and total time says that the faulty rule is
better.

**The fingerprint.** A high preemption rate, and gaps of many seconds inside
answers. **The guard.** Alert on the preemption rate and on the longest gap
of each request.
''',
    'd2': '''
### What happens

On my card:

- One pass of 12,000 tokens: **1,876 ms**. Every other stream waits that long.
- In chunks of 512: 24 steps, the longest 138 ms, the total 2,240 ms.
- The matmul FLOP alone would take 842 ms at 49 TFLOP/s. The rest is the
  attention, which grows with the square of the length, and the efficiency.

Chunking costs 19% more time in total, and it cuts the longest freeze by
14x. The stall of one pass is the fingerprint of Ticket 2: every stream
freezes for the time of one whole prefill.
''',
    'd3': '''
### What happens

- 5.2 million gaps: p50 22 ms, **p99 22 ms, p99.9 22 ms**, max 1,898 ms.
- The frozen gaps are 0.011% of all the gaps.
- **64 of 64** streams saw a gap above 1 s.

Every user sees the freeze, and no percentile over the steps can see it. The
p99.9 ignores the worst 0.1%, and the freezes are ten times rarer than that.

**The guard.** The worst gap **of each request**, and the fraction of
requests with a gap above 1 s.
''',
    'd4': '''
### What happens

| budget | prefill tokens per step | one step | document TTFT |
|---|---|---|---|
| 128 | 64 | 50 ms | 12.4 s |
| 256 | 192 | 62 ms | 5.2 s |
| 512 | 448 | 102 ms | 3.7 s |
| 1,024 | 960 | 229 ms | 3.9 s |
| 2,048 | 1,984 | 478 ms | 4.3 s |

With a budget of 128, the 64 decodes take half of each step, and the document
needs 250 steps: 12.4 s. The steps at the small budgets are also expensive
for the few tokens they carry, because each step reads all the weights.

Above 512 the document gets **no faster**. A larger chunk at a context of
8,000 tokens costs more than linearly, because the attention of each chunk
reads the whole context. The chat users pay for it: 478 ms between their
tokens at 2,048. On this card, 512 is the knee of the dial.
''',
    'd5': '''
### What happens

| policy | preemptions | the most for one request | the slowest request | all done |
|---|---|---|---|---|
| admit on the prompt, victim to the back | 4,848 | 23 | 357 s | 935 s |
| admit on the prompt, victim to the front | 8,440 | **70** | 605 s | 1,199 s |
| admit on the final length | 0 | 0 | 123 s | 720 s |

**The surprise.** The fix that the engineer of Ticket 5 proposes makes the
loop **worse**. At the front of the queue, the preempted request comes back at
once, into a pool that is still too full, and it is again the newest request,
so it is the next victim. One request was preempted 70 times.

The loop comes from the admission. With room for the final length, no request
is preempted, and every number is better. This result changed Ticket 5 of the
incident file.
''',
    'd6': '''
### What happens

- **Long prompt** (163 tokens, 3 chunks). With the correct positions: `'Le
  café ouvre à dix.'`. With the positions reset: `"Okay, let's see. The user
  provided a series of sentences that are a bit confusing and have some typos
  or formatting issues."` The model saw three chunks that all claim to start
  at position 0, as if the text were three texts on top of each other. It
  forgot the instruction to reply in French, and it calls the prompt
  confusing.
- **Short prompt** (28 tokens, 1 chunk): identical. One chunk has correct
  positions.

**The fingerprint.** Only prompts longer than one chunk fail, and the model
behaves as if the prompt were garbled. **The guard.** A chunked prefill
against a prefill in one pass, in fp32.
''',
    'd7': '''
### What happens

On my laptop:

| host memory | the call returned after | the copy took | GB/s |
|---|---|---|---|
| pageable | 134.6 ms | 134.6 ms | 10.2 |
| pinned | 0.0 ms | 105.7 ms | 13.0 |

- With pageable memory, `non_blocking=True` does nothing: the CPU waits for
  the whole copy. With pinned memory, the call returns at once, and the CPU
  can schedule the next step during the copy.
- The bandwidth differs less than in Ticket 8 (1.3x, not 4x). The bus of this
  laptop is the limit, at 13 GB/s. On a server with a PCIe 4.0 x16 slot, the
  pinned copy is faster, and the gap grows.

So on this machine, the main cost of pageable memory is not the bandwidth. It
is the 135 ms that the scheduler thread cannot use.
''',
}

MYSTERY_SOL = '''
### The three mysteries

**`SchedulerA`: the shortest prompt first.** It sorts the queue by the
pending tokens. On a light load it is identical to the lab scheduler. The
experiment: a mix of long and short prompts under load, for example one prompt
of 6,000 tokens in every 20, a request every 50 ms, and a pool of 3,000
blocks. On my run, the median time to the first token **fell** from 13.9 s to
0.4 s, and the long prompts waited 38 s at the median instead of 14 s. A
fault that improves the headline number is the hardest kind to find. Look at
the tail of each class of request, not at the median of all.

**`SchedulerB`: decodes do not count against the budget.** On a light load,
the steps are small anyway. The experiment: many running decodes and a
prefill at the same time, then read the step log. On my run, with 400 requests
and a budget of 256, one step carried 448 tokens. The budget is the promise
about the time between tokens, and a scheduler that breaks it breaks that
promise with no error.

**`SchedulerC`: preemption does not reset the computed tokens.** It frees the
blocks and keeps the count, so on resume the request skips its recompute, and
it gets new blocks full of garbage. The experiment: force preemption, then
count the work. On my run, with the workload of Exercise 1: 106 preemptions,
and the total computed tokens were **exactly** the minimum, 474,810. A
preemption by recompute that costs nothing is impossible. In a real engine
this request would attend to empty blocks and write nonsense.

### The method

| Mystery | The experiment |
|---|---|
| A | a mix of sizes under load: compare the tail of each class |
| B | many decodes: check the invariant `tokens per step <= budget` in the log |
| C | force preemption: compare the work with the minimum |

C is Part 1, Ticket 4, again: a result better than the physics allows is a
broken measurement or a broken engine.
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| admit on the prompt | no | under load | preemptions, 35% recompute, gaps of 49 s | preemption rate; worst gap of each request |
| a recompute in one piece | no | when a long request resumes | every stream freezes for one prefill | prefill tokens per step <= budget |
| a percentile over all the steps | no | always | p99.9 = 22 ms, and 64 of 64 streams froze | the worst gap of each request |
| a budget of 128 tokens | no | long prompts | TTFT 12.4 s, 3x the knee | TTFT per prompt length, and the step time |
| admission that loops preemptions | no | under load | one request preempted 23 to 70 times | preemptions per request |
| every chunk starts at position 0 | no | prompts longer than one chunk | the prompt reads as garbled | chunked against one pass, in fp32 |
| a swap into pageable memory | no | every swap | the CPU waits the whole copy | the time the call takes to return |

Two of these faults made a headline number better: the admission on the
prompt improved the median TTFT, and the shortest-first mystery improved it
more. A scheduler must be judged on the tail of each kind of request.
'''
