"""Part 2, break it on purpose: batching."""

PART = 2
NAME = 'Batching'
FOLDER = 'course/Part2_Batching/3_incidents'
IMPORTS = 'from transformers import AutoModelForCausalLM, AutoTokenizer'

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into a working batch, a working scheduler or a working
plan, and you watch what it does.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** Which row, which token, which minute?
- **What?** What does the wrong result look like?
- **Which guard?** Which check would catch it?

The reference for a row of a batch is the same prompt run **alone**. Some
exercises compare in float32 with Qwen3-0.6B, because in bfloat16 a batch
changes the last bits of the arithmetic (Part 1, Ticket 5). In float32 a
correct batch gives the same tokens as the row alone.

Two exercises simulate a server with `lab.serve`. A simulation is the right
tool when the fault lives in the policy, not in the arithmetic.

This notebook needs a GPU with about 10 GB free.
'''

LOAD = '''
### run this cell

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'
TOKENS = 20

PROMPTS = ['The three largest cities in Japan are',
           'My favourite recipe for pancakes is',
           'The capital of France is',
           'Water boils at',
           'A good name for a black cat that likes to sleep in the sun all day is',
           'The first person to walk on the moon was',
           'The chemical symbol for gold is',
           'In 1492, Columbus']
lengths = [len(tokenizer(p).input_ids) for p in PROMPTS]
alone = [lab.greedy_alone(model, tokenizer, p, TOKENS) for p in PROMPTS]
print('prompt lengths:', lengths)

def compare(name, rows, reference, lengths):
  """One line for each row: the pads and the first different token."""
  print(name)
  for row, (got, want) in enumerate(zip(rows, reference)):
    print(f'  row {row}: {max(lengths) - lengths[row]:2d} pads, first difference at {lab.first_difference(got, want)}'
          f'   {tokenizer.decode(got)[:60]!r}')
'''

DRILLS = [
    ('d1', '''
# Exercise 1: pad on the right, read position -1

The batch pads on the right, and the loop reads the logits of the last
position of each row. Everything else is correct: the mask is there, and it
grows at each step.

This is Ticket 1 of the incident file. Predict which rows go wrong, and at
which token.
''', '''
@torch.inference_mode()
def fault_right_padding(prompts, max_tokens):
  tokenizer.padding_side = 'right'                                    # THE FAULT
  batch = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
  mask = batch.attention_mask
  out = model(batch.input_ids, attention_mask=mask, use_cache=True)
  cache = out.past_key_values
  next_tokens = out.logits[:, -1].argmax(-1)
  rows = [[] for _ in prompts]
  for _ in range(max_tokens):
    for row, token in enumerate(next_tokens.tolist()):
      rows[row].append(token)
    mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=1)
    out = model(next_tokens[:, None], attention_mask=mask, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    next_tokens = out.logits[:, -1].argmax(-1)
  tokenizer.padding_side = 'left'
  return rows

compare('right padding', fault_right_padding(PROMPTS, TOKENS), alone, lengths)
'''),

    ('d2', '''
# Exercise 2: the mask only in the prefill

The fix of Exercise 1: left padding, and the mask in the prefill. But the
decode steps do not pass the mask.

This is Ticket 2. Predict the first different token of each row. Run it in
float32 on Qwen3-0.6B, so that a correct row matches exactly.
''', '''
exact = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).cuda().eval()
alone_fp32 = [lab.greedy_alone(exact, tokenizer, p, TOKENS) for p in PROMPTS]

@torch.inference_mode()
def fault_prefill_mask_only(m, prompts, max_tokens):
  tokenizer.padding_side = 'left'
  batch = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
  out = m(batch.input_ids, attention_mask=batch.attention_mask, use_cache=True)
  cache = out.past_key_values
  next_tokens = out.logits[:, -1].argmax(-1)
  rows = [[] for _ in prompts]
  for _ in range(max_tokens):
    for row, token in enumerate(next_tokens.tolist()):
      rows[row].append(token)
    out = m(next_tokens[:, None], past_key_values=cache, use_cache=True)   # THE FAULT: no mask
    cache = out.past_key_values
    next_tokens = out.logits[:, -1].argmax(-1)
  return rows

compare('mask in the prefill only, float32', fault_prefill_mask_only(exact, PROMPTS, TOKENS), alone_fp32, lengths)
compare('the correct batch, float32', lab.batched_greedy(exact, tokenizer, PROMPTS, TOKENS), alone_fp32, lengths)
del exact; torch.cuda.empty_cache()
'''),

    ('d3', '''
# Exercise 3: batch 80 against batch 64

Measure one decode step at several batch sizes, with 512 tokens of context in
each sequence. Compare each with the floor: the weights plus the KV cache of
every sequence, divided by the bandwidth of `./vc info`.

This is Ticket 3. Predict the gain in tokens/s from 32 to 64, and from 64 to
80.
''', '''
BANDWIDTH = 294e9            # replace with the streaming bandwidth of ./vc info on your card
CONTEXT = 512
weights = sum(p.numel() * p.element_size() for p in model.parameters())
kv_per_token = 2 * 28 * 8 * 128 * 2

steps = {}
previous = None
for batch in (1, 8, 32, 64, 80):
  ms = lab.decode_step_ms(model, batch, CONTEXT)
  steps[batch] = ms
  kv = batch * CONTEXT * kv_per_token
  floor = (weights + kv) / BANDWIDTH * 1000
  rate = batch / ms * 1000
  gain = f'  x{rate / previous:.2f}' if previous else ''
  previous = rate
  print(f'batch {batch:3d}: KV {kv / 1e9:4.2f} GB   step {ms:5.1f} ms   floor {floor:5.1f} ms '
        f'({floor / ms:.0%})   {rate:6.0f} tok/s{gain}')
'''),

    ('d4', '''
# Exercise 4: the static batch with one long answer

Simulate the nightly job of Ticket 4: 10,000 summaries in static batches of
32. 96% of the answers have about 100 tokens, and 4% run to 2,000. A static
batch runs until its longest answer ends.

Predict the steps for each batch, and the fraction of the slots that do
useful work.
''', '''
import random
rng = random.Random(0)
answers = [2000 if rng.random() < 0.04 else rng.randint(60, 140) for _ in range(10_000)]

steps_static = useful = 0
for start in range(0, len(answers), 32):
  batch = answers[start:start + 32]
  steps_static += max(batch)                                          # THE FAULT: wait for the slowest
  useful += sum(batch)
batches = math.ceil(len(answers) / 32)
mean = sum(answers) / len(answers)
print(f'mean answer {mean:.0f} tokens; the plan: {batches} batches x {mean:.0f} steps = {batches * mean:,.0f} steps')
print(f'static batches: {steps_static:,} steps, {steps_static / (batches * mean):.1f}x the plan')
print(f'useful slots: {useful / (steps_static * 32):.0%}')
print(f'the same answers sorted by length first: '
      f'{sum(max(sorted(answers)[s:s + 32]) for s in range(0, len(answers), 32)):,} steps')
'''),

    ('d5', '''
# Exercise 5: continuous batching that forgets the stop token

A simulated server with 32 slots and a backlog of 1,000 requests. The answers
have a median of about 180 tokens, and `max_tokens` is 1,024. The fault: a
request leaves its slot only at `max_tokens`, not when it emits the stop
token.

This is Ticket 5. Predict the useful fraction of the decoded tokens, and the
slowdown of the whole backlog.
''', '''
import random
rng = random.Random(1)
backlog = [(0.0, min(1024, max(8, int(rng.lognormvariate(math.log(180), 0.6)))), 1024) for _ in range(1000)]

for evict in (True, False):
  result = lab.serve(backlog, slots=32, step_ms=25, evict_on_stop=evict)   # THE FAULT when False
  print(f'evict on stop = {evict!s:5s}: backlog done after {result["end"] / 60:5.1f} min, '
        f'useful tokens {result["useful"] / result["total"]:.0%}')
'''),

    ('d6', '''
# Exercise 6: 12 arrivals each second for 10 each second

A simulated server with 16 slots, 25 ms for each step, and answers of 64
tokens. So it finishes 16 / 0.025 / 64 = 10 requests each second. For 30
minutes, 12 requests arrive each second.

This is Ticket 6. Predict the time to the first token at minutes 5, 15 and
30.
''', '''
arrivals = lab.poisson_arrivals(rate=12, seconds=30 * 60)
requests = [(t, 64, 64) for t in arrivals]
result = lab.serve(requests, slots=16, step_ms=25)
for minute in (1, 5, 15, 30):
  window = [f - a for (a, _, _), f in zip(requests, result['first'])
            if minute * 60 - 30 <= a < minute * 60]
  print(f'arrived at minute {minute:2d}: time to the first token {sum(window) / len(window):6.1f} s')
print(f'predicted slope: (12 - 10) / 10 = 0.2 s of wait for each second of overload')
'''),

    ('d7', '''
# Exercise 7: the pad token is the end-of-turn token

A multi-turn chat and a short chat share a batch. The code uses `<|im_end|>`
(151645) as the pad token, and it builds the mask as `ids != pad_id`. In a
Qwen3 chat, `<|im_end|>` also ends every turn.

This is Ticket 7. Predict the zeros of the mask of the long chat, and how its
answer changes. Run in float32.
''', '''
exact = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).cuda().eval()
long_chat = [{'role': 'user', 'content': 'My name is Priya and I live in Lisbon.'},
             {'role': 'assistant', 'content': 'Nice to meet you, Priya!'},
             {'role': 'user', 'content': 'I have a dog called Bento.'},
             {'role': 'assistant', 'content': 'Bento is a lovely name for a dog.'},
             {'role': 'user', 'content': 'What is my name, where do I live, and what is my dog called?'}]
short_chat = [{'role': 'user', 'content': 'Say hello.'}]
texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
         for c in (long_chat, short_chat)]
reference = [lab.greedy_alone(exact, tokenizer, t, 40) for t in texts]

PAD = 151645                                                          # <|im_end|>
token_lists = [tokenizer(t).input_ids for t in texts]
width = max(len(t) for t in token_lists)
ids = torch.tensor([[PAD] * (width - len(t)) + t for t in token_lists], device=device)
mask = (ids != PAD).long()                                            # THE FAULT
print('pads:', [width - len(t) for t in token_lists], '  zeros in the mask:', (mask == 0).sum(1).tolist())

@torch.inference_mode()
def run(ids, mask, max_tokens):
  out = exact(ids, attention_mask=mask, use_cache=True)
  cache, rows = out.past_key_values, [[] for _ in ids]
  next_tokens = out.logits[:, -1].argmax(-1)
  for _ in range(max_tokens):
    for row, token in enumerate(next_tokens.tolist()):
      rows[row].append(token)
    mask = torch.cat([mask, mask.new_ones(len(ids), 1)], dim=1)
    out = exact(next_tokens[:, None], attention_mask=mask, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    next_tokens = out.logits[:, -1].argmax(-1)
  return rows

rows = run(ids, mask, 40)
for name, got, want in zip(['long chat', 'short chat'], rows, reference):
  got = got[:got.index(PAD)] if PAD in got else got
  print(f'{name}: first difference at {lab.first_difference(got, want)}')
  print('   batch:', repr(tokenizer.decode(got)))
  print('   alone:', repr(tokenizer.decode(want)))
del exact; torch.cuda.empty_cache()
'''),

    ('d8', '''
# Exercise 8: the fastest batch against the promise to each user

Take the step times of Exercise 3. The promise to the users is 25 tokens each
second for each user, so a step must take at most 40 ms.

This is Ticket 8. Predict the largest batch that keeps the promise, and how
much total throughput the promise costs.
''', '''
PROMISE = 25                                                          # tokens each second for each user
best = max(steps, key=lambda b: b / steps[b])
for batch, ms in steps.items():
  per_user = 1000 / ms
  flag = '' if per_user >= PROMISE else '   <- breaks the promise'
  print(f'batch {batch:3d}: {per_user:5.1f} tok/s for each user, {batch * per_user:6.0f} tok/s in total{flag}')
kept = max(b for b in steps if 1000 / steps[b] >= PROMISE)
print(f'the fastest batch: {best}. The largest batch that keeps the promise: {kept}, '
      f'which costs {1 - (kept / steps[kept]) / (best / steps[best]):.0%} of the total throughput')
'''),
]

MYSTERY = '''
# Exercise 9: three mystery batches

The module `mystery.py` holds three batched loops: `mystery_a`, `mystery_b`
and `mystery_c`. Each one takes a list of prompts and returns one answer for
each prompt, like `lab.batched_greedy`. Each one has one fault. **Do not open
the file.**

For each loop:

1. Run it on `PROMPTS`, and compare each row with `alone`.
2. Design a second experiment that makes the fault visible. Which variable do
   you change? The order of the prompts? Their number? The kind of prompt?
3. Write your diagnosis: the fault, and the experiment that proved it.
4. Only then, open `mystery.py` and check.

A hint about the method: each answer of one loop is a good answer, to some
prompt. One loop depends on the other prompts in the batch. One loop depends
on the size of the batch.
'''

MYSTERY_CODE = '''
from mystery import mystery_a, mystery_b, mystery_c

for name, loop in [('a', mystery_a), ('b', mystery_b), ('c', mystery_c)]:
  compare(f'mystery_{name}', loop(model, tokenizer, PROMPTS, TOKENS), alone, lengths)
'''

MYSTERY_NAMES = ['mystery_a', 'mystery_b', 'mystery_c']

FINGERPRINT_ROWS = ['right padding, read position -1', 'the mask only in the prefill',
                    'a larger batch that reads more KV', 'a static batch with one long answer',
                    'no eviction at the stop token', 'arrivals above the service rate',
                    'the pad id is the end-of-turn id', 'the fastest batch breaks the promise']

SOLUTIONS = {
    'd1': '''
### What happens

- **Crash?** No.
- **When?** At token 0, in 6 of the 7 rows that have pads. The one row with 0
  pads (row 4, the longest prompt) matched the reference exactly.
- **What?** Three rows collapsed into `' the the the the ...'`. The others
  wrote text that starts oddly: `', and. The three largest cities in Japan
  are ...'`. Position -1 of a padded row is a pad token, so the first token
  is the prediction after a pad.
- **The surprise.** Row 3 (14 pads) was correct for 5 tokens. With the mask,
  the pads are invisible to the attention, so the prediction at a pad
  position is not always garbage. It depends on the prompt. You cannot rely
  on garbage to find this bug.

**The fingerprint.** Only the padded rows break, and the row with 0 pads is
perfect.

**The guard.** Compare each row of a batch with the row alone.
''',
    'd2': '''
### What happens

In float32 on Qwen3-0.6B:

| row | pads | first difference, mask in the prefill only | the correct batch |
|---|---|---|---|
| 0 | 10 | 8 | none |
| 1 | 11 | 5 | none |
| 2 | 12 | 5 | none |
| 3 | 14 | 6 | none |
| 4 | 0 | none | none |
| 5 | 8 | 2 | none |
| 6 | 11 | none | none |
| 7 | 9 | 4 | none |

- **Crash?** No.
- **When?** Never at token 0: the prefill has the mask. The rows with pads
  leave the reference 2 to 8 tokens later. Row 4, with 0 pads, is exact.
- **What?** Fluent text that is a little different: `' Paris. The capital of
  France is Paris.'` instead of `' Paris. The capital of Italy is Rome.'`.
- **The surprise.** Row 6 has 11 pads and still matched for 20 tokens. The
  pads get a small weight in the attention, and for some prompts that does
  not change the argmax. A test with one prompt can miss this bug.

The correct batch matched **every** row exactly in float32. That is the
test to keep.
''',
    'd3': '''
### What happens

On my card (294 GB/s), with 512 tokens of context:

| batch | KV | step | floor | % of the floor | tok/s | gain |
|---|---|---|---|---|---|---|
| 1 | 0.06 GB | 13.1 ms | 11.9 ms | 91% | 77 | |
| 8 | 0.47 GB | 19.3 ms | 13.3 ms | 69% | 415 | x5.42 |
| 32 | 1.88 GB | 37.4 ms | 18.1 ms | 48% | 856 | x2.07 |
| 64 | 3.76 GB | 64.5 ms | 24.5 ms | 38% | 992 | x1.16 |
| 80 | 4.70 GB | 82.3 ms | 27.7 ms | 34% | 972 | x0.98 |

The gain falls fast, as Ticket 3 predicts: each sequence adds its own KV
bytes, and only the weights are shared. From 64 to 80 the throughput did not
grow at all.

But the fraction of the floor falls too, from 91% to 34%. Physics does not
explain that part. A profile of the step at batch 64 does:

| kernel | GPU time |
|---|---|
| `torch.cat` in the KV cache | 30.5 ms (55%) |
| flash attention | 11.5 ms (21%) |
| the matmuls | about 10 ms |

HF's `DynamicCache` grows by concatenation. At each step it copies the whole
cache of each layer to append one token: it reads 3.76 GB and writes 3.76 GB,
at about 246 GB/s. The attention itself reads the cache at 327 GB/s, close to
the card.

So this ticket has two answers on this code. The shape of the curve is
physics. The distance from the floor is the cache layout, and that is the
problem that Part 3 solves: a preallocated, paged cache that never copies.
''',
    'd4': '''
### What happens

- The plan: 313 batches x 171 steps (the mean answer) = 53,475 steps.
- The static batches: 467,739 steps, **8.7x** the plan. The same factor as in
  Ticket 4.
- Only **11%** of the slots did useful work.
- Sorted by length first: 55,938 steps, only 5% above the plan. The long
  answers share a few batches, and the other batches end early.

**The fingerprint.** A batch time that follows the **maximum** answer, and a
plan that used the mean.
''',
    'd5': '''
### What happens

| | the backlog is done after | useful decoded tokens |
|---|---|---|
| evict at the stop token | 3.0 min | 100% |
| evict only at `max_tokens` | 13.7 min | 20% |

A median answer of about 180 tokens stays for 1,024 steps. 80% of the decode
work goes to tokens that the detokenizer throws away, and the backlog takes
4.6x longer. The answers that the users see are correct.

**The fingerprint.** The slots are always full, the queue is long, and the
answers are correct. Only a count of the tokens after the stop token shows it.
''',
    'd6': '''
### What happens

| arrived at | time to the first token |
|---|---|
| minute 1 | 7.9 s |
| minute 5 | 51.6 s |
| minute 15 | 184.4 s |
| minute 30 | 367.3 s |

The prediction was 0.2 s of wait for each second of overload: 60 s at minute
5, 180 s at minute 15, and 360 s at minute 30. The simulation gives 52, 184
and 367 s. The Poisson arrivals add noise, and the line is the same.

The engine is never slow in this simulation. Each step is 25 ms. The wait
is only the queue.

**The fingerprint.** A time to the first token that grows linearly with the
time since the overload began, with a flat time for each token.
''',
    'd7': '''
### What happens

- The mask of the long chat has **5** zeros and 0 pads: the 5 `<|im_end|>`
  tokens that end its turns. The mask of the short chat has 72 zeros: 71
  pads and 1 end of turn.
- The short chat is identical to the reference.
- The long chat changed from token 0. Alone: `'Your name is Priya, you live
  in Lisbon, and your dog is called Bento.'`. In the batch: `'Hi Priya! Your
  name is **Priya**, you live in **Lisbon**, ...'`, a longer answer with
  bold text, and it did not stop at the same place.

**The surprise.** The facts are still correct. Qwen3 marks the **start** of
each turn with `<|im_start|>`, which the mask did not hide, so the model still
found the turns. The damage is a different answer, not a wrong one. On a
model with fewer turn markers, or a longer chat, the damage is larger. A
different answer is a fault even when it reads well: the output depends on
the batch.

**The fingerprint.** Only the rows with several turns differ, from token 0.
The count of zeros in the mask = pads + turns.
''',
    'd8': '''
### What happens

| batch | tok/s for each user | tok/s in total |
|---|---|---|
| 1 | 76.5 | 77 |
| 8 | 51.8 | 415 |
| 32 | 26.8 | 856 |
| 64 | 15.5 | 992 |
| 80 | 12.1 | 972 |

The fastest batch is 64. The largest batch that keeps the promise of 25
tokens each second is 32. The promise costs 14% of the total throughput.

On this card and this code, 14% is a large price, because the step grows
fast (Exercise 3). With a cache that does not copy, the step grows more
slowly, and the same promise allows a larger batch. The rule is the same:
choose the batch from the promise to each user, then maximize the total.
''',
}

MYSTERY_SOL = '''
### The three mysteries

**`mystery_a`: the answers come back in the wrong order.** It sorts the
prompts by length to pad less, and it forgets to put the answers back. Each
answer is a correct answer, to another prompt: row 0 asked about the cities of
Japan and got the answer about boiling water. The experiment: compare each
answer with **every** reference, not only with its own row. A match in
another row proves a permutation. On my run, row 2 matched by chance, because
the sort left it in place. In production this is a data leak: user A gets the
answer of user B.

**`mystery_b`: the whole batch stops at the first stop token.** On `PROMPTS`
it looks perfect, because no prompt stops within 20 tokens. The experiment:
put a prompt that stops early into the batch, such as a chat turn "Say hi".
Then every other row is cut at that step. The fault depends on the **other
prompts** in the batch.

**`mystery_c`: sub-batches of 4 share one cache.** Rows 0 to 3 are perfect.
Rows 4 to 7 attend to the cache of rows 0 to 3, and they borrow from it: row
4 wrote `'...? 10,000, and Osaka is 13,000'`, with Osaka from row 0. The
experiment: run 4 prompts, then 8 prompts, then the same 8 in another order.
The fault depends on the **size** of the batch, and it is Part 1, Ticket 8
in a new place.

### The method

| Mystery | The variable to change |
|---|---|
| a | compare each answer with every prompt, not with its own row |
| b | the other prompts in the batch: add one that stops early |
| c | the size of the batch |
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| right padding, read position -1 | no | token 0, padded rows | odd starts, `the the the` | each row against the row alone |
| the mask only in the prefill | no | tokens 2 to 8, padded rows | fluent, a little different | the same, in float32, for 20 tokens |
| a larger batch that reads more KV | no | above batch 32 | tok/s stops growing | the floor with the KV term |
| a static batch with one long answer | no | every batch with a long answer | 8.7x the plan, 11% useful | the useful fraction of the slots |
| no eviction at the stop token | no | always | 20% useful, correct answers | tokens after the stop = 0 |
| arrivals above the service rate | no | from the first minute | a wait that grows linearly | arrival rate against service rate |
| the pad id is the end-of-turn id | no | token 0, chats with turns | a different answer | zeros in the mask = pads |
| the fastest batch breaks the promise | no | above batch 32 here | slow text for each user | TPOT against the promise |

Exercise 3 found a second problem that the ticket did not describe: a cache
that grows by concatenation spends more than half of each step on copies.
That is where Part 3 begins.
'''
