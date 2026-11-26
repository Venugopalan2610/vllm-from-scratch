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
lengths = [len(tokenizer(prompt).input_ids) for prompt in PROMPTS]
alone = [lab.greedy_alone(model, tokenizer, prompt, TOKENS) for prompt in PROMPTS]
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
  forward_pass = model(batch.input_ids, attention_mask=mask, use_cache=True)
  cache = forward_pass.past_key_values
  next_tokens = forward_pass.logits[:, -1].argmax(-1)
  rows = [[] for _ in prompts]
  for _ in range(max_tokens):
    for row, token in enumerate(next_tokens.tolist()):
      rows[row].append(token)
    mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=1)
    forward_pass = model(next_tokens[:, None], attention_mask=mask, past_key_values=cache, use_cache=True)
    cache = forward_pass.past_key_values
    next_tokens = forward_pass.logits[:, -1].argmax(-1)
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
alone_fp32 = [lab.greedy_alone(exact, tokenizer, prompt, TOKENS) for prompt in PROMPTS]

@torch.inference_mode()
def fault_prefill_mask_only(model_under_test, prompts, max_tokens):
  tokenizer.padding_side = 'left'
  batch = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
  forward_pass = model_under_test(batch.input_ids, attention_mask=batch.attention_mask, use_cache=True)
  cache = forward_pass.past_key_values
  next_tokens = forward_pass.logits[:, -1].argmax(-1)
  rows = [[] for _ in prompts]
  for _ in range(max_tokens):
    for row, token in enumerate(next_tokens.tolist()):
      rows[row].append(token)
    forward_pass = model_under_test(next_tokens[:, None], past_key_values=cache, use_cache=True)   # THE FAULT: no mask
    cache = forward_pass.past_key_values
    next_tokens = forward_pass.logits[:, -1].argmax(-1)
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
import cudalib
BANDWIDTH = cudalib.read_bandwidth()     # bytes/s of a streaming read on this card, measured now
CONTEXT = 512
weights = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
kv_per_token = 2 * 28 * 8 * 128 * 2

steps = {}
previous = None
for batch in (1, 8, 32, 64, 80):
  step_ms = lab.decode_step_ms(model, batch, CONTEXT)
  steps[batch] = step_ms
  kv = batch * CONTEXT * kv_per_token
  floor = (weights + kv) / BANDWIDTH * 1000
  rate = batch / step_ms * 1000
  gain = f'  x{rate / previous:.2f}' if previous else ''
  previous = rate
  print(f'batch {batch:3d}: KV {kv / 1e9:4.2f} GB   step {step_ms:5.1f} ms   floor {floor:5.1f} ms '
        f'({floor / step_ms:.0%})   {rate:6.0f} tok/s{gain}')
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
      f'{sum(max(sorted(answers)[batch_start:batch_start + 32]) for batch_start in range(0, len(answers), 32)):,} steps')
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
  served = lab.serve(backlog, slots=32, step_ms=25, evict_on_stop=evict)   # THE FAULT when False
  print(f'evict on stop = {evict!s:5s}: backlog done after {served["end"] / 60:5.1f} min, '
        f'useful tokens {served["useful"] / served["total"]:.0%}')
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
requests = [(arrival, 64, 64) for arrival in arrivals]
served = lab.serve(requests, slots=16, step_ms=25)
for minute in (1, 5, 15, 30):
  window = [first_token_at - arrival for (arrival, _, _), first_token_at in zip(requests, served['first'])
            if minute * 60 - 30 <= arrival < minute * 60]
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
texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
         for chat in (long_chat, short_chat)]
reference = [lab.greedy_alone(exact, tokenizer, text, 40) for text in texts]

PAD = 151645                                                          # <|im_end|>
token_lists = [tokenizer(text).input_ids for text in texts]
width = max(len(token_list) for token_list in token_lists)
ids = torch.tensor([[PAD] * (width - len(token_list)) + token_list for token_list in token_lists], device=device)
mask = (ids != PAD).long()                                            # THE FAULT
print('pads:', [width - len(token_list) for token_list in token_lists], '  zeros in the mask:', (mask == 0).sum(1).tolist())

@torch.inference_mode()
def run(ids, mask, max_tokens):
  forward_pass = exact(ids, attention_mask=mask, use_cache=True)
  cache, rows = forward_pass.past_key_values, [[] for _ in ids]
  next_tokens = forward_pass.logits[:, -1].argmax(-1)
  for _ in range(max_tokens):
    for row, token in enumerate(next_tokens.tolist()):
      rows[row].append(token)
    mask = torch.cat([mask, mask.new_ones(len(ids), 1)], dim=1)
    forward_pass = exact(next_tokens[:, None], attention_mask=mask, past_key_values=cache, use_cache=True)
    cache = forward_pass.past_key_values
    next_tokens = forward_pass.logits[:, -1].argmax(-1)
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

Take the step times of Exercise 3. The promise to the users: each user gets at
least a third of the speed of a user who is alone on the card. A promise in
tokens each second would give a different answer on every card; a promise
relative to batch 1 gives the same kind of answer everywhere.

This is Ticket 8. Predict the largest batch that keeps the promise, and how
much total throughput the promise costs.
''', '''
PROMISE = (1000 / steps[1]) / 3                                       # a third of the speed alone
print(f'alone: {1000 / steps[1]:.0f} tok/s for the user; the promise: {PROMISE:.0f} tok/s for each user')
best = max(steps, key=lambda batch_size: batch_size / steps[batch_size])
for batch, step_ms in steps.items():
  per_user = 1000 / step_ms
  flag = '' if per_user >= PROMISE else '   <- breaks the promise'
  print(f'batch {batch:3d}: {per_user:5.1f} tok/s for each user, {batch * per_user:6.0f} tok/s in total{flag}')
kept = max(batch_size for batch_size in steps if 1000 / steps[batch_size] >= PROMISE)
print(f'the fastest batch: {best}. The largest batch that keeps the promise: {kept}, '
      f'which costs {1 - (kept / steps[kept]) / (best / steps[best]):.0%} of the total throughput')
'''),
    ('d9', '''
# Exercise 9: the order of eight additions

Exercises 9, 10 and 11 have no bug in the code. The fault is an assumption:
"at temperature 0, the same prompt gives the same tokens at every batch
size". Part 1, Ticket 5 showed the symptom. These three exercises show the
cause, one level at a time: one sum, one layer, one whole answer.

A **reduction** combines many values into one value. A sum is a reduction.
The **reduction order** is the sequence in which the additions happen. In
exact arithmetic the order has no effect. In floating point, the hardware
rounds the result after each addition, so the order can change the result.

**Given:**

- float32 holds 24 bits of precision. Near 100,000,000 the distance between
  two neighbouring float32 numbers is 8. So float32 holds 100,000,000 and
  100,000,008, and no number between them.
- The eight values `[100000000, 3, 3, 3, 3, 3, 3, 3]`. The exact sum is
  100,000,021.
- Three orders of the same additions:
  1. Left to right, one value at a time.
  2. Right to left, one value at a time.
  3. Two **tiles** of 4 values: the sum of values 0 to 3, the sum of values 4
     to 7, then the sum of the two tile sums. A GPU kernel adds in tiles like
     this.

**Asked:** the float32 result of each order. This cell runs on the CPU.
''', '''
values = torch.tensor([100_000_000.0, 3, 3, 3, 3, 3, 3, 3], dtype=torch.float32)

def sum_in_order(terms):
  """Add the terms one at a time, in float32, in the order given."""
  running_sum = torch.tensor(0.0, dtype=torch.float32)
  for term in terms:
    running_sum = running_sum + term                                  # rounds to float32 after each addition
  return running_sum

spacing_near_1e8 = (torch.nextafter(values[0], torch.tensor(float('inf'))) - values[0]).item()
tile_sums = torch.stack([sum_in_order(values[0:4]), sum_in_order(values[4:8])])
print(f'distance between neighbouring float32 numbers near 1e8: {spacing_near_1e8:.0f}')
print(f'exact sum:     {sum(int(value) for value in values.tolist()):,}')
print(f'left to right: {int(sum_in_order(values)):,}')
print(f'right to left: {int(sum_in_order(values.flip(0))):,}')
print(f'tiles of 4:    {int(sum_in_order(tile_sums)):,}   (the tile sums: {[int(tile_sum) for tile_sum in tile_sums]})')
'''),
    ('d10', '''
# Exercise 10: one layer at six batch sizes

**Given:**

- `gate_proj` of layer 0 of Qwen3-1.7B is a linear layer with 2048 inputs and
  6144 outputs. Each output is a dot product of 2048 terms, so each output is
  a reduction.
- One input row is "the user". The other rows of the batch are random rows:
  the other users.
- The cell computes the output of the user row at batch sizes 1, 2, 8, 32, 64
  and 128, in bfloat16. It compares each output with the output at batch 1,
  bit for bit. The input of the user row is the same at every batch size.
- The profiler prints the name of each CUDA kernel that ran. A name with
  `gemv` is a matrix times one vector (**GEMV**). A name with `gemm` is a
  matrix times a matrix (**GEMM**). A kernel named `splitKreduce` means
  **split-K**: the kernel divides the terms of each dot product among
  several thread blocks, and a second kernel adds their partial sums.
- Then the cell does the same in float32, with `gate_proj` of Qwen3-0.6B
  (1024 inputs, 3072 outputs).

**Asked:**

- Is the user row at batch 64 bit-identical to the user row at batch 1?
- How many different kernels run over the six batch sizes?
- Does float32 remove the differences, or only make them smaller?
''', '''
from torch.profiler import profile, ProfilerActivity

def kernel_names(layer, inputs):
  """The names of the CUDA kernels that one call of the layer runs, cut to 60 characters."""
  with profile(activities=[ProfilerActivity.CUDA]) as trace:
    layer(inputs)
    torch.cuda.synchronize()
  return [event.key[:60] for event in trace.key_averages() if event.device_type.name == 'CUDA']

@torch.inference_mode()
def one_row_at_each_batch(layer, dtype):
  torch.manual_seed(0)
  user_row = torch.randn(1, layer.in_features, device=device, dtype=dtype)
  user_output_alone = layer(user_row)[0]
  for batch in (1, 2, 8, 32, 64, 128):
    other_rows = torch.randn(batch - 1, layer.in_features, device=device, dtype=dtype)
    inputs = torch.cat([user_row, other_rows])
    user_output_in_batch = layer(inputs)[0]
    outputs_that_differ = (user_output_in_batch != user_output_alone).sum().item()
    largest_difference = (user_output_in_batch.float() - user_output_alone.float()).abs().max().item()
    print(f'  batch {batch:3d}: {outputs_that_differ:4d} of {user_output_alone.numel()} outputs differ, '
          f'largest {largest_difference:.2e}   {kernel_names(layer, inputs)}')

print('Qwen3-1.7B, gate_proj of layer 0, bfloat16')
one_row_at_each_batch(model.model.layers[0].mlp.gate_proj, torch.bfloat16)
exact = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).cuda().eval()
print('Qwen3-0.6B, gate_proj of layer 0, float32')
one_row_at_each_batch(exact.model.layers[0].mlp.gate_proj, torch.float32)
'''),
    ('d11', '''
# Exercise 11: the same prompt at batch 1 and batch 64

**Given:**

- `lab.batched_greedy` is correct. In Exercise 2 it matched every row alone
  in float32.
- The batch holds the **same prompt** many times. So the batch has no pads,
  and every row has exactly the same input.
- Three prompts, 40 tokens each, at batch sizes 8, 32, 48, 64 and 96. First
  in bfloat16 on Qwen3-1.7B, then in float32 on Qwen3-0.6B.
- `lab.top_two_logits` repeats the run alone, and gives the largest and the
  second largest logit at each token. Their distance is the **gap**. A small
  gap means that the argmax was close to another token.
- `largest_logit_change` runs one prompt at batch 1 and at batch 64, and
  gives the largest change of any logit at the first decode step: the noise
  that the batch size adds after all 28 layers.
- One bfloat16 step is the distance between two neighbouring bfloat16
  numbers. Between 16 and 32 it is 0.125. Between 32 and 64 it is 0.25.

**Asked:**

- At which batch sizes does row 0 differ from the row alone?
- Inside one batch, do the rows differ from each other?
- What is the gap at the first different token, in bfloat16 steps?
- Does a larger batch always differ more?
''', '''
PROMPTS_REPEATED = ['The three largest cities in Japan are',
                    'My favourite recipe for pancakes is',
                    'The first person to walk on the moon was']
ANSWER_TOKENS = 40

def step_at(value, dtype):
  """The distance from `value` to the next larger number of `dtype`."""
  number = torch.tensor(value, dtype=dtype)
  return (torch.nextafter(number, torch.tensor(float('inf'), dtype=dtype)) - number).item()

@torch.inference_mode()
def largest_logit_change(model_under_test, prompt, batch):
  """The largest change of any logit at the first decode step, between batch 1 and `batch`."""
  ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  first_token = model_under_test(ids).logits[:, -1].argmax(-1, keepdim=True)
  logits_by_batch = []
  for rows in (1, batch):
    prefill = model_under_test(ids.repeat(rows, 1), use_cache=True)
    decode = model_under_test(first_token.repeat(rows, 1), past_key_values=prefill.past_key_values, use_cache=True)
    logits_by_batch.append(decode.logits[0, -1].float())
  return (logits_by_batch[1] - logits_by_batch[0]).abs().max().item()

def same_prompt_at_each_batch(model_under_test, dtype, name):
  print(name)
  for prompt in PROMPTS_REPEATED:
    reference = lab.greedy_alone(model_under_test, tokenizer, prompt, ANSWER_TOKENS)
    top_two = lab.top_two_logits(model_under_test, tokenizer, prompt, ANSWER_TOKENS)
    print(f'  {prompt!r}: smallest gap in the run alone {min(largest - second for largest, second in top_two):.3f}, '
          f'largest logit change from batch 1 to batch 64 {largest_logit_change(model_under_test, prompt, 64):.2e}')
    for batch in (8, 32, 48, 64, 96):
      rows = lab.batched_greedy(model_under_test, tokenizer, [prompt] * batch, ANSWER_TOKENS)
      rows_agree = all(row == rows[0] for row in rows)
      position = lab.first_difference(rows[0], reference)
      if position is None:
        where = 'never'
      else:
        largest, second = top_two[position]
        where = (f'at token {position}: gap alone {largest - second:.3f} = '
                 f'{(largest - second) / step_at(largest, dtype):.1f} steps of {dtype} at {largest:.2f}')
      print(f'    batch {batch:3d}: rows of the batch all equal: {rows_agree!s:5s}   first difference from alone: {where}')

same_prompt_at_each_batch(model, torch.bfloat16, 'Qwen3-1.7B, bfloat16')
same_prompt_at_each_batch(exact, torch.float32, 'Qwen3-0.6B, float32')
del exact; torch.cuda.empty_cache()
'''),
]

MYSTERY = '''
# Exercise 12: three mystery batches

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
                    'the pad id is the end-of-turn id', 'the fastest batch breaks the promise',
                    'temperature 0 at another batch size']

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

The gain in tokens/s from each batch size to the next, in the reference runs
(two days on one card, before and after a repair that made its memory 30%
faster):

| from | to | gain |
|---|---|---|
| batch 1 | batch 8 | x5.4 to x5.6 |
| batch 8 | batch 32 | x2.0 to x2.1 |
| batch 32 | batch 64 | x1.16 to x1.20 |
| batch 64 | batch 80 | x0.98 to x1.11 |

The gain falls fast, as Ticket 3 predicts: each sequence adds its own KV
bytes, and only the weights are shared. From 64 to 80 the throughput hardly
grew.

The fraction of the floor falls too: from about 70% to 90% at batch 1 to about
30% at batch 80. Physics does not explain that part. A profile of the step at
batch 64 does: `torch.cat` in the KV cache takes about 55% of the GPU time,
flash attention about 20%, and the matmuls the rest.

HF's `DynamicCache` grows by concatenation. At each step it copies the whole
cache of each layer to append one token. The attention itself reads the cache
at close to the bandwidth of the card.

So this ticket has two answers on this code. The shape of the curve is
physics. The distance from the floor is the cache layout, and that is the
problem that Part 3 solves: a preallocated, paged cache that never copies.

The fraction at batch 1 depends on the card and even on the day: after the
repair the memory was 30% faster, and the step at batch 1 only 5% faster,
because at batch 1 the CPU work of HF's eager step is a large part of it. So
compare the gains and the shape, not the percentages.
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

In the reference runs, a user alone gets about 80 tokens/s, so the promise is
about 27 tokens/s for each user:

| batch | speed for each user, against alone | total, against batch 1 |
|---|---|---|
| 1 | 1.00 | 1x |
| 8 | 0.68 to 0.71 | 5.4x to 5.6x |
| 32 | 0.35 to 0.37 | 11x to 12x |
| 64 | 0.20 to 0.22 | 13x to 14x |
| 80 | 0.16 to 0.20 | 13x to 16x |

The largest batch that keeps the promise is 32. The fastest batch is 64 or 80,
and the promise costs 14% to 25% of the total throughput.

That is a large price, because the step of this code grows fast (Exercise 3).
With a cache that does not copy, the step grows more slowly, and the same
promise allows a larger batch. The rule is the same: choose the batch from the
promise to each user, then maximize the total.
''',
    'd9': '''
### What happens

| Order | The steps | float32 result |
|---|---|---|
| left to right | `100000000 + 3` = 100,000,003, which rounds to 100,000,000. The same happens 7 times. | 100,000,000 |
| tiles of 4 | Tile 1: each `+ 3` rounds away, so 100,000,000. Tile 2: `3+3+3+3` = 12, exact. Then `100000000 + 12` = 100,000,012, which is halfway between 100,000,008 and 100,000,016. The tie rounds to the even neighbour: 100,000,016. | 100,000,016 |
| right to left | `3+3+...+3` = 21, exact. Then `21 + 100000000` = 100,000,021, which rounds to 100,000,024. | 100,000,024 |

The same eight values give three different sums. None of them is the exact
sum, 100,000,021. Each order lost a different amount in the rounding.

- **Crash?** No.
- **When?** Each time that a small value meets a large running sum.
- **What?** A result that depends on the order, with no error and no warning.
- **The surprise.** The right-to-left order is the closest. The small values
  first add up to a number large enough to survive the rounding. No order is
  "the correct one" in float32. Each order is a different, valid rounding.

**The fingerprint.** The same values, a different order, a different last
bit.
''',
    'd10': '''
### What happens

In the reference run (one RTX 4080 Laptop GPU, torch 2.13):

| batch | bfloat16: outputs that differ from batch 1 | float32: outputs that differ from batch 1 | the kernel, bfloat16 | the kernel, float32 |
|---|---|---|---|---|
| 1 | 0 of 6144 | 0 of 3072 | GEMV | GEMV |
| 2 | 10, by 0.0078 | 2672, by 1.2e-6 | CUTLASS GEMM | `gemmSN` GEMM |
| 8 | 10, by 0.0078 | 2672, by 1.2e-6 | CUTLASS GEMM | `gemmSN` GEMM |
| 32 | 10, by 0.0078 | 2804, by 2.2e-6 | CUTLASS GEMM | `sgemm 128x32` + `splitKreduce` |
| 64 | 10, by 0.0078 | 2804, by 2.2e-6 | `gemm 128x64` | `sgemm 128x64` + `splitKreduce` |
| 128 | 10, by 0.0078 | 2804, by 2.2e-6 | `gemm 128x128` | `sgemm 128x64` |

- **Crash?** No.
- **When?** From batch 2 on. The input of the user row never changed. Only
  the number of the other rows changed.
- **What?** cuBLAS selects a kernel from the shape of the whole matrix. Batch
  1 is a matrix times one vector, so it gets a GEMV kernel. Batch 2 and more
  get a GEMM kernel. In bfloat16 there are 5 kernels over the 6 batch sizes,
  with tiles of 128x64 and 128x128. Each kernel adds the 2048 terms of a dot
  product in its own order. In float32, batch 32 and 64 use split-K: the
  `splitKreduce` kernel adds the partial sums of several thread blocks. At
  batch 128 there are enough output tiles to fill the card, and split-K goes
  away again.
- **The surprise.** bfloat16 shows **fewer** differences than float32: 10 of
  6144 against about 2700 of 3072. The kernels accumulate in float32 inside,
  and then round the output to bfloat16. A bfloat16 number has only 8 bits of
  precision, so most of the small float32 differences round to the same
  bfloat16 number. But a difference that survives is a whole bfloat16 step,
  0.0078, not a small fraction of one. float32 differs almost everywhere, by
  0.0000012 to 0.0000022.

So float32 does not remove the differences. It makes each one 3,500 to 6,500
times smaller.

**The fingerprint.** The row is the same, the batch size is different, the
kernel name is different, and the last bits are different.
''',
    'd11': '''
### What happens

In the reference run (the same card):

| prompt | batch 8 | batch 32 | batch 48 | batch 64 | batch 96 | gap alone at the first difference |
|---|---|---|---|---|---|---|
| cities of Japan, bfloat16 | same | same | same | token 14 | token 14 | 0.125 = 1 bfloat16 step at 27.75 |
| pancakes, bfloat16 | same | same | token 23 | token 23 | token 23 | 0.125 = 1 bfloat16 step at 23.75 |
| moon, bfloat16 | same | token 13 | token 13 | same | same | 0.25 = 2 bfloat16 steps at 19.25 |
| all three, float32 | same | same | same | same | same | (the smallest gaps were 0.026 to 0.072) |

- **Crash?** No.
- **When?** Late in the answer: at token 13, 14 or 23 of 40. The answers are
  identical up to that token.
- **What?** Fluent text that is different from that token on. One different
  token changes the input of every step after it, so the answers never join
  again.
- **The rows inside a batch.** They were **always** equal. Each row runs
  through the same kernel with the same tiles, and the rows do not mix in a
  matmul. So the result depends on the **size** of the batch, not on the
  position of the row in it.
- **The gap and the noise.** At each first difference, the gap of the run
  alone was 1 or 2 bfloat16 steps. The noise is larger than one layer
  suggests: from batch 1 to batch 64, the largest logit changed by 0.19 to
  0.38. Exercise 10 showed 10 outputs of one layer that moved by one step.
  Each layer passes its changes to the next, and after 28 layers some logits
  move by 1 to 3 bfloat16 steps. That is enough to swap two tokens with a gap
  of 1 or 2 steps. This is the number of Part 1, Ticket 5.
- **The surprise.** A larger batch does not always differ more. The moon
  prompt differs at batch 32 and 48, and it is identical at 64 and 96. Batch
  64 uses another kernel (Exercise 10), and its rounding at token 13 happens
  to agree with batch 1. The variable is the kernel, not the size.

The pancakes prompt had a gap of 0.000 at one token, an exact tie in
bfloat16, and that token did not change. The tie was the same in both runs,
and the argmax took the same index. A small gap is necessary for a swap. It
is not sufficient.

In float32 every answer matched, although Exercise 10 showed that the bits
differ. The largest logit change was 0.00002 to 0.00003, and the smallest gap
was 0.026: about 1,000 times larger than the noise. That is why the tests of
this repo compare in float32: the bits still move, and the tokens do not. It
is a margin, not a guarantee. A prompt with a gap below 0.0001 can still
swap in float32.

**The guard.** Never test "temperature 0 gives the same tokens" across batch
sizes in bfloat16. Compare in float32. In production, log the gap of the top
two logits. A first difference at a gap of 1 or 2 bfloat16 steps is the
signature of the reduction order, not of a bug.

**The fix, when a customer needs it.** A **batch-invariant** kernel uses one
reduction order for a row at every batch size: no GEMV for batch 1, no
split-K that depends on the shape, and fixed tiles. It costs speed, because
split-K and the special kernels exist to make small batches fast.
''',
}

MYSTERY_SOL = '''
### The three mysteries

**`mystery_a`: the answers come back in the wrong order.** It sorts the
prompts by length to pad less, and it forgets to put the answers back. Each
answer is a correct answer, to another prompt: row 0 asked about the cities of
Japan and got the answer about boiling water. The experiment: compare each
answer with **every** reference, not only with its own row. A match in
another row proves a permutation. In the reference run, row 2 matched by chance, because
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
| temperature 0 at another batch size | no | late, at a gap of 1 or 2 bf16 steps | fluent, different from that token on | compare in float32; log the top-two gap |

Exercise 3 found a second problem that the ticket did not describe: a cache
that grows by concatenation spends more than half of each step on copies.
That is where Part 3 begins.
'''
