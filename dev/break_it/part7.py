"""Part 7, break it on purpose: speculation, quantization, guided decoding, tensor parallel."""

PART = 7
NAME = 'Modern vLLM'
FOLDER = 'course/Part7_ModernVLLM/5_incidents'
IMPORTS = '''import random
import cudalib
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache'''

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into working speculation, quantization, guided
decoding or tensor parallelism, and you watch what it does.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** Which task, which layer, which token?
- **What?** What does the wrong result look like: slower, less exact, an
  invalid answer, a wrong number?
- **Which guard?** Which check would catch it?

Each feature of this Part makes a promise: the same output, almost the same
distribution, always valid, the same result on two ranks. The exercises test
the promises.

This notebook needs a GPU with about 9 GB free.
'''

LOAD = '''
### run this cell

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'

def chat(text):
  text = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
  return tokenizer(text, return_tensors='pt').input_ids.to(device)

def seconds(fn):
  torch.cuda.synchronize()
  start = time.perf_counter()
  result = fn()
  torch.cuda.synchronize()
  return result, time.perf_counter() - start

PARAGRAPH = ('The old cat sat by the window every morning. The cat watched the birds in the garden, '
             'and the cat never tried to catch them. In the evening the cat slept on the warm stones near the fire.')
EDIT = chat(PARAGRAPH + '\\n\\nRepeat the text above exactly, but replace every "cat" with "dog".')
POEM = chat('Write a short, original poem about the sea at night.')
_ = lab.greedy(model, EDIT, 5)                                         # warm up
'''

DRILLS = [
    ('d1', '''
# Exercise 1: speculation on the wrong task

N-gram speculation with 4 draft tokens, on two tasks: an edit that repeats
its input, and a new poem. Then the poem again, with an eager drafter that
matches only the last token (`n=1`), so that it proposes much more often.

This is Ticket 1 of the incident file. Predict the acceptance rate, the tokens
for each step and the speedup of each run, and whether the output equals
plain greedy.
''', '''
for name, ids, n in [('edit, n=3', EDIT, 3), ('poem, n=3', POEM, 3), ('poem, n=1', POEM, 1)]:
  plain, t_plain = seconds(lambda: lab.greedy(model, ids, 120))
  (fast, stats), t_fast = seconds(lambda: lab.speculative(model, ids, 120, n=n))
  rate = stats['accepted'] / max(stats['proposed'], 1)
  print(f'{name}: {stats["proposed"]:3d} drafts, accepted {rate:4.0%}, {stats["tokens_per_step"]:.2f} tokens '
        f'per step, speedup {t_plain / t_fast:.2f}x, the same as greedy: {fast == plain}')
'''),

    ('d2', '''
# Exercise 2: after a rejection, sample from the target again

A target distribution with two tokens: A with 0.6, B with 0.4. The draft
always proposes A. The correct rule accepts A with probability
min(1, p/q), and after a rejection it samples from the residual
max(0, p - q). The fault samples from p again. Draw 100,000 tokens with each
rule.

This is Ticket 2. Predict the frequency of A for each rule.
''', '''
rng = random.Random(0)
p, q = {'A': 0.6, 'B': 0.4}, {'A': 1.0, 'B': 0.0}
def draw(dist):
  return 'A' if rng.random() < dist['A'] else 'B'

for name, after_rejection in [('the residual', 'residual'), ('p again', 'p')]:   # 'p': THE FAULT
  count = 0
  for _ in range(100_000):
    if rng.random() < min(1.0, p['A'] / q['A']):
      token = 'A'
    elif after_rejection == 'p':
      token = draw(p)
    else:
      residual = {t: max(0.0, p[t] - q[t]) for t in p}
      total = sum(residual.values())
      token = draw({t: v / total for t, v in residual.items()})
    count += token == 'A'
  print(f'after a rejection, sample {name:12s}: A in {count / 100_000:.4f} of the draws (the target is 0.6)')
'''),

    ('d3', '''
# Exercise 3: dequantize, then multiply

One matrix of the size of the Llama-3-8B MLP (4,096 x 14,336), at batch 1.
Time three versions: bf16, int8 that is converted to bf16 before the matmul,
and an int8 GEMV that applies the scale at the end, in a register. The first
run compiles the kernel.

This is Ticket 3. Predict the time of each, against bf16.
''', '''
gemv = cudalib.build_source('inc7_gemv_int8', lab.GEMV_INT8)
W = torch.randn(4096, 14336, device=device, dtype=torch.bfloat16) * 0.02
x = torch.randn(1, 14336, device=device, dtype=torch.bfloat16)
q, scale = lab.quantize(W, per_channel=True)
scale = scale.flatten().contiguous()
y = torch.empty(4096, device=device, dtype=torch.bfloat16)
gemv.run(q, scale, x.flatten(), y)
print('the int8 GEMV agrees with the dequantized matmul:',
      torch.allclose(y.float(), (x.float() @ (q.float() * scale[:, None]).t()).flatten(), atol=0.05))

params = W.numel()
versions = [('bf16', lambda: x @ W.t()),
            ('dequantize, then matmul', lambda: x @ (q.to(torch.bfloat16) * scale[:, None].to(torch.bfloat16)).t()),  # THE FAULT
            ('int8 GEMV, scale at the end', lambda: gemv.run(q, scale, x.flatten(), y))]
base = None
for name, fn in versions:
  us = lab.ms_per_call(fn, 50) * 1000
  base = base or us
  print(f'{name:28s}: {us:7.1f} us, {base / us:4.2f}x bf16')
'''),

    ('d4', '''
# Exercise 4: one scale for a whole matrix

Quantize every linear layer of the model to int8, first with one scale for
each matrix, then with one scale for each output channel. Measure the KL of
the next-token distribution against bf16 on six prompts, and the fraction of
the weights that round to 0.

This is Ticket 4. Predict both KL values, and which layer loses the most
weights.
''', '''
PROMPTS = ['The three largest cities in Japan are', 'My favourite recipe for pancakes is',
           'The capital of France is', 'def fibonacci(n):\\n    if n <= 1:\\n        return',
           'Once upon a time, in a small village,', 'The chemical symbol for gold is']
reference = lab.log_probs(model, PROMPTS, tokenizer)
linears = [(name, m) for name, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and 'lm_head' not in name]
original = {name: m.weight.data.clone().cpu() for name, m in linears}

for per_channel in (False, True):                                      # False: THE FAULT
  zeros = []
  for name, m in linears:
    q, s = lab.quantize(m.weight.data, per_channel)
    zeros.append(((q == 0).float().mean().item(), name, m.weight.abs().max().item()))
    m.weight.data = (q.float() * s).to(torch.bfloat16)
  kl = lab.next_token_kl(model, PROMPTS, tokenizer, reference)
  worst = max(zeros)
  print(f'{"one scale per channel" if per_channel else "one scale per matrix "}: KL {kl:.4f} nats, '
        f'weights at 0: {100 * sum(z[0] for z in zeros) / len(zeros):.1f}% on average, '
        f'{100 * worst[0]:.1f}% in {worst[1]} (its largest weight {worst[2]:.2f})')
  for name, m in linears:
    m.weight.data = original[name].to(device)
'''),

    ('d5', '''
# Exercise 5: the scale of an FP8 KV cache

Fill the cache with a prompt, round every K and V to FP8 e4m3, and measure
the KL of the next token against the bf16 cache. First look at the largest
|K| of each layer, for this prompt and for a prompt of code. Then quantize
with a scale of 1, as the code of Ticket 5 does, with a scale for each layer
(the largest value / 448), and with a scale of 0.5: a scale that is too
small, as a calibration on the wrong data gives.

Predict the largest |K|, and the three KL values.
''', '''
ids = tokenizer(' '.join(PROMPTS), return_tensors='pt').input_ids.to(device)
code = tokenizer('def merge(a, b):\\n    out = []\\n    while a and b:\\n        out.append(a.pop(0))\\n' * 10,
                 return_tensors='pt').input_ids.to(device)
with torch.inference_mode():
  cache, code_cache = DynamicCache(), DynamicCache()
  model(ids[:, :-1], past_key_values=cache, use_cache=True)
  model(code, past_key_values=code_cache, use_cache=True)
keys = [layer.keys.clone() for layer in cache.layers]
values = [layer.values.clone() for layer in cache.layers]
print('the largest |K| of each layer:', [round(k.abs().max().item()) for k in keys])
print('the same for the code prompt :', [round(layer.keys.abs().max().item()) for layer in code_cache.layers][:6], '...')

def fp8(x, scale):
  return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)

@torch.inference_mode()
def next_logp(k_list, v_list):
  c = DynamicCache()
  for layer, (k, v) in enumerate(zip(k_list, v_list)):
    c.update(k, v, layer)
  return model(ids[:, -1:], past_key_values=c).logits[0, -1].float().log_softmax(-1)

good = next_logp(keys, values)
def kl(logp):
  return (good.exp() * (good - logp)).sum().item()

scale_of = lambda t: (t.abs().max().float() / 448).item()
for name, scale in [('a scale of 1', lambda t: 1.0),                       # THE FAULT of Ticket 5
                    ('a scale for each layer', scale_of),
                    ('a scale of 0.5, too small', lambda t: 0.5)]:
  clamped = sum((k.abs() / scale(k) > 448).sum().item() for k in keys)
  q_keys = [fp8(k, scale(k)) for k in keys]
  q_values = [fp8(v, scale(v)) for v in values]
  print(f'FP8 with {name:24s}: KL {kl(next_logp(q_keys, q_values)):.4f} nats, {clamped} key values clamped at 448')
'''),

    ('d6', '''
# Exercise 6: a valid prefix is not a valid answer

Guided decoding with an automaton for `{"skills": ["word", "word", ...]}`.
The mask allows only tokens that keep the text a valid prefix. Ask for a long
list with `max_tokens=30`. Then the same with a forced close: when the budget
is almost gone, the loop writes the characters that close the object.

This is Ticket 6. Predict the answer, and whether it parses, in each case.
The first run decodes every token of the vocabulary once.
''', '''
import json
texts = lab.token_texts(tokenizer)
lists = lab.Automaton('list')
masks = {}
VOCAB = model.config.vocab_size                                        # the logits are padded beyond len(tokenizer)
def padded(allowed):
  return torch.cat([allowed, torch.zeros(VOCAB - len(allowed), dtype=torch.bool)]).to(device)
def mask_for(automaton, state):
  if state not in masks:
    masks[state] = padded(lab.build_mask(automaton, state, texts))
  return masks[state]

CLOSE = {'word': '"]}', 'after': ']}', 'comma': ' "x"]}', 'next': '"x"]}', 'item': ']}', 'close': '}'}

@torch.inference_mode()
def guided(ids, automaton, max_tokens, force_close):
  state, text = automaton.start(), ''
  cache = DynamicCache()
  out = model(ids, past_key_values=cache, use_cache=True)
  for used in range(max_tokens):
    suffix = CLOSE.get(state[0], '')
    if force_close and len(tokenizer(suffix).input_ids) >= max_tokens - used:
      return text + suffix
    logits = out.logits[0, -1].masked_fill(~mask_for(automaton, state), float('-inf'))
    token = int(logits.argmax())
    text += texts[token]
    state = automaton.walk(state, texts[token])
    if automaton.finished(state):
      return text
    out = model(torch.tensor([[token]], device=device), past_key_values=cache, use_cache=True)
  return text                                                           # THE FAULT: cut at the limit

ask = chat('List twenty skills of a good chef, as JSON: {"skills": [...]}. Lowercase words only.')
for force in (False, True):
  answer = guided(ask, lists, 30, force)
  try:
    json.loads(answer)
    valid = 'parses'
  except ValueError:
    valid = 'does NOT parse'
  print(f'forced close {str(force):5s}: {valid:15s} {answer!r}')
'''),

    ('d7', '''
# Exercise 7: build the mask at every step

Time the build of one mask: one walk of the automaton over each of the
151,669 tokens. Then count how many different states the answer of
Exercise 6 visited.

This is Ticket 7. Predict the cost of one build, and what a cache of masks by
state saves over an answer of 30 tokens.
''', '''
start = time.perf_counter()
lab.build_mask(lists, ('word', 3), texts)                               # THE FAULT: once for every step
build = time.perf_counter() - start
print(f'one mask build: {build * 1000:.0f} ms, against a decode step of about 13 ms')
print(f'masks in the cache after Exercise 6: {len(masks)} different states')
print(f'30 tokens with a build at every step: {30 * build:.1f} s of CPU; with the cache: {len(masks) * build:.1f} s, once')
'''),

    ('d8', '''
# Exercise 8: check the first character only

An enum: `{"color": "green" | "red" | "blue"}`. Ask the color of an elephant.
The faulty mask tests only the **first character** of each token against the
automaton. Then the correct mask, which walks every character.

This is Ticket 8. Predict the answer of each.
''', '''
colors = lab.Automaton('enum', ['green', 'red', 'blue'])
ask = chat('What color is an elephant? Answer as JSON: {"color": ...}')

@torch.inference_mode()
def enum_answer(first_char_only, max_tokens=12):
  state, text, cache = colors.start(), '', DynamicCache()
  out = model(ask, past_key_values=cache, use_cache=True)
  for _ in range(max_tokens):
    allowed = padded(lab.build_mask(colors, state, texts, first_char_only=first_char_only))   # THE FAULT when True
    token = int(out.logits[0, -1].masked_fill(~allowed, float('-inf')).argmax())
    text += texts[token]
    state = colors.walk(state, texts[token])
    if state is None or colors.finished(state):
      break
    out = model(torch.tensor([[token]], device=device), past_key_values=cache, use_cache=True)
  return text, state

for first in (True, False):
  text, state = enum_answer(first)
  print(f'{"first character only" if first else "every character     "}: {text!r:28s} '
        f'{"VALID" if colors.finished(state) else "INVALID: the automaton rejects it"}')
'''),

    ('d9', '''
# Exercise 9: tensor parallel for a small model

Measure one all-reduce of one token's hidden state between two processes on
this machine. You have one GPU, so the two processes talk through the CPU
(gloo), not through NCCL. Then predict the decode step of Qwen3-0.6B and of
Llama-3-8B on two GPUs: half the weight read, plus 2 all-reduces for each
layer. Do it with your measured latency and with the 25 us of the reference
sheet.

This is Ticket 9. Predict whether two GPUs are faster.
''', '''
measured = lab.allreduce_latency_us(1024)                             # 1,024 values, as floats: 4 KB
print(f'one all-reduce between 2 CPU processes (gloo) here: {measured:.0f} us')
BANDWIDTH = 294e9
for label, latency in [('gloo on this CPU', measured), ('NCCL over PCIe, the reference sheet', 25)]:
  print(f'with {label} ({latency:.0f} us):')
  for name, weight_bytes, layers in [('Qwen3-0.6B', 1.5e9, 28), ('Llama-3-8B', 16.1e9, 32)]:
    one = weight_bytes / BANDWIDTH * 1000
    two = weight_bytes / 2 / BANDWIDTH * 1000 + layers * 2 * latency / 1000   # THE FAULT: shard a small model
    print(f'   {name}: 1 GPU {one:5.1f} ms, 2 GPUs {two:5.1f} ms -> {one / two:.2f}x')
'''),

    ('d10', '''
# Exercise 10: cut the fused QKV weight in the middle

Take the real q, k and v weights of layer 0 and fuse them into one matrix, as
vLLM does: 2,048 rows of Q, then 1,024 of K, then 1,024 of V. Shard it for
two ranks. Each rank reads its 2,048 rows with the shapes that the config
promises: 8 query heads (1,024 rows), then 4 KV heads of K (512 rows), then 4
of V (512 rows). Then run a simplified attention with the real weights (no
RoPE and no norm), sum the two ranks through their halves of `o_proj`, as the
all-reduce does, and compare with one rank.

This is Ticket 10. Predict what rank 0 reads as K, and both errors.
''', '''
attn = model.model.layers[0].self_attn
Wq, Wk, Wv, Wo = (attn.q_proj.weight.float(), attn.k_proj.weight.float(),
                  attn.v_proj.weight.float(), attn.o_proj.weight.float())
fused = torch.cat([Wq, Wk, Wv])                                        # 2,048 + 1,024 + 1,024 rows

def kind(row):
  return 'Q' if row < 2048 else 'K' if row < 3072 else 'V'

def attention(x, wq, wk, wv, heads, kv_heads):
  L = x.shape[0]
  q = (x @ wq.t()).view(L, heads, 128).transpose(0, 1)
  k = (x @ wk.t()).view(L, kv_heads, 128).transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
  v = (x @ wv.t()).view(L, kv_heads, 128).transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
  mask = torch.ones(L, L, device=x.device).triu(1).bool()
  scores = (q @ k.transpose(-1, -2) / math.sqrt(128)).masked_fill(mask, float('-inf'))
  return (scores.softmax(-1) @ v).transpose(0, 1).reshape(L, heads * 128)

x = torch.randn(12, Wq.shape[1], device=device) * 0.5
full = attention(x, Wq, Wk, Wv, 16, 8) @ Wo.t()

for name in ('chunk(2)', 'by heads'):
  total = torch.zeros_like(full)
  for rank in range(2):
    if name == 'chunk(2)':                                             # THE FAULT
      rows = list(range(rank * 2048, (rank + 1) * 2048))
    else:
      rows = (list(range(rank * 1024, (rank + 1) * 1024)) + list(range(2048 + rank * 512, 2048 + (rank + 1) * 512))
              + list(range(3072 + rank * 512, 3072 + (rank + 1) * 512)))
    shard = fused[rows]
    wq, wk, wv = shard[:1024], shard[1024:1536], shard[1536:]         # what the config promises
    reads = {part: sorted({kind(r) for r in rows[a:b]}) for part, (a, b) in
             {'as Q': (0, 1024), 'as K': (1024, 1536), 'as V': (1536, 2048)}.items()}
    print(f'{name} rank {rank} reads: {reads}')
    total += attention(x, wq, wk, wv, 8, 4) @ Wo[:, rank * 1024:(rank + 1) * 1024].t()   # the all-reduce
  print(f'   relative error against one rank: {((total - full).norm() / full.norm()).item():.2e}')
'''),
]

MYSTERY = '''
# Exercise 11: three mystery speculators

The module `mystery.py` holds three speculative decoders: `spec_a`, `spec_b`
and `spec_c`. Each has the signature of `lab.speculative` and returns the
tokens and the statistics. Each one has one fault. **Do not open the file.**

The cell below runs each one on a short question, and compares it with plain
greedy. All three agree.

For each speculator:

1. Design an experiment that makes the fault visible. Which variable do you
   change? The task? The length? What do you measure: the tokens, the speed,
   the statistics?
2. Write your diagnosis: the fault, and the experiment that proved it.
3. Only then, open `mystery.py` and check.

A hint about the method: two faults change the output, but only when a draft
is rejected after some of it was accepted. One fault changes nothing in the
output.
'''

MYSTERY_CODE = '''
from mystery import spec_a, spec_b, spec_c

ask = chat('Name one planet.')
plain = lab.greedy(model, ask, 20)
for name, spec in [('a', spec_a), ('b', spec_b), ('c', spec_c)]:
  tokens, stats = spec(model, ask, 20)
  print(f'spec_{name}: the same as greedy: {tokens == plain}, {stats}')
'''

MYSTERY_NAMES = ['spec_a', 'spec_b', 'spec_c']

FINGERPRINT_ROWS = ['speculation on the wrong task', 'after a rejection, sample from p again',
                    'dequantize, then multiply', 'one scale for a whole matrix',
                    'an FP8 KV cache with the wrong scale', 'a valid prefix is not a valid answer',
                    'build the mask at every step', 'check the first character only',
                    'tensor parallel for a small model', 'cut the fused QKV in the middle']

SOLUTIONS = {
    'd1': '''
### What happens

| run | drafts | accepted | tokens per step | speedup | the same as greedy |
|---|---|---|---|---|---|
| edit, n=3 | 32 | 59% | 1.83 | 1.69x | yes |
| poem, n=3 | 16 | 0% | 1.00 | 1.00x | yes |
| poem, n=1 | 187 | 3% | 1.04 | 0.99x | **no** |

- The edit repeats its input, and the drafts hit. The poem is new text, and
  they miss.
- **The surprise.** Wrong drafts cost almost nothing here: 187 drafts at 3%
  acceptance, and the speed did not move. At batch 1 on this card, a verify
  of 5 tokens reads the same weights as a step of 1 token, so it costs about
  the same. The loss of Ticket 1 appears when the step is closer to compute,
  at a larger batch or on a faster card.
- **The second surprise.** The eager drafter changed the greedy text. A
  verify of 5 tokens at once uses other kernel shapes, and in bf16 that
  changes the last bits, and near-ties flip (Part 1, Ticket 5). The promise
  "the same output" holds only in exact arithmetic.
''',
    'd2': '''
### What happens

- The residual: A in **0.6001** of 100,000 draws. The target is 0.6.
- p again: A in **0.8403**. The formula of Ticket 2 predicts 0.6 + 0.4 x 0.6
  = 0.84.

The draft token gets two chances: the acceptance, and the new sample after a
rejection. A test with a one-hot draft and a known target finds it in one
second.
''',
    'd3': '''
### What happens

On my card, one matrix of 4,096 x 14,336 at batch 1:

| version | time | against bf16 |
|---|---|---|
| bf16 | 281 us | 1.00x |
| dequantize, then matmul | 1,350 us | **0.21x** |
| int8 GEMV, scale at the end | 169 us | 1.67x |

The dequantized path is 4.8x **slower** than bf16. It runs three kernels:
the conversion (read 1 byte, write 2), the scale (read 2, write 2) and the
matmul (read 2). That is 9 bytes for each weight, against 2 for bf16. The
fused kernel reads 1 byte for each weight, and applies the scale to one
register value at the end.

The fused kernel reaches 1.67x, not 2x, because it also reads the input and
spends time on the conversion of each int8 value.
''',
    'd4': '''
### What happens

| scales | KL | weights at 0, on average | the worst layer |
|---|---|---|---|
| one for each matrix | 0.0340 nats | 5.9% | 15.9% of `layers.27.self_attn.k_proj`, whose largest weight is 1.60 |
| one for each output channel | 0.0031 nats | 1.4% | 1.9% |

One scale for each matrix is 11x worse. The worst layer is a `k_proj` with a
largest weight of 1.60: one large weight sets the step for the whole matrix.

This model suffers less than Ticket 4 describes: 0.034 nats is above the
target of 0.015 of stage 24, and far from a broken model. A model with larger
outliers suffers more. The measurement, not the model name, decides.
''',
    'd5': '''
### What happens

- The largest |K| is **366** in layer 0 for this prompt, and **402** for the
  code prompt. The other layers stay below 200. FP8 e4m3 ends at 448: Qwen3
  fits with a scale of 1, with 10% headroom. Layer 0 decides, because it holds
  the very large key of the first token (the attention sink).
- A scale of 1: KL 0.0113 nats, 0 values clamped.
- A scale for each layer: KL 0.0095 nats.
- A scale of 0.5: 61 key values clamped, KL 0.0153 nats.

**The honest result.** On this model, the scale matters little. The clamp of
Ticket 5 needs keys that are larger than 448, and Qwen3-1.7B does not have
them, today, on these prompts. That headroom of 10% is the danger: a new
checkpoint, or a new kind of prompt, can cross it with no warning. So the
guard of Ticket 5 still applies: calibrate the scale on the real traffic,
and count the clamped values at runtime.
''',
    'd6': '''
### What happens

- `max_tokens=30`, no forced close: `'{"skills": ["cleaning", "cooking",
  ..., "grilling", "measuring'`. Every token was legal, and the answer does
  **not** parse. The list was not finished when the budget ended.
- With a forced close: `'{"skills": [..., "grilling", "x"]}'`, which parses.
  The loop saw that the budget was almost gone, and it wrote the closing
  characters itself.

The forced close of this exercise is crude: after `, ` a list needs one more
string, so it writes `"x"`. A real engine closes with the shortest valid
suffix, and it chooses it before the model starts a new item.
''',
    'd7': '''
### What happens

- One mask build: **138 ms**, a walk of the automaton over 151,669 tokens in
  Python. A decode step takes about 13 ms. A build at every step makes each
  token 10x slower.
- The answer of Exercise 6 visited **7 states**. With a cache of masks by
  state, the answer builds 7 masks, once, and every later answer reuses them.
''',
    'd8': '''
### What happens

- First character only: `'{"color": "gray'`, and the automaton rejects it.
  The token `gray` starts with `g`, which `green` allows.
- Every character: `'{"color": "blue"}'`, valid. The model wanted a grey
  elephant, and the mask gave it the closest legal answer.

**The fingerprint.** The invalid answers always contain a token of several
characters whose first character was legal.
''',
    'd9': '''
### What happens

One all-reduce between 2 CPU processes (gloo) on my machine: **566 us**.

| latency | model | 1 GPU | 2 GPUs | speedup |
|---|---|---|---|---|
| gloo, 566 us | Qwen3-0.6B | 5.1 ms | 34.3 ms | 0.15x |
| gloo, 566 us | Llama-3-8B | 54.8 ms | 63.6 ms | 0.86x |
| NCCL over PCIe, 25 us | Qwen3-0.6B | 5.1 ms | 4.0 ms | 1.29x |
| NCCL over PCIe, 25 us | Llama-3-8B | 54.8 ms | 29.0 ms | 1.89x |

- With a slow link, even the 8B model gets slower. The all-reduces are
  latency, not bandwidth: 4 KB each, 56 or 64 times for each token.
- With 25 us, the 0.6B model **gains** 1.29x on this card. The step of this
  card is slow (294 GB/s), so half the weight read saves 2.55 ms, more than
  the 1.4 ms of the all-reduces. On the L40S of Ticket 9 (864 GB/s), the same
  model saves only 0.87 ms, and it loses.

The answer of Ticket 9 is right for its card and wrong for this one. The
formula, not the conclusion, is what travels.
''',
    'd10': '''
### What happens

| sharding | rank 0 reads as Q, K, V | rank 1 reads as Q, K, V | error |
|---|---|---|---|
| `chunk(2)` | Q, **Q**, **Q** | **K**, V, V | 1.10 |
| by heads | Q, K, V | Q, K, V | 3.1e-7 |

With `chunk(2)`, rank 0 gets all 2,048 rows of Q. It reads the first half as
its queries, which is correct, and the second half as its keys and values,
which are queries. Rank 1 reads the keys as queries. Every shape is correct,
and the output is wrong by 110%. Split by heads, the two ranks sum to one rank
within float32 rounding.

**The fingerprint.** Correct with tp = 1, wrong from token 0 with tp = 2, and
every shape passes its check.
''',
}

MYSTERY_SOL = '''
### The three mysteries

All three agree with greedy on "Name one planet.", because that answer has
no repeated 3-gram, so no draft is ever proposed. The experiment for all three
is the task where drafts are proposed and **partly** rejected: the edit of
Exercise 1. On my run, against plain greedy on the edit:

| | the same as greedy | first difference | tokens per step | speedup |
|---|---|---|---|---|
| `lab.speculative` | yes | | 1.83 | 1.75x |
| `spec_a` | no | token 11 | 1.41 | 0.50x |
| `spec_b` | no | token 12 | 2.00 | **2.20x** |
| `spec_c` | yes | | 1.83 | 0.82x |

**`spec_a` keeps the first rejected draft token.** The draft comes from the
prompt, so it proposes `cat` where the model wants `dog`. `spec_a` keeps
`cat`: the answer says `'The cat watched the birds'`. A drafter that copies
the input, and an off-by-one that keeps its first wrong token, together copy
the input into the output.

**`spec_b` does not crop the cache after a rejection.** The K and V of the
rejected drafts stay in the cache, at positions that the next tokens then
reuse. The answer skips words: `'The dog the garden, and the cat never
tried'`. And it is the **fastest** of the four, at 2.20x. A fault that makes
a benchmark better is the most dangerous kind, because nobody looks for it.

**`spec_c` verifies the drafts one token at a time.** Its output and its
statistics are identical to the correct speculator. Only the clock shows the
fault: 0.82x, slower than no speculation. The whole point of speculation is
one forward pass for k tokens.

### The method

| Mystery | What you measure |
|---|---|
| a | the output against greedy, on a task with partial rejections |
| b | the same, and do not trust the speedup |
| c | the time: the tokens and the statistics are all correct |
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| speculation on the wrong task | no | new text | 0% accepted, and the greedy text can change in bf16 | the acceptance rate per workload |
| after a rejection, sample from p again | no | at temperature > 0 | the draft token at 0.84 instead of 0.6 | a distribution test with a known target |
| dequantize, then multiply | no | always | 0.21x bf16 | the quantized path must beat bf16 at batch 1 |
| one scale for a whole matrix | no | layers with an outlier | 11x the KL of per-channel | KL and zeros for each layer |
| an FP8 KV cache with the wrong scale | no | keys above 448 | little on Qwen3, which has 10% headroom | calibrate; count the clamped values |
| a valid prefix is not a valid answer | no | long answers | JSON that ends in the middle | finish reason `length` for guided requests |
| build the mask at every step | no | always | 138 ms of CPU for each token | the mask time, and the cache hit rate |
| check the first character only | no | multi-character tokens | `gray` in an enum of `green` | walk every character |
| tensor parallel for a small model | no | slow links, fast cards | 2 GPUs slower than 1 | the formula before you shard |
| cut the fused QKV in the middle | no | tp = 2 | error 1.10 with every shape right | tp = 2 against tp = 1 in fp32 |

Three of these faults depend on the machine or the model: the cost of wrong
drafts, the FP8 headroom, and the speedup of tensor parallel. For those, a
conclusion from another card is not evidence. Measure on yours.
'''
