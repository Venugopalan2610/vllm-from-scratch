"""Part 5, break it on purpose: CUDA graphs, sampling and detokenization."""

PART = 5
NAME = 'Making It Fast'
FOLDER = 'course/Part5_MakingItFast/4_incidents'
IMPORTS = 'from transformers import AutoModelForCausalLM, AutoTokenizer'

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into a working graph, sampler or detokenizer, and you
watch what it does.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** Which token, which batch size, which request?
- **What?** What does the wrong result look like?
- **Which guard?** Which check would catch it?

`lab.Graphed(model, batch)` captures one decode step of an HF model as a CUDA
graph, with a static KV cache. Its static inputs are `g.ids` and
`g.position`, and it writes `g.logits`. `lab.greedy_graphed` is the correct
loop. The reference for a graph is the same static cache run eager, because
in bfloat16 a static cache and a growing cache can differ in the last bits.

The model is Qwen3-0.6B: it is small, so the CPU cost of the launches is easy
to see.

This notebook needs a GPU with about 4 GB free.
'''

LOAD = '''
### run this cell

MODEL = 'Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'
PROMPTS = ['The three largest cities in Japan are', 'My favourite recipe for pancakes is',
           'The capital of France is', 'Water boils at', 'The first person on the moon was']
g1 = lab.Graphed(model, 1)
print('graph captured for batch 1')
'''

DRILLS = [
    ('d1', '''
# Exercise 1: a new input tensor for each step

The loop makes a new tensor for the token at each step, and then replays the
graph. Run it for five prompts, and compare with the same loop without the
graph.

This is Ticket 1 of the incident file. Predict what the five answers look
like.
''', '''
@torch.inference_mode()
def fault_new_tensor(g, ids, max_tokens):
  token = g.prefill(ids)
  out = [int(token)]
  for i in range(max_tokens - 1):
    input_ids = torch.tensor([[int(token)]], device=device)            # THE FAULT: a new address
    g.position.fill_(ids.shape[1] + i)
    g.graph.replay()
    token = g.logits.argmax(-1)
    out.append(int(token))
  return out

for prompt in PROMPTS:
  ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  print(f'graph: {tokenizer.decode(fault_new_tensor(g1, ids, 15))!r}')
print('g1.ids after the loop:', g1.ids.tolist())
for prompt in PROMPTS:
  ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  print(f'eager: {tokenizer.decode(lab.greedy_eager_static(g1, ids, 15)[0])!r}')
'''),

    ('d2', '''
# Exercise 2: a padding row that writes to slot 0

A toy engine step: each row writes its K into a flat cache at its slot. The
step is captured as a graph for a bucket of 4 rows. A batch of 3 real rows
is padded to 4, and the padding row gets the slot 0. Slot 0 belongs to
request X, which is in the middle of its answer.

This is Ticket 2. Predict what happens to request X, and what changes with a
scratch slot for the padding.
''', '''
SLOTS, DIM, BUCKET = 64, 8, 4
cache = torch.zeros(SLOTS + 1, DIM, device=device)                      # the last slot is scratch
cache[0] = 7.0                                                          # request X owns slot 0
static_slots = torch.zeros(BUCKET, dtype=torch.long, device=device)
static_k = torch.zeros(BUCKET, DIM, device=device)

def step():
  cache.index_copy_(0, static_slots, static_k)
step()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
  step()

for padding_slot, name in [(0, 'padding to slot 0'), (SLOTS, 'padding to the scratch slot')]:
  cache[0] = 7.0
  real_slots, real_k = [10, 11, 12], torch.randn(3, DIM, device=device)
  static_slots.copy_(torch.tensor(real_slots + [padding_slot], device=device))   # THE FAULT when 0
  static_k.copy_(torch.cat([real_k, torch.zeros(1, DIM, device=device)]))
  graph.replay()
  torch.cuda.synchronize()
  print(f'{name:28s}: slot 0 of request X = {cache[0].tolist()[:3]}...   the real rows written: '
        f'{torch.equal(cache[10:13], real_k)}')
'''),

    ('d3', '''
# Exercise 3: graphs at batch 1 and at a large batch

Time one decode step, eager and as a graph, at several batch sizes.

This is Ticket 3. Predict the speedup of the graph at each batch size.
''', '''
import gc
for batch in (1, 4, 16, 64, 128):
  g = lab.Graphed(model, batch, max_len=128)
  g.prefill(torch.randint(0, 1000, (batch, 16), device=device))    # each timed call appends one token
  g.position.fill_(16)
  eager = lab.ms_per_call(g.eager_step)
  graphed = lab.ms_per_call(g.graph.replay)
  print(f'batch {batch:3d}: eager {eager:5.1f} ms, graph {graphed:5.1f} ms, speedup {eager / graphed:.2f}x')
  del g
  gc.collect()
  torch.cuda.empty_cache()
'''),

    ('d4', '''
# Exercise 4: a Python loop over the requests

Sample one token for each row of a batch of logits, with a temperature and a
top-p for each row. The fault: a loop over the rows, with `.item()` for each
one. The reference: `lab.sample`, one vectorized pass. A third version,
`lab.sample_top_candidates`, sorts only the 1,024 most likely tokens of each
row.

This is Ticket 4. Predict the ms of each version at batch 8, 64 and 128.
''', '''
VOCAB = model.config.vocab_size

def loop_sample(logits, temperature, top_p):
  out = []
  for row in range(logits.shape[0]):                                    # THE FAULT: one row at a time
    probs = torch.softmax(logits[row].float() / temperature[row], -1)
    sorted_probs, order = probs.sort(descending=True)
    before = sorted_probs.cumsum(-1) - sorted_probs
    sorted_probs[before > top_p[row]] = 0
    out.append(order[torch.multinomial(sorted_probs, 1)].item())
  return out

with torch.inference_mode():                                            # real next-token logits
  real = torch.cat([model(tokenizer(p, return_tensors='pt').input_ids.to(device)).logits[:, -1] for p in PROMPTS])
for batch in (8, 64, 128):
  logits = real.repeat(math.ceil(batch / len(PROMPTS)), 1)[:batch].float()
  temperature = torch.full((batch,), 0.8, device=device)
  top_p = torch.full((batch,), 0.9, device=device)
  generators = [torch.Generator(device=device).manual_seed(i) for i in range(batch)]
  slow = lab.ms_per_call(lambda: loop_sample(logits, temperature, top_p), iters=10)
  fast = lab.ms_per_call(lambda: lab.sample(logits, temperature, top_p, generators).tolist(), iters=10)
  top = lab.ms_per_call(lambda: lab.sample_top_candidates(logits, temperature, top_p, generators)[0].tolist(), iters=10)
  exact = lab.sample_top_candidates(logits, temperature, top_p, generators)[1].all().item()
  sort = lab.ms_per_call(lambda: logits.sort(-1, descending=True), iters=10)
  print(f'batch {batch:3d}: loop {slow:5.1f} ms ({slow / batch:.2f} per request), vectorized {fast:5.1f} ms '
        f'(of which the sort {sort:4.1f}), top 1,024 only {top:4.1f} ms (exact: {exact})')
'''),

    ('d5', '''
# Exercise 5: top-p removes the token that crosses the line

The top-p mask removes every token whose **cumulative** probability is above
`top_p`, which includes the token that crosses the line. Sample 30 tokens
with `top_p=0.5` for prompts where the model is sure, and for one where it is
not.

This is Ticket 5. Predict the text for each prompt.
''', '''
def fault_top_p(logits, top_p, generator):
  probs = torch.softmax(logits.float(), -1)
  sorted_probs, order = probs.sort(descending=True)
  cumulative = sorted_probs.cumsum(-1)
  sorted_probs[cumulative > top_p] = 0                                  # THE FAULT
  probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
  return int((probs / torch.empty_like(probs).exponential_(generator=generator)).argmax())

@torch.inference_mode()
def generate(prompt, top_p, tokens=30):
  generator = torch.Generator(device=device).manual_seed(0)
  ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  out = model(ids, use_cache=True)
  cache, text, sure = out.past_key_values, [], 0
  for _ in range(tokens):
    logits = out.logits[0, -1]
    sure += int(torch.softmax(logits.float(), -1).max() > top_p)
    token = fault_top_p(logits, top_p, generator)
    text.append(token)
    out = model(torch.tensor([[token]], device=device), past_key_values=cache, use_cache=True)
    cache = out.past_key_values
  return tokenizer.decode(text), sure

for prompt in ['The capital of France is', 'def fibonacci(n):\\n    if n <= 1:\\n        return',
               'Once upon a time']:
  text, sure = generate(prompt, 0.5)
  print(f'{prompt[:24]!r:28s} steps with top probability > 0.5: {sure:2d}/30   {text!r}')
'''),

    ('d6', '''
# Exercise 6: a seed in the global generator

A request asks for `seed=42`. The sampler seeds the global generator with it,
and then samples the whole batch. Run the request alone, and then in two
batches with other requests. Then do the same with one generator for each
request.

This is Ticket 6. Predict which runs agree.
''', '''
torch.manual_seed(0)
logits = torch.randn(6, 1000, device=device) * 2

def global_seed(rows, seeds):
  for seed in seeds:
    if seed is not None:
      torch.manual_seed(seed)                                           # THE FAULT: one global generator
  probs = torch.softmax(logits[rows], -1)
  return torch.multinomial(probs, 5, replacement=True)[0].tolist()     # the draws of the first row

def own_generator(rows, seeds):
  draws = []
  for row, seed in zip(rows, seeds):
    generator = torch.Generator(device=device).manual_seed(seed if seed is not None else 1000 + row)
    probs = torch.softmax(logits[row], -1)
    draws.append(torch.multinomial(probs, 5, replacement=True, generator=generator).tolist())
  return draws[0]

for name, sampler in [('global generator', global_seed), ('one generator for each request', own_generator)]:
  alone = sampler([0], [42])
  batch_a = sampler([0, 1, 2], [42, None, 7])
  batch_b = sampler([0, 3, 4, 5], [42, 13, None, None])
  print(f'{name}: alone {alone}, batch A {batch_a}, batch B {batch_b}')
'''),

    ('d7', '''
# Exercise 7: decode one token at a time

Stream a text with two rare characters by decoding each token alone. Then
stream it with `lab.Detokenizer`.

This is Ticket 7. Predict what the stream shows for 🫠 and for 𝄞.
''', '''
text = 'Melting 🫠 and a clef 𝄞 in one line.'
tokens = tokenizer(text).input_ids
print('the tokens:', [tokenizer.decode([t]) for t in tokens])
one_at_a_time = ''.join(tokenizer.decode([t]) for t in tokens)          # THE FAULT
detokenizer = lab.Detokenizer(tokenizer)
incremental = ''.join(detokenizer.push(t) for t in tokens)
print('one token at a time:', repr(one_at_a_time))
print('incremental        :', repr(incremental))
print('the same as the full decode:', incremental == tokenizer.decode(tokens))
'''),

    ('d8', '''
# Exercise 8: the same stream with another tokenizer

Stream the same way with the tokenizer of Llama-2, a SentencePiece
tokenizer. Then with `lab.Detokenizer`.

This is Ticket 8. Predict the spaces in each stream.
''', '''
llama = AutoTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
text = 'The capital of France is Paris.'
tokens = llama(text, add_special_tokens=False).input_ids
print('the pieces:', llama.convert_ids_to_tokens(tokens))
one_at_a_time = ''.join(llama.decode([t]) for t in tokens)             # THE FAULT
detokenizer = lab.Detokenizer(llama)
incremental = ''.join(detokenizer.push(t) for t in tokens)
for name, stream in [('one token at a time', one_at_a_time), ('incremental', incremental)]:
  print(f'{name:20s}: {stream!r}, {stream.count(" ")} spaces')
qwen_stream = ''.join(tokenizer.decode([t]) for t in tokenizer(text).input_ids)
print(f'{"Qwen3, one at a time":20s}: {qwen_stream!r}, {qwen_stream.count(" ")} spaces')
'''),

    ('d9', '''
# Exercise 9: a stop string across two tokens

The stop string is `###`. The model writes it in two ways: as the one token
`###` (14374), or as `##` (565) and then `#` (2). The check looks for the
stop string in the text of the **new token** only.

This is Ticket 9. Predict which stream stops, and what the user sees.
''', '''
head = tokenizer('The answer is 42.\\n').input_ids
tail = tokenizer(' Next section: more text').input_ids
streams = {'one token': head + [14374] + tail, 'two tokens': head + [565, 2] + tail}
STOP = '###'

def per_token(tokens):
  shown = ''
  for t in tokens:
    piece = tokenizer.decode([t])
    if STOP in piece:                                                   # THE FAULT: one token only
      return shown, 'stopped'
    shown += piece
  return shown, 'did not stop'

def on_the_text(tokens):
  text = ''
  for t in tokens:
    text += tokenizer.decode([t])
    if STOP in text:
      return text[:text.index(STOP)], 'stopped'
  return text, 'did not stop'

for name, tokens in streams.items():
  for check in (per_token, on_the_text):
    shown, result = check(tokens)
    print(f'{name:10s} {check.__name__:12s}: {result:12s} the user sees {shown!r}')
'''),
]

MYSTERY = '''
# Exercise 10: three mystery samplers

The module `mystery.py` holds three samplers: `sampler_a`, `sampler_b` and
`sampler_c`. Each takes `(logits, params, generator)`, where `params` is a
list with one dict for each row: `temperature`, `top_k` (0 means off), `top_p`
and `penalty` with the list `seen` of the tokens that the row already wrote.
Each one has one fault. **Do not open the file.**

`lab.frequencies(sampler, logits, params)` draws many samples and returns the
frequency of each token for each row. `lab.target(logits[row], **param)` is
the exact distribution of a row. A correct sampler matches the target within
the noise of the draws.

The cell below runs each sampler on a small vocabulary with the same simple
parameters for every row. All three pass.

For each sampler:

1. Design a test that makes the fault visible. Which parameter do you change?
   For which row?
2. Write your diagnosis: the fault, and the experiment that proved it.
3. Only then, open `mystery.py` and check.

A hint about the method: one fault needs rows with different parameters. One
fault needs a special value of one parameter. One fault needs a token whose
logit is negative.
'''

MYSTERY_CODE = '''
from mystery import sampler_a, sampler_b, sampler_c

torch.manual_seed(1)
small = torch.randn(4, 8, device=device) * 1.5
params = [dict(temperature=0.8, top_k=5, top_p=1.0, penalty=1.0, seen=[]) for _ in range(4)]
targets = torch.stack([lab.target(small[r], **params[r]) for r in range(4)])
for name, sampler in [('a', sampler_a), ('b', sampler_b), ('c', sampler_c)]:
  measured = lab.frequencies(sampler, small, params, draws=5000)
  print(f'sampler_{name}: largest difference from the target {(measured - targets).abs().max().item():.3f}')
'''

MYSTERY_NAMES = ['sampler_a', 'sampler_b', 'sampler_c']

FINGERPRINT_ROWS = ['a new input tensor for each step', 'a padding row that writes to slot 0',
                    'a graph on a GPU-bound step', 'a Python loop over the requests',
                    'top-p that removes the crossing token', 'a seed in the global generator',
                    'decode one token at a time (bytes)', 'decode one token at a time (spaces)',
                    'a stop string checked in one token']

SOLUTIONS = {
    'd1': '''
### What happens

- **Crash?** No.
- **When?** Token 0 is correct in all five answers: it comes from the eager
  prefill. From token 1, the first token of the graph, the answers fall apart:
  `' Tokyo A A A (!!!!!!!!!!'`, `' Paris A A A ( (!!!!!!!!!'`.
- **What?** The graph read `g1.ids`, which still held `[[0]]` after the loop.
  Token 0 is `!`. So the graph fed `!` at every step, and the model wrote what
  follows a run of `!`. The answers differ a little at first, because each
  cache holds its own prompt, and all of them end in `!`.

The same loop that copies into `g1.ids` gives the correct answers, below the
broken ones.

**The fingerprint.** Token 0 right, then text that ignores the prompt.
**The guard.** Graph against eager for 20 tokens, and the `data_ptr()` of
each graph input at capture and at replay.
''',
    'd2': '''
### What happens

- **Padding to slot 0:** slot 0 of request X changed from 7.0 to 0.0. The
  three real rows were written correctly, so every check of the current batch
  passes. The damage is in a request that is **not in the batch**.
- **Padding to the scratch slot:** slot 0 stays 7.0.

In a real engine, request X then attends to a K of zeros at position 0, and
its answer drifts, a few tokens later, with no error.

**The fingerprint.** A victim that was not in the damaged step, and damage
only in steps whose batch is not a bucket size. **The guard.** Fill a block
with a known pattern, replay a padded step, and check the pattern.
''',
    'd3': '''
### What happens

On my card, Qwen3-0.6B:

| batch | eager | graph | speedup |
|---|---|---|---|
| 1 | 10.5 ms | 5.4 ms | 1.96x |
| 4 | 11.0 ms | 6.3 ms | 1.74x |
| 16 | 10.9 ms | 8.6 ms | 1.27x |
| 64 | 20.7 ms | 20.2 ms | 1.03x |
| 128 | 36.9 ms | 36.2 ms | 1.02x |

The eager step stays at about 11 ms from batch 1 to batch 16. That is the time
that the CPU needs to launch the kernels of one step, and it does not depend
on the batch. The graph shows the GPU time, which grows with the batch. At
batch 64 the GPU needs 20 ms, more than the CPU, and the graph has nothing
left to remove.

With Qwen3-1.7B at batch 1, the graph gave only 1.07x (11.8 ms against 12.6
ms): the GPU reads 3.4 GB of weights, and that takes longer than the launches.
A graph helps when the CPU is the slower side, and only then.
''',
    'd4': '''
### What happens

On my card, with real next-token logits:

| batch | loop | vectorized (its sort) | top 1,024 only |
|---|---|---|---|
| 8 | 1.8 ms | 0.6 ms (0.2) | 0.4 ms |
| 64 | 12.4 ms | 7.0 ms (4.1) | 0.9 ms |
| 128 | 19.6 ms | 16.5 ms (10.0) | 2.4 ms |

The loop costs about 0.15 to 0.23 ms for each request, a constant: the
fingerprint of Ticket 4.

**The surprise.** The vectorized pass is only 1.2x faster at batch 128. It
sorts 128 x 151,936 values, and that sort takes 10 ms. Removing the loop was
not enough: the next bottleneck was the algorithm. Sorting only the 1,024 most
likely tokens of each row gives 2.4 ms, and it was exact here, because those
tokens held the top-p mass of every row. Real engines use the same idea, or a
sampler that needs no sort at all.

**The lesson.** Time the parts of a fix before you call it a fix.
''',
    'd5': '''
### What happens

With `top_p=0.5`:

| prompt | steps where the top token had more than 0.5 | the text |
|---|---|---|
| The capital of France is | 17 of 30 | `'! 10,0! 0!!!!! 1!!! 1!!!!!!!!!'` |
| def fibonacci | 11 of 30 | `" n!\\n! I think I'm making! mistake! ..."` |
| Once upon a time | 10 of 30 | `'! The language! The culture!! ...'` |

Every step where the model is sure removes all the tokens, and the sampler
returns token 0, `!`. The more predictable the text, the more `!`. The first
token of the France prompt, `' Paris'`, was the most certain token of the run,
and it became `!`.

**The fingerprint.** `!` exactly where the model is most sure. **The guard.**
Assert that each row keeps at least one token.
''',
    'd6': '''
### What happens

| sampler | alone | batch A | batch B |
|---|---|---|---|
| global generator | 600, 11, 441, 42, 223 | 971, 521, 971, 553, 738 | 518, 943, 841, 527, 738 |
| one generator for each request | 600, 11, 441, 42, 223 | the same | the same |

With the global generator, the request with `seed=42` gets its own draws only
when it is alone. In batch A, another request seeded the generator with 7
after it. In batch B, with 13. The seed of one request controls nothing when
another request can seed after it.

With one generator for each request, all three runs agree.
''',
    'd7': '''
### What happens

- 🫠 and 𝄞 are 3 tokens each: `' �', '�', '�'`.
- One token at a time: `'Melting ��� and a clef ��� in one line.'` Three
  replacement characters for each rare character, one for each token.
- `lab.Detokenizer`: `'Melting 🫠 and a clef 𝄞 in one line.'`, the same
  as the full decode. It holds back the text while it ends in `�`.
''',
    'd8': '''
### What happens

- The Llama-2 pieces: `['▁The', '▁capital', '▁of', '▁France', '▁is',
  '▁Paris', '.']`.
- One token at a time: `'ThecapitalofFranceisParis.'`, 0 spaces.
- `lab.Detokenizer`: `'The capital of France is Paris.'`, 5 spaces.
- Qwen3, one token at a time: 5 spaces, correct. Its tokens carry the space
  as a byte, so the same bug does not show.

The bug is in the stream code, and the tokenizer decides whether you see it.
A test with one tokenizer cannot pass for another.
''',
    'd9': '''
### What happens

| the model writes `###` as | the check on the new token | the check on the text |
|---|---|---|
| one token (14374) | stopped | stopped |
| two tokens (565, 2) | **did not stop**: the user sees `'...\\n### Next section: more text'` | stopped |

The check on the text stops in both cases. It still has a smaller fault: it
only knows the stop string after the `#` arrives, so a stream that sends each
piece at once has already sent `##`. The fix is to hold back the end of the
text while it could be the start of a stop string.
''',
}

MYSTERY_SOL = '''
### The three mysteries

**`sampler_a`: one temperature for the whole batch.** It uses the temperature
of row 0 for every row. When all rows have the same temperature, it is
correct. The experiment: give the rows different temperatures, for example
0.3, 1.0, 2.0 and 0.8. On my run, the largest difference from the target was
0.004 for row 0 and 0.43, 0.54 and 0.22 for the others.

**`sampler_b`: the penalty divides every seen logit.** For a positive logit,
dividing lowers it, which is correct. For a negative logit, dividing brings it
closer to 0, which **raises** it. The experiment: a seen token with a negative
logit, and a large penalty. On my run, a token with the logit -0.44 and a
penalty of 2.0 had the probability 0.007 without a penalty, the target 0.004
with the penalty, and `sampler_b` gave it 0.008. The penalty made the repeated
token more likely.

**`sampler_c`: `top_k=0` keeps no token.** The API says that 0 means "off".
The code passes 0 to the filter, which then keeps nothing, and the row
returns token 0 every time. The experiment: `top_k=0` for one row. On my run
that row drew token 0 in 100% of the draws. This is Exercise 5 in another
form: an empty set, and `argmax` of zeros.

### The method

| Mystery | The variable to change |
|---|---|
| a | different parameters in different rows |
| b | a seen token with a negative logit |
| c | the special value of a parameter: 0 for "off" |

A distribution test catches all three, but only with parameters that reach the
fault. A test with the same simple parameters on every row passed all three.
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| a new input tensor for each step | no | token 1 | text that ignores the prompt, then `!` | graph against eager; input addresses |
| a padding row that writes to slot 0 | no | batches that are not a bucket | a request outside the batch drifts | a pattern in a block, replay, check |
| a graph on a GPU-bound step | no | large batch or large model | a speedup near 1x | CPU time and GPU time of a step |
| a Python loop over the requests | no | large batch | 0.15 ms or more for each request | the sampler time at batch 128 |
| top-p that removes the crossing token | no | when the model is sure | `!` exactly at the certain tokens | at least one token in each row |
| a seed in the global generator | no | in a batch with other seeds | a seeded request that changes | the same seed in two batches |
| decode one token at a time (bytes) | no | rare characters | `���` in the stream only | stream against the full decode |
| decode one token at a time (spaces) | no | SentencePiece tokenizers | words with no spaces | the same test for each tokenizer |
| a stop string checked in one token | no | a stop string across two tokens | the stop string, then more text | a stop string split by force |

Exercise 4 found a fault that the ticket did not name: after the loop, the
sort. Every fix has a next bottleneck, and you only see it with a timer.
'''
