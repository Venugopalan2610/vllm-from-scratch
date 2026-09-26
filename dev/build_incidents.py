"""Build the Part 1 notebook "break it on purpose", and its solution.

The incident file of every Part comes from its tickets.yaml. See
dev/build_incident_files.py.
"""
import json
from pathlib import Path

OUT = Path('/home/venugopalan/vllm-from-scratch/course/Part1_TheNaiveLoop/4_incidents')
(OUT / 'solutions').mkdir(parents=True, exist_ok=True)


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')}


def code(text=''):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.strip('\n')}


def header(lecture):
    return md(f'''
|<h2>Course:</h2>|<h1><a href="https://derivingsystems.com/course.html" target="_blank">Build your own vLLM: inference engines from the memory system up</a></h1>|
|-|:-:|
|<h2>Part 1:</h2>|<h1>The Naive Loop<h1>|
|<h2>Section:</h2>|<h1>Incidents<h1>|
|<h2>Lecture:</h2>|<h1><b>{lecture}<b></h1>|

<br>

<h5><b>Course repo:</b> <a href="https://github.com/Venugopalan2610/vllm-from-scratch" target="_blank">github.com/Venugopalan2610/vllm-from-scratch</a></h5>
<h5><b>The derivations:</b> <a href="https://derivingsystems.com" target="_blank">derivingsystems.com</a></h5>
<i>The notebooks build the intuition. The ladder in app/ makes you build the thing.</i>
''')


def save(name, cells):
    nb = {'cells': [dict(cell) for cell in cells],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    for cell in nb['cells']:
        lines = cell['source'].split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# Notebook 2: break it on purpose. GPU.
# ---------------------------------------------------------------------------

SETUP = code('''
# find the repo root. The directory you start from does not matter.
import sys
from pathlib import Path
ROOT = next(folder for folder in [Path.cwd(), *Path.cwd().parents]
            if (folder/'cudalib').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'course'/'Part1_TheNaiveLoop'/'4_incidents'))

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# your own stage 02 is the reference. It passed its checks.
from app.s02_cache import cached_generate as reference
''')

LOAD = code('''
### run this cell

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'

PROMPT = 'The three largest cities in Japan are'
TOKENS = 30

def report(name, generated, expected):
  """Print the first token that differs from `expected`, and the text."""
  first = next((position for position, (got, want) in enumerate(zip(generated, expected))
                if got != want), None)
  if first is None and len(generated) != len(expected):
    first = min(len(generated), len(expected))
  print(f'{name}: {len(generated)} tokens, first difference at: {first}')
  print('  ', repr(tokenizer.decode(generated)))

expected = reference(model, tokenizer, PROMPT, TOKENS)
report('reference', expected, expected)
''')

INTRO_2 = md('''
In the incident file you went from a symptom to a cause. Here you go the other
way: from a cause to a symptom. You put one fault into a working loop, and you
watch what it does.

This builds a table in your head: **this bug looks like this**. After this
notebook, when you see "fluent text that repeats the prompt", you know where to
look first.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** At which token does the output first differ from the reference?
- **What?** What does the wrong text look like: nonsense, repetition, fluent
  but wrong, or too long?
- **Which guard?** Which check of stages 01 to 03 catches it? If none: what
  check would?

The reference is your own `cached_generate` from stage 02.
''')

PREDICT = md('''
**Your prediction** (write it before you run the cell)

- Crash?
- When?
- What?
- Which guard?

**What happened, and where you were wrong:**
''')

DRILLS = [
    ('d1', md('''
# Exercise 1: feed the whole sequence on each decode step

The loop keeps the cache. But on each decode step it feeds the whole sequence,
not only the new token. The cache already holds the prefix, so the cache
appends the prefix again.

The Exercise 2 text of `part1_kv_3_CCwriteBothLoops` warned about this trap.
'''), code('''
@torch.inference_mode()
def fault_whole_sequence(prompt, max_tokens):
  token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  output = model(token_ids, use_cache=True)
  cache = output.past_key_values
  next_token = output.logits[:, -1:].argmax(-1)
  token_ids = torch.cat([token_ids, next_token], dim=1)
  generated = [next_token.item()]
  for _ in range(max_tokens - 1):
    output = model(token_ids, past_key_values=cache, use_cache=True)    # THE FAULT
    cache = output.past_key_values
    next_token = output.logits[:, -1:].argmax(-1)
    token_ids = torch.cat([token_ids, next_token], dim=1)
    generated.append(next_token.item())
  print(f'cache length {cache.get_seq_length()}, real sequence length {token_ids.shape[1]}')
  return generated

report('whole sequence', fault_whole_sequence(PROMPT, TOKENS), expected)
''')),

    ('d2', md('''
# Exercise 2: the position never moves

Each decode step tells the model that the new token is at position `L`, the
length of the prompt. The position does not advance. RoPE uses the position to
rotate Q and K, so every new token gets the same rotation.

Your stage 02 docstring says that HF finds the position from the length of the
cache. Here the code overrides it.
'''), code('''
@torch.inference_mode()
def fault_stuck_position(prompt, max_tokens):
  token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  prompt_length = token_ids.shape[1]
  output = model(token_ids, use_cache=True)
  cache = output.past_key_values
  next_token = output.logits[:, -1:].argmax(-1)
  generated = [next_token.item()]
  for _ in range(max_tokens - 1):
    output = model(next_token, past_key_values=cache, use_cache=True,
                   position_ids=torch.tensor([[prompt_length]], device=device))   # THE FAULT
    cache = output.past_key_values
    next_token = output.logits[:, -1:].argmax(-1)
    generated.append(next_token.item())
  return generated

report('stuck position', fault_stuck_position(PROMPT, TOKENS), expected)
''')),

    ('d3', md('''
# Exercise 3: the logits of the wrong row

After prefill, the code reads the logits of row 0, not of the last row. Row 0
is the prediction after the first prompt token only. The decode steps are
correct.
'''), code('''
@torch.inference_mode()
def fault_first_row(prompt, max_tokens):
  token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  output = model(token_ids, use_cache=True)
  cache = output.past_key_values
  next_token = output.logits[:, :1].argmax(-1)          # THE FAULT: row 0, not row -1
  generated = [next_token.item()]
  for _ in range(max_tokens - 1):
    output = model(next_token, past_key_values=cache, use_cache=True)
    cache = output.past_key_values
    next_token = output.logits[:, -1:].argmax(-1)
    generated.append(next_token.item())
  return generated

report('first row', fault_first_row(PROMPT, TOKENS), expected)
''')),

    ('d4', md('''
# Exercise 4: feed the token of the step before

A refactor moved one line. Now each decode step feeds the token of the step
before, not the token that the model just chose. The cache and the positions
are correct.
'''), code('''
@torch.inference_mode()
def fault_previous_token(prompt, max_tokens):
  token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  output = model(token_ids, use_cache=True)
  cache = output.past_key_values
  next_token = output.logits[:, -1:].argmax(-1)
  generated = [next_token.item()]
  previous = next_token
  for _ in range(max_tokens - 1):
    output = model(previous, past_key_values=cache, use_cache=True)   # THE FAULT
    cache = output.past_key_values
    previous = next_token
    next_token = output.logits[:, -1:].argmax(-1)
    generated.append(next_token.item())
  return generated

report('previous token', fault_previous_token(PROMPT, TOKENS), expected)
''')),

    ('d5', md('''
# Exercise 5: one cache for everyone

The cache is global. Request A runs first. Then request B runs, on the same
cache. Compare request B with request B run alone.

This is Ticket 8 of the incident file. Predict **how soon** B goes wrong.
'''), code('''
shared_cache = DynamicCache()

@torch.inference_mode()
def fault_shared_cache(prompt, max_tokens):
  token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  output = model(token_ids, past_key_values=shared_cache, use_cache=True)   # THE FAULT
  next_token = output.logits[:, -1:].argmax(-1)
  generated = [next_token.item()]
  for _ in range(max_tokens - 1):
    output = model(next_token, past_key_values=shared_cache, use_cache=True)
    next_token = output.logits[:, -1:].argmax(-1)
    generated.append(next_token.item())
  return generated

PROMPT_B = 'My favourite recipe for pancakes is'
expected_b = reference(model, tokenizer, PROMPT_B, TOKENS)

report('request A, shared cache', fault_shared_cache(PROMPT, TOKENS), expected)
report('request B, shared cache', fault_shared_cache(PROMPT_B, TOKENS), expected_b)
report('request B, alone       ', expected_b, expected_b)
print('shared cache length:', shared_cache.get_seq_length())
''')),

    ('d6', md('''
# Exercise 6: the stop check that never fires

The loop compares the new token with `eos_token_id` using `==`. For Qwen3,
`eos_token_id` is a list. The prompt is a chat turn that asks for a short
answer, so a correct loop stops early.

This is Ticket 1. Predict the length of the output, and what the text looks
like after the answer ends.
'''), code('''
chat_prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'Say hi in two words.'}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
print('eos_token_id:', model.generation_config.eos_token_id)

@torch.inference_mode()
def fault_stop_check(prompt, max_tokens):
  stop = model.generation_config.eos_token_id
  token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  output = model(token_ids, use_cache=True)
  cache = output.past_key_values
  next_token = int(output.logits[0, -1].argmax())
  generated = []
  for _ in range(max_tokens):
    if next_token == stop:                               # THE FAULT
      break
    generated.append(next_token)
    output = model(torch.tensor([[next_token]], device=device),
                   past_key_values=cache, use_cache=True)
    cache = output.past_key_values
    next_token = int(output.logits[0, -1].argmax())
  return generated

correct = reference(model, tokenizer, chat_prompt, 60)
report('stop check with ==', fault_stop_check(chat_prompt, 60), correct)
report('reference         ', correct, correct)
''')),

    ('d7', md('''
# Exercise 7: the silent truncation

The tokenizer call has `truncation=True, max_length=8`. The prompt is longer
than 8 tokens.

This is Ticket 6. Predict what the model answers.
'''), code('''
QUESTION = 'Answer in one word. What is the capital of the country whose largest city is Osaka?'

@torch.inference_mode()
def fault_truncation(prompt, max_tokens):
  token_ids = tokenizer(prompt, return_tensors='pt', truncation=True,
                        max_length=8).input_ids.to(device)      # THE FAULT
  print('the model sees:', repr(tokenizer.decode(token_ids[0])))
  output = model(token_ids, use_cache=True)
  cache = output.past_key_values
  next_token = output.logits[:, -1:].argmax(-1)
  generated = [next_token.item()]
  for _ in range(max_tokens - 1):
    output = model(next_token, past_key_values=cache, use_cache=True)
    cache = output.past_key_values
    next_token = output.logits[:, -1:].argmax(-1)
    generated.append(next_token.item())
  return generated

full = reference(model, tokenizer, QUESTION, 15)
report('truncated', fault_truncation(QUESTION, 15), full)
report('full     ', full, full)
''')),

    ('d8', md('''
# Exercise 8: time it without synchronize

Time one prefill of 2048 tokens, with and without
`torch.cuda.synchronize()`. Then do it again with more iterations. Convert
each time into TFLOP/s, and compare with the sustained peak of your card from
`./vc info`.

This is Ticket 4. Predict how the error changes with the number of iterations.
'''), code('''
PEAK_TFLOPS = 49      # replace with the sustained number of `./vc info` on your card

params = sum(p.numel() for p in model.parameters())
prompt_2048 = torch.randint(0, 1000, (1, 2048), device=device)
flop = 2 * params * 2048

@torch.inference_mode()
def time_prefill(iters, synchronize):
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(iters):
    model(prompt_2048, use_cache=False)
  if synchronize:                                        # THE FAULT when False
    torch.cuda.synchronize()
  ms = (time.perf_counter() - start) / iters * 1000
  torch.cuda.synchronize()
  return ms

for _ in range(3):
  time_prefill(1, True)                                  # warm-up

for iters in (1, 3, 10, 40):
  for synchronize in (False, True):
    ms = time_prefill(iters, synchronize)
    tflops = flop / (ms / 1000) / 1e12
    flag = '   <-- above the peak: impossible' if tflops > PEAK_TFLOPS else ''
    print(f'iters {iters:2d}  synchronize {str(synchronize):5s}  {ms:6.1f} ms  {tflops:5.1f} TFLOP/s{flag}')
''')),

    ('d9', md('''
# Exercise 9: the logits that nobody counted

Run one prefill of 8192 tokens two times. The first time, keep the logits for
every position. The second time, pass `logits_to_keep=1`. Measure the peak
memory of each.

This is Ticket 9. Predict both numbers before you run. You have every term:
the KV cache, the logits, and the vocabulary size.
'''), code('''
LENGTH = 8192
vocab = model.config.vocab_size
long_prompt = torch.randint(0, 1000, (1, LENGTH), device=device)

@torch.inference_mode()
def peak_gb(**kwargs):
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  before = torch.cuda.memory_allocated()
  output = model(long_prompt, use_cache=True, **kwargs)
  shape = tuple(output.logits.shape)
  del output
  return (torch.cuda.max_memory_allocated() - before) / 1e9, shape

for kwargs in ({}, {'logits_to_keep': 1}):
  gb, shape = peak_gb(**kwargs)
  print(f'{str(kwargs):22s} logits {str(shape):20s} peak extra memory {gb:.2f} GB')
print(f'the logits of every position, computed: {LENGTH * vocab * 2 / 1e9:.2f} GB')
''')),

    ('d10', md('''
# Exercise 10: the first call

Two experiments. First, start a **new process**, load the model, and time
four prefills of 2048 tokens. Then, in this warm process, time five calls at a
shape that the model has never seen.

This is Ticket 10. Predict both ratios: the first call against the calls
after it, in the new process and in the warm process.
'''), code('''
import subprocess, textwrap

script = textwrap.dedent("""
    import time, torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).cuda().eval()
    prompt = torch.randint(0, 1000, (1, 2048), device='cuda')
    with torch.inference_mode():
        for _ in range(4):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(prompt, use_cache=False)
            torch.cuda.synchronize()
            print(round((time.perf_counter() - start) * 1000))
""").replace('MODEL_NAME', repr(MODEL))

# a second copy of the weights must fit next to this one
torch.cuda.empty_cache()
result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
cold = [int(line) for line in result.stdout.split()]
print('new process, ms per call:', cold)
print(f'first / last: {cold[0] / cold[-1]:.2f}x')

@torch.inference_mode()
def one_call(tokens):
  torch.cuda.synchronize()
  start = time.perf_counter()
  model(tokens, use_cache=False)
  torch.cuda.synchronize()
  return (time.perf_counter() - start) * 1000

new_shape = torch.randint(0, 1000, (3, 777), device=device)   # a shape not seen before
warm = [one_call(new_shape) for _ in range(5)]
print('warm process, new shape, ms per call:', [round(t) for t in warm])
print(f'first / median of the rest: {warm[0] / sorted(warm[1:])[2]:.2f}x')
''')),
]

MYSTERY = md('''
# Exercise 11: three mystery loops

The module `mystery.py` holds three loops: `mystery_a`, `mystery_b` and
`mystery_c`. Each one has the same signature as your stage 02, and each one
has one fault. **Do not open the file.**

This is the real skill. You get a broken thing and no source. You must design
the experiments that expose the fault.

For each loop:

1. Run it on one prompt, and compare with `reference`. Maybe it looks fine.
2. Design a second experiment that makes the fault visible. Which variable do
   you change? The prompt? Its length? The order of the calls? The number of
   calls?
3. Write your diagnosis: the fault, and the experiment that proved it.
4. Only then, open `mystery.py` and check.

A hint about the method, not about the answers: one of the loops is correct
on its first call. One of them writes fluent text. One of them depends on a
property of the prompt.
''')

MYSTERY_CODE = code('''
from mystery import mystery_a, mystery_b, mystery_c

for name, loop in [('a', mystery_a), ('b', mystery_b), ('c', mystery_c)]:
  report(f'mystery_{name}', loop(model, tokenizer, PROMPT, TOKENS), expected)
''')

MYSTERY_ANSWER = md('''
**Your diagnosis**

- `mystery_a`: the fault, and the experiment that proves it:
- `mystery_b`:
- `mystery_c`:
''')

FINGERPRINTS = md('''
# Your fingerprint table

Fill in this table from what you saw, not from what you predicted. It is the
real product of this notebook.

| Fault | Crash? | First wrong token | What the text looks like | The guard |
|---|---|---|---|---|
| whole sequence on decode | | | | |
| stuck position | | | | |
| logits of row 0 | | | | |
| previous token | | | | |
| shared cache | | | | |
| stop check with `==` | | | | |
| truncation | | | | |
| no synchronize | | | | |
| all the logits | | | | |
| the first call | | | | |
| position off by one | | | | |
''')

SOLUTIONS_2 = {
    'd1': '''
### What happens

- **Crash?** No. The cache grows much faster than the sequence. On my run the
  cache had 645 entries for a sequence of 37 tokens.
- **When?** At token 7 on my run. The first few tokens are correct, because the
  real context is still at the start of the cache.
- **What?** Fluent, but it **repeats the prompt**: "Tokyo, Osaka, and Kyoto.
  The three largest cities in Japan are Tokyo, Osaka, and Kyoto. The three
  largest cities in Japan are...". The model sees the prompt many times, so the
  most likely continuation is one more copy of the prompt.
- **Which guard?** The stage 02 check that compares the cached loop with the
  naive loop. A cheaper guard: assert that the cache length equals the
  sequence length after each step.

**The fingerprint.** Fluent text that loops back to the prompt.
''',
    'd2': '''
### What happens

- **Crash?** No.
- **When?** At token 4 on my run.
- **What?** Repetition that gets worse: "Tokyo, Osaka, Osaka, Kyoto, and
  Kyoto, and Kyoto, and... and... and". Each new token has the same rotation,
  so to the model every new token sits at the same place. The attention cannot
  tell the new tokens apart, and the output collapses into a loop.
- **Which guard?** The stage 02 comparison with the naive loop. In stage 07
  you give the positions yourself, and this is the bug that you must avoid
  there.

**The fingerprint.** A collapse into a loop of short tokens that gets worse
over time.
''',
    'd3': '''
### What happens

- **Crash?** No.
- **When?** At token 0. On my run the first token was ":" and not " Tokyo".
- **What?** Only the first token is wrong. After it, the text recovers: ":
  Tokyo, Osaka, and Kyoto. The population of Tokyo...". The decode steps are
  correct, and the prompt is still in the cache. The model simply continues
  from the odd first token.
- **Which guard?** The stage 01 and 02 checks that compare the first tokens
  with HF's `generate`.

**The fingerprint.** One odd first token, then good text. This bug is easy to
miss in a demo, because the answer "looks fine".
''',
    'd4': '''
### What happens

- **Crash?** No.
- **When?** At token 2 on my run.
- **What?** Nonsense with **doubled words**: "cities cities in in Japan Japan".
  Each token goes into the cache one step late, and so the model sees each
  token two times in a row.
- **Which guard?** The stage 02 comparison with the naive loop.

**The fingerprint.** Words that stutter.
''',
    'd5': '''
### What happens

- **Crash?** No.
- **When?** Request A is perfect. Request B went wrong at token 4 on my run.
- **What?** Fluent, and plausible, and wrong. On my run B wrote "a secret. I
  have a cat named Momo. I have a dog named Kiki". The Japanese names come
  from request A. The shared cache length at the end was 71: the sum of both
  requests.
- **Which guard?** A stage 02 check runs two requests one after the other for
  exactly this reason. In production: assert `cache_length == prompt_length`
  after prefill.

**The fingerprint.** The first request is fine. Later requests are fluent but
borrow words from other requests. This is the most dangerous bug in the table,
because it leaks data and still looks like a normal answer.
''',
    'd6': '''
### What happens

- **Crash?** No.
- **When?** The text is correct. The problem is its **length**. The reference
  stops after "Hello!". The broken loop runs to the limit of 60 tokens.
- **What?** The answer, then the stop token as text, then the answer again:
  "Hello!<|im_end|>\\n\\nHello!<|im_end|>\\n\\nHello!...". The model chose the
  stop token every time. The loop did not see it.
- **Which guard?** Your stage 01 `stop_token_ids` normalizes the list to a
  set. An alert on the fraction of responses that hit `max_tokens`.

**The fingerprint.** Every response has the maximum length, and the stop token
appears inside the text.
''',
    'd7': '''
### What happens

- **Crash?** No. No warning either.
- **When?** At token 0.
- **What?** The model sees only "Answer in one word. What is the". It then
  invents a question. On my run it asked about "the name of the first element
  in the periodic table".
- **Which guard?** None of stages 01 to 03, because they never truncate. The
  guard is to log the prompt length and alert on a spike at one value.

**The fingerprint.** A fluent answer to a different question. Short prompts
work, and long prompts fail.
''',
    'd8': '''
### What happens

With one iteration and no synchronize, the time is too short, and the TFLOP/s
can be above the peak of your card. As the number of iterations grows, the
error shrinks.

On my card (RTX 4080 Laptop GPU, sustained peak 49 TFLOP/s), over two runs:

| iterations | no synchronize | synchronize |
|---|---|---|
| 1 | 60 to 123 ms, 57 to 118 TFLOP/s | 188 ms, 37 TFLOP/s |
| 3 | 142 ms | 186 ms |
| 10 | 173 ms | 186 ms |
| 40 | 183 ms | 185 ms |

The row with one iteration changes a lot from run to run. A broken
measurement is not only wrong, it is also unstable. The synchronized column
stays within 1%.

**Why the error shrinks.** The CPU can queue only a limited amount of work
ahead of the GPU. After the queue is full, each new launch waits for the GPU.
So the CPU clock follows the GPU clock again, and only the work still in the
queue at the end goes missing. The size of that missing work stays about the
same, and more iterations divide it into smaller parts.

This is why a wrong benchmark can look right. A long benchmark hides the bug,
and a short one exposes it.

**The fingerprint.** A speedup that goes away when you run longer, or a
number above the peak.
''',
    'd9': '''
### What happens

On my run, at 8192 tokens:

| | logits shape | peak extra memory |
|---|---|---|
| all the logits | (1, 8192, 151936) | 3.46 GB |
| `logits_to_keep=1` | (1, 1, 151936) | 1.38 GB |

The logits of every position cost 8192 x 151936 x 2 = 2.49 GB. The KV cache
for 8192 tokens is only 0.94 GB. So the tensor that nobody counted is more
than twice the size of the cache.

**The fingerprint.** OOM that appears only above some prompt length, with one
allocation of size `prompt_len x vocab x bytes`.
''',
    'd10': '''
### What happens

The two ratios are very different.

**A new process pays a large cost one time.** On my card, the first prefill of
2048 tokens in a new process took 644 to 700 ms, and the calls after it took
186 to 202 ms: 3.5x in both runs. CUDA creates its context, loads the kernel modules and the cuBLAS
handles, and grows the memory pool. All of that happens in the first call.

**A new shape in a warm process costs almost nothing in eager PyTorch.** On my
card the ratio was between 0.90x and 1.01x over two runs. That is noise. The kernels are already loaded, and the allocator
already has memory. Eager PyTorch simply launches the same kernels with other
sizes.

That second result is "not a bug", and it is important. It is not true
everywhere:

- On the JAX track, each new shape is a new compile. The first call at a new
  shape can cost seconds.
- With CUDA graphs (stage 12), a new batch size has no graph. So vLLM captures
  a fixed set of batch sizes at startup, and pads each batch up to the nearest
  one.

So the question "is the first call slow?" has three answers: in a new process,
yes. At a new shape in eager PyTorch, no. At a new shape with compilation or
graphs, yes.

**The fingerprint.** A slow first request on each new process: warm up before
you report ready.
''',
}

MYSTERY_SOL = md('''
### The three mysteries

**`mystery_a`: a shared cache.** On the first call it is perfect, so one run
tells you nothing. The experiment: call it two times with two different
prompts, and compare the second call with the reference. It goes wrong after a
few tokens, and borrows words from the first prompt. The proof: call it two
times with **the same** prompt. The second answer differs from the first. A
pure function cannot do that, so the loop has state.

**`mystery_b`: the position is off by one.** The text stays fluent, and it is
a sensible answer. On my runs it differed from the reference after 10 to 14
tokens. For example, it wrote "3.7 million" where the reference wrote
"37,400,000". Nothing looks broken, so only a comparison token by token finds
it. The experiment: compare with the reference on several prompts, and look at
where the first difference is. It is always after a few correct tokens, and
never at token 0.

Why it is fluent: RoPE encodes the **relative** distance between two tokens.
An offset of one changes only the distance between the prompt and the new
tokens, and only by one. The distance between two new tokens is still correct.
With a larger offset the damage grows. In my test, an offset of 100 gave "a a
a a" on one prompt, and an offset of 5000 gave "Tokyo Tokyo Tokyo".

This is the most dangerous kind of bug: a small error in the numerics that
produces good text. Only a test that compares against an exact reference
catches it. A person who reads the output does not.

**`mystery_c`: truncation to 24 tokens.** It is correct for every prompt
shorter than 24 tokens, so the default prompt shows nothing. The experiment:
make the prompt longer and longer. Above 24 tokens, the answer stops following
the end of the prompt. The proof: put the important instruction at the end of
a long prompt. The loop ignores it.

### The method

All three needed a second experiment. The first run looked fine each time.
In each case you changed one variable and held the others fixed:

| Mystery | The variable to change |
|---|---|
| a | the order and the number of calls |
| b | the prompt, with a comparison token by token |
| c | the length of the prompt |

When a bug does not show, ask: **what is the variable that my test holds
constant?** The bug usually lives there.
''')

FINGERPRINTS_SOL = md('''
# The fingerprint table

| Fault | Crash? | First wrong token | What the text looks like | The guard |
|---|---|---|---|---|
| whole sequence on decode | no | after a few tokens | fluent, repeats the prompt | cache length == sequence length |
| stuck position | no | after a few tokens | collapses into a loop | compare with the naive loop |
| logits of row 0 | no | 0 | one odd token, then good text | compare the first token with HF |
| previous token | no | 1 or 2 | words that stutter | compare with the naive loop |
| shared cache | no | request 2, after a few tokens | fluent, borrows words from other users | cache length == prompt length after prefill |
| stop check with `==` | no | none, but too long | the answer, the stop token, the answer again | alert on `finish_reason="length"` |
| truncation | no | 0 | fluent answer to a different question | alert on a spike in the prompt lengths |
| no synchronize | no | not applicable | faster than the hardware allows | fail when above the peak |
| all the logits | OOM for long prompts | not applicable | not applicable | `prompt_len x vocab x bytes` in the memory plan |
| the first call | no | not applicable | a slow first request | warm up before ready |
| position off by one | no | after 10 to 14 tokens | fluent and plausible | compare token by token with an exact reference |

Look at the "Crash?" column. Only one fault out of eleven raises an error.
That is the main lesson of this notebook. **Inference bugs do not crash. They
give you text.** Only a comparison with a reference, or a check of a number,
finds them.

Now go to Part 2:

    ./vc guide 4
''')


def break_it(with_solutions):
    title = ('CodeChallenge: break it on purpose' if with_solutions
             else 'CodeChallenge HELPER: break it on purpose')
    cells = [header(title), SETUP, INTRO_2, LOAD]
    for key, text, drill in DRILLS:
        cells += [text, PREDICT, drill] if not with_solutions else [text, drill, md(SOLUTIONS_2[key])]
    cells += [MYSTERY, MYSTERY_CODE]
    if with_solutions:
        cells.append(MYSTERY_SOL)
        cells.append(FINGERPRINTS_SOL)
    else:
        cells += [code(), code(), MYSTERY_ANSWER, FINGERPRINTS]
    return cells


save('part1_inc_2_CCbreakItOnPurpose_helper.ipynb', break_it(False))
save('solutions/part1_inc_2_CCbreakItOnPurpose.ipynb', break_it(True))
print('done')
