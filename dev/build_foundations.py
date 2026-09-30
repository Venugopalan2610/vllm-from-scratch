"""Build the two sections of Part 0 that come before and after the GPU
section: 0_map (the map, the words, and where your instinct is wrong) and
4_measure (how to ask a machine a question).

    python dev/build_foundations.py

The demos then need to run once, so that they keep their output:

    .venv/bin/jupyter nbconvert --to notebook --execute --inplace <demo>.ipynb
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from break_it.common import ROOT, code, md, save  # noqa: E402

PART0 = 'course/Part0_FromAProgramToAModel'
MAP = f'{PART0}/0_map'
MEASURE = f'{PART0}/4_measure'


def header(section, lecture):
    return md(f'''
|<h2>Course:</h2>|<h1><a href="https://derivingsystems.com/course.html" target="_blank">Build your own vLLM: inference engines from the memory system up</a></h1>|
|-|:-:|
|<h2>Part 0:</h2>|<h1>From a Program to a Model<h1>|
|<h2>Section:</h2>|<h1>{section}<h1>|
|<h2>Lecture:</h2>|<h1><b>{lecture}<b></h1>|

<br>

<h5><b>Course repo:</b> <a href="https://github.com/Venugopalan2610/vllm-from-scratch" target="_blank">github.com/Venugopalan2610/vllm-from-scratch</a></h5>
<h5><b>The derivations:</b> <a href="https://derivingsystems.com" target="_blank">derivingsystems.com</a></h5>
<i>The notebooks build the intuition. The ladder in app/ makes you build the thing.</i>
''')


SETUP = '''
# find the repo root. The directory you start from does not matter.
import sys
from pathlib import Path
ROOT = next(folder for folder in [Path.cwd(), *Path.cwd().parents]
            if (folder/'cudalib').is_dir())
sys.path.insert(0, str(ROOT))
'''

GPU_SETUP = SETUP + '''
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cudalib

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
'''


def demo(path, section, lecture, cells):
    save(ROOT / path, [header(section, lecture)] + cells)


def challenge(helper_path, solution_path, section, title, cells):
    """cells: ('md', text), ('code', helper, solution) or ('solution', text)."""
    helper = [header(section, f'CodeChallenge: {title}')]
    solution = [header(section, f'CodeChallenge: {title}')]
    for cell in cells:
        if cell[0] == 'md':
            helper.append(md(cell[1]))
            solution.append(md(cell[1]))
        elif cell[0] == 'code':
            helper.append(code(cell[1]))
            solution.append(code(cell[2] if len(cell) > 2 else cell[1]))
        elif cell[0] == 'solution':
            solution.append(md(cell[1]))
    save(ROOT / helper_path, helper)
    save(ROOT / solution_path, solution)


# ================================================================ 0_map / 1

LAYERS = '''
```
  client ── POST /v1/chat/completions
     │
┌────▼──────────────────────────────────────────────────────────┐
│ ROUTER      the API server: parse, stream, stop on disconnect │
├───────────────────────────────────────────────────────────────┤
│ SERVICE     the engine loop, one step after another           │
│             ├── the scheduler: who runs in this step?         │
│             ├── the sampler: scores → one token for each row  │
│             └── the detokenizer: tokens → text                │
├───────────────────────────────────────────────────────────────┤
│ REPO        the KV cache manager: the memo tables of all the  │
│             requests, in fixed-size pages                     │
├───────────────────────────────────────────────────────────────┤
│ DB DRIVER   the model runner: one flat batch for each step    │
├───────────────────────────────────────────────────────────────┤
│ DB ENGINE   the model, and the GPU functions under it         │
└───────────────────────────────────────────────────────────────┘
  SCHEMA      the model config: it sets the size of everything
```
'''

map_1 = [
    md('''
# You already know this shape

A web service has a shape that is in your fingers by now: a router, a
service, a repo, a database. The models and the migrations live in their own
files. When a request fails, the shape tells you where to look first.

An **inference** engine has a shape too. It is the server that runs a trained
**model** (a large function that predicts the next piece of text) and answers
requests with it. Most courses give it to you at the end,
after you build the parts. This course gives it to you **first**. Every stage
after this notebook builds one box of it, and each stage guide starts with
"YOU ARE HERE".

This notebook needs no **GPU** (the processor on the graphics card, which
runs the same arithmetic on much data at once) and has no code. Read it slowly. The same map is
on one page in `docs/MAP.md`. Keep that page open while you work.
'''),
    md('''
# The words first

GPU people use many words that sound hard. Most of them are new names for
ideas that you already know. Here is the idea first, and then the name.

| The idea, in software words | The name that GPU people use |
|---|---|
| A function. Text goes in, and a score for each possible next piece of text comes out. | the **model** |
| The piece of text that the model reads and writes: a word or a part of a word, as an integer id. | a **token** |
| A large read-only table of numbers. The model reads all of it to make one token. | the **weights** |
| The requests that the engine serves in the same call. | the **batch** |
| The first call for a request. It reads the whole prompt at once. | the **prefill** |
| Each call after that. It makes one new token for each request. | a **decode** step |
| A memo table for each request. The model keeps its work on the earlier tokens there, so that it never does that work again. | the **KV cache** |
| A fixed-size page of that memo table. | a **block** |
| A function that runs on the GPU. | a **kernel** |
| A call from the CPU that starts a kernel. It has a fixed cost, like an RPC. | a **kernel launch** |
| The main memory of the GPU. | **HBM** |
| Code that waits for memory reads, not for its arithmetic. The GPU version of "I/O-bound". | **memory-bound** |
| Evict a request from memory, and run it again later. | **preemption** |
| A slow version of the code that you trust. You compare the fast version with it. | an **oracle** |

Three of these carry a real idea that is new to you: the KV cache,
memory-bound, and the batch. The course spends most of its time on those
three. The other names are only labels.
'''),
    md('''
# The layers

Here is the engine, in the words of a web service. The left column is the
word that you know. Each box is a part that you will build.
''' + LAYERS + '''
The analogy is not exact. `docs/MAP.md` has a table of each pattern, with the
nearest software pattern and the point where the analogy stops. That point is
usually the lesson of the stage.
'''),
    md('''
# The life of one request

Follow one chat request through the boxes. Each step names the box.

1. **Router.** The request arrives. The server checks the limits and puts an
   "add this request" command in a queue.
2. **Service.** The engine loop takes the command at the start of its next
   step. It turns the prompt into tokens and puts the request in the waiting
   queue.
3. **Service → Repo.** The scheduler plans the step. The running requests get
   one new token each. Waiting requests join, until the step is full. The repo
   gives each request the pages of memo table that it needs.
4. **Repo is full.** No free pages? The engine evicts a request, and runs it
   again later.
5. **DB driver.** The runner puts all the requests of the step into one flat
   batch.
6. **DB engine.** The model runs one step for the whole batch. It reads all
   its weights one time, and it reads the memo table of each request.
7. **Service.** The **sampler** picks one token for each request from the
   scores. The **detokenizer** turns the new tokens into text.
8. **Router.** The new text goes to the client as a stream.
9. **The end.** The request finishes, or the client leaves. The repo takes
   its pages back.

Steps 3 to 8 are **one step** of the engine. They repeat, for all the running
requests together, until nothing is left.
'''),
    md('''
### What to remember from this notebook

- The engine has five layers, and you know their shape: router, service, repo,
  DB driver, DB engine. The model config is the schema.
- One step serves every running request together. This one fact is the source
  of most of the surprises in this course.
- When something breaks, the first question is the same as in your web
  services: **which layer?** The challenge of this section trains that.

The shape of your instinct transfers. Some of the assumptions under it do
not. The next notebook finds them, and you measure each one.
'''),
]

# ================================================================ 0_map / 2

map_2 = [
    md('''
# Your instinct, measured

Your instinct as a software engineer has two parts. The **shape of your
search** (layers, narrowing, patterns) transfers to an engine completely. The
last notebook gave you the layers.

The **assumptions under your search** do not transfer. You learned them on
CPUs, networks and databases. This notebook takes six of them. For each one:

1. You **predict** with your old instinct. Write the number in the cell
   before you run the measurement. Do not skip this. A prediction that you
   write after the result teaches you nothing.
2. The next cell **measures** it on your GPU.
3. You see where the instinct fails, and why.

A wrong prediction is the point of this notebook, not a failure. When one of
yours is wrong, write it down with `./vc note "..."`.
'''),
    code(GPU_SETUP + '''
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
WEIGHT_BYTES = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

# Two numbers of YOUR card. Part 0, section 3, measures them in detail.
PEAK_BANDWIDTH = cudalib.peak_bandwidth() * 1e9     # bytes/s
PEAK_FLOPS = cudalib.matmul_flops()                  # FLOP/s, bf16
print(f'weights:   {WEIGHT_BYTES/1e9:.2f} GB')
print(f'bandwidth: {PEAK_BANDWIDTH/1e9:.0f} GB/s')
print(f'compute:   {PEAK_FLOPS/1e12:.1f} TFLOP/s')

PREDICTED, MEASURED = {}, {}
'''),
    md('''
# 1. "Each request costs its own share"

In a web service, 32 requests cost about 32 times the work of one request.

One **decode** step makes one new token for each request in the batch.
Predict: how long is one step for 32 requests, divided by one step for 1
request?
'''),
    code('''
# PREDICT, before you run the next cell.
PREDICTED['1. 32 requests / 1 request'] = None      # your number
'''),
    code('''
@torch.inference_mode()
def step_ms(model, batch_size):
    """One decode step: one new token for each of batch_size requests."""
    tokens = torch.randint(0, 1000, (batch_size, 1), device='cuda')
    return cudalib.bench_ms(lambda: model(tokens, use_cache=False), iters=20, warmup=5, best_of=3)

batch_sizes = [1, 2, 4, 8, 16, 32]
step_times = [step_ms(model, batch_size) for batch_size in batch_sizes]
for batch_size, milliseconds in zip(batch_sizes, step_times):
    print(f'{batch_size:>3} requests: {milliseconds/step_times[0]:5.2f}x the time of 1 request')
MEASURED['1. 32 requests / 1 request'] = step_times[-1] / step_times[0]
'''),
    md('''
The requests do not each pay for their own step. The model reads all its
weights one time for each step, and every request in the batch uses that one
read. So the 32nd request is almost free.

The other side of the same fact: all the requests move together, in the same
step. If one request makes the step slow, every other request waits. Stage 11
is about that.
'''),
    md('''
# 2. "Computation is the expensive part"

In a web service, you make code faster when it does less work.

At batch 1, one step does two things. It **reads** every weight from memory,
and it does the **arithmetic**: one multiply and one add for each weight.
Predict: what percentage of the step time is the arithmetic?
'''),
    code('''
PREDICTED['2. arithmetic, % of the step'] = None      # your number, 0 to 100
'''),
    code('''
read_ms = WEIGHT_BYTES / PEAK_BANDWIDTH * 1e3
parameters = WEIGHT_BYTES / 2                          # 2 bytes for each bf16 weight
arithmetic_ms = 2 * parameters / PEAK_FLOPS * 1e3
print(f'the read of the weights, at full bandwidth: {read_ms:7.2f} ms')
print(f'the arithmetic, at full speed:              {arithmetic_ms:7.2f} ms')
print(f'the measured step at batch 1:               {step_times[0]:7.2f} ms')
MEASURED['2. arithmetic, % of the step'] = 100 * arithmetic_ms / step_times[0]
print(f'-> the arithmetic is {MEASURED["2. arithmetic, % of the step"]:.1f}% of the step')
'''),
    md('''
The step waits for bytes. The arithmetic units have almost nothing to do. So
"do less work" does not help here. Two things help: read fewer bytes (Part 7
makes the weights smaller), or serve more requests with each read (the batch,
from assumption 1).

Code that waits for memory, not for arithmetic, is **memory-bound**. It is the
GPU version of "I/O-bound". Part 1 derives the exact point where this changes.
'''),
    md('''
# 3. "The state of one request is small"

In a web service, the state of one request is a few kilobytes. The big state
is in the database.

For each token of a conversation, the model keeps a memo of its work, so that
it never does that work again. That memo is the **KV cache**. Its size for one
token comes from the model config: the schema.

Predict: how many tokens long must one conversation be before its memo is as
large as the whole model?
'''),
    code('''
PREDICTED['3. tokens until the memo = the model'] = None      # your number
'''),
    code('''
config = model.config
# K and V, for each layer, for each KV head, head_dim numbers of 2 bytes
kv_bytes_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 2
print(f'memo for one token: {kv_bytes_per_token/1024:.0f} KB   (from the config)')

# Check the formula against the real memo of a 4096-token prompt.
with torch.inference_mode():
    tokens = torch.randint(0, 1000, (1, 4096), device='cuda')
    cache = model(tokens, use_cache=True).past_key_values
    real_bytes = sum(layer.keys.numel() * layer.keys.element_size()
                     + layer.values.numel() * layer.values.element_size()
                     for layer in cache.layers)
print(f'memo for 4096 tokens: {real_bytes/1e6:.0f} MB measured, '
      f'{kv_bytes_per_token*4096/1e6:.0f} MB from the formula')
del cache, tokens
torch.cuda.empty_cache()

MEASURED['3. tokens until the memo = the model'] = WEIGHT_BYTES / kv_bytes_per_token
print(f'-> at {MEASURED["3. tokens until the memo = the model"]:,.0f} tokens, '
      'one conversation needs as much memory as the whole model')
'''),
    md('''
So on a GPU, the memory, not the CPU, sets how many users fit at the same
time. Every request carries a large state, and it grows by one token at each
step. Stage 06 manages that memory in pages, the way an operating system
manages the memory of processes.
'''),
    md('''
# 4. "A bug raises an error"

In a web service, most bugs raise an exception, or return a 500.

Here is a real bug from a real engine: it gives a new request the memo table
of the request before it. Predict: what does the user see? Write `'error'`,
`'garbage'` (text that makes no sense) or `'fluent'` (normal text).
'''),
    code('''
PREDICTED['4. what the user sees'] = None      # 'error', 'garbage' or 'fluent'
'''),
    code('''
@torch.inference_mode()
def generate(prompt, num_tokens=40, old_cache=None):
    """Greedy generation with an explicit memo table. With `old_cache`, it
    reuses the memo of an earlier request: the bug."""
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    start = old_cache.get_seq_length() if old_cache is not None else 0
    forward_pass = model(prompt_ids, past_key_values=old_cache, use_cache=True)
    cache, token, tokens = forward_pass.past_key_values, None, []
    for index in range(num_tokens):
        token = forward_pass.logits[0, -1].argmax().item()
        tokens.append(token)
        position = torch.tensor([[start + prompt_ids.shape[1] + index]], device='cuda')
        forward_pass = model(torch.tensor([[token]], device='cuda'), position_ids=position,
                       past_key_values=cache, use_cache=True)
        cache = forward_pass.past_key_values
    return tokens, cache

PROMPT = 'The three most important ideas in operating systems are'
right, _ = generate(PROMPT)
_, earlier_cache = generate('My favourite recipe for a chocolate cake needs')
wrong, _ = generate(PROMPT, old_cache=earlier_cache)          # no exception here

print('correct:  ', repr(tokenizer.decode(right)))
print('with bug: ', repr(tokenizer.decode(wrong)))
different = sum(right_token != wrong_token for right_token, wrong_token in zip(right, wrong))
print(f'\\nexceptions: 0.   tokens that differ from the oracle: {different} of {len(right)}')
MEASURED['4. what the user sees'] = 'fluent'
'''),
    md('''
No error, and the text is fluent. Nothing in a log tells you that it is
wrong. Only a comparison with a version that you trust finds it.
That trusted version is an **oracle**. Stage 02 has a check for exactly this
bug, and the course keeps an oracle for every fast part that you build.
'''),
    md('''
# 5. "The framework overhead is small"

In a web service, 1 ms of Python is nothing next to a 50 ms database call.

One decode step is about a thousand small GPU functions (**kernels**). Python
starts each one with a call, a **kernel launch**, and each call has a fixed
cost. A **CUDA graph** records all the launches of one step one time, and
then replays them with no Python at all. It is like a prepared statement.

Take a smaller model, Qwen3-0.6B, so the GPU has less work in each step.
Predict: what percentage of its step is Python and launches?
'''),
    code('''
PREDICTED['5. Python and launches, % of the step'] = None      # your number, 0 to 100
'''),
    code('''
@torch.inference_mode()
def eager_and_graph_ms(model):
    tokens = torch.randint(0, 1000, (1, 1), device='cuda')
    eager = cudalib.bench_ms(lambda: model(tokens, use_cache=False), iters=20, warmup=5, best_of=3)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                     # warm up before the capture
        for _ in range(3):
            model(tokens, use_cache=False)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model(tokens, use_cache=False)
    replay = cudalib.bench_ms(graph.replay, iters=20, warmup=5, best_of=3)
    del graph
    return eager, replay

small = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16).cuda().eval()
for name, some_model in [('Qwen3-0.6B', small), ('Qwen3-1.7B', model)]:
    eager, replay = eager_and_graph_ms(some_model)
    overhead = 100 * (eager - replay) / eager
    print(f'{name}: step {eager:6.2f} ms, the same step replayed {replay:6.2f} ms '
          f'-> Python and launches are {overhead:4.1f}% of the step')
    if name == 'Qwen3-0.6B':
        MEASURED['5. Python and launches, % of the step'] = overhead
del small
torch.cuda.empty_cache()
'''),
    md('''
On the small model, Python and the launches are a large part of the step.
On the larger model, they almost disappear. Why?

The launches are **asynchronous**. Python puts each kernel in a queue and
continues at once, while the GPU works on the queue. So a step costs about
**the larger of** the Python time and the GPU time, not the sum. When the
GPU work is long, it hides the Python. When the GPU work is short, the Python
is the step.

Stage 12 builds the graph replay. Part 0, section 4, shows you how this same
queue can make a stopwatch lie to you.
'''),
    md('''
# 6. "The same input gives the same output"

In a web service, a pure function gives the same result for the same input.

**Greedy decoding** always takes the top choice, so it has no randomness.
Run it on one prompt two times: alone, and in a batch of 2 copies of the
same prompt. Predict: how many of the 64 tokens differ?
'''),
    code('''
PREDICTED['6. tokens that differ, of 64'] = None      # your number
'''),
    code('''
@torch.inference_mode()
def greedy(batch_ids, num_tokens=64):
    """Greedy tokens of row 0, and the top-2 scores at each position."""
    forward_pass = model(batch_ids, use_cache=True)
    cache, tokens, top2 = forward_pass.past_key_values, [], []
    for _ in range(num_tokens):
        scores = forward_pass.logits[:, -1].float()
        top2.append(scores[0].topk(2).values.tolist())
        next_tokens = scores.argmax(-1, keepdim=True)
        tokens.append(next_tokens[0].item())
        forward_pass = model(next_tokens, past_key_values=cache, use_cache=True)
        cache = forward_pass.past_key_values
    return tokens, top2

def differ(prompt):
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    alone, top2 = greedy(prompt_ids)
    in_batch, _ = greedy(prompt_ids.repeat(2, 1))
    first = next((i for i, (alone_token, batch_token) in enumerate(zip(alone, in_batch)) if alone_token != batch_token), None)
    return sum(alone_token != batch_token for alone_token, batch_token in zip(alone, in_batch)), first, top2

num_different, first, top2 = differ(PROMPT)
MEASURED['6. tokens that differ, of 64'] = num_different
print(f'{num_different} of 64 tokens differ.')
if first is not None:
    print(f'The first difference is at token {first}. There, the top two scores '
          f'at batch 1 were {top2[first][0]:.3f} and {top2[first][1]:.3f}.')

# One prompt is one sample. Try more.
PROMPTS = ['The capital of France is', 'def fibonacci(n):', 'Water boils at',
           'The best way to learn a language is', 'In 1969, the first person to walk on the moon',
           'A good unit test should', 'The difference between a list and a tuple in Python is']
counts = [differ(prompt)[0] for prompt in PROMPTS]
print(f'\\nOther prompts: {sum(num_changed > 0 for num_changed in counts)} of {len(PROMPTS)} changed somewhere '
      f'in 64 tokens. Tokens that differ: {counts}')
'''),
    md('''
The math of one row does not depend on the other rows. But the GPU adds the
numbers of a matrix product in a different order when the batch size
changes. Floating-point addition is not associative: `(a + b) + c` and
`a + (b + c)` can round differently. The difference is tiny. At a position
where the top two choices are almost equal, it is enough to flip the choice,
and after that the two texts are different conversations.

Users report this as "the same prompt gives a different answer when the
server is busy". `LORE.md`, section 12, measures how often it happens.
'''),
    md('''
# Your predictions, against the measurements
'''),
    code('''
print(f"{'assumption':<42} {'you predicted':>14} {'measured':>12}")
for key in MEASURED:
    predicted, measured = PREDICTED.get(key), MEASURED[key]
    show = lambda value: f'{value:.2f}' if isinstance(value, float) else str(value)
    print(f'{key:<42} {show(predicted):>14} {show(measured):>12}')
'''),
    md('''
Which predictions were far from the measurement? For each one, write one line:

```bash
./vc note "predicted 32x for 32 requests, measured about 1x: the weights are read one time for the whole batch"
```

### What to remember from this notebook

| Your instinct says | On a GPU engine |
|---|---|
| Each request costs its own share | All the requests share one step, and one read of the weights. |
| Computation is the expensive part | The step waits for bytes. The arithmetic is a small part. |
| The state of one request is small | The memo of one long conversation can be larger than the model. |
| A bug raises an error | A bug gives fluent, wrong text. Keep an oracle. |
| The framework overhead is small | Python and launches can be a large part of a step. |
| The same input gives the same output | A different batch size can change the text. |

`docs/METHOD.md` has this table, and the six moves that the rest of the
course uses to find more assumptions like these.
'''),
]

# ================================================================ 0_map / 3

TICKETS = [
    ('a', 'When one user pastes a 20-page document, every other user sees their '
          'stream stop for a moment.', 'SERVICE',
     'The scheduler put the whole long prompt into one step, and every other request '
     'waits for that step. Stage 11 cuts a long prompt into chunks.'),
    ('b', 'The answers read well, but sometimes a new answer continues the topic of an '
          'earlier user.', 'REPO',
     'A request got memo pages that belong to another request: a leak, or a wrong key '
     'in the prefix cache. Stage 09 has this bug on purpose.'),
    ('c', 'The model writes its private reasoning into the answer, but only through the '
          'chat endpoint.', 'ROUTER',
     'The router builds the chat prompt from the messages. A wrong chat template, or a '
     'missing option of it, changes what the model sees. Stage 27.'),
    ('d', 'At batch 1, the new kernel for **attention** (the part of the model that '
          'reads the memo table) uses only a small part of the memory **bandwidth** '
          '(bytes per second) of the card. Each thread of it is fast.', 'DB ENGINE',
     'At batch 1 the kernel has too little parallel work, and most of the GPU waits. '
     'Stage 08c splits the work so that the whole GPU has some.'),
    ('e', 'After a change to how the step is captured, one user\'s answer changes a few '
          'tokens after another user joins the batch.', 'DB DRIVER',
     'The runner fills the empty rows of a fixed-size batch. If an empty row writes '
     'into page 0, it overwrites the memo of a real request. Stage 23.'),
    ('f', 'All the streams stop together for about 20 ms at a time, and each stop '
          'matches one step of the engine.', 'SERVICE',
     'The engine loop runs its step inside the event loop, so nothing else can run '
     'during the step. Stage 27 moves the step to its own thread.'),
    ('g', 'With the same traffic, the server holds fewer requests every hour. No client '
          'disconnects.', 'REPO',
     'Pages that nobody uses never go back to the free list: a leak in the repo. '
     'The stage 22 checks count the free pages after each run.'),
    ('h', 'Characters of Hindi text appear broken in the stream, but the final text is '
          'correct.', 'SERVICE',
     'The detokenizer turns one token at a time into text, and one character can use '
     'more than one token. Stage 14.'),
]

map_3 = [
    ('md', '''
# Which layer?

In your web services, the first question after a failure is "which layer?".
The answer tells you which file to open. Here is the same question for an
engine.

Each ticket below is a symptom, as a user or a dashboard reports it. For each
one, name the layer where you would look first. Use one of these five:

    ROUTER, SERVICE, REPO, DB DRIVER, DB ENGINE

You do not know how to fix any of them yet. You do not need to. A good first
guess of the layer is the skill. Use the map of the first notebook of this
section, or `docs/MAP.md`.
'''),
    ('md', '''
### Given

- The five layers of the map, and what each one does.
- Eight symptoms, in the words of the person who saw them.

### Asked

- For each symptom, the one layer where you look first.
- You know that you are correct when the check prints `8 of 8`.
'''),
    ('code', SETUP + '''
import hashlib

def check(answers):
    """The answers are stored as hashes, so that this cell does not give them away."""
    right = 0
    for ticket, layer in answers.items():
        digest = hashlib.sha256(f'{ticket}:{str(layer).strip().upper()}'.encode()).hexdigest()[:12]
        ok = digest == EXPECTED[ticket]
        right += ok
        print(f'  ticket {ticket}: {"correct" if ok else "not yet"}')
    print(f'{right} of {len(EXPECTED)}')

EXPECTED = ''' + repr({ticket: __import__('hashlib').sha256(f'{ticket}:{layer}'.encode()).hexdigest()[:12]
                     for ticket, _, layer, _ in TICKETS}) + '''
'''),
    ('md', '### The tickets\n\n' + '\n'.join(f'**{ticket}.** {text}\n' for ticket, text, _, _ in TICKETS)),
    ('code', 'ANSWERS = {\n' + ''.join(f"    '{ticket}': '',\n" for ticket, _, _, _ in TICKETS) + '}\ncheck(ANSWERS)',
             'ANSWERS = {\n' + ''.join(f"    '{ticket}': '{layer}',\n" for ticket, _, layer, _ in TICKETS) + '}\ncheck(ANSWERS)'),
    ('solution', '### Why\n\n' + '\n'.join(f'- **{ticket}. {layer}.** {why}' for ticket, _, layer, why in TICKETS)),
    ('md', '''
### After the check

For each ticket that you got wrong, find the box on the map that you should
have chosen, and read what it does. Two of the tickets have the same layer as
another ticket, but a different cause. The layer tells you where to look. It
does not tell you what you will find.
'''),
]

# ============================================================= 4_measure / 1

measure_1 = [
    md('''
# A stopwatch that lies

The rest of this course measures the GPU, again and again. Before you trust
a single number, you must know three ways that a measurement of a GPU goes
wrong. Each one gave a wrong number to people with years of experience.

Every move of the method in `docs/METHOD.md` depends on a measurement that
you can trust. This notebook is about that trust. Each section below asks you
to predict first.
'''),
    code(GPU_SETUP),
    md('''
# 1. The GPU works from a queue

When Python calls a GPU function, the call puts the work in a queue and
returns at once. The GPU does the work later. This is **asynchronous**
execution. It is the same queue that hid the Python time in assumption 5
of section `0_map`.

Here is a stopwatch around 50 multiplications of two 4096 x 4096 matrices.
The first version stops the stopwatch when Python returns. The second version
waits for the GPU first, with `torch.cuda.synchronize()`.

Predict: the time with the wait, divided by the time without it.
'''),
    code('''
PREDICTED_RATIO = None      # your number
'''),
    code('''
left_matrix = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
right_matrix = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(3):                                   # warm up: section 2 explains why
    left_matrix @ right_matrix
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(50):
    left_matrix @ right_matrix
no_wait = time.perf_counter() - start                # Python returned. The GPU did not finish.
torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    left_matrix @ right_matrix
torch.cuda.synchronize()                             # wait for the GPU
with_wait = time.perf_counter() - start

print(f'without the wait: {no_wait*1e3:8.2f} ms')
print(f'with the wait:    {with_wait*1e3:8.2f} ms')
print(f'-> ratio {with_wait/no_wait:.0f}x.  You predicted {PREDICTED_RATIO}.')
'''),
    md('''
The first number measures how fast Python puts work in the queue. It says
nothing about the GPU. Code that looks 10 or 100 times faster after a change
often has only lost its wait.

**Rule 1.** Wait for the GPU before you stop the stopwatch: call
`torch.cuda.synchronize()`, or use CUDA events, which time the GPU itself.
'''),
    md('''
# 2. The first call is different

The first call of a GPU function with a new shape does extra work: it
chooses an algorithm, allocates memory, and sometimes compiles code.

Predict: the time of the first call of a new shape, divided by the time of a
later call.
'''),
    code('''
PREDICTED_FIRST = None      # your number
'''),
    code('''
def one_call_ms(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3

left_new_shape = torch.randn(3001, 2999, device='cuda', dtype=torch.bfloat16)     # a shape not used before
right_new_shape = torch.randn(2999, 3003, device='cuda', dtype=torch.bfloat16)
first = one_call_ms(lambda: left_new_shape @ right_new_shape)
later = np.median([one_call_ms(lambda: left_new_shape @ right_new_shape) for _ in range(20)])
print(f'first call: {first:7.2f} ms')
print(f'later call: {later:7.2f} ms')
print(f'-> ratio {first/later:.1f}x.  You predicted {PREDICTED_FIRST}.')
'''),
    md('''
The ratio changes from card to card and from run to run. It is never 1 by
design. If your timing includes the first call, you time the setup, not the
work.

**Rule 2.** Warm up: call the function a few times before you start the
stopwatch.
'''),
    md('''
# 3. One number is not a measurement

Run the same work 300 times, and you get 300 different times. Other
programs, the temperature of the chip, and its clock speed all move them. A
laptop GPU raises its clock when it is cool, and lowers it when it is hot.

Predict: the slowest 1% of the runs (the **p99**), divided by the middle run
(the **median**).
'''),
    code('''
PREDICTED_TAIL = None      # your number
'''),
    code('''
square_matrix = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
for _ in range(10):
    square_matrix @ square_matrix
times = []
for _ in range(300):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    square_matrix @ square_matrix
    end.record()
    end.synchronize()
    times.append(start.elapsed_time(end))
times = np.array(times)

for name, value in [('min', times.min()), ('median', np.median(times)),
                    ('mean', times.mean()), ('p99', np.percentile(times, 99))]:
    print(f'{name:>7}: {value*1e3:7.1f} us')
print(f'-> p99 / median = {np.percentile(times, 99)/np.median(times):.2f}.  You predicted {PREDICTED_TAIL}.')

plt.figure(figsize=(7.5, 3.6))
plt.hist(times * 1e3, bins=60)
plt.axvline(np.median(times) * 1e3, color='k', ls='--', label='median')
plt.axvline(np.percentile(times, 99) * 1e3, color='r', ls='--', label='p99')
plt.xlabel('microseconds for one call')
plt.ylabel('calls')
plt.legend()
plt.grid(alpha=.3)
plt.show()
'''),
    md('''
Each summary answers a different question. Choose the one that matches your
question:

| The question | The number | Why |
|---|---|---|
| What can the hardware do? Which of two kernels is faster? | the **min**, or the best of some rounds | The fastest run is the one that the noise changed least. |
| What does a typical call cost? | the **median** | Half the runs are faster and half are slower. One slow run cannot move it. |
| What do the unluckiest users feel? | the **p99**, or the p99.9 | A service is judged by its slow requests. Stage 16 measures them. |
| (Almost never) | the mean | A few slow runs pull it up. It describes no real run. |

**Rule 3.** Repeat the measurement, and report the summary that answers your
question. Compare two versions on the same machine, one after the other, and
more than once. `cudalib.compare_ms` does this.
'''),
    md('''
### What to remember from this notebook

1. **Wait for the GPU** before you stop the stopwatch.
2. **Warm up** before you start it.
3. **Repeat**, and choose the min, the median or the p99 by your question.

`cudalib.bench_ms`, which the rest of the course uses, does all three. Now you
know what it does, and why. When a number in this course surprises you,
check these three rules before you check anything else.
'''),
]

# ============================================================= 4_measure / 2

measure_2 = [
    md('''
# How does the cost grow?

A number is true at one size. Production runs at the next size. So the
question that matters most is often not "how long does this take?", but
"what happens to the time when the input grows?"

This is **move 4** of `docs/METHOD.md`: read the shape. This notebook gives
you three tools for it, from the simplest to the most formal. Use them in
this order.
'''),
    code(GPU_SETUP),
    md('''
Three functions, each of a length `L`:

- `project(L)`: each of `L` tokens goes through one layer of weights. The work
  grows with `L`.
- `scores(L)`: each of `L` tokens is compared with every other token. The work
  grows with `L x L`.
- `model_like(L)`: a fixed read of 192 MB, like the weights of a small model,
  plus both of the above. This one is like one real forward pass.

The code of the first two tells you their shape. The third is the kind of
function that you meet in real work: a mix of terms, and you do not know
which one wins.
'''),
    code('''
WIDTH = 2048
W = torch.randn(WIDTH, WIDTH, device='cuda', dtype=torch.bfloat16)
FIXED = torch.randn(96 * 2**20, device='cuda', dtype=torch.bfloat16)      # 192 MB

def project(L):
    activations = torch.randn(L, WIDTH, device='cuda', dtype=torch.bfloat16)
    return lambda: activations @ W

def scores(L):
    q = torch.randn(L, 128, device='cuda', dtype=torch.bfloat16)
    return lambda: q @ q.T

def model_like(L):
    run_project, run_scores = project(L), scores(L)
    return lambda: (FIXED.sum(), run_project(), run_scores())
'''),
    md('''
# Tool 1: double it, and divide

Time the function at `L`, and at `2L`. Divide the second time by the first:

| `t(2L) / t(L)` | The shape |
|---|---|
| about 1 | flat: the size does not matter |
| about 2 | linear |
| about 4 | quadratic |

You can do this in your head, and it needs no plot. Predict the ratio of
`model_like` for a doubling from 1024 to 2048, and from 8192 to 16384.
'''),
    code('''
PREDICTED_SMALL = None      # model_like: t(2048) / t(1024)
PREDICTED_LARGE = None      # model_like: t(16384) / t(8192)
'''),
    code('''
LENGTHS = np.array([512, 1024, 2048, 4096, 8192, 16384])
TIMES = {name: np.array([cudalib.bench_ms(function(int(L)), iters=20, warmup=5, best_of=3)
                         for L in LENGTHS])
         for name, function in [('project', project), ('scores', scores), ('model_like', model_like)]}

print(f"{'L':>6}" + ''.join(f'{name:>22}' for name in TIMES))
for index, L in enumerate(LENGTHS):
    row = f'{L:>6}'
    for name, times in TIMES.items():
        doubling_label = f'x{times[index]/times[index-1]:.2f}' if index else ''
        row += f'{times[index]:>12.3f} ms {doubling_label:>6}'
    print(row)
'''),
    md('''
Read the `model_like` column from top to bottom. At small `L` the doubling
ratio is near 1: the function looks flat. At large `L` it grows past 2, and
it continues to move toward 4.

Both readings are true. At small `L` the fixed 192 MB read is almost all of
the time, so the other terms hide. At large `L` the `L x L` term wins. **The
shape of a mix changes with the size.** A table of small sizes shows you only
the term that wins at small sizes.

This is exactly how a quadratic cost reaches production. At the sizes that
the tests use, it looks flat.

`scores` also shows something at small `L`: its ratio is not 4 there. At
small sizes a fixed cost, such as the kernel launch, is a large part of a
tiny time. Always read the ratio at the large end.
'''),
    md('''
# Tool 2: a log-log plot

On a plot with both axes in log scale, `t = c x L^k` is a straight line, and
its slope is the exponent `k`. So the slope tells you the shape, at each
size, and you can see where it bends.
'''),
    code('''
plt.figure(figsize=(7.5, 4.6))
for name, times in TIMES.items():
    plt.plot(LENGTHS, times, 'o-', label=name)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('L')
plt.ylabel('ms')
plt.title('slope 1 = linear, slope 2 = quadratic')
plt.legend()
plt.grid(alpha=.3, which='both')
plt.show()

for name, times in TIMES.items():
    slopes = np.diff(np.log(times)) / np.diff(np.log(LENGTHS))
    print(f'{name:>10}: slope between each pair of sizes  ' + '  '.join(f'{slope:4.2f}' for slope in slopes))
'''),
    md('''
The slope is the doubling test in a different form: a ratio of 2 is a slope
of 1, and a ratio of 4 is a slope of 2. The plot adds one thing: you see the
bend of `model_like`, and where it is.
'''),
    md('''
# Tool 3: a fit

The first two tools tell you the shape. A **fit** tells you the size of each
term, so that you can predict a size that you did not measure.

Suppose that the time is `t(L) = c + b L + a L^2`. `np.polyfit` finds the
`c`, `b` and `a` that are closest to the measurements. The code names them
`constant_ms`, `l_coef` and `l_squared_coef`. The raw value of `l_squared_coef`
means nothing by itself, because it depends on the units. Ask a better
question: **at each length, what share of the time is the `L^2` term?**
'''),
    code('''
l_squared_coef, l_coef, constant_ms = np.polyfit(LENGTHS, TIMES['model_like'], 2)
print(f't(L) = {constant_ms:.3f} + {l_coef:.2e} L + {l_squared_coef:.2e} L^2   (ms)\\n')
for L in [1024, 4096, 16384, 65536]:
    quadratic = l_squared_coef * L * L
    predicted_ms = constant_ms + l_coef * L + quadratic
    measured = '' if L > LENGTHS[-1] else '   (measured)'
    print(f'at L = {L:>6}: the L^2 term is {100*quadratic/predicted_ms:5.1f}% of the time{measured}')
'''),
    md('''
At 65536, which you did not measure, the fit predicts that the `L^2` term is
most of the time. That is the value of the fit: it goes beyond the
data. It is also the risk: a fit is only as good as the sizes that you gave
it. Check a prediction of a fit with one real measurement when you can.

### When to use which tool

| Tool | Use it when | Trap |
|---|---|---|
| Double it, and divide | always first: in your head, or on paper | small sizes, where a fixed cost hides the shape |
| A log-log plot | you want to see where the shape changes | a straight line through three points proves little |
| A fit | you must predict a size that you cannot measure | a fit on small sizes, where the big term is hidden |

### What to remember from this notebook

- Ask "what happens when it doubles?" before any other question about size.
- The shape of a mix changes with the size. Read the ratio at the large end.
- The fit comes last. It separates the terms, and it predicts beyond the data.

Part 1, section 3, uses these three tools on the real model, to find the cost
of a loop that forgets its work.
'''),
]

# ============================================================= 4_measure / 3

MYSTERIES = '''"""Three functions for the challenge "measure it yourself".

Do not read this file before you finish the challenge. The point of the
challenge is to find the shape of each function from the outside, with
measurements only, the way you find the shape of a system that you did not
write.
"""
import torch

_FIXED = None


def _fixed():
    global _FIXED
    if _FIXED is None:
        _FIXED = torch.randn(128 * 2**20, device='cuda', dtype=torch.bfloat16)
    return _FIXED


def mystery_1(n):
    points = torch.randn(n, 64, device='cuda')
    return lambda: torch.cdist(points, points)


def mystery_2(n):
    small = torch.randn(n, device='cuda')
    return lambda: (_fixed().sum(), small.sum())


def mystery_3(n):
    values = torch.randn(n * 2**16, device='cuda')
    return lambda: values.sum()
'''

SHAPES = {'mystery_1': 'quadratic', 'mystery_2': 'flat', 'mystery_3': 'linear'}

measure_3 = [
    ('md', '''
# Measure it yourself

`mysteries.py`, next to this notebook, has three functions. Each takes a size
`n` and returns a function to time. Do not open the file. Find the shape of
each function from the outside, with measurements only: `flat`, `linear` or
`quadratic`.

This is the situation of your real work: a system that you did not write,
and a question about how it grows.
'''),
    ('code', GPU_SETUP + '''
sys.path.insert(0, str(ROOT/'course/Part0_FromAProgramToAModel/4_measure'))
import hashlib
from mysteries import mystery_1, mystery_2, mystery_3
'''),
    ('md', '''
# Exercise 1: an honest stopwatch

**Given:** a GPU function with no arguments, the three rules of the first
notebook of this section, and `cudalib.bench_ms` to compare with.

**Asked:** `honest_ms(function)`, the milliseconds of one call. You know that
it is correct when the check prints "your stopwatch is honest".

Use the three rules:

1. Warm up: call `function` 3 times first.
2. Wait for the GPU before you read the time.
3. Repeat: time 5 rounds of 10 calls, and return the **median** of the
   milliseconds for one call.
'''),
    ('code', '''
def honest_ms(function):
    """Milliseconds for one call of `function`: warmed up, waited for, repeated."""
    pass
''', '''
def honest_ms(function):
    """Milliseconds for one call of `function`: warmed up, waited for, repeated."""
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10):
            function()
        torch.cuda.synchronize()
        rounds.append((time.perf_counter() - start) * 1e3 / 10)
    return float(np.median(rounds))
'''),
    ('code', '''
# The check: your stopwatch must agree with the stopwatch of the course.
work = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
yours = honest_ms(lambda: work @ work)
course = cudalib.bench_ms(lambda: work @ work, iters=10, warmup=3, best_of=5)
print(f'yours: {yours:.3f} ms   cudalib.bench_ms: {course:.3f} ms')
assert yours is not None, 'honest_ms returned nothing'
assert 0.7 < yours / course < 1.5, 'too far from cudalib.bench_ms. Did you wait for the GPU?'
print('your stopwatch is honest')
'''),
    ('md', '''
# Exercise 2: the shape of each mystery

**Given:** three functions that you cannot read, your `honest_ms`, and three
sizes.

**Asked:** the shape of each function: `'flat'`, `'linear'` or `'quadratic'`.
You know that you are correct when the check prints `correct` three times.

Time each mystery at `n = 4096`, `8192` and `16384`. Compute the doubling
ratios. Then write the shape of each one: `'flat'`, `'linear'` or
`'quadratic'`.

Read the ratio at the large end. Remember the trap of the last notebook: at
small sizes, a fixed cost can hide the real shape.
'''),
    ('code', '''
SIZES = [4096, 8192, 16384]
for name, mystery in [('mystery_1', mystery_1), ('mystery_2', mystery_2), ('mystery_3', mystery_3)]:
    times = None            # the time at each size, with honest_ms
    ms_2n_over_n = None           # t(2n) / t(n), for each doubling
    print(name, times, ms_2n_over_n)
''', '''
SIZES = [4096, 8192, 16384]
for name, mystery in [('mystery_1', mystery_1), ('mystery_2', mystery_2), ('mystery_3', mystery_3)]:
    times = [honest_ms(mystery(size)) for size in SIZES]
    ms_2n_over_n = [round(later / earlier, 2) for earlier, later in zip(times, times[1:])]
    print(f'{name}: ' + '  '.join(f'{mystery_ms:.3f} ms' for mystery_ms in times) + f'   ratios {ms_2n_over_n}')
'''),
    ('code', '''
SHAPES = {
    'mystery_1': '',
    'mystery_2': '',
    'mystery_3': '',
}
''', '''
SHAPES = {
    'mystery_1': 'quadratic',
    'mystery_2': 'flat',
    'mystery_3': 'linear',
}
'''),
    ('code', '''
# The check. The answers are stored as hashes.
EXPECTED = ''' + repr({name: __import__('hashlib').sha256(f'{name}:{shape}'.encode()).hexdigest()[:12]
                      for name, shape in SHAPES.items()}) + '''
for name, shape in SHAPES.items():
    ok = hashlib.sha256(f'{name}:{shape.strip().lower()}'.encode()).hexdigest()[:12] == EXPECTED[name]
    print(f'{name}: {"correct" if ok else "not yet"}')
'''),
    ('solution', '''
### Why

- **mystery_1** is quadratic. It computes the distance between every pair of
  `n` points, and it writes an `n x n` result. At small `n` the ratio can be
  larger than 4: the result fits in the cache of the GPU at small sizes, and
  not at large ones. That is a second shape inside the first.
- **mystery_2** is flat. Almost all of its time is a fixed read of 256 MB. The
  part that depends on `n` is too small to see at these sizes.
- **mystery_3** is linear. It reads `n x 65536` numbers one time.

Now open `mysteries.py`, and check your reading against the code.
'''),
    ('md', '''
### After the check

You found the shape of three systems from the outside, with measurements
only. That is the whole of move 4. The course uses it on a transformer in
Part 1, on a scheduler in Part 2, and on your own engine in Part 8.
'''),
]


def main():
    demo(f'{MAP}/part0_map_1_theMapInWordsYouKnow.ipynb', 'The map', 'The map, in words that you know', map_1)
    demo(f'{MAP}/part0_map_2_whereYourInstinctBreaks.ipynb', 'The map', 'Where your instinct breaks: six predictions', map_2)
    challenge(f'{MAP}/part0_map_3_CCwhichLayer_challenge.ipynb',
              f'{MAP}/solutions/part0_map_3_CCwhichLayer.ipynb', 'The map', 'which layer?', map_3)
    demo(f'{MEASURE}/part0_meas_1_aStopwatchThatLies.ipynb', 'How to measure', 'A stopwatch that lies', measure_1)
    demo(f'{MEASURE}/part0_meas_2_howDoesItGrow.ipynb', 'How to measure', 'How does the cost grow?', measure_2)
    challenge(f'{MEASURE}/part0_meas_3_CCmeasureItYourself_challenge.ipynb',
              f'{MEASURE}/solutions/part0_meas_3_CCmeasureItYourself.ipynb', 'How to measure',
              'measure it yourself', measure_3)
    (ROOT / MEASURE / 'mysteries.py').write_text(MYSTERIES)


if __name__ == '__main__':
    main()
