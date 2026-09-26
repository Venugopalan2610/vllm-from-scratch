"""Part 0, break it on purpose: the model, attention and the GPU."""

PART = 0
NAME = 'From a Program to a Model'
FOLDER = 'course/Part0_FromAProgramToAModel/4_incidents'
IMPORTS = '''import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer'''

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into a working model, a working loop or a working
kernel, and you watch what it does.
''' + '''
The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** When does the output first differ from the reference: which
  token, which prompt, which size?
- **What?** What does the wrong result look like?
- **Which guard?** Which check would catch it?

The reference is a correct greedy loop in `lab.py`, on the attention of HF.
`lab.use_attention(model, fn)` puts your own attention function into every
layer of the model, and `lab.use_attention(model, None)` takes it out.

This notebook needs a GPU with about 8 GB free.
'''

LOAD = '''
### run this cell

MODEL = 'Qwen/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
device = 'cuda'

PROMPT = 'The three largest cities in Japan are'
TOKENS = 30

def report(name, generated, expected):
  lab.report(tokenizer, name, generated, expected)

expected = lab.greedy(model, tokenizer, PROMPT, TOKENS)
report('reference', expected, expected)
'''

DRILLS = [
    ('d1', '''
# Exercise 1: the tokenizer of another model

Encode the prompt with the tokenizer of Llama-2, feed the ids to Qwen3, and
decode the answer with the same wrong tokenizer. Then do it again with the
tokenizer of Qwen2.5, an older model of the same family.

This is Ticket 1 of the incident file. Predict both results. Does the second
tokenizer break anything?
''', '''
llama = AutoTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')   # the Llama-2 tokenizer
older = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
print('entries:  Llama-2', len(llama), '  Qwen2.5', len(older), '  Qwen3', len(tokenizer),
      '  the model', model.config.vocab_size)

for name, wrong in [('Llama-2', llama), ('Qwen2.5', older)]:
  ids = wrong(PROMPT, return_tensors='pt').input_ids.to(device)       # THE FAULT
  print(f'\\n{name} ids: {ids[0].tolist()}')
  print('  what Qwen3 reads:', repr(tokenizer.decode(ids[0])))
  generated = lab.greedy_ids(model, ids, TOKENS)
  print('  largest id written:', max(generated))
  try:
    print('  decoded with the wrong tokenizer:', repr(wrong.decode(generated)))
  except Exception as error:
    print('  decode raised', type(error).__name__, error)
'''),

    ('d2', '''
# Exercise 2: exp() in float16

Load a second copy of the model in float16. First run it with the attention
of HF. Then put in an attention that calls `torch.exp` on the raw scores,
with no subtraction of the maximum. Run a short prompt and a long one, and
record the largest score that each layer sees.

This is Ticket 2. Predict which prompt fails, and what the text looks like.
''', '''
model16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
LONG = ('Summarize the following report. ' + 'The committee met on Tuesday to review the budget, '
        'the hiring plan and the new office lease, and it approved all three with minor changes. ' * 12)
largest = []

def naive_exp_attention(q, k, v):
  scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
  scores = scores + lab.causal_bias(scores)
  largest.append(scores.max().item())
  weights = torch.exp(scores)                                        # THE FAULT: no max subtracted
  weights = weights / weights.sum(-1, keepdim=True)
  return weights @ v

for name, prompt in [('short', PROMPT), ('long', LONG)]:
  ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
  lab.use_attention(model16, None)
  hf = lab.greedy_ids(model16, ids, 15)
  lab.use_attention(model16, naive_exp_attention)
  largest.clear()
  with torch.inference_mode():
    logits = model16(ids).logits
  layer_max = largest[:model.config.num_hidden_layers]
  naive = lab.greedy_ids(model16, ids, 15)
  print(f'{name}: {ids.shape[1]} tokens, largest score {max(layer_max):.1f} '
        f'(layer {layer_max.index(max(layer_max))}), finite logits: {torch.isfinite(logits).all().item()}')
  print('   HF float16:   ', repr(tokenizer.decode(hf)))
  print('   naive float16:', repr(tokenizer.decode(naive)))
print('exp() overflows float16 above', round(math.log(torch.finfo(torch.float16).max), 2))
lab.use_attention(model16, None)
del model16; torch.cuda.empty_cache()
'''),

    ('d3', '''
# Exercise 3: the attention with no causal mask

The function forgets the causal mask. First run the unit test of Ticket 3:
one query against 50 keys. Then run the same test with 20 queries. Then put
the function into the model.

Predict: at which token does the model go wrong? The loop has a KV cache, so
every decode step has exactly one query.
''', '''
def no_mask_attention(q, k, v):
  scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
  return torch.softmax(scores.float(), -1).to(q.dtype) @ v            # THE FAULT: no mask

torch.manual_seed(0)
for queries in (1, 20):
  q = torch.randn(1, 16, queries, 128, device=device)
  k = torch.randn(1, 16, 50, 128, device=device)
  v = torch.randn(1, 16, 50, 128, device=device)
  want = lab.correct_attention(q, k, v)
  print(f'unit test with {queries:2d} queries: largest difference {(no_mask_attention(q, k, v) - want).abs().max().item():.4f}')

lab.use_attention(model, no_mask_attention)
report('no mask', lab.greedy(model, tokenizer, PROMPT, TOKENS), expected)
lab.use_attention(model, None)
'''),

    ('d4', '''
# Exercise 4: the kernel at a fraction of a percent of peak

Time the RMSNorm of the model on 32,768 tokens x 2,048 values. Report it in
TFLOP/s, as the team of Ticket 4 did, and in GB/s. Compare it with a plain
copy of the same tensor, with the same RMSNorm after `torch.compile`, and
with the numbers of `./vc info`.

The tensor is 134 MB, larger than the L2 cache of the card (`./vc info`). A
tensor that fits in L2 does not measure the memory.

Predict the fractions: of the compute peak, and of the copy bandwidth.
''', '''
PEAK_TFLOPS, PEAK_GBS = 49, 294        # the sustained numbers of ./vc info on your card

x = torch.randn(32768, 2048, dtype=torch.bfloat16, device=device)
norm = model.model.norm
compiled = torch.compile(norm)

@torch.inference_mode()
def ms_per_call(fn, iters=50):
  for _ in range(5):
    fn()
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - start) / iters * 1000

bytes_moved = 2 * x.numel() * x.element_size()           # read x, write y
flop = 4 * x.numel()
for name, fn in [('RMSNorm, HF eager', lambda: norm(x)), ('RMSNorm, compiled', lambda: compiled(x)),
                 ('clone            ', lambda: x.clone())]:
  ms = ms_per_call(fn)
  print(f'{name} {ms:.3f} ms   {flop / ms / 1e9:6.3f} TFLOP/s = {flop / ms / 1e9 / PEAK_TFLOPS:.2%} of peak   '
        f'{bytes_moved / ms / 1e6:5.0f} GB/s = {bytes_moved / ms / 1e6 / PEAK_GBS:.0%} of ./vc info')
'''),

    ('d5', '''
# Exercise 5: load with no dtype

Load the model on the CPU two times: once with no dtype, as the code of
Ticket 5 does in an old version of transformers, and once in bfloat16. Count
the bytes of the weights, and the bytes of one MLP matrix.

Predict the two totals before you run. The model has 1.72 B parameters.
''', '''
def model_bytes(m):
  return sum(p.numel() * p.element_size() for p in m.parameters())

for dtype in (torch.float32, torch.bfloat16):                       # float32: THE FAULT of the old default
  cpu_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
  matrix = cpu_model.model.layers[0].mlp.up_proj.weight
  print(f'{str(dtype):15s} {model_bytes(cpu_model) / 1e9:5.2f} GB, one MLP matrix {tuple(matrix.shape)} = '
        f'{matrix.numel() * matrix.element_size() / 2**20:.0f} MiB')
  del cpu_model
'''),

    ('d6', '''
# Exercise 6: generate() with no sampling arguments

Call `model.generate` three times with only `max_new_tokens`, as the harness
of Ticket 6 does. Then three times with `torch.manual_seed(0)` before each
call. Then with `do_sample=False`.

Predict which of the three groups agree with each other, and which one
agrees with the greedy reference.
''', '''
ids = tokenizer(PROMPT, return_tensors='pt').to(device)
print('generation_config:', {k: getattr(model.generation_config, k)
                             for k in ('do_sample', 'temperature', 'top_k', 'top_p')})

def run(**kwargs):
  out = model.generate(**ids, max_new_tokens=TOKENS, **kwargs)        # THE FAULT: no do_sample
  return out[0, ids.input_ids.shape[1]:].tolist()

plain = [run() for _ in range(3)]
seeded = []
for _ in range(3):
  torch.manual_seed(0)
  seeded.append(run())
greedy = run(do_sample=False)
for name, group in [('no arguments', plain), ('seed 0 each time', seeded)]:
  print(f'{name}: {len({tuple(g) for g in group})} different answers of 3; first differences with the reference:',
        [lab.first_difference(g, expected) for g in group])
report('do_sample=False', greedy, expected)
'''),

    ('d7', '''
# Exercise 7: generate() with no length

Call `model.generate` with no length at all, for prompts of several lengths.
Count the prompt tokens and the new tokens.

This is Ticket 7. Predict the new tokens for each prompt.
''', '''
import warnings
for prompt in ['Hello', 'The three largest cities in Japan are',
               'Write a long story about a robot who learns to paint, and about its first exhibition']:
  ids = tokenizer(prompt, return_tensors='pt').to(device)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = model.generate(**ids, do_sample=False)                      # THE FAULT: no max_new_tokens
  prompt_len = ids.input_ids.shape[1]
  new = out.shape[1] - prompt_len
  print(f'prompt {prompt_len:2d} + new {new:2d} = {prompt_len + new}')
print('warnings:', {str(w.message)[:90] for w in caught})
'''),

    ('d8', '''
# Exercise 8: the chat model with no chat template

Send a question as raw text, then the same question in the chat template.

This is Ticket 8. Predict the first words of each answer.
''', '''
question = 'What is the capital of France?'
raw = tokenizer(question, return_tensors='pt').input_ids.to(device)             # THE FAULT
chat = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                     add_generation_prompt=True, enable_thinking=False,
                                     return_tensors='pt', return_dict=True).input_ids.to(device)
stop = set(model.generation_config.eos_token_id)
for name, ids in [('raw text', raw), ('chat template', chat)]:
  count = (ids == 151644).sum().item()
  answer = lab.greedy_ids(model, ids, 40, stop)
  print(f'{name}: {ids.shape[1]} prompt tokens, <|im_start|> appears {count} times')
  print('   ', repr(tokenizer.decode(answer)))
'''),

    ('d9', '''
# Exercise 9: the grid that rounds down

A vector add in CUDA with `blocks = n / threads`, and the bounds check
inside the kernel. Run it for several sizes, and count the wrong elements.

This is Ticket 9. Predict the count for each size before you run. The first
run compiles the kernel, which takes 20 to 40 seconds.
''', '''
import cudalib

KERNEL = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__global__ void add(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

void launch(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  int n = a.numel();
  int threads = 256;
  int blocks = n / threads;                          // THE FAULT: rounds down
  if (blocks > 0)
    add<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""
extension = cudalib.build_source('inc0_vector_add', KERNEL)

for n in (1024, 4096, 65536, 1000, 5000, 1009, 100):
  a, b = torch.rand(n, device=device), torch.rand(n, device=device)
  out = torch.zeros(n, device=device)
  extension.launch(a, b, out)
  wrong = (out != a + b).nonzero().flatten()
  where = f'{wrong[0].item()} to {wrong[-1].item()}' if len(wrong) else '-'
  print(f'n = {n:6d}: {len(wrong):4d} wrong   ({where})')
'''),
]

MYSTERY = '''
# Exercise 10: three mystery attentions

The module `mystery.py` holds three attention functions: `attention_a`,
`attention_b` and `attention_c`. Each one has one fault. **Do not open the
file.** Put each one into the model with `lab.use_attention`, and find the
fault from its behaviour.

For each function:

1. Run it on the default prompt, and compare with the reference.
2. Design a second experiment that makes the fault visible. Which variable do
   you change? The prompt? Its length? The number of queries?
3. Write your diagnosis: the fault, and the experiment that proved it.
4. Only then, open `mystery.py` and check.

A hint about the method: one function is correct for short prompts. One
function is wrong only when there is more than one query. One function breaks
everything, but you must still explain **how**.
'''

MYSTERY_CODE = '''
from mystery import attention_a, attention_b, attention_c

for name, attention in [('a', attention_a), ('b', attention_b), ('c', attention_c)]:
  lab.use_attention(model, attention)
  report(f'attention_{name}', lab.greedy(model, tokenizer, PROMPT, TOKENS), expected)
lab.use_attention(model, None)
'''

MYSTERY_NAMES = ['attention_a', 'attention_b', 'attention_c']

FINGERPRINT_ROWS = ['a tokenizer of another model', 'exp() in float16', 'no causal mask',
                    'a memory-bound kernel on the FLOP ruler', 'no dtype at load',
                    'generate() with the model defaults', 'generate() with no length',
                    'no chat template', 'a grid that rounds down']

SOLUTIONS = {
    'd1': '''
### What happens

- **Crash?** No. On my run the decode did not raise either, because the model
  wrote only ids below 32,000 (the largest was 16,252).
- **When?** At token 0.
- **What?** The Llama-2 ids mean other pieces to Qwen3. The model read
  `'"deatabaseENCE� o counter int'`, not the prompt, and the answer decoded to
  a mix of scripts: `'Anкоde��� Tвиtembre des��++...'`.
- **The second tokenizer.** Qwen2.5 made **the same ids** as Qwen3 for this
  prompt, and the answer was identical to the reference. Every one of the
  151,665 entries of Qwen2.5 has the same id in Qwen3. Qwen3 adds only 4
  special tokens, such as `<think>` (151667), which Qwen2.5 encodes as 3
  ordinary tokens. So this mismatch is invisible until a prompt contains one
  of the 4.

**The fingerprint.** Text in many scripts from token 0. And the dangerous
version: a mismatch that gives correct text on every test prompt.

**The guard.** Compare the ids of a known text with the known ids, and
include a special token in that text.
''',
    'd2': '''
### What happens

- **Crash?** No. The logits are `nan`, and `argmax` returns 0.
- **When?** At token 0, for **both** prompts.
- **What?** `'!!!!!!!!!!!!!!!'`, token 0 of the vocabulary, over and over.

The surprise is the short prompt. On my run the largest score was 31.5 in
layer 0 for a prompt of 7 tokens, and 32.5 for 344 tokens. Both are far above
11.09, so `exp()` overflowed for both. Large scores are normal in a real
model. The first layers often put one very large score on the first token.

The HF attention in float16 gave the correct text for both prompts, on the
same weights. It subtracts the maximum before `exp()`. Only the naive
function breaks.

**The fingerprint.** Only `!` (token 0), from the first token: `nan` logits.

**The guard.** `torch.isfinite(logits).all()` in every test.
''',
    'd3': '''
### What happens

- **Crash?** No.
- **When?** The unit test with 1 query passed with a difference of 0.0000.
  With 20 queries the difference was 1.13. In the model, the text was wrong
  from token 0.
- **What?** The model **repeated the prompt**: `' the three largest cities in
  Japan are the three largest cities in Japan are ...'`. Each prompt token
  saw the tokens after it, so in the prefill each position "knew" its own
  next token. The model then learned nothing useful at the last position.

The decode steps have one query each, so they do not need a mask. But they
read the K and V that the broken prefill wrote. The damage is in the cache.

**The fingerprint.** The prompt comes back, from token 0.

**The guard.** A test with the prefill shape: many queries. And a test of
the whole model against HF.
''',
    'd4': '''
### What happens

On my card (RTX 4080 Laptop GPU), in two runs on two days:

| | run 1 | run 2 | TFLOP/s (run 2) | % of the compute peak |
|---|---|---|---|---|
| RMSNorm, HF eager | 7.31 ms, 37 GB/s | 6.26 ms, 43 GB/s | 0.043 | 0.09% |
| RMSNorm, compiled | 0.92 ms, 293 GB/s | 0.71 ms, 380 GB/s | 0.380 | 0.78% |
| clone | 0.88 ms, 305 GB/s | 0.71 ms, 380 GB/s | 0.380 | 0.78% |

On the FLOP ruler, all three look terrible, and the ruler cannot tell them
apart. On the GB/s ruler, the compiled norm is as fast as a copy in both runs:
nothing is left to win. The **eager** norm is 8 to 9 times slower than a copy.
That is a real problem, and only the right ruler shows it.

The absolute numbers moved by 25% between the two runs, and the copy passed the
294 GB/s that `./vc info` measured on another day. A laptop changes its
memory clock with its temperature and its power mode. That is why the exercise
times a copy **in the same run**: the ratio to the copy is stable, and the
absolute number is not.

Why the eager norm is slow: it runs five separate operations (convert to
float32, square, mean, multiply, convert back, multiply by the weight). Each
one reads and writes the whole tensor, some in float32. Its time says that it
moves about 8 times the bytes of one read and one write. `torch.compile` fuses them into one
kernel.

My first version of this drill used 8,192 tokens, 33.5 MB. That fits in the
50 MB L2 cache of this card, and the copy then ran at 139% of `./vc info`.
A benchmark that fits in the cache measures the cache.

**The fingerprint.** A tiny fraction of the compute peak means nothing for a
memory-bound kernel. Measure GB/s, and compare with a copy.
''',
    'd5': '''
### What happens

- **Crash?** No, on the CPU.
- **What?** 6.88 GB in float32, 3.44 GB in bfloat16. One MLP matrix of
  6,144 x 2,048 is 48 MiB in float32 and 24 MiB in bfloat16. The total
  doubles, and so does every single allocation.

**The fingerprint.** Twice the expected bytes, and an allocation that is
twice the size of a known matrix.
''',
    'd6': '''
### What happens

- **Crash?** No.
- **What?** The config of Qwen3 says `do_sample: True`, temperature 0.6. On
  my run, the three plain calls gave 3 different answers. They left the
  reference at tokens 7, 14 and 16.
- With `torch.manual_seed(0)` before each call, the three answers were
  **identical** to each other, and all three left the reference at token 14.
  A fixed seed makes a sampler repeat itself. It does not make it greedy.
- With `do_sample=False`, the answer equals the reference.

**The fingerprint.** Answers that change from run to run, and stop changing
with a fixed seed. That is a random number generator, not numerics.
''',
    'd7': '''
### What happens

In transformers 5.14 on my machine, every call gave **20 new tokens**, for
prompts of 1, 7 and 17 tokens. Old versions (4.x) used `max_length = 20`
instead, which counts the prompt too. That is Ticket 7: the new tokens were
20 minus the prompt.

Both defaults have the same fingerprint: a wall at one number. The versions
differ in which number is constant: the new tokens, or the total.

**The guard.** Always pass `max_new_tokens`. Record why each answer ended.
''',
    'd8': '''
### What happens

- **Raw text.** 7 prompt tokens, and `<|im_start|>` appears 0 times. The model
  continued the question with more questions: `' Also, can you tell me about
  the famous French painter who painted the Mona Lisa? What is the largest
  planet ...'`.
- **Chat template.** 19 prompt tokens, `<|im_start|>` appears 2 times. The
  answer: `'The capital of France is Paris.'`, and then the stop token.

**The fingerprint.** The assistant writes the next part of the document, not
an answer. The count of the special tokens proves it.
''',
    'd9': '''
### What happens

| n | wrong elements | where |
|---|---|---|
| 1,024, 4,096, 65,536 | 0 | |
| 1,000 | 232 | 768 to 999 |
| 5,000 | 136 | 4,864 to 4,999 |
| 1,009 | 241 | 768 to 1,008 |
| 100 | 100 | all of them |

Exactly `n mod 256` elements are wrong, always at the end. Every size that
passed is a multiple of 256. For n = 100 the grid has 0 blocks, and nothing
runs.

**The fingerprint.** The last `n mod block_size` elements are wrong. The
tests with round sizes pass.
''',
}

MYSTERY_SOL = '''
### The three mysteries

**`attention_a`: a window of 64 keys.** It is perfect on the default prompt.
The prompt and the answer together have 37 positions, fewer than 64, so the
window cuts nothing. The experiment: a prompt longer than 64 tokens, with the
important fact at its start. Or generate more than 57 tokens and watch the
answer lose the start of the prompt. The fault depends on a property of the
input, its length, and a short test cannot see it.

**`attention_b`: each query also sees the next token.** On my run it left the
reference at token 5, with fluent text: `' Tokyo, Osaka, and Osaka. What is
the total number of letters ...'`. The experiment: compare the logits of
**every** prefill position with the reference, not only the last one. The last
position has no next token, so its row is correct in layer 0. The other rows
are wrong. In the decode steps, one query sees every key legally, so the
decode arithmetic is correct. Only the K and V from the prefill are wrong.
This is Exercise 3 in a mild form.

**`attention_c`: the softmax over the queries.** On my run: `' and and and
and ...,,,,'` from token 0. The explanation is the part to find. In a decode
step there is one query, so a softmax over the queries gives 1.0 to **every**
key. The output is the **sum** of all the values, not an average. In the
prefill, each column is normalized over the queries instead of each row over
the keys. The experiment: call the function on a tiny q, k and v, and check
that each row of the weights sums to 1. It does not.

### The method

| Mystery | The variable that the default test held constant |
|---|---|
| a | the length of the context |
| b | the number of queries: the test compared only the last position |
| c | nothing: the fault is loud. The work is to explain it |
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| a tokenizer of another model | no | token 0, or never | text in many scripts, or nothing at all | known text, known ids, with a special token |
| exp() in float16 | no | token 0 | only `!` | `isfinite(logits)` |
| no causal mask | no | token 0 | the prompt comes back | a test with many queries |
| a memory-bound kernel on the FLOP ruler | no | never | a tiny % of peak for a good kernel and a bad one | GB/s against a copy |
| no dtype at load | OOM on a small card | at load | twice the bytes | assert the dtype |
| generate() with the model defaults | no | a few tokens in | answers that change, and stop changing with a seed | log the effective settings |
| generate() with no length | no | at 20 tokens | a wall at one number | `max_new_tokens`, and the finish reason |
| no chat template | no | token 0 | the question continues | count the special tokens |
| a grid that rounds down | no | the last `n mod 256` elements | zeros at the end | test sizes that are not round |

Only one of the nine faults can crash, and only on a small card. The others
give output. A good part of this table is about the ruler, not the code: a
FLOP/s number, a seed, and a round test size all hid a fault.
'''
