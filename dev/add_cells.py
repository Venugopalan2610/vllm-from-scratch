"""Add the pictures, the bridge tables and other new cells to the notebooks.

Each entry inserts cells after a cell that starts with (or, for a code cell,
contains) a known text, or at the end of the notebook. A picture cell keeps
its drawn output, as a demo must. A code cell that needs the GPU gets no
output here: run the notebook after this script.

The script is idempotent. It skips an entry whose first new cell is already
in the notebook. It never touches a challenge notebook, because a learner
writes in those.

    .venv/bin/python dev/add_cells.py
"""

import io
import json
import secrets
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from course import pictures  # noqa: E402

FIND_ROOT = """import sys
from pathlib import Path
sys.path.insert(0, str(next(folder for folder in [Path.cwd(), *Path.cwd().parents]
                            if (folder/'cudalib').is_dir())))
"""

P0 = 'course/Part0_FromAProgramToAModel'
P1 = 'course/Part1_TheNaiveLoop'
P2 = 'course/Part2_Batching'
P3 = 'course/Part3_PagedAttention'
END = None            # the anchor that means: at the end of the notebook
BRIDGE = '### Where this goes in the ladder'


def md(text):
    return ('markdown', text, None)


def picture(name, *args, find_root=False):
    """A code cell that draws pictures.<name>(*args). args are source text."""
    call = f'pictures.{name}({", ".join(args)})'
    source = ('# The picture. course/pictures.py draws it. It needs no GPU.\n'
              + (FIND_ROOT if find_root else '') + f'from course import pictures\n{call}')
    return ('code', source, name)


def code(source):
    return ('code', source, None)


def bridge(rows):
    """The last cell of a section: which stage uses what, and what is new."""
    lines = [BRIDGE, '',
             'The stages in `app/` that use this section. Read the right column before you',
             'start the stage: the section does not teach it.', '',
             '| Stage | It uses from this section | New in the stage |',
             '|---|---|---|']
    lines += [f'| {stage} | {uses} | {new} |' for stage, uses, new in rows]
    return md('\n'.join(lines))


# The arguments of a picture, when its notebook gives the numbers.
PICTURE_ARGS = {'two_waits': (14e9, 380e9, 7e9, 100e12)}

ENTRIES = [
    # ---------------------------------------------------------------- Part 0
    (f'{P0}/1_model/part0_mdl_1_aModelIsAFunction.ipynb', 'Row *i* of the output predicts', [
        md("""The picture shows the same call with the shapes. The prompt has 5 tokens, so
the output has 5 rows. Each row has one score for each of the 151,936 tokens
of the vocabulary. Stage 01 uses only the last row: `logits[0, -1]`."""),
        picture('forward_shapes')]),

    (f'{P0}/1_model/part0_mdl_2_theGenerationLoop.ipynb', 'eos_id = tokenizer.eos_token_id', [
        md("""**One EOS is not always enough.** The tokenizer gives one EOS id. The model
can have more than one stop token. Read them from
`model.generation_config.eos_token_id`. For Qwen3 it is a LIST:
`[151645, 151643]`, which is `<|im_end|>` and `<|endoftext|>`. On another
model it is one int. Stage 01 must accept both forms."""),
        code("print('the tokenizer:        ', tokenizer.eos_token_id)\n"
             "print('generation_config:    ', model.generation_config.eos_token_id)")]),

    (f'{P0}/1_model/part0_mdl_2_theGenerationLoop.ipynb', '### Count the work', [
        md("""The picture shows the loop with a prompt of P = 3 tokens. Each row is one call
of the model. Each square is one token that goes through all the layers. The
row grows by one square at each step, and the loop computes every square
again. This is the loop of stage 01."""),
        picture('naive_loop')]),

    (f'{P0}/1_model/part0_mdl_2_theGenerationLoop.ipynb', END, [bridge([
        ('01 `naive_generate`',
         '"The simplest loop" and "The reply must stop" in this notebook. Exercise 2 of the challenge.',
         'The input is a tensor of shape `(1, L)` on the device of the model, not a Python list. '
         'Pass `use_cache=False`. The stop tokens are a list. Return only the new ids.'),
        ('03 `model_bytes`',
         '`count_weights` in notebook 1, Step 5.',
         'Add `numel() * element_size()` for each tensor, so that the dtype gives the bytes.'),
        ('13 `sample`',
         '"Choose at random" in this notebook. Exercise 3 of the challenge.',
         'Each row of the batch has its own temperature, [top-k](../../GLOSSARY.md#top-k) and [top-p](../../GLOSSARY.md#top-p).'),
    ])]),

    (f'{P0}/2_attention/part0_att_2_headsAndTheCache.ipynb', 'So there is no reason to compute them again', [
        md("""The picture puts the two loops side by side, with the same prompt of 3 tokens.
On the left, each step computes the whole text again. On the right, the grey
squares come from the cache. Each step after the prefill computes ONE square.
The right side is the loop of stage 02."""),
        picture('with_and_without_cache', find_root=True)]),

    (f'{P0}/2_attention/part0_att_2_headsAndTheCache.ipynb', END, [bridge([
        ('02 `cached_generate`',
         'Step 3: the K and V of a token never change. The count of token passes. '
         'Exercise 2 of the challenge.',
         'The HuggingFace cache, `past_key_values`. A NEW cache for each call. The model gets '
         'the position of a new token from the length of the cache (RoPE, notebook 1).'),
        ('02 `kv_bytes_per_token`',
         'Step 4: the formula.',
         'Read `head_dim` from the config. Part 1, `3_kvCache`, shows why.'),
        ('07 `paged_attention`',
         'Causal attention, notebook 1. GQA, Step 2.',
         'Query head h reads KV head `h // (num_heads // num_kv_heads)`. K and V come from '
         'scattered blocks.'),
    ])]),

    (f'{P0}/3_gpu/part0_gpu_2_aKernelIsALoopBody.ipynb', END, [bridge([
        ('03 all four functions',
         'Notebook 1, Step 3: the ridge point. Step 4: one token is a chain of GEMVs.',
         'Measure it on the real model, and compare the GB/s of decode with the peak.'),
        ('08 `paged_attention_cuda`',
         'Step 1: how a thread finds its index. The grid and the thread blocks.',
         'A real kernel, compiled with nvcc, that must match the stage 07 oracle.'),
        ('08b `paged_attention_vec`',
         'Step 3: the threads of a warp read together. A sector is 32 bytes.',
         'Coalesced, 128-bit loads, and the accumulator in [registers](../../GLOSSARY.md#registers).'),
        ('12 `CUDAGraphRunner`',
         'Step 2: a launch has a fixed cost.',
         'Capture the decode step one time for each batch size, and replay it.'),
    ])]),

    (f'{P0}/4_measure/part0_meas_1_aStopwatchThatLies.ipynb', 'The first number measures how fast Python', [
        md("""The picture shows the two stopwatches. Python launches 5 matmuls and returns
before the GPU does the second one. Stopwatch A stops when Python returns.
Stopwatch B stops after `torch.cuda.synchronize()`, when the GPU finished.
Stage 03 times prefill and decode. It needs stopwatch B."""),
        picture('async_timeline')]),

    (f'{P0}/4_measure/part0_meas_2_howDoesItGrow.ipynb', END, [bridge([
        ('03 `time_prefill`, `time_decode`',
         'Notebook 1: the three rules. Wait for the GPU, warm up, repeat.',
         'Time the real model. Give each decode pass the cache of the pass before it. '
         'Do not copy the cache in the timed loop.'),
        ('02 the checks',
         'Tool 1: double it, and divide.',
         'A check times your cached loop at 32 and 64 tokens. The ratio must be below 2.8: '
         'linear is about 2, quadratic is about 4.'),
        ('04 and later `four_lines`',
         'Tool 1, and the habit: predict first, then measure.',
         'Predict the floor, measure, divide, double. Most stages from 04 on ask for it.'),
    ])]),

    # ---------------------------------------------------------------- Part 1
    (f'{P1}/1_arithmetic/part1_arith_1_readVsCompute.ipynb', 'compute = 100e12', [
        md("""The picture puts the two times of one step side by side, with the numbers of
this notebook. The read and the arithmetic overlap, so the step takes the
LONGER bar. One token is a decode step. 512 tokens is a prefill. The read
is the same for both. Stage 03 measures this difference on your card."""),
        picture('two_waits', 'weight_bytes', 'bandwidth', 'params', 'compute', find_root=True)]),

    (f'{P1}/1_arithmetic/part1_arith_2_whatTheCacheCosts.ipynb', END, [bridge([
        ('03 `achieved_gbs`',
         'Notebook 1, Step 1: time = bytes / bandwidth.',
         'Turn it around: bandwidth = bytes / time, from a time that you measured.'),
        ('02 `kv_bytes_per_token`',
         '"Grouped-query attention is not a detail": use `num_key_value_heads`.',
         'Read `head_dim` from the config too.'),
        ('04 `static_batch_generate`',
         'Notebook 1, Step 3: the batch moves the workload toward the ridge.',
         'A real batch: left [padding](../../GLOSSARY.md#padding), an attention mask, and the padding waste.'),
        ('06 `capacity_contiguous`, `capacity_paged`',
         '"What fits": the batch that memory permits.',
         'Count the sequences that fit with a reservation of max_len, and with blocks.'),
    ])]),

    (f'{P1}/2_roofline/part1_roof_2_theKnee.ipynb', END, [bridge([
        ('03 `time_prefill`, `time_decode`, `achieved_gbs`',
         'Notebook 1: your peak bandwidth. This notebook: the GB/s column. '
         'The challenge: prefill and decode, timed.',
         'The checks call your functions on the real model. `cudalib.bench_ms` and '
         '`cudalib.peak_bandwidth` are given.'),
        ('04 `static_batch_generate`',
         'The knee: below the ridge, a larger batch costs almost no time.',
         'Measure throughput against batch size on the model, not on one matmul.'),
    ])]),

    (f'{P1}/3_kvCache/part1_kv_2_theCostOfRemembering.ipynb', "weight = model.model.layers[0].self_attn.k_proj.weight", [
        md("""The picture shows the cache of one layer, with the shapes of Qwen3-1.7B. The
blue rows are the K and V of earlier tokens. The orange rows are what one new
token adds. Multiply by the 28 layers, and you get the number that
`kv_bytes_per_token` returns in stage 02."""),
        picture('cache_tensor')]),

    (f'{P1}/3_kvCache/part1_kv_2_theCostOfRemembering.ipynb', END, [
        md("""# One cache for each request

A cache holds the K and V of every token that went into it. That is the
point of a cache. It is also a trap.

**Given:** request A, `'The capital of France is'`, and the cache that it
filled. Request B, `'The language that people speak there is'`, from a
different user.

**Asked:** request B uses the cache of A by mistake. This happens when a
cache lives in a global variable, or in a default argument. What does B
see? Does its next token change?

Write your prediction first."""),
        code("my_prediction = None      # the same next token as a new cache? yes or no"),
        picture('stale_cache'),
        code("""from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')

ids_a = tokenizer('The capital of France is', return_tensors='pt').input_ids.cuda()
ids_b = tokenizer('The language that people speak there is', return_tensors='pt').input_ids.cuda()

with torch.inference_mode():
  cache_of_a = model(ids_a, use_cache=True).past_key_values        # request A
  fresh = model(ids_b, use_cache=True)                             # request B, a NEW cache
  stale = model(ids_b, past_key_values=cache_of_a, use_cache=True) # request B, the cache of A

def next_token(model_output):
  return repr(tokenizer.decode(int(model_output.logits[0, -1].argmax())))

print(f'request B, a new cache:     {fresh.past_key_values.get_seq_length():>2} tokens in the cache, '
      f'next token {next_token(fresh)}')
print(f'request B, the cache of A:  {stale.past_key_values.get_seq_length():>2} tokens in the cache, '
      f'next token {next_token(stale)}')
print(f'\\nyou predicted: {my_prediction}')"""),
        md("""The old cache does not crash. It returns a fluent, plausible answer, and the
answer is wrong for user B. "There" now means France, because B reads the K
and V of A.

Prove that claim with a number. If B reads A, then the logits of B with the
old cache must equal the logits of the text A + B, read as one text."""),
        code("""with torch.inference_mode():
  joined = model(torch.cat([ids_a, ids_b], dim=1), use_cache=False).logits[0, -1].float()
stale_logits = stale.logits[0, -1].float()
fresh_logits = fresh.logits[0, -1].float()
print(f'largest difference, the old cache against A + B as one text: {(stale_logits - joined).abs().max():6.3f}')
print(f'largest difference, the old cache against B alone:            {(stale_logits - fresh_logits).abs().max():6.3f}')"""),
        md("""The first difference is small. It is the rounding of bf16: a cached pass and a
full pass add the numbers in a different order (the challenge of this
section, Exercise 5). The second difference is many times larger. The old
cache made B into the end of A.

**Rule.** Make a new cache for each request. Stage 02 has a check that runs
two requests in a row, and it fails on exactly this bug. In stage 06 the
same rule comes back: a freed block holds old K and V, and nothing may read
it before a new token writes it."""),
        bridge([
            ('02 `cached_generate`',
             'The challenge of this section: both loops, and the proof that they agree. '
             'This notebook: one cache for each request.',
             'Stop at the stop tokens. Return only the new ids. A check runs two requests '
             'in a row.'),
            ('02 `kv_bytes_per_token`',
             'This notebook: read `head_dim` from the config. The picture of the cache.',
             'Nothing. The function is the formula.'),
            ('03 `time_decode`',
             'Notebook 1, `decode_ms`: give each pass the cache of the pass before it.',
             'Time `steps` passes and divide. Compare the GB/s with the peak.'),
            ('06 the block allocator',
             'This notebook, "The bill": 60 to 80 percent of KV memory goes to reservation.',
             'Blocks, [block tables](../../GLOSSARY.md#block-table) and a free list, in plain Python.'),
        ])]),

    # ---------------------------------------------------------------- Part 2
    (f'{P2}/1_padding/part2_pad_2_theSlotThatWaited.ipynb', '### What one static batch wastes', [
        md("""The picture shows one small batch: 4 sequences of 8, 4, 6 and 2 tokens. The
batch runs for 8 steps, because A needs 8. Each pink square is a slot that the
batch pays for, after its sequence emitted EOS. `static_waste` below counts
the same thing for real lengths."""),
        picture('static_batch_slots', find_root=True)]),

    (f'{P2}/2_continuous/part2_cont_1_iterationLevel.ipynb', '### Two schedulers, side by side', [
        md("""First the idea, in a picture with 4 slots. On the left, the static batch of
the last notebook. On the right, the same 4 slots under a scheduler that
decides at every step. When a sequence emits EOS, a waiting sequence takes
its slot at the next step. No slot waits."""),
        picture('static_and_continuous', find_root=True)]),

    # ---------------------------------------------------------------- Part 3
    (f'{P3}/1_blocks/part3_blk_2_thePageTable.ipynb', '# A page table for tokens', [
        md("""The picture shows two sequences of 7 and 10 tokens. It uses blocks of 4
slots, so that each slot is visible. The engine and this notebook use 16.
Read the last line: that one lookup is what stage 06 asks for in
`BlockTable.slot`."""),
        picture('block_table', find_root=True)]),

    (f'{P3}/2_gather/part3_gat_2_throughThePageTable.ipynb', '# Attention that reads through the page table', [
        md("""The picture shows one sequence of 10 tokens in blocks of 4. Its blocks are
5, 2 and 7, in that order, and they are not next to each other. The gather
puts them back in table order. Block 7 holds only 2 tokens. Its last 2 slots
hold old data, and the masking trap below is about them."""),
        picture('paged_gather', find_root=True)]),
]


def drawn_output(name):
    """Run one picture function. -> a notebook output with the SVG."""
    captured = {}

    def capture(*args, **kwargs):
        figure = plt.gcf()
        buffer = io.StringIO()
        figure.savefig(buffer, format='svg', bbox_inches='tight')
        width, height = figure.get_size_inches() * figure.dpi
        captured['output'] = {
            'output_type': 'display_data', 'metadata': {},
            'data': {'image/svg+xml': buffer.getvalue(),
                     'text/plain': f'<Figure size {width:.0f}x{height:.0f} with 1 Axes>'}}
        plt.close('all')

    original, plt.show = plt.show, capture
    try:
        getattr(pictures, name)(*PICTURE_ARGS.get(name, ()))
    finally:
        plt.show = original
    return captured['output']


def printed(source):
    """Run the EOS cell, which only prints. -> a notebook stream output."""
    from transformers import AutoTokenizer, GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
    model = type('Model', (), {'generation_config':
                               GenerationConfig.from_pretrained('Qwen/Qwen3-1.7B')})
    buffer = io.StringIO()
    original, sys.stdout = sys.stdout, buffer
    try:
        exec(source, {'tokenizer': tokenizer, 'model': model})
    finally:
        sys.stdout = original
    return {'output_type': 'stream', 'name': 'stdout', 'text': lines(buffer.getvalue())}


def lines(text):
    parts = text.split('\n')
    return [part + '\n' for part in parts[:-1]] + [parts[-1]]


def make_cell(kind, source, name):
    cell = {'cell_type': kind, 'id': secrets.token_hex(4), 'metadata': {}, 'source': lines(source)}
    if kind == 'code':
        cell['execution_count'] = None
        if name:
            cell['outputs'] = [drawn_output(name)]
        elif 'generation_config.eos_token_id' in source:
            cell['outputs'] = [printed(source)]
        else:
            cell['outputs'] = []
    return cell


def main():
    for notebook, anchor, cells in ENTRIES:
        assert '_challenge' not in notebook, notebook
        path = ROOT / notebook
        book = json.loads(path.read_text())
        first = cells[0][1].strip()
        if any(''.join(cell['source']).strip() == first for cell in book['cells']):
            print(f'already there: {path.name}: {first.splitlines()[0][:50]}')
            continue
        if anchor is END:
            index = len(book['cells']) - 1
            while index > 0 and not ''.join(book['cells'][index]['source']).strip():
                index -= 1                    # after the last cell that is not empty
        else:
            index = next(i for i, cell in enumerate(book['cells'])
                         if ''.join(cell['source']).lstrip().startswith(anchor)
                         or (cell['cell_type'] == 'code' and anchor in ''.join(cell['source'])))
        book['cells'][index + 1:index + 1] = [make_cell(*cell) for cell in cells]
        path.write_text(json.dumps(book, indent=1, ensure_ascii=False) + '\n')
        print(f'added: {path.name}: {first.splitlines()[0][:50]}')


if __name__ == '__main__':
    main()
