"""The pictures of the notebooks: drawings of a mechanism, not plots of data.

Each function draws one picture with matplotlib, and needs no GPU and no
model. A notebook calls it in one line:

    from course import pictures
    pictures.naive_loop()

The rules are the rules of docs/PICTURES.md:

- A picture shows the mechanism with real numbers, not a metaphor.
- Each symbol has its definition in the picture.
- Two different quantities never have the same example number.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROMPT_COLOR = '#4c78a8'       # a token of the prompt
NEW_COLOR = '#f58518'          # a token that the loop made
COMPUTED_EDGE = '#222222'
CACHED_COLOR = '#d9d9d9'       # read from the cache, not computed
HIGHLIGHT = '#e45756'
WASTE_COLOR = '#fbe3e1'      # a slot that is paid for and not used
TEXT = '#222222'


def _canvas(width, height):
    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect('equal')
    axis.axis('off')
    return figure, axis


def _box(axis, x, y, width, height, color, text='', text_color='white', size=9,
         edge=COMPUTED_EDGE, weight='normal', line=0.8):
    axis.add_patch(Rectangle((x, y), width, height, facecolor=color,
                             edgecolor=edge, linewidth=line))
    if text:
        axis.text(x + width / 2, y + height / 2, text, ha='center', va='center',
                  color=text_color, fontsize=size, weight=weight)


def _arrow(axis, start, end, color=TEXT, style='-|>', width=1.2):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                   color=color, linewidth=width))


def _show(figure):
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Part 0, 1_model: one forward pass, as shapes
# --------------------------------------------------------------------------

def forward_shapes():
    """Token ids (1, L) go in. Logits (1, L, vocab) come out. Only the last
    row chooses the next token."""
    pieces = ['The', ' capital', ' of', ' France', ' is']
    ids = [785, 6722, 315, 9625, 374]
    figure, axis = _canvas(12, 5.4)

    # The input: one row of L = 5 token ids.
    axis.text(0.3, 5.15, 'input_ids: shape (1, L) = (1, 5)', fontsize=10, weight='bold')
    for position, (piece, token_id) in enumerate(zip(pieces, ids)):
        y = 4.0 - 0.75 * position
        _box(axis, 0.3, y, 1.2, 0.6, PROMPT_COLOR, str(token_id))
        axis.text(1.6, y + 0.3, f'{piece!r}', va='center', fontsize=8, color=TEXT)
        axis.text(0.15, y + 0.3, str(position), va='center', ha='right', fontsize=8, color='#666')
    axis.text(0.15, 4.75, 'pos', ha='right', fontsize=8, color='#666')

    # The model.
    _arrow(axis, (2.9, 2.75), (3.9, 2.75))
    _box(axis, 3.9, 1.9, 1.6, 1.7, '#54a24b', 'model(\ninput_ids\n).logits', size=9)
    _arrow(axis, (5.5, 2.75), (6.3, 2.75))

    # The output: one row of vocab scores for each position.
    axis.text(6.3, 5.15, 'logits: shape (1, L, vocab) = (1, 5, 151936)', fontsize=10, weight='bold')
    predicts = [' capital?', ' of?', ' France?', ' is?', ' Paris']
    for position in range(5):
        y = 4.0 - 0.75 * position
        last = position == 4
        for column in range(10):
            shade = HIGHLIGHT if (last and column == 6) else ('#f2b8b5' if last else '#e8e8e8')
            _box(axis, 6.3 + 0.32 * column, y, 0.32, 0.6, shade, line=0.4, edge='#999')
        axis.text(9.6, y + 0.3, '...', va='center', fontsize=9)
        axis.text(10.0, y + 0.3, f'row {position} scores the token at pos {position + 1}',
                  va='center', fontsize=8, color=HIGHLIGHT if last else '#666',
                  weight='bold' if last else 'normal')
    axis.text(6.3, 4.72, 'one column = one token of the vocabulary (151,936 columns)',
              fontsize=8, color='#666')

    # The choice.
    _arrow(axis, (8.4, 0.95), (8.4, 0.5), color=HIGHLIGHT)
    axis.text(6.3, 0.15, "logits[0, -1]  ->  argmax  ->  the id of ' Paris'  "
              "(the red column: the highest score of the LAST row)",
              fontsize=9, color=HIGHLIGHT, weight='bold')
    axis.set_title('One forward pass: L token ids in, L rows of scores out', fontsize=11)
    _show(figure)


# --------------------------------------------------------------------------
# Part 0, 1_model and 2_attention: the loop with and without a cache
# --------------------------------------------------------------------------

def _token_grid(axis, x0, y0, prompt_len, steps, cached, cell=0.55):
    """Draw one row for each step. Return the number of computed cells."""
    computed = 0
    for step in range(1, steps + 1):
        length = prompt_len + step - 1             # the tokens in the input of this step
        y = y0 - (step - 1) * (cell + 0.18)
        axis.text(x0 - 0.15, y + cell / 2, f'step {step}', ha='right', va='center', fontsize=8)
        for position in range(length):
            is_prompt = position < prompt_len
            is_new_this_step = (step == 1) or position == length - 1
            if cached and not is_new_this_step:
                color, text_color = CACHED_COLOR, '#777'
            else:
                color, text_color = (PROMPT_COLOR if is_prompt else NEW_COLOR), 'white'
                computed += 1
            label = f'p{position}' if is_prompt else f'n{position - prompt_len + 1}'
            _box(axis, x0 + position * cell, y, cell, cell, color, label, text_color, size=7)
        new_cells = length if (step == 1 or not cached) else 1
        kind = 'prefill' if step == 1 else ('decode' if cached else 'full pass')
        axis.text(x0 + length * cell + 0.15, y + cell / 2,
                  f'{new_cells} computed ({kind}) -> makes n{step}', va='center', fontsize=8)
    return computed


def naive_loop(prompt_len=3, steps=5):
    """Stage 01: each step runs the model on the WHOLE text again."""
    figure, axis = _canvas(10, 5.2)
    total = _token_grid(axis, 1.3, 4.0, prompt_len, steps, cached=False)
    lengths = ' + '.join(str(prompt_len + step - 1) for step in range(1, steps + 1))
    axis.text(0.2, 0.55, f'p = a prompt token (P = {prompt_len}).   '
              'n = a token that the loop made.   Each colored square is one token through '
              'all the layers.', fontsize=8)
    axis.text(0.2, 0.1, f'{steps} steps compute {lengths} = {total} token passes. '
              f'Step t computes P + t - 1. N steps cost about N x N / 2: O(N^2).',
              fontsize=9, weight='bold')
    axis.text(6.6, 4.65, 'after each step:', fontsize=8, weight='bold')
    axis.text(6.6, 4.3, 'argmax of the last row = the next token,\nappended to the input of the next step',
              fontsize=8, va='top')
    axis.text(6.6, 3.5, 'stop when the next token is in\n'
              'model.generation_config.eos_token_id\n= [151645, 151643] for Qwen3\n'
              '(a LIST: two stop tokens)', fontsize=8, va='top', color=HIGHLIGHT)
    axis.set_title('The naive loop: the whole text through the model, at every step', fontsize=11)
    _show(figure)


def with_and_without_cache(prompt_len=3, steps=5):
    """Stage 02: compute the K and V of each token once, and keep them."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    totals = []
    for axis, cached, title in zip(axes, (False, True),
                                   ('WITHOUT a cache (stage 01)', 'WITH a KV cache (stage 02)')):
        axis.set_xlim(0, 7.6)
        axis.set_ylim(0, 4.8)
        axis.set_aspect('equal')
        axis.axis('off')
        totals.append(_token_grid(axis, 1.0, 3.9, prompt_len, steps, cached=cached))
        axis.set_title(title, fontsize=11)
    axes[0].text(0.1, 0.1, f'token passes: {totals[0]}', fontsize=10, weight='bold')
    axes[1].text(0.1, 0.1, f'token passes: {totals[1]}   (prefill {prompt_len}, then 1 for each step)',
                 fontsize=10, weight='bold')
    axes[1].text(0.1, 0.55, 'grey = K and V read from the cache, not computed again',
                 fontsize=8, color='#555')
    _show(figure)


# --------------------------------------------------------------------------
# Part 1, 3_kvCache: what one token adds to the cache
# --------------------------------------------------------------------------

def cache_tensor(layers=28, kv_heads=8, head_dim=128, tokens=6, bytes_per_value=2):
    """The cache of one layer is (batch, kv_heads, tokens, head_dim), for K and
    again for V. A new token adds one row of head_dim to each KV head."""
    figure, axis = _canvas(13, 8.2)
    slab, gap, row = 1.15, 0.18, 0.3
    for tensor_index, tensor in enumerate(('K', 'V')):
        y_top = 7.0 - tensor_index * (tokens * row + 0.9)
        axis.text(0.15, y_top - tokens * row / 2, tensor, fontsize=14, weight='bold', va='center')
        for head in range(kv_heads):
            x = 0.7 + head * (slab + gap)
            if tensor_index == 0:
                axis.text(x + slab / 2, y_top + 0.12, f'KV head {head}', ha='center', fontsize=7)
            for token in range(tokens):
                new = token == tokens - 1
                _box(axis, x, y_top - (token + 1) * row, slab, row,
                     NEW_COLOR if new else PROMPT_COLOR, edge='white', line=0.6)
        x_end = 0.7 + kv_heads * (slab + gap)
        axis.text(x_end, y_top - (tokens - 0.5) * row, '<- the NEW token: one row of\n'
                  f'   {head_dim} numbers in each KV head', fontsize=8, color=NEW_COLOR, va='center')
        axis.text(x_end, y_top - 1.5 * row, f'<- tokens 0 to {tokens - 2}:\n   already in the cache',
                  fontsize=8, va='center')
    axis.text(0.7, 7.65, f'ONE layer. Each tensor has the shape (batch=1, kv_heads={kv_heads}, '
              f'tokens, head_dim={head_dim}). The tokens axis grows by 1 at each step.',
              fontsize=9, weight='bold')
    per_token = 2 * layers * kv_heads * head_dim * bytes_per_value
    lines = [
        'Bytes that ONE new token adds to the cache, in all layers:',
        f'      2   K and V',
        f'  x {layers:>4}   layers     config.num_hidden_layers',
        f'  x {kv_heads:>4}   KV heads   config.num_key_value_heads  (NOT num_attention_heads)',
        f'  x {head_dim:>4}   head_dim   config.head_dim  (read it, do not derive it)',
        f'  x {bytes_per_value:>4}   bytes for one bf16 number',
        f'  = {per_token:,} bytes = {per_token // 1024} KiB',
    ]
    axis.text(0.7, 2.15, '\n'.join(lines), fontsize=9, family='monospace', va='top')
    axis.set_title(f'The KV cache: the orange rows are what one token costs (x {layers} layers)',
                   fontsize=11)
    _show(figure)


# --------------------------------------------------------------------------
# Part 0, 4_measure: why a stopwatch needs torch.cuda.synchronize()
# --------------------------------------------------------------------------

def async_timeline(num_kernels=5):
    """Python puts work in a queue and returns. The GPU does the work later."""
    figure, axis = _canvas(12, 4.6)
    launch, kernel = 0.35, 1.5
    cpu_y, gpu_y = 3.0, 1.7
    axis.text(0.2, cpu_y + 0.25, 'CPU\n(Python)', va='center', fontsize=9, weight='bold')
    axis.text(0.2, gpu_y + 0.25, 'GPU', va='center', fontsize=9, weight='bold')
    x_cpu, x_gpu = 1.6, 1.8
    for index in range(num_kernels):
        _box(axis, x_cpu, cpu_y, launch, 0.5, PROMPT_COLOR, 'L', size=8)
        start = max(x_gpu, x_cpu + launch)
        _box(axis, start, gpu_y, kernel, 0.5, NEW_COLOR, f'matmul {index + 1}', size=8)
        _arrow(axis, (x_cpu + launch / 2, cpu_y), (start + 0.1, gpu_y + 0.5), color='#999', width=0.7)
        x_cpu += launch + 0.05
        x_gpu = start + kernel
    cpu_end, gpu_end = x_cpu, x_gpu
    for x, color, label, y in (
            (cpu_end, HIGHLIGHT, 'stopwatch A stops here:\nPython returned.\nThe GPU is not finished.', 1.0),
            (gpu_end, '#54a24b', 'stopwatch B stops here,\nafter torch.cuda.synchronize():\nthe GPU finished.', 1.0)):
        axis.plot([x, x], [0.95, 3.7], color=color, linestyle='--', linewidth=1.5)
        axis.text(x + 0.08, y - 0.15, label, fontsize=8, color=color, va='top')
    axis.plot([1.6, 1.6], [0.95, 3.7], color=TEXT, linestyle=':', linewidth=1)
    axis.text(1.65, 3.75, 'start', fontsize=8)
    axis.text(1.6, 4.25, 'L = a launch: Python puts one matmul in the queue, and returns at once.',
              fontsize=8)
    axis.set_title('Asynchronous execution: the CPU runs ahead of the GPU', fontsize=11)
    _show(figure)


# --------------------------------------------------------------------------
# Part 1, 1_arithmetic: what one step waits for
# --------------------------------------------------------------------------

def two_waits(weight_bytes, bandwidth, params, compute, token_counts=(1, 512)):
    """One step needs two things: read every weight, and do the arithmetic.
    The two overlap, so the step takes the LONGER of the two."""
    read_ms = 1000 * weight_bytes / bandwidth
    rows = []
    for tokens in token_counts:
        rows.append((tokens, read_ms, 1000 * 2 * params * tokens / compute))
    longest = max(max(read, math) for _, read, math in rows)
    figure, axis = plt.subplots(figsize=(10, 1.4 + 1.3 * len(rows)))
    labels = []
    for index, (tokens, read, math) in enumerate(rows):
        y = len(rows) - 1 - index
        axis.barh(y + 0.18, read, height=0.32, color=PROMPT_COLOR)
        axis.barh(y - 0.18, math, height=0.32, color=NEW_COLOR)
        axis.text(read + longest * 0.01, y + 0.18, f'read the weights: {read:.1f} ms',
                  va='center', fontsize=8)
        axis.text(math + longest * 0.01, y - 0.18, f'the arithmetic: {math:.2f} ms',
                  va='center', fontsize=8)
        waits_on = 'memory' if read > math else 'arithmetic'
        step = max(read, math)
        axis.plot([step, step], [y - 0.4, y + 0.4], color=HIGHLIGHT, linewidth=2)
        name = 'a decode step' if tokens == 1 else 'a prefill'
        labels.append(f'{name}\n{tokens} token{"s" if tokens > 1 else ""}\n'
                      f'step = {step:.1f} ms\nwaits on {waits_on}')
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(labels[::-1], fontsize=8)
    axis.set_xlim(0, longest * 1.45)
    axis.set_xlabel('Milliseconds. The red line is the time of the step: the longer bar.')
    axis.set_title('The same weights, read once. Only the arithmetic grows with the tokens.',
                   fontsize=11)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    _show(figure)


# --------------------------------------------------------------------------
# Part 1, 3_kvCache: a cache from an earlier request
# --------------------------------------------------------------------------

def stale_cache():
    """Stage 02 trap: a cache that an earlier request filled. The new request
    reads the old tokens, and it gets the wrong positions."""
    request_a = ['The', ' capital', ' of', ' France', ' is']
    request_b = ['The', ' language', ' that', ' people', ' speak', ' there', ' is']
    figure, axis = _canvas(13, 4.6)
    cell = 0.78
    rows = (('A NEW cache for request B', [], 2.9),
            ('the OLD cache of request A, used again', request_a, 1.0))
    for title, old_tokens, y in rows:
        axis.text(0.3, y + 1.0, title, fontsize=10, weight='bold',
                  color=TEXT if not old_tokens else HIGHLIGHT)
        tokens = [(piece, CACHED_COLOR, '#555') for piece in old_tokens] + \
                 [(piece, PROMPT_COLOR, 'white') for piece in request_b]
        for position, (piece, color, text_color) in enumerate(tokens):
            x = 0.3 + position * cell
            _box(axis, x, y, cell, 0.6, color, piece.strip(), text_color, size=7)
            axis.text(x + cell / 2, y - 0.2, f'pos {position}', ha='center', fontsize=7, color='#666')
        end = 0.3 + len(tokens) * cell
        if old_tokens:
            axis.text(end + 0.2, y + 0.3, f'the cache holds {len(tokens)} tokens.\n'
                      'B reads the K and V of A.\nB starts at pos 5, not pos 0.\n'
                      'The model reads A and B as one text.',
                      va='center', fontsize=8, color=HIGHLIGHT)
        else:
            axis.text(end + 0.2, y + 0.3, f'the cache holds {len(tokens)} tokens: only B, '
                      'from pos 0.', va='center', fontsize=8)
    axis.text(0.3, 0.1, 'grey = K and V that request A left in the cache.   '
              'blue = the tokens of request B.', fontsize=8)
    axis.set_title('One cache for each request: an old cache changes the answer', fontsize=11)
    _show(figure)


# --------------------------------------------------------------------------
# Part 2: the slots of a batch
# --------------------------------------------------------------------------

def _slot_grid(axis, x0, y0, slots, steps, cell=0.55):
    """slots: one list for each slot. Each item is (label, kind), kind is
    'token', 'eos' or 'waste'."""
    colors = {'token': PROMPT_COLOR, 'eos': NEW_COLOR, 'waste': WASTE_COLOR}
    for step in range(steps):
        axis.text(x0 + step * cell + cell / 2, y0 + 0.2, str(step + 1), ha='center', fontsize=7)
    axis.text(x0 - 0.15, y0 + 0.2, 'step', ha='right', fontsize=7)
    for index, (name, row) in enumerate(slots):
        y = y0 - (index + 1) * (cell + 0.1)
        axis.text(x0 - 0.15, y + cell / 2, name, ha='right', va='center', fontsize=8)
        for step, (label, kind) in enumerate(row):
            _box(axis, x0 + step * cell, y, cell, cell, colors[kind],
                 '' if kind == 'waste' else label, 'white', size=7,
                 edge=HIGHLIGHT if kind == 'waste' else COMPUTED_EDGE)


def static_batch_slots():
    """Stage 04: a static batch runs until its slowest sequence ends."""
    lengths = {'A': 8, 'B': 4, 'C': 6, 'D': 2}
    steps = max(lengths.values())
    slots = []
    for name, length in lengths.items():
        row = [(name, 'token')] * (length - 1) + [('x', 'eos')] + [('.', 'waste')] * (steps - length)
        if length == steps:
            row[-1] = ('x', 'eos')
        slots.append((f'seq {name}: {length} tokens', row))
    figure, axis = _canvas(10, 4.0)
    _slot_grid(axis, 2.2, 3.3, slots, steps)
    used, paid = sum(lengths.values()), len(lengths) * steps
    lines = [f'used: {" + ".join(str(n) for n in lengths.values())} = {used} slots',
             f'paid: {len(lengths)} sequences x {steps} steps = {paid} slots',
             f'waste: {paid - used} / {paid} = {100 * (paid - used) / paid:.1f} %']
    axis.text(7.0, 2.6, '\n'.join(lines), fontsize=9, family='monospace', va='top')
    axis.text(0.2, 0.15, 'blue = a token.  orange x = the EOS of the sequence.  '
              'pink = a slot that the batch pays for and does not use.', fontsize=8)
    axis.set_title('A static batch: every slot waits for the slowest sequence (A)', fontsize=11)
    _show(figure)


def static_and_continuous():
    """Stage 05: the scheduler decides at every step, so a free slot takes the
    next waiting sequence."""
    steps = 8
    static = []
    for name, length in (('A', 8), ('B', 4), ('C', 6), ('D', 2)):
        row = [(name, 'token')] * (length - 1) + [('x', 'eos')] + [('.', 'waste')] * (steps - length)
        static.append((f'slot {len(static)}', row))
    # In each slot: (sequence, its length from its first step), in order.
    plan = [[('A', 8)], [('B', 4), ('F', 4)], [('C', 6), ('G', 2)], [('D', 2), ('H', 6)]]
    continuous = []
    for index, sequences in enumerate(plan):
        row = []
        for name, length in sequences:
            row += [(name, 'token')] * (length - 1) + [(name, 'eos')]
        continuous.append((f'slot {index}', row[:steps]))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for axis, slots, title in ((axes[0], static, 'STATIC (stage 04): 20 of 32 slots used'),
                               (axes[1], continuous, 'CONTINUOUS (stage 05): 32 of 32 slots used')):
        axis.set_xlim(0, 6.5)
        axis.set_ylim(-1.1, 3.6)
        axis.set_aspect('equal')
        axis.axis('off')
        _slot_grid(axis, 1.0, 3.0, slots, steps)
        axis.set_title(title, fontsize=10)
    axes[1].text(0.1, -0.15, 'B ends at step 4, F joins at step 5.  D ends at step 2, H joins at step 3.\n'
                 'Orange = the last token of a sequence (its EOS). A waiting sequence takes the slot\n'
                 'at the next step.', fontsize=8, va='top')
    axes[0].text(0.1, -0.15, 'pink = a slot that waits for the slowest sequence (A).',
                 fontsize=8, va='top')
    _show(figure)


# --------------------------------------------------------------------------
# Part 3: the block table and the gather
# --------------------------------------------------------------------------

SEQUENCE_COLORS = {'A': '#4c78a8', 'B': '#54a24b'}


def block_table():
    """Stage 06: cut the cache into fixed blocks, and give them out on demand."""
    block_size, max_len = 4, 40
    figure, axis = _canvas(13, 6.6)
    # Contiguous.
    axis.text(0.3, 6.2, f'CONTIGUOUS: reserve max_len = {max_len} slots for each sequence',
              fontsize=10, weight='bold')
    for row, (name, used) in enumerate((('A', 7), ('B', 10))):
        y = 5.5 - row * 0.5
        axis.text(0.3, y + 0.15, f'seq {name}', fontsize=8, va='center')
        for slot in range(max_len):
            filled = slot < used
            _box(axis, 1.1 + slot * 0.25, y, 0.25, 0.32,
                 SEQUENCE_COLORS[name] if filled else '#ffffff', edge='#bbb', line=0.4)
        axis.text(11.2, y + 0.15, f'{used} of {max_len} used', fontsize=8, va='center')
    axis.text(1.1, 4.35, 'reserved 80, used 17, waste 63 = 79 %', fontsize=9, color=HIGHLIGHT)

    # Paged.
    axis.text(0.3, 3.75, f'PAGED: blocks of block_size = {block_size} slots '
              '(the real engine uses 16)', fontsize=10, weight='bold')
    pool = {0: ('A', 4), 1: ('B', 4), 2: ('B', 2), 3: ('A', 3), 4: ('B', 4)}
    for block in range(8):
        x = 1.1 + block * 1.35
        axis.text(x + 0.6, 3.3, f'block {block}', ha='center', fontsize=7)
        owner, used = pool.get(block, (None, 0))
        for slot in range(block_size):
            filled = slot < used
            _box(axis, x + slot * 0.3, 2.75, 0.3, 0.4,
                 SEQUENCE_COLORS[owner] if filled else '#ffffff', edge='#bbb', line=0.4)
    axis.text(0.3, 2.95, 'pool', fontsize=8, va='center')
    lines = ['block table A = [0, 3]       7 tokens -> ceil(7 / 4)  = 2 blocks',
             'block table B = [1, 4, 2]   10 tokens -> ceil(10 / 4) = 3 blocks',
             'free list     = [5, 6, 7]',
             'reserved 20, used 17, waste 3 = 15 %  (less than one block for each sequence)',
             '',
             'token i of a sequence lives at:',
             '   block  = block_table[i // block_size]',
             '   offset = i % block_size',
             'example: token 9 of B -> block_table[9 // 4] = block_table[2] = block 2, offset 9 % 4 = 1']
    axis.text(1.1, 2.4, '\n'.join(lines), fontsize=9, family='monospace', va='top')
    axis.set_title('Blocks, block tables and a free list: a page table for the KV cache', fontsize=11)
    _show(figure)


def paged_gather():
    """Stage 07: gather K and V through the block table, then do ordinary
    attention."""
    block_size, context_len, table = 4, 10, [5, 2, 7]
    figure, axis = _canvas(13, 6.0)
    axis.text(0.3, 5.6, f'One sequence, one KV head. block_size = {block_size}. '
              f'context_len = {context_len}. block_table = {table}', fontsize=10, weight='bold')
    # The pool, in physical order.
    axis.text(0.3, 4.75, 'key_cache\n(the pool)', fontsize=8, va='center')
    position_of = {}
    for logical, block in enumerate(table):
        for offset in range(block_size):
            token = logical * block_size + offset
            if token < context_len:
                position_of[(block, offset)] = token
    for block in range(9):
        x = 1.5 + block * 1.25
        axis.text(x + 0.5, 5.2, f'blk {block}', ha='center', fontsize=7,
                  weight='bold' if block in table else 'normal')
        for offset in range(block_size):
            token = position_of.get((block, offset))
            if token is not None:
                color, label, text_color = PROMPT_COLOR, f't{token}', 'white'
            elif block in table:
                color, label, text_color = '#ffffff', '?', HIGHLIGHT
            else:
                color, label, text_color = CACHED_COLOR, '', '#555'
            _box(axis, x + offset * 0.25, 4.5, 0.25, 0.5, color, label, text_color, size=6,
                 edge='#999', line=0.4)
    # The gathered K.
    axis.text(0.3, 2.45, 'K, gathered\nin table order', fontsize=8, va='center')
    for token in range(context_len):
        _box(axis, 1.5 + token * 0.5, 2.2, 0.5, 0.5, NEW_COLOR, f't{token}', size=7)
    for logical, block in enumerate(table):
        source_x = 1.5 + block * 1.25 + 0.5
        first = logical * block_size
        target_x = 1.5 + (first + min(block_size, context_len - first) / 2) * 0.5
        _arrow(axis, (source_x, 4.45), (target_x, 2.75), color='#888', width=0.9)
        axis.text(source_x, 4.25, f'table[{logical}]', ha='center', fontsize=7,
                  color='#555', backgroundcolor='white')
    axis.text(6.7, 2.45, f'= {context_len} rows x head_dim', fontsize=8, va='center')
    lines = ['token i -> block_table[i // block_size], offset i % block_size',
             f'blk 7 holds tokens 8 and 9. Its last 2 slots ("?") hold old data:',
             'do NOT score them. Stop at context_len.',
             '',
             'q (1 x head_dim) -> softmax(q K^T / sqrt(head_dim)) V -> out',
             'the check: paged out == reference_attention out on a dense cache']
    axis.text(1.5, 1.6, '\n'.join(lines), fontsize=9, family='monospace', va='top')
    axis.text(8.6, 5.45, 'grey = blocks of other sequences', fontsize=7, color='#555')
    axis.set_title('Paged attention: the arithmetic is the same, the addresses change', fontsize=11)
    _show(figure)
