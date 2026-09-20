"""Shared helpers for the checks: paged caches, timing, a JSON validator and
the capstone oracle."""

import math
import os
import random
import time
from pathlib import Path

import torch

MODEL = os.environ.get("VC_MODEL", "Qwen/Qwen3-0.6B")


# ---- paged caches (stages 07, 08, 09) --------------------------------


def shuffled_block_ids(num_seqs, blocks_per_seq, shuffle=True, seed=0):
    """-> block_ids[seq][block]: a random physical block for each logical
    block. The order is random on purpose. A correct implementation must not
    depend on it."""
    physical_ids = list(range(num_seqs * blocks_per_seq))
    if shuffle:
        random.Random(seed).shuffle(physical_ids)
    return [physical_ids[seq * blocks_per_seq:(seq + 1) * blocks_per_seq]
            for seq in range(num_seqs)]


def build_paged(keys, values, block_size, shuffle=True, seed=0):
    """Scatter dense (seqs, kv_heads, context, D) K and V into a paged cache.
    -> (key_cache, value_cache, block_tables, context_lens).
    """
    num_seqs, num_kv_heads, context_len, head_dim = keys.shape
    blocks_per_seq = math.ceil(context_len / block_size)
    block_ids = shuffled_block_ids(num_seqs, blocks_per_seq, shuffle, seed)

    key_cache = torch.zeros(num_seqs * blocks_per_seq, num_kv_heads,
                            block_size, head_dim, dtype=keys.dtype,
                            device=keys.device)
    value_cache = torch.zeros_like(key_cache)
    for seq in range(num_seqs):
        for block, block_id in enumerate(block_ids[seq]):
            start = block * block_size
            end = min(start + block_size, context_len)
            key_cache[block_id, :, :end - start] = keys[seq, :, start:end]
            value_cache[block_id, :, :end - start] = values[seq, :, start:end]

    block_tables = torch.tensor(block_ids, dtype=torch.int32,
                                device=keys.device)
    context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32,
                              device=keys.device)
    return key_cache, value_cache, block_tables, context_lens


def random_kv(num_seqs, num_kv_heads, context_len, head_dim, device,
              dtype=torch.float32, seed=1234):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (num_seqs, num_kv_heads, context_len, head_dim)
    keys = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    values = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return keys, values


def random_query(num_seqs, num_heads, head_dim, device, dtype=torch.float32):
    return torch.randn(num_seqs, num_heads, head_dim, device=device,
                       dtype=dtype)


def paged_problem(num_seqs, num_heads, num_kv_heads, head_dim, context_len,
                  block_size, device, dtype=torch.float32):
    """A random decode-attention problem.
    -> (query, keys, values, (key_cache, value_cache, block_tables,
    context_lens)). Give the last tuple to a paged kernel with a star."""
    keys, values = random_kv(num_seqs, num_kv_heads, context_len, head_dim,
                             device, dtype=dtype)
    query = random_query(num_seqs, num_heads, head_dim, device, dtype)
    return query, keys, values, build_paged(keys, values, block_size)


def poison_past_context(key_cache, value_cache, block_tables, context_len,
                        block_size, poison=999.0):
    """Write `poison` into every slot at or after context_len, in place. A
    correct kernel gives the same output after it."""
    for row in block_tables.tolist():
        for block, block_id in enumerate(row):
            first_dead = max(context_len - block * block_size, 0)
            key_cache[block_id, :, first_dead:] = poison
            value_cache[block_id, :, first_dead:] = poison


def kv_bytes(context_lens, num_kv_heads, head_dim, itemsize):
    """The K and V bytes that a paged decode MUST read for these contexts.

    This is the minimum of the algorithm, not the traffic of the hardware.
    With GQA, several query heads read the same KV head. So the hardware can
    read fewer bytes than this (the L2 cache kept them) or more (it did not).
    """
    num_tokens = int(context_lens.sum().item())
    return 2 * num_tokens * num_kv_heads * head_dim * itemsize


def elapsed_ms(function):
    """-> (result, milliseconds) of one call, with a CUDA sync on each side."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def bench_ms(function, iters=30, warmup=5):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_spread(function, rounds=3, iters=30, warmup=5):
    """Run `rounds` separate timings. -> (median_ms, min_ms, max_ms, spread_pct).
    Laptops and power-limited GPUs throttle: reporting median and spread
    tells the student whether a failed gate is noise or a real problem."""
    times = [bench_ms(function, iters=iters, warmup=warmup) for _ in range(rounds)]
    times.sort()
    median_ms = times[len(times) // 2]
    min_ms, max_ms = times[0], times[-1]
    spread_pct = ((max_ms - min_ms) / median_ms * 100) if median_ms > 0 else 0.0
    return median_ms, min_ms, max_ms, spread_pct


# ---- CUDA tools -----------------------------------------------------


def report_kernel_stats(stats):
    """Print the registers, shared memory and spills of each kernel.
    -> the names of the kernels that spill to local memory."""
    print()
    for name, kernel in sorted(stats.items()):
        if "EmptyKernel" in name:
            continue
        print(f"  {name[-40:]:>40}  {kernel['reg']:>3} regs  "
              f"{kernel['shared']:>5}B shared  {kernel['local']:>4}B spilled")
    return [name for name, kernel in stats.items() if kernel.get("local")]


def probe_script(file_name, body):
    """Write a small script for an outside tool (ncu, compute-sanitizer)
    into the build cache. -> its path. The script can import the repo."""
    import cudalib
    from cudalib.build import CACHE

    path = CACHE / file_name
    path.parent.mkdir(exist_ok=True)
    path.write_text("import sys, torch\n"
                    f"sys.path.insert(0, {str(cudalib.ROOT)!r})\n" + body
                    + "torch.cuda.synchronize()\n")
    return path


# ---- the measurement log ----------------------------------------------
#
# A stage writes a number here, and a later stage compares against it.


def record_measurement(label, ms, tokens=None):
    from cudalib.probe import write_measurement

    tok_s = round(tokens / (ms / 1000), 1) if tokens else None
    write_measurement(label, {"ms": round(ms, 2), "tok_s": tok_s})


def measurement(label):
    from cudalib.probe import read_measurement

    return read_measurement(label)


# ---- a JSON prefix validator (stage 19) ------------------------------
#
# We give you this: a grammar compiler is a different subject. Stage 19
# is about the alignment of a validator to the TOKENIZER, and about masks
# off the critical path.

WHITESPACE = " \t\n\r"
LITERALS = ("true", "false", "null")


class _Incomplete(Exception):
    """The text ends inside a token. It is a prefix."""


class _Invalid(Exception):
    """No continuation can make the text valid."""


class JsonPrefixScanner:
    """Scan a text one JSON token at a time. `expect` is what comes next:
    value, value_or_end, key, key_or_end, colon or comma_or_end."""

    def __init__(self, text):
        self.text = text
        self.index = 0
        self.stack = []             # 'object' or 'array'
        self.expect = "value"

    def state(self):
        try:
            while self._skip_whitespace():
                self._step()
        except _Incomplete:
            return "prefix"
        except _Invalid:
            return "invalid"
        if not self.stack and self.expect == "comma_or_end":
            return "valid"
        return "prefix"

    def _skip_whitespace(self):
        """-> True if a character remains."""
        while self.index < len(self.text) and self.text[self.index] in WHITESPACE:
            self.index += 1
        return self.index < len(self.text)

    def _step(self):
        if self.expect in ("value", "value_or_end"):
            self._value()
        elif self.expect in ("key", "key_or_end"):
            self._key()
        elif self.expect == "colon":
            self._colon()
        else:
            self._comma_or_end()

    def _close(self, container):
        if not self.stack or self.stack[-1] != container:
            raise _Invalid
        self.stack.pop()
        self.index += 1
        self.expect = "comma_or_end"

    def _open(self, container, expect):
        self.stack.append(container)
        self.index += 1
        self.expect = expect

    def _value(self):
        char = self.text[self.index]
        if self.expect == "value_or_end" and char == "]":
            self._close("array")
        elif char == "{":
            self._open("object", "key_or_end")
        elif char == "[":
            self._open("array", "value_or_end")
        elif char == '"':
            self._string()
            self.expect = "comma_or_end"
        elif char == "-" or char.isdigit():
            self._number()
            self.expect = "comma_or_end"
        else:
            self._literal()
            self.expect = "comma_or_end"

    def _key(self):
        if self.expect == "key_or_end" and self.text[self.index] == "}":
            self._close("object")
            return
        if self.text[self.index] != '"':
            raise _Invalid
        self._string()
        self.expect = "colon"

    def _colon(self):
        if self.text[self.index] != ":":
            raise _Invalid
        self.index += 1
        self.expect = "value"

    def _comma_or_end(self):
        char = self.text[self.index]
        if not self.stack:
            raise _Invalid                  # text after a complete value
        if char == ",":
            self.index += 1
            self.expect = "key" if self.stack[-1] == "object" else "value"
        elif char == "}":
            self._close("object")
        elif char == "]":
            self._close("array")
        else:
            raise _Invalid

    def _string(self):
        """At the opening quote. Move past the closing quote."""
        position = self.index + 1
        while position < len(self.text):
            if self.text[position] == "\\":
                if position + 1 >= len(self.text):
                    raise _Incomplete
                position += 2
            elif self.text[position] == '"':
                self.index = position + 1
                return
            else:
                position += 1
        raise _Incomplete

    def _digits(self, position):
        start = position
        while position < len(self.text) and self.text[position].isdigit():
            position += 1
        return position, position > start

    def _number(self):
        position = self.index
        if self.text[position] == "-":
            position += 1
        position, has_digits = self._digits(position)
        if not has_digits:
            raise _Invalid
        if position < len(self.text) and self.text[position] == ".":
            position, has_digits = self._digits(position + 1)
            if not has_digits:
                raise _Incomplete
        if position < len(self.text) and self.text[position] in "eE":
            position += 1
            if position < len(self.text) and self.text[position] in "+-":
                position += 1
            position, has_digits = self._digits(position)
            if not has_digits:
                raise _Incomplete
        if position >= len(self.text):
            raise _Incomplete               # more digits can follow
        self.index = position

    def _literal(self):
        rest = self.text[self.index:]
        for literal in LITERALS:
            if rest.startswith(literal):
                self.index += len(literal)
                return
            if literal.startswith(rest):
                raise _Incomplete
        raise _Invalid


def json_prefix_state(text):
    """'valid' (complete JSON), 'prefix' (it can still become valid), or
    'invalid' (no continuation can make it valid)."""
    return JsonPrefixScanner(text).state()


# ---- the capstone (stages 21 to 28) ----------------------------------


def hf_greedy_ids(model, prompt_ids, num_tokens):
    """Exactly num_tokens greedy tokens from HF after a list of token ids,
    with EOS ignored."""
    prompt = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        output = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                max_new_tokens=num_tokens,
                                min_new_tokens=num_tokens, do_sample=False)
    return output[0, len(prompt_ids):].tolist()


def hf_greedy(model, tokenizer, prompt, num_tokens):
    """Exactly num_tokens greedy tokens from HF, with EOS ignored. The oracle
    of every capstone check that compares tokens."""
    return hf_greedy_ids(model, tokenizer(prompt).input_ids, num_tokens)


# Real prose, for the accuracy checks. A text that repeats itself is too
# easy: the model predicts it almost perfectly, and the check proves nothing.
PROSE_SAMPLE = (Path(__file__).resolve().parent.parent
                / "LORE.md").read_text()[2000:9000]

# The workload of the accuracy checks has the shape of serving: a prompt of
# ISL tokens, then OSL tokens that the model generates. The checks compare two
# models on the OSL positions only, because those are what a user reads.
ISL, OSL, NUM_PROMPTS = 256, 64, 4


def prose_prompts(tokenizer, num_prompts=NUM_PROMPTS, isl=ISL):
    """-> num_prompts token lists of isl tokens, from different parts of the
    prose sample."""
    token_ids = tokenizer(PROSE_SAMPLE).input_ids
    stride = (len(token_ids) - isl) // max(1, num_prompts - 1)
    return [token_ids[index * stride:index * stride + isl]
            for index in range(num_prompts)]


def hf_workload(model, tokenizer, osl=OSL):
    """-> one token list for each prose prompt: the prompt, then the greedy
    continuation of the bf16 HF model."""
    return [prompt + hf_greedy_ids(model, prompt, osl)
            for prompt in prose_prompts(tokenizer)]

CAPSTONE_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
    "The best way to learn a language is",
    "Photosynthesis is the process",
    "Once upon a time",
]
