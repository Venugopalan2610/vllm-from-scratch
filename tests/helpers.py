"""Shared test fixtures for the paged-attention stages (07, 08, 09)."""

import math
import random

import torch


def build_paged(keys, values, block_size, shuffle=True, seed=0):
    """Scatter dense (S, KVH, L, D) K/V into a paged cache with a random block
    layout. Returns (key_cache, value_cache, block_tables, context_lens).

    Deliberately shuffles physical blocks -- a correct implementation must not
    care what order they land in.
    """
    S, KVH, L, D = keys.shape
    nb_per_seq = math.ceil(L / block_size)
    total = S * nb_per_seq
    ids = list(range(total))
    if shuffle:
        random.Random(seed).shuffle(ids)

    kc = torch.zeros(total, KVH, block_size, D, dtype=keys.dtype, device=keys.device)
    vc = torch.zeros_like(kc)
    bt = torch.zeros(S, nb_per_seq, dtype=torch.int32, device=keys.device)

    for s in range(S):
        for b in range(nb_per_seq):
            phys = ids[s * nb_per_seq + b]
            bt[s, b] = phys
            lo, hi = b * block_size, min((b + 1) * block_size, L)
            kc[phys, :, : hi - lo] = keys[s, :, lo:hi]
            vc[phys, :, : hi - lo] = values[s, :, lo:hi]

    ctx = torch.full((S,), L, dtype=torch.int32, device=keys.device)
    return kc, vc, bt, ctx


def rand_kv(S, KVH, L, D, dev, dtype=torch.float32, seed=1234):
    g = torch.Generator(device=dev).manual_seed(seed)
    k = torch.randn(S, KVH, L, D, generator=g, device=dev, dtype=dtype)
    v = torch.randn(S, KVH, L, D, generator=g, device=dev, dtype=dtype)
    return k, v


def bench_ms(fn, iters=30, warmup=5):
    import time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


# ---- JSON prefix validator (fixture for stage 19) --------------------
# Supplied for you: compiling a grammar is a separate discipline. Stage 19 is
# about aligning a validator to the TOKENIZER and keeping masks off the
# critical path.

_WS = " \t\n\r"


def json_prefix_state(s):
    """Classify s as 'valid' (complete JSON), 'prefix' (could still become
    valid), or 'invalid' (no continuation can rescue it)."""
    stack = []          # 'obj' / 'arr'
    expect = "value"    # value | key | colon | comma_or_end
    i, n = 0, len(s)

    def ws(i):
        while i < n and s[i] in _WS:
            i += 1
        return i

    def scan_string(i):
        """i points at the opening quote. -> (end_index, complete?)"""
        j = i + 1
        while j < n:
            if s[j] == "\\":
                if j + 1 >= n:
                    return j, False
                j += 2
                continue
            if s[j] == '"':
                return j + 1, True
            j += 1
        return j, False

    def scan_number(i):
        j = i
        if j < n and s[j] == "-":
            j += 1
        start_digits = j
        while j < n and s[j].isdigit():
            j += 1
        if j == start_digits:
            return None, False
        if j < n and s[j] == ".":
            j += 1
            d = j
            while j < n and s[j].isdigit():
                j += 1
            if j == d:
                return j, False
        if j < n and s[j] in "eE":
            j += 1
            if j < n and s[j] in "+-":
                j += 1
            d = j
            while j < n and s[j].isdigit():
                j += 1
            if j == d:
                return j, False
        return j, True

    while True:
        i = ws(i)
        if i >= n:
            if not stack and expect == "comma_or_end":
                return "valid"
            return "prefix"

        c = s[i]

        if expect in ("value", "value_or_end"):
            if expect == "value_or_end" and c == "]":
                if not stack or stack[-1] != "arr":
                    return "invalid"
                stack.pop()
                expect = "comma_or_end"
                i += 1
                continue
            if c == "{":
                stack.append("obj")
                expect = "key_or_end"
                i += 1
            elif c == "[":
                stack.append("arr")
                expect = "value_or_end"
                i += 1
            elif c == '"':
                end, complete = scan_string(i)
                if not complete:
                    return "prefix"
                i = end
                expect = "comma_or_end"
            elif c == "-" or c.isdigit():
                end, complete = scan_number(i)
                if end is None:
                    return "invalid"
                if not complete or end >= n:
                    return "prefix"
                i = end
                expect = "comma_or_end"
            else:
                for lit in ("true", "false", "null"):
                    if s.startswith(lit, i):
                        i += len(lit)
                        expect = "comma_or_end"
                        break
                    if lit.startswith(s[i:]):
                        return "prefix"
                else:
                    return "invalid"

        elif expect in ("key", "key_or_end"):
            if expect == "key_or_end" and c == "}":
                if not stack or stack[-1] != "obj":
                    return "invalid"
                stack.pop()
                expect = "comma_or_end"
                i += 1
                continue
            if c != '"':
                return "invalid"
            end, complete = scan_string(i)
            if not complete:
                return "prefix"
            i = end
            expect = "colon"

        elif expect == "colon":
            if c != ":":
                return "invalid"
            i += 1
            expect = "value"

        elif expect == "comma_or_end":
            if not stack:
                return "invalid"        # trailing junk after a complete value
            if c == ",":
                i += 1
                expect = "key" if stack[-1] == "obj" else "value"
            elif c == "}" and stack[-1] == "obj":
                stack.pop()
                i += 1
            elif c == "]" and stack[-1] == "arr":
                stack.pop()
                i += 1
            else:
                return "invalid"
