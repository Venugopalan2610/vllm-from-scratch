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
