"""Reference solution, stage 03."""

import time

import torch


def model_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _sync_time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


@torch.inference_mode()
def time_prefill(model, n_tokens: int, iters: int = 10) -> float:
    dev = next(model.parameters()).device
    ids = torch.randint(0, 1000, (1, n_tokens), device=dev)
    return _sync_time(lambda: model(ids, use_cache=True), iters)


@torch.inference_mode()
def time_decode(model, ctx_len: int, steps: int = 40) -> float:
    dev = next(model.parameters()).device
    ids = torch.randint(0, 1000, (1, ctx_len), device=dev)
    past = model(ids, use_cache=True).past_key_values
    one = torch.randint(0, 1000, (1, 1), device=dev)

    # warmup, and let the cache grow naturally -- no copying in the loop
    for _ in range(5):
        past = model(one, past_key_values=past, use_cache=True).past_key_values

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        past = model(one, past_key_values=past, use_cache=True).past_key_values
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps * 1000


def achieved_gbs(nbytes: int, ms_per_token: float) -> float:
    return nbytes / (ms_per_token / 1000) / 1e9
