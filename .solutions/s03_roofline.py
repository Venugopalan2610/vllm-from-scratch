"""Reference solution, stage 03."""

import time

import torch

WARMUP_CALLS = 5


def model_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _milliseconds_per_call(call, num_calls):
    for _ in range(WARMUP_CALLS):
        call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_calls * 1000


def _random_ids(model, num_tokens):
    device = next(model.parameters()).device
    return torch.randint(0, 1000, (1, num_tokens), device=device)


@torch.inference_mode()
def time_prefill(model, num_tokens: int, iters: int = 10) -> float:
    """Milliseconds for one forward pass over num_tokens."""
    prompt_ids = _random_ids(model, num_tokens)
    return _milliseconds_per_call(lambda: model(prompt_ids, use_cache=True),
                                  iters)


@torch.inference_mode()
def time_decode(model, context_len: int, steps: int = 40) -> float:
    """Milliseconds for each token of decode, at a context of context_len."""
    cache = model(_random_ids(model, context_len),
                  use_cache=True).past_key_values
    one_token = _random_ids(model, 1)

    def decode_step():
        nonlocal cache
        cache = model(one_token, past_key_values=cache,
                      use_cache=True).past_key_values

    return _milliseconds_per_call(decode_step, steps)


def achieved_gbs(num_bytes: int, ms_per_token: float) -> float:
    return num_bytes / (ms_per_token / 1000) / 1e9
