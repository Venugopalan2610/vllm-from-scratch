"""Stage 02 (JAX) - The KV cache, preallocated.

The spec:

    app/j02_cache.py must define

        cached_generate(model, prompt: str, max_tokens: int) -> list[int]
        kv_bytes_per_token(config) -> int

Rules:
  - One prefill over the whole prompt, then ONE TOKEN per forward.
  - You own the cache: allocate it, write into it, track cache_len.
  - Output must be token-identical to stage 01.

Spec and traps in app/j02_cache.py.
"""

import time

import jax
import pytest

from app.j01_naive import naive_generate
from app.j02_cache import cached_generate, kv_bytes_per_token

LONG_PROMPT = "The history of computing began " * 200     # ~1000 tokens


def test_identical_to_stage_01(jmodel_exact, prompts):
    """Caching is an optimization, not a behavior change. Same tokens, exactly."""
    model = jmodel_exact
    for p in prompts[:2]:
        want = naive_generate(model, p, max_tokens=12)
        got = cached_generate(model, p, max_tokens=12)
        assert got == want, (
            f"\nprompt: {p!r}\n"
            f"uncached: {model.decode(want)!r}\n"
            f"cached:   {model.decode(got)!r}\n"
            "Drift a few tokens in almost always means the decode step's "
            "`positions` are wrong -- they must be the ABSOLUTE position "
            "(cache_len), not 0."
        )


def test_cache_is_not_leaked_between_calls(jmodel_exact):
    """Calling twice must not contaminate the second result."""
    model = jmodel_exact
    a = cached_generate(model, "The capital of France is", max_tokens=10)
    _ = cached_generate(model, "Something else entirely, quite different", 10)
    b = cached_generate(model, "The capital of France is", max_tokens=10)
    assert a == b, (
        "state leaked across calls -- init_cache() a fresh buffer each "
        "generate(), and start cache_len at 0"
    )


def test_does_not_overflow_the_cache(jmodel_exact):
    """max_len is a hard ceiling that does NOT raise when you cross it.

    dynamic_update_slice CLAMPS an out-of-range start, so a cache sized for the
    prompt but not for the output writes every token past the end into the same
    final slot, and the model attends to a slot that keeps changing under it.

    Checked as a prefix property: greedy decoding is deterministic, so the
    first 8 tokens of a 48-token generation must be the first 8 tokens of an
    8-token generation. A cache that overflows partway through the long run
    breaks that; nothing else does.
    """
    model = jmodel_exact
    p = "One two three four five six seven"
    short = cached_generate(model, p, 8)
    long = cached_generate(model, p, 48)

    assert len(short) > 0, "generated nothing"
    assert long[:len(short)] == short, (
        f"\n 8 tokens:  {model.decode(short)!r}"
        f"\n48 tokens:  {model.decode(long[:len(short)])!r}\n"
        "The same prompt decoded greedily has to start the same way. It "
        "diverging on the longer run means the cache ran out of slots part "
        "way through -- max_len must cover prompt + max_tokens."
    )


def test_kv_bytes_per_token(jmodel):
    cfg = jmodel.config
    got = kv_bytes_per_token(cfg)
    want = (2 * cfg.num_hidden_layers * cfg.num_key_value_heads
            * cfg.head_dim * 2)
    assert got == want, (
        f"got {got}, want {want}. (layers={cfg.num_hidden_layers} "
        f"kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim} -- did "
        f"you use num_attention_heads={cfg.num_attention_heads} by mistake?)"
    )

    # The model's own accounting must agree with yours. If it does not, one of
    # you is wrong about GQA, and it is worth finding out which before stage 06
    # budgets VRAM with this number.
    assert jmodel.cache_bytes(batch=1, max_len=1) == got

    print(f"\n  \033[36mKV per token\033[0m: {got / 1024:.1f} KB")
    print(f"  \033[36m4096-token sequence\033[0m: {got * 4096 / 1e6:.0f} MB")
    dev = jax.devices()[0]
    total = dev.memory_stats().get("bytes_limit") if dev.memory_stats() else None
    if total:
        print(f"  \033[2mAt 4k context, ~{int(total * 0.7 / (got * 4096))} such "
              "sequences fit in 70% of your VRAM. That number is your "
              "throughput.\033[0m")


def test_is_actually_faster(jmodel):
    """The whole point -- measured warm, so it is the algorithm and not XLA.

    Both paths are run once first to fill the compilation cache. What is left
    is arithmetic: the uncached loop re-reads the whole prefix every step, the
    cached one reads one token.
    """
    model, N = jmodel, 8

    def ms(fn):
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1000

    ms(lambda: naive_generate(model, LONG_PROMPT, N))      # warm
    ms(lambda: cached_generate(model, LONG_PROMPT, N))     # warm
    uncached = ms(lambda: naive_generate(model, LONG_PROMPT, N))
    cached = ms(lambda: cached_generate(model, LONG_PROMPT, N))

    speedup = uncached / cached
    print(f"\n  uncached: {uncached:7.0f} ms")
    print(f"  cached:   {cached:7.0f} ms")
    print(f"\n  \033[1mSpeedup at a ~1000-token prompt: {speedup:.1f}x\033[0m")
    assert speedup > 3.0, (
        f"only {speedup:.1f}x. Are you re-running the full prefix each step, "
        "or rebuilding the cache inside the loop?"
    )


def test_it_also_stopped_recompiling(jmodel):
    """The other half of the win, and the one that is specific to JAX.

    The uncached loop compiles once per generated token, forever, for every
    request. The cached loop compiles a prefill and a decode step and then
    reuses both -- and if you bucketed max_len, it reuses them across prompts
    of different lengths too.
    """
    from jvllm.model import _forward

    model = jmodel
    p1 = "Recompilation is measured here " * 11            # ~55 tokens
    p2 = "Recompilation is measured here " * 13            # ~65 tokens

    cached_generate(model, p1, 6)                          # warm
    before = _forward._cache_size()
    cached_generate(model, p1, 6)
    same_prompt = _forward._cache_size() - before

    before = _forward._cache_size()
    cached_generate(model, p2, 6)
    new_length = _forward._cache_size() - before

    print(f"\n  re-running the same prompt: {same_prompt} new compilations")
    print(f"  a prompt of a new length:   {new_length} new compilations")

    assert same_prompt == 0, (
        f"{same_prompt} recompilations on an identical repeat call. Something "
        "in your loop changes shape between steps -- a growing positions "
        "array, or a cache sized from a Python int that moves."
    )
    if new_length == 0:
        print("  \033[2mZero: you bucketed max_len, so a longer prompt reuses")
        print("  the same compiled decode step. That is stage 12's trick,")
        print("  arrived at early.\033[0m")
    else:
        print(f"  \033[2m{new_length} compilations for a prompt {10} tokens longer.")
        print("  Correct, but every new prompt length will keep paying it.")
        print("  Round max_len up to a bucket and this goes to zero.\033[0m")


def test_scaling_is_linear_not_quadratic(jmodel):
    """Uncached is O(N^2) in generated length; cached is O(N). Prove it."""
    model = jmodel

    def ms_for(n):
        cached_generate(model, LONG_PROMPT, max_tokens=n)   # warm this shape
        t = time.perf_counter()
        cached_generate(model, LONG_PROMPT, max_tokens=n)
        return (time.perf_counter() - t) * 1000

    a, b = ms_for(8), ms_for(16)
    ratio = b / a
    print(f"\n   8 tok: {a:.0f} ms | 16 tok: {b:.0f} ms | ratio {ratio:.2f}x")
    print("  \033[2mLinear would be ~2.0x. Quadratic would be ~4.0x.\033[0m")
    assert ratio < 2.8, f"scaling looks quadratic ({ratio:.2f}x for 2x the tokens)"
