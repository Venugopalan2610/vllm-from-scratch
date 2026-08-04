"""Stage 02 - The KV cache.

The spec:

    app/s02_cache.py must define

        cached_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]
        kv_bytes_per_token(config) -> int

Rules:
  - One prefill forward over the whole prompt, then ONE TOKEN per forward.
  - You own the cache. Hold the past K/V and feed it back in.
  - Output must be token-identical to stage 01.

kv_bytes_per_token must compute, from a HF config:

    2 (K and V) * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes

The trap is num_key_value_heads. Modern models use GQA, so KV heads are
often 4-8x fewer than attention heads. Get this wrong and every memory
budget you compute from stage 06 onward is wrong by the GQA factor.
"""

import pytest
import torch

from app.s01_naive import naive_generate
from app.s02_cache import cached_generate, kv_bytes_per_token
from tests.conftest import measurement


def test_identical_to_stage_01(hf_exact, prompts, dev):
    """Caching is an optimization, not a behavior change. Same tokens, exactly."""
    model, tok = hf_exact
    for p in prompts:
        want = naive_generate(model, tok, p, max_tokens=24)
        got = cached_generate(model, tok, p, max_tokens=24)
        assert got == want, (
            f"\nprompt: {p!r}\n"
            f"uncached: {tok.decode(want)!r}\n"
            f"cached:   {tok.decode(got)!r}\n"
            "A drifting cache usually means you fed the wrong position ids "
            "or reused a stale cache between calls."
        )


def test_cache_is_not_leaked_between_calls(hf_exact, dev):
    """Calling twice must not contaminate the second result."""
    model, tok = hf_exact
    a = cached_generate(model, tok, "The capital of France is", max_tokens=12)
    _ = cached_generate(model, tok, "Something else entirely, quite different", max_tokens=12)
    b = cached_generate(model, tok, "The capital of France is", max_tokens=12)
    assert a == b, "state leaked across calls - build a fresh cache each generate()"


def test_kv_bytes_per_token(hf):
    model, tok = hf
    cfg = model.config
    got = kv_bytes_per_token(cfg)

    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    want = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2

    assert got == want, (
        f"got {got}, want {want}. "
        f"(layers={cfg.num_hidden_layers} kv_heads={cfg.num_key_value_heads} "
        f"head_dim={head_dim} -- did you use num_attention_heads"
        f"={cfg.num_attention_heads} by mistake?)"
    )

    print(f"\n  \033[36mKV per token\033[0m: {got / 1024:.1f} KB")
    print(f"  \033[36m4096-token sequence\033[0m: {got * 4096 / 1e6:.0f} MB")
    free = torch.cuda.get_device_properties(0).total_memory
    print(f"  \033[2mAt 4k context, ~{int(free * 0.7 / (got * 4096))} such sequences "
          f"fit in 70% of your VRAM. That number is your throughput.\033[0m")


LONG_PROMPT = "The history of computing began " * 200  # ~1000 tokens


def test_is_actually_faster(hf, dev, timer):
    """The whole point -- measured at a prompt length where it matters."""
    import time
    model, tok = hf
    N = 64

    cached_generate(model, tok, "warmup", 4)  # warm the kernels

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    naive_generate(model, tok, LONG_PROMPT, max_tokens=N)
    torch.cuda.synchronize()
    uncached_ms = (time.perf_counter() - t0) * 1000

    with timer("s02_cached_1k_prompt") as t:
        out = cached_generate(model, tok, LONG_PROMPT, max_tokens=N)
    t.report(tokens=len(out))

    speedup = uncached_ms / t.ms
    print(f"  uncached: {uncached_ms:.0f} ms")
    print(f"\n  \033[1mSpeedup at a ~1000-token prompt: {speedup:.1f}x\033[0m")
    assert speedup > 3.0, (
        f"only {speedup:.1f}x. Are you re-running the full prefix each step?"
    )


def test_the_win_grows_with_context(hf, dev):
    """Why the speedup above was measured at 1000 tokens and not 6.

    The cache saves you from recomputing the PREFIX, so its value is
    proportional to prefix length. At a 6-token prompt on a 0.6B model there is
    almost nothing to save, and both paths are bound by Python and kernel-launch
    overhead instead of by arithmetic -- you are nowhere near the memory roofline.

    That overhead floor is a bug you will fix much later, with CUDA graphs, in
    stage 12. Notice it now: `./vc info` printed your batch-1 ceiling, and this
    test is running at a fraction of it.
    """
    import time
    model, tok = hf
    base = "The history of computing began "

    def ms(fn, p):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(model, tok, p, 32)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    ms(cached_generate, "warmup")
    rows = []
    for mult in (1, 60, 200):
        p = base * mult
        n = len(tok(p).input_ids)
        rows.append((n, ms(naive_generate, p) / ms(cached_generate, p)))

    print()
    for n, sp in rows:
        print(f"  prompt {n:>5} tok -> cache is {sp:5.1f}x faster")

    assert rows[-1][1] > rows[0][1], "speedup should grow with prompt length"
    print("\n  \033[2mThe KV cache is not a constant-factor win. It is a win that"
          "\n  grows with context -- which is exactly why long-context serving"
          "\n  lives or dies on how well you manage KV memory.\033[0m")


def test_scaling_is_linear_not_quadratic(hf, dev):
    """Uncached is O(N^2) in generated length; cached is O(N). Prove it."""
    import time
    model, tok = hf
    prompt = LONG_PROMPT

    def ms_for(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        cached_generate(model, tok, prompt, max_tokens=n)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    ms_for(8)  # warmup
    a, b = ms_for(32), ms_for(64)
    ratio = b / a
    print(f"\n  32 tok: {a:.0f} ms | 64 tok: {b:.0f} ms | ratio {ratio:.2f}x")
    print(f"  \033[2mLinear would be ~2.0x. Quadratic would be ~4.0x.\033[0m")
    assert ratio < 2.8, f"scaling looks quadratic ({ratio:.2f}x for 2x the tokens)"


def test_bf16_divergence_is_real(hf, dev):
    """Not a pass/fail on your code. A fact you need to know about.

    In bf16 the cached and uncached paths can produce DIFFERENT TEXT from the
    same prompt, because they reduce in different orders and argmax flips at a
    near-tie. This is why correctness above is checked in fp32, and why
    "same model, same prompt, different batch size, different output" is
    normal in production serving rather than a bug.
    """
    model, tok = hf  # bf16 on purpose
    p = "The capital of France is"
    a = naive_generate(model, tok, p, max_tokens=24)
    b = cached_generate(model, tok, p, max_tokens=24)
    if a != b:
        i = next(k for k in range(min(len(a), len(b))) if a[k] != b[k])
        print(f"\n  \033[33mbf16 paths diverged at token {i}\033[0m")
        print(f"    uncached: {tok.decode(a)!r}")
        print(f"    cached:   {tok.decode(b)!r}")
        print("  \033[2mSame weights, same prompt, different sentence. Remember this"
              "\n  the next time a serving bug 'only happens at batch size 8'.\033[0m")
    else:
        print("\n  \033[2mbf16 paths agreed on this prompt (they often won't).\033[0m")
