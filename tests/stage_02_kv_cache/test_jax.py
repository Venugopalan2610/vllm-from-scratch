"""Stage 02 (JAX) - the KV cache, preallocated.

The spec:

    app/j02_cache.py must define

        cached_generate(model, prompt: str, max_tokens: int) -> list[int]
        kv_bytes_per_token(config) -> int

Rules:
  - One prefill over the whole prompt, then ONE TOKEN for each pass.
  - You own the cache: allocate it, write into it, and keep cache_len.
  - The output must be the same tokens as stage 01.

The spec and the traps are in app/j02_cache.py.
"""

import jax

from app.j01_naive import naive_generate
from app.j02_cache import cached_generate, kv_bytes_per_token
from tests.helpers import elapsed_ms

LONG_PROMPT = "The history of computing began " * 200     # about 1000 tokens


def _warm_ms(function):
    """Run one time to fill the compilation cache, then time a second run."""
    function()
    return elapsed_ms(function)[1]


def test_identical_to_stage_01(jmodel_exact, prompts):
    """A cache is an optimization, not a change of behavior. It must give
    exactly the same tokens."""
    model = jmodel_exact
    for prompt in prompts[:2]:
        uncached = naive_generate(model, prompt, max_tokens=12)
        cached = cached_generate(model, prompt, max_tokens=12)
        assert cached == uncached, (
            f"\nprompt:   {prompt!r}\n"
            f"uncached: {model.decode(uncached)!r}\n"
            f"cached:   {model.decode(cached)!r}\n"
            "A drift after a few tokens almost always means that the "
            "`positions` of the decode step are wrong. They must be the "
            "ABSOLUTE position (cache_len), not 0.")


def test_cache_is_not_leaked_between_calls(jmodel_exact):
    """A second call must give a clean result. The first call must not
    change it."""
    model = jmodel_exact
    prompt = "The capital of France is"
    first = cached_generate(model, prompt, max_tokens=10)
    cached_generate(model, "Something else, quite different", 10)
    again = cached_generate(model, prompt, max_tokens=10)
    assert first == again, (
        "state leaked across calls. Call init_cache() for a new buffer in "
        "each generate(), and start cache_len at 0.")


def test_does_not_overflow_the_cache(jmodel_exact):
    """max_len is a hard limit that does NOT raise an error when you cross it.

    dynamic_update_slice CLAMPS a start that is out of range. So a cache
    that has room for the prompt but not for the output writes every token
    past the end into the same last slot. The model then attends to a slot
    that changes all the time.

    The check uses a prefix property. Greedy decode is deterministic, so the
    first 8 tokens of a 48-token generation must be the tokens of an 8-token
    generation. A cache that overflows during the long run breaks that.
    Nothing else does.
    """
    model = jmodel_exact
    prompt = "One two three four five six seven"
    short = cached_generate(model, prompt, 8)
    long = cached_generate(model, prompt, 48)

    assert short, "generated nothing"
    assert long[:len(short)] == short, (
        f"\n 8 tokens:  {model.decode(short)!r}"
        f"\n48 tokens:  {model.decode(long[:len(short)])!r}\n"
        "A greedy decode of the same prompt must start the same way. A "
        "difference on the longer run means that the cache ran out of slots. "
        "max_len must cover the prompt + max_tokens.")


def test_kv_bytes_per_token(jmodel):
    config = jmodel.config
    measured = kv_bytes_per_token(config)
    expected = (2 * config.num_hidden_layers * config.num_key_value_heads
                * config.head_dim * 2)
    assert measured == expected, (
        f"got {measured}, expected {expected}. "
        f"(layers={config.num_hidden_layers} "
        f"kv_heads={config.num_key_value_heads} head_dim={config.head_dim}. "
        f"Did you use num_attention_heads={config.num_attention_heads} by "
        "mistake?)")

    # The model must agree with your count. If they do not agree, one of
    # them is wrong about GQA. Find which one before stage 06 uses this
    # number for VRAM budgets.
    assert jmodel.cache_bytes(batch=1, max_len=1) == measured

    sequence_bytes = measured * 4096
    print(f"\n  \033[36mKV per token\033[0m: {measured / 1024:.1f} KB")
    print(f"  \033[36m4096-token sequence\033[0m: {sequence_bytes / 1e6:.0f} MB")
    memory_stats = jax.devices()[0].memory_stats()
    memory_limit = memory_stats.get("bytes_limit") if memory_stats else None
    if memory_limit:
        print(f"  \033[2mAt 4k context, about "
              f"{int(memory_limit * 0.7 / sequence_bytes)} such sequences fit "
              "in 70% of your VRAM. That number is your throughput.\033[0m")


def test_is_actually_faster(jmodel):
    """This is the point of the stage. Measure it warm, so that you see the
    algorithm and not XLA.

    Both paths run one time first to fill the compilation cache. Only the
    arithmetic remains: the uncached loop reads the whole prefix again at
    each step, and the cached loop reads one token.
    """
    uncached_ms = _warm_ms(lambda: naive_generate(jmodel, LONG_PROMPT, 8))
    cached_ms = _warm_ms(lambda: cached_generate(jmodel, LONG_PROMPT, 8))

    speedup = uncached_ms / cached_ms
    print(f"\n  uncached: {uncached_ms:7.0f} ms")
    print(f"  cached:   {cached_ms:7.0f} ms")
    print(f"\n  \033[1mSpeedup at a prompt of about 1000 tokens: "
          f"{speedup:.1f}x\033[0m")
    assert speedup > 3.0, (
        f"only {speedup:.1f}x. Do you run the full prefix again at each step, "
        "or make the cache again inside the loop?")


def test_it_also_stopped_recompiling(jmodel):
    """The other half of the gain, and the half that is special to JAX.

    The uncached loop compiles one time for each generated token, for every
    request. The cached loop compiles a prefill and a decode step, then uses
    both again. If you put max_len in buckets, it also uses them again for
    prompts of different lengths.
    """
    from jvllm.model import _forward

    shorter = "Recompilation is measured here " * 11          # about 55 tokens
    longer = "Recompilation is measured here " * 13           # about 65 tokens

    def new_compiles(prompt):
        compiles_before = _forward._cache_size()
        cached_generate(jmodel, prompt, 6)
        return _forward._cache_size() - compiles_before

    new_compiles(shorter)
    same_prompt = new_compiles(shorter)
    new_length = new_compiles(longer)

    print(f"\n  the same prompt again:     {same_prompt} new compilations")
    print(f"  a prompt of a new length:  {new_length} new compilations")
    assert same_prompt == 0, (
        f"{same_prompt} compilations on the same call again. A shape in your "
        "loop changes between steps: a positions array that grows, or a "
        "cache size from a Python int that changes.")
    if new_length == 0:
        print("  \033[2mZero: you put max_len in buckets, so a longer prompt")
        print("  uses the same compiled decode step. That is the method of")
        print("  stage 12, found early.\033[0m")
    else:
        print(f"  \033[2m{new_length} compilations for a prompt 10 tokens "
              "longer.")
        print("  Correct, but every new prompt length pays it again.")
        print("  Round max_len up to a bucket, and this becomes zero.\033[0m")


def test_scaling_is_linear_not_quadratic(jmodel):
    """With no cache, the cost is O(N^2) in the generated length. With a
    cache it is O(N). Prove it."""
    short_ms = _warm_ms(lambda: cached_generate(jmodel, LONG_PROMPT, 8))
    long_ms = _warm_ms(lambda: cached_generate(jmodel, LONG_PROMPT, 16))
    ratio = long_ms / short_ms
    print(f"\n   8 tok: {short_ms:.0f} ms | 16 tok: {long_ms:.0f} ms | "
          f"ratio {ratio:.2f}x")
    print("  \033[2mLinear is about 2.0x. Quadratic is about 4.0x.\033[0m")
    assert ratio < 2.8, (
        f"the scaling looks quadratic ({ratio:.2f}x for 2x the tokens)")
