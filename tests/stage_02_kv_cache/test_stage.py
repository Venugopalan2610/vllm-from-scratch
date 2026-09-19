"""Stage 02 - the KV cache.

The spec:

    app/s02_cache.py must define

        cached_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]
        kv_bytes_per_token(config) -> int

Rules:
  - One prefill pass over the whole prompt, then ONE TOKEN for each pass.
  - You own the cache. Keep the past K and V, and give them back.
  - The output must be the same tokens as stage 01.

kv_bytes_per_token must compute, from a HF config:

    2 (K and V) * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes

The trap is num_key_value_heads. Modern models use GQA, so there are
frequently 4 to 8 times fewer KV heads than attention heads. If this is
wrong, every memory budget from stage 06 on is wrong by the GQA factor.
"""

import torch

from app.s01_naive import naive_generate
from app.s02_cache import cached_generate, kv_bytes_per_token
from tests.helpers import elapsed_ms

LONG_PROMPT = "The history of computing began " * 200      # about 1000 tokens


def test_identical_to_stage_01(hf_exact, prompts, device):
    """A cache is an optimization, not a change of behavior. It must give
    exactly the same tokens."""
    model, tokenizer = hf_exact
    for prompt in prompts:
        uncached = naive_generate(model, tokenizer, prompt, max_tokens=24)
        cached = cached_generate(model, tokenizer, prompt, max_tokens=24)
        assert cached == uncached, (
            f"\nprompt:   {prompt!r}\n"
            f"uncached: {tokenizer.decode(uncached)!r}\n"
            f"cached:   {tokenizer.decode(cached)!r}\n"
            "A cache that drifts usually means wrong position ids, or an old "
            "cache from an earlier call.")


def test_cache_is_not_leaked_between_calls(hf_exact, device):
    """A second call must give a clean result. The first call must not
    change it."""
    model, tokenizer = hf_exact
    prompt = "The capital of France is"
    first = cached_generate(model, tokenizer, prompt, max_tokens=12)
    cached_generate(model, tokenizer, "Something else, quite different",
                    max_tokens=12)
    again = cached_generate(model, tokenizer, prompt, max_tokens=12)
    assert first == again, (
        "state leaked across calls. Make a new cache in each generate().")


def test_kv_bytes_per_token(hf):
    model, _ = hf
    config = model.config
    measured = kv_bytes_per_token(config)
    head_dim = (getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads)
    expected = (2 * config.num_hidden_layers * config.num_key_value_heads
                * head_dim * 2)

    assert measured == expected, (
        f"got {measured}, expected {expected}. "
        f"(layers={config.num_hidden_layers} "
        f"kv_heads={config.num_key_value_heads} head_dim={head_dim}. Did you "
        f"use num_attention_heads={config.num_attention_heads} by mistake?)")

    sequence_bytes = measured * 4096
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"\n  \033[36mKV per token\033[0m: {measured / 1024:.1f} KB")
    print(f"  \033[36m4096-token sequence\033[0m: {sequence_bytes / 1e6:.0f} MB")
    print(f"  \033[2mAt 4k context, about "
          f"{int(total_memory * 0.7 / sequence_bytes)} such sequences fit in "
          "70% of your VRAM. That number is your throughput.\033[0m")


def test_is_actually_faster(hf, device, timer):
    """This is the point of the stage. Measure it at a prompt length where
    it is important."""
    model, tokenizer = hf
    cached_generate(model, tokenizer, "warmup", 4)
    _, uncached_ms = elapsed_ms(
        lambda: naive_generate(model, tokenizer, LONG_PROMPT, max_tokens=64))

    with timer("s02_cached_1k_prompt") as timing:
        generated = cached_generate(model, tokenizer, LONG_PROMPT,
                                    max_tokens=64)
    timing.report(tokens=len(generated))

    speedup = uncached_ms / timing.ms
    print(f"  uncached: {uncached_ms:.0f} ms")
    print(f"\n  \033[1mSpeedup at a prompt of about 1000 tokens: "
          f"{speedup:.1f}x\033[0m")
    assert speedup > 3.0, (
        f"only {speedup:.1f}x. Do you run the full prefix again at each step?")


def test_the_win_grows_with_context(hf, device):
    """Why the measurement above uses 1000 tokens, and not 6.

    The cache saves the work on the PREFIX, so the prefix length controls
    its value. A prompt of 6 tokens on a 0.6B model saves almost nothing.

    At that size, Python and the kernel launches limit both paths. The
    arithmetic does not. You are far from the memory roofline.

    You fix that overhead floor much later, with CUDA graphs, in stage 12.
    Note it now: `./vc info` printed your batch-1 ceiling, and this check
    runs at a fraction of it.
    """
    model, tokenizer = hf

    def generate_ms(generate, prompt):
        return elapsed_ms(lambda: generate(model, tokenizer, prompt, 32))[1]

    generate_ms(cached_generate, "warmup")
    speedups = []
    for repeats in (1, 60, 200):
        prompt = "The history of computing began " * repeats
        prompt_len = len(tokenizer(prompt).input_ids)
        speedups.append((prompt_len, generate_ms(naive_generate, prompt)
                         / generate_ms(cached_generate, prompt)))

    print()
    for prompt_len, speedup in speedups:
        print(f"  prompt {prompt_len:>5} tok -> cache is {speedup:5.1f}x faster")
    assert speedups[-1][1] > speedups[0][1], (
        "the speedup must increase with the prompt length")
    print("\n  \033[2mThe KV cache is not a constant-factor gain. The gain"
          "\n  increases with context. That is why long-context serving"
          "\n  depends on how well you manage KV memory.\033[0m")


def test_scaling_is_linear_not_quadratic(hf, device):
    """With no cache, the cost is O(N^2) in the generated length. With a
    cache it is O(N). Prove it."""
    model, tokenizer = hf

    def generate_ms(num_tokens):
        return elapsed_ms(lambda: cached_generate(
            model, tokenizer, LONG_PROMPT, max_tokens=num_tokens))[1]

    generate_ms(8)
    short_ms, long_ms = generate_ms(32), generate_ms(64)
    ratio = long_ms / short_ms
    print(f"\n  32 tok: {short_ms:.0f} ms | 64 tok: {long_ms:.0f} ms | "
          f"ratio {ratio:.2f}x")
    print("  \033[2mLinear is about 2.0x. Quadratic is about 4.0x.\033[0m")
    assert ratio < 2.8, (
        f"the scaling looks quadratic ({ratio:.2f}x for 2x the tokens)")


def test_bf16_divergence_is_real(hf, device):
    """Not a pass or a fail on your code. A fact that you must know.

    In bf16, the cached and uncached paths can give DIFFERENT TEXT from the
    same prompt. They add in different orders, and argmax changes where two
    tokens are almost equal. That is why the correctness checks above use
    fp32. It is also why "same model, same prompt, different batch size,
    different output" is normal in production serving, not a bug.
    """
    model, tokenizer = hf            # bf16 on purpose
    prompt = "The capital of France is"
    uncached = naive_generate(model, tokenizer, prompt, max_tokens=24)
    cached = cached_generate(model, tokenizer, prompt, max_tokens=24)
    if uncached == cached:
        print("\n  \033[2mThe bf16 paths agreed on this prompt. Frequently "
              "they do not.\033[0m")
        return
    first_difference = next(index for index in range(min(len(uncached),
                                                         len(cached)))
                            if uncached[index] != cached[index])
    print(f"\n  \033[33mThe bf16 paths diverged at token "
          f"{first_difference}\033[0m")
    print(f"    uncached: {tokenizer.decode(uncached)!r}")
    print(f"    cached:   {tokenizer.decode(cached)!r}")
    print("  \033[2mSame weights, same prompt, different sentence. Remember"
          "\n  this when a serving bug 'only occurs at batch size 8'.\033[0m")
