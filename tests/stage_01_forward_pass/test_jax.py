"""Stage 01 (JAX) - greedy decode with no cache.

The spec:

    app/j01_naive.py must define

        naive_generate(model, prompt: str, max_tokens: int) -> list[int]

    It returns the list of GENERATED token ids (without the prompt).

Rules:
  - Greedy: always the argmax of the logits at the last position.
  - NO KV cache. Every step runs the model again over the whole prefix.
  - Stop early if you emit an eos token.

Why you must do it the slow way: a check of stage 02 asserts a speedup over
this number. If you cache here, you cannot pass stage 02.
"""

import jax.numpy as jnp

from app.j01_naive import naive_generate
from tests.helpers import elapsed_ms, record_measurement

CHAT_QUESTION = "What is 2+2? Reply with just the number."


def _reference_greedy(model, prompt, max_tokens):
    """A different code path from the expected solution, on purpose.

    This asks for the logits at EVERY position and takes the last one. It
    does not let the model gather the last position. The tokens are the
    same and the HLO is different. That also shows that the two forms agree.
    """
    token_ids = [int(token) for token in model.encode(prompt)]
    generated = []
    for _ in range(max_tokens):
        logits, _ = model.forward(jnp.asarray([token_ids], jnp.int32),
                                  logits_index=None)
        next_token = int(logits[0, -1].argmax())
        if next_token in model.eos_ids:
            break
        generated.append(next_token)
        token_ids.append(next_token)
    return generated


def test_matches_a_reference_greedy_decode(jmodel_exact, prompts):
    """The same tokens as a simple greedy loop, in fp32."""
    model = jmodel_exact
    for prompt in prompts[:2]:
        expected = _reference_greedy(model, prompt, 12)
        generated = naive_generate(model, prompt, max_tokens=12)
        assert generated == expected, (
            f"\nprompt:   {prompt!r}\n"
            f"expected: {model.decode(expected)!r}\n"
            f"got:      {model.decode(generated)!r}"
        )


def test_respects_max_tokens(jmodel_exact):
    generated = naive_generate(jmodel_exact, "Count: 1 2 3 4", max_tokens=7)
    assert len(generated) <= 7, (
        f"emitted {len(generated)} tokens, max_tokens was 7")


def test_stops_at_eos(jmodel_exact):
    """A question in chat format must stop, and not continue to max_tokens.

    Trap: `model.eos_ids` is a SET, because this model has several stop ids
    (end of text and end of turn). Test membership. Do not compare with one
    int.
    """
    model = jmodel_exact
    prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": CHAT_QUESTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    generated = naive_generate(model, prompt, max_tokens=40)
    assert len(generated) < 40, (
        f"never stopped. model.eos_ids is {sorted(model.eos_ids)}. Note that "
        "it is a set of several ids.")


def test_prompt_is_not_echoed(jmodel_exact):
    """Return only what you generated. The prompt ids are not output."""
    model = jmodel_exact
    prompt = "The capital of France is"
    prompt_len = len(model.encode(prompt))
    generated = naive_generate(model, prompt, max_tokens=8)
    assert len(generated) <= 8, (
        f"returned {len(generated)} ids for max_tokens=8. The prompt "
        f"({prompt_len} ids) is probably still in the output.")


def test_every_new_length_costs_a_compile(jmodel):
    """The cost that is special to JAX, measured.

    Each step gives a prefix one token longer than the step before. So each
    step is a new SHAPE, and each new shape is a new XLA compilation: one
    for each token, seconds each, before any arithmetic.

    The prompt has an odd length that no other check uses, on purpose. So
    the compilation cache cannot already hold these shapes.
    """
    from jvllm.model import _forward

    num_tokens = 6
    prompt = "The history of computing began " * 30      # about 150 tokens
    compiles_before = _forward._cache_size()
    _, wall_ms = elapsed_ms(
        lambda: naive_generate(jmodel, prompt, max_tokens=num_tokens))
    new_compiles = _forward._cache_size() - compiles_before

    print(f"\n  {new_compiles} new XLA compilations for {num_tokens} generated "
          f"tokens ({wall_ms / 1000:.1f}s wall)")
    print("  \033[2mA preallocated cache (stage 02) has ONE shape, so it")
    print("  compiles one time only. On short prompts, that alone is most of")
    print("  the speedup of stage 02, before one flop is saved.\033[0m")
    assert new_compiles >= num_tokens, (
        f"expected about {num_tokens} new compilations, one for each prefix "
        f"length, but the cache grew by {new_compiles}. Do you run the whole "
        "prefix again at each step, with no cache?")


def test_baseline_throughput(jmodel):
    """Not a pass or a fail. It records your slowest number, two times.

    COLD includes the XLA compile of one program for each prefix length.
    WARM runs the same lengths again, so each compile is a cache hit, and
    only the arithmetic remains. Stage 02 compares with the WARM number. A
    30x gain that is really "I stopped the compiler" tells you nothing about
    the KV cache.

    Both numbers are real, and a server pays the cold one.
    """
    prompt = "The history of computing began"
    generated, cold_ms = elapsed_ms(
        lambda: naive_generate(jmodel, prompt, max_tokens=24))
    _, warm_ms = elapsed_ms(
        lambda: naive_generate(jmodel, prompt, max_tokens=24))

    print(f"\n  \033[36mj01 cold\033[0m: {cold_ms:8.0f} ms  "
          f"({len(generated) / (cold_ms / 1000):6.1f} tok/s)   compile included")
    print(f"  \033[36mj01 warm\033[0m: {warm_ms:8.0f} ms  "
          f"({len(generated) / (warm_ms / 1000):6.1f} tok/s)   compile cached")
    print(f"\n  \033[2mThe compiler cost you {cold_ms / warm_ms:.0f}x on the "
          "first run of a")
    print("  shape that it had not seen. That is not the algorithm. It is the")
    print("  cost of each new sequence length that reaches your server.\033[0m")
    record_measurement("j01_naive_warm_ms", warm_ms)
    assert generated, "generated nothing"
