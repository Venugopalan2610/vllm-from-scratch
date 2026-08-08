"""Stage 01 (JAX) - Greedy decode with no cache.

The spec:

    app/j01_naive.py must define

        naive_generate(model, prompt: str, max_tokens: int) -> list[int]

    returning the list of GENERATED token ids (not including the prompt).

Rules:
  - Greedy: always argmax of the final-position logits.
  - NO kv cache. Every step re-runs the model over the entire prefix.
  - Stop early if you emit an eos token.

Why you must do it the slow way: stage 02's test asserts a speedup over this
number. Cache here and you cannot pass stage 02.
"""

import time

import jax
import jax.numpy as jnp
import pytest

from app.j01_naive import naive_generate


def _reference_greedy(model, prompt, max_tokens):
    """Deliberately a different code path from the expected solution.

    This one asks for logits at EVERY position and takes the last, instead of
    letting the model gather the final position for you. Same tokens, different
    HLO -- which is also a small demonstration that the two spellings agree.
    """
    ids = [int(t) for t in model.encode(prompt)]
    out = []
    for _ in range(max_tokens):
        logits, _ = model.forward(jnp.asarray([ids], jnp.int32), logits_index=None)
        nxt = int(logits[0, -1].argmax())
        if nxt in model.eos_ids:
            break
        out.append(nxt)
        ids.append(nxt)
    return out


def test_matches_a_reference_greedy_decode(jmodel_exact, prompts):
    """Token-identical to a straightforward greedy loop, in fp32."""
    model = jmodel_exact
    for p in prompts[:2]:
        want = _reference_greedy(model, p, 12)
        got = naive_generate(model, p, max_tokens=12)
        assert got == want, (
            f"\nprompt: {p!r}\n"
            f"want: {model.decode(want)!r}\n"
            f"got:  {model.decode(got)!r}"
        )


def test_respects_max_tokens(jmodel_exact):
    out = naive_generate(jmodel_exact, "Count: 1 2 3 4", max_tokens=7)
    assert len(out) <= 7, f"emitted {len(out)} tokens, max_tokens was 7"


def test_stops_at_eos(jmodel_exact):
    """A chat-formatted question must terminate, not ramble to max_tokens.

    Trap: `model.eos_ids` is a SET, because this model has several stop ids
    (end-of-text vs end-of-turn). Check membership, do not compare to one int.
    """
    model = jmodel_exact
    prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    out = naive_generate(model, prompt, max_tokens=40)
    assert len(out) < 40, (
        f"never stopped. model.eos_ids is {sorted(model.eos_ids)} -- "
        "note that it is a set of several ids."
    )


def test_prompt_is_not_echoed(jmodel_exact):
    """Return only what you generated. The prompt ids are not output."""
    model = jmodel_exact
    p = "The capital of France is"
    n_prompt = len(model.encode(p))
    out = naive_generate(model, p, max_tokens=8)
    assert len(out) <= 8, (
        f"returned {len(out)} ids for max_tokens=8 -- the prompt "
        f"({n_prompt} ids) looks like it is still in there"
    )


def test_every_new_length_costs_a_compile(jmodel):
    """The JAX-specific tax, measured.

    Each step feeds a prefix one token longer than the last, so each step is a
    new SHAPE, so each step is a fresh XLA compilation. One per token, seconds
    each, all of it before any arithmetic happens.

    The prompt here is deliberately an odd length no other check has used, so
    the compilation cache cannot already be warm for these shapes.
    """
    from jvllm.model import _forward

    model = jmodel
    prompt = "The history of computing began " * 30      # ~150 tokens
    n = 6

    before = _forward._cache_size()
    t0 = time.perf_counter()
    naive_generate(model, prompt, max_tokens=n)
    elapsed = time.perf_counter() - t0
    grew = _forward._cache_size() - before

    print(f"\n  {grew} new XLA compilations for {n} generated tokens "
          f"({elapsed:.1f}s wall)")
    print("  \033[2mA preallocated cache (stage 02) has ONE shape, so it")
    print("  compiles once and never again. On short prompts that alone is")
    print("  most of stage 02's speedup -- before a single flop is saved.\033[0m")
    assert grew >= n, (
        f"expected ~{n} new compilations, one per prefix length, but the "
        f"cache grew by {grew}. Are you actually re-running the whole prefix "
        "each step -- i.e. actually running without a cache?"
    )


def test_baseline_throughput(jmodel):
    """Not pass/fail. This records your rock-bottom number -- twice.

    COLD includes XLA compiling one program per prefix length. WARM re-runs the
    identical lengths, so every compile is a cache hit and what is left is the
    arithmetic. Stage 02 is measured against the WARM number, because a 30x
    win that is really "I stopped invoking the compiler" would tell you nothing
    about the KV cache.

    Both numbers are real, though, and in a server you would pay the cold one.
    """
    model = jmodel
    prompt = "The history of computing began"
    N = 24

    t0 = time.perf_counter()
    out = naive_generate(model, prompt, max_tokens=N)
    cold = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    naive_generate(model, prompt, max_tokens=N)
    warm = (time.perf_counter() - t0) * 1000

    print(f"\n  \033[36mj01 cold\033[0m: {cold:8.0f} ms  "
          f"({len(out) / (cold / 1000):6.1f} tok/s)   compile included")
    print(f"  \033[36mj01 warm\033[0m: {warm:8.0f} ms  "
          f"({len(out) / (warm / 1000):6.1f} tok/s)   compile cached")
    print(f"\n  \033[2mThe compiler cost you {cold / warm:.0f}x on the first run of a")
    print("  shape it had not seen. That is not the algorithm -- it is the")
    print("  price of every new sequence length reaching your server.\033[0m")
    _record_jax("j01_naive_warm_ms", warm)
    assert out, "generated nothing"


def _record_jax(label, ms):
    """Bank a number so a later stage can compare against it."""
    import json
    from pathlib import Path

    f = Path(__file__).resolve().parents[2] / ".measurements.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[label] = {"ms": round(ms, 2), "tok_s": None}
    f.write_text(json.dumps(d, indent=2))
