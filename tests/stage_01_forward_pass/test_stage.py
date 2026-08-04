"""Stage 01 - Greedy decode with no cache.

The spec:

    app/s01_naive.py must define

        naive_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]

    returning the list of GENERATED token ids (not including the prompt).

Rules:
  - Greedy: always argmax of the final-position logits.
  - NO kv cache. Every step re-runs the model over the entire prefix
    (prompt + everything generated so far). Pass use_cache=False.
  - Stop early if you emit the eos token.

Why you must do it the slow way: stage 02's test asserts a speedup over
this number. If you cache here, you cannot pass stage 02.
"""

import pytest
import torch

from app.s01_naive import naive_generate


def test_matches_reference_greedy(hf_exact, prompts, dev):
    """Your output must be token-identical to HuggingFace's greedy decode.

    Checked in fp32 -- see the hf_exact fixture for why.
    """
    model, tok = hf_exact
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        want = model.generate(
            ids, max_new_tokens=24, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )[0, ids.shape[1]:].tolist()

        got = naive_generate(model, tok, p, max_tokens=24)

        assert got == want, (
            f"\nprompt: {p!r}\n"
            f"want: {tok.decode(want)!r}\n"
            f"got:  {tok.decode(got)!r}"
        )


def test_respects_max_tokens(hf_exact, dev):
    model, tok = hf_exact
    out = naive_generate(model, tok, "Count: 1 2 3 4", max_tokens=7)
    assert len(out) <= 7, f"emitted {len(out)} tokens, max_tokens was 7"


def test_stops_at_eos(hf_exact, dev):
    """A chat-formatted question must terminate, not ramble to max_tokens.

    Trap: this model's generation_config.eos_token_id is a LIST of ids, not a
    single int. Models routinely have several stop ids (end-of-text vs
    end-of-turn). Handle both shapes, or you will hang on every chat request.
    """
    model, tok = hf_exact
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    out = naive_generate(model, tok, prompt, max_tokens=40)
    assert len(out) < 40, (
        "never stopped. model.generation_config.eos_token_id is "
        f"{model.generation_config.eos_token_id} -- note it is a list."
    )
    # Note: we assert only that it STOPPED. What a 0.6B model thinks 2+2 is
    # is not your problem. (It says 2. Scale is a hell of a drug.)


def test_baseline_throughput(hf, dev, timer):
    """Not pass/fail. This records your rock-bottom number."""
    model, tok = hf
    N = 64
    with timer("s01_naive_nocache") as t:
        out = naive_generate(model, tok, "The history of computing began", max_tokens=N)
    t.report(tokens=len(out))
    print(f"\n  \033[2mThis is the slowest this will ever be. Remember it.\033[0m")
