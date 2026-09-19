"""Stage 01 - greedy decode with no cache.

The spec:

    app/s01_naive.py must define

        naive_generate(model, tokenizer, prompt: str, max_tokens: int) -> list[int]

    It returns the list of GENERATED token ids (without the prompt).

Rules:
  - Greedy: always the argmax of the logits at the last position.
  - NO KV cache. Every step runs the model again over the whole prefix
    (the prompt and all the tokens so far). Give use_cache=False.
  - Stop early if you emit the eos token.

Why you must do it the slow way: a check of stage 02 asserts a speedup over
this number. If you cache here, you cannot pass stage 02.
"""

from app.s01_naive import naive_generate

CHAT_QUESTION = "What is 2+2? Reply with just the number."


def test_matches_reference_greedy(hf_exact, prompts, device):
    """Your output must be the same tokens as the greedy decode of
    HuggingFace.

    The check uses fp32. The hf_exact fixture tells why.
    """
    model, tokenizer = hf_exact
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        expected = model.generate(
            prompt_ids, max_new_tokens=24, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0, prompt_ids.shape[1]:].tolist()

        generated = naive_generate(model, tokenizer, prompt, max_tokens=24)

        assert generated == expected, (
            f"\nprompt:   {prompt!r}\n"
            f"expected: {tokenizer.decode(expected)!r}\n"
            f"got:      {tokenizer.decode(generated)!r}"
        )


def test_respects_max_tokens(hf_exact, device):
    model, tokenizer = hf_exact
    generated = naive_generate(model, tokenizer, "Count: 1 2 3 4", max_tokens=7)
    assert len(generated) <= 7, (
        f"emitted {len(generated)} tokens, max_tokens was 7")


def test_stops_at_eos(hf_exact, device):
    """A question in chat format must stop, and not continue to max_tokens.

    Trap: the generation_config.eos_token_id of this model is a LIST of ids,
    not one int. Models frequently have several stop ids (end of text and
    end of turn). Accept both forms. If not, every chat request continues to
    max_tokens.
    """
    model, tokenizer = hf_exact
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": CHAT_QUESTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    generated = naive_generate(model, tokenizer, prompt, max_tokens=40)
    assert len(generated) < 40, (
        "never stopped. model.generation_config.eos_token_id is "
        f"{model.generation_config.eos_token_id}. Note that it is a list.")
    # This asserts only that it STOPPED. The answer of a 0.6B model to 2+2
    # is not your problem.


def test_baseline_throughput(hf, device, timer):
    """Not a pass or a fail. This records your lowest number."""
    model, tokenizer = hf
    with timer("s01_naive_nocache") as timing:
        generated = naive_generate(model, tokenizer,
                                   "The history of computing began",
                                   max_tokens=64)
    timing.report(tokens=len(generated))
    print("\n  \033[2mThis is the slowest that it will be. Remember it.\033[0m")
