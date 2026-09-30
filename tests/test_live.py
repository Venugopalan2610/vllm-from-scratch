"""Checks on the live engine (tvllm/live.py) itself, not on any stage.

The live engine is glue. These checks prove that the glue changes nothing:
with only given parts it must give the tokens of the oracle exactly, in fp32.
With the reference solutions in app/ (dev/verify.sh puts them there), it
must also run every part from 06 to 19 and stay with the oracle.
"""

import pytest

from tvllm.live import LIVE_STAGES, LiveEngine, LiveStall, Parts, oracle_greedy

PROMPTS = ["The capital of France is", "def fibonacci(n):", "A good unit test should"]
NEAR_TIE = 0.25


def test_with_no_stage_every_slot_is_given():
    assert all(whose == "given" for _, whose, _ in Parts().table())
    assert Parts().attention() is None


def test_the_given_engine_matches_the_oracle(tmodel_exact):
    model = tmodel_exact
    engine = LiveEngine(model, Parts(), num_blocks=64, max_model_len=256, ignore_eos=True)
    prompts = [model.tokenizer(text).input_ids for text in PROMPTS]
    for index, ids in enumerate(prompts):
        engine.add_request(index, ids, 12)
    outputs = engine.run_to_completion()
    for index, ids in enumerate(prompts):
        assert outputs[index] == oracle_greedy(model, ids, 12)[0], (
            "the given engine changed the tokens: the glue of tvllm/live.py is wrong")


def test_a_request_that_can_never_fit_stops_with_a_reason(tmodel_exact):
    engine = LiveEngine(tmodel_exact, Parts(), num_blocks=4, max_model_len=1024)
    engine.add_request("big", [1, 2, 3], 4)
    with pytest.raises(LiveStall, match="stopped making progress"):
        engine.run_to_completion(max_idle_steps=5)


def test_every_part_that_you_finished_runs_inside_the_engine(tmodel_exact):
    """With the reference solutions in app/, every slot holds a real part.
    With stubs (a learner's checkout), a part raises NotImplementedError, and
    this check skips."""
    model = tmodel_exact
    prompts = [model.tokenizer(text).input_ids for text in PROMPTS]
    for upto in (1, 2, 6, 7, 8, len(LIVE_STAGES) - 5):
        parts = Parts(LIVE_STAGES[:upto])
        try:
            engine = LiveEngine(model, parts, num_blocks=64, max_model_len=256, ignore_eos=True)
            for index, ids in enumerate(prompts):
                engine.add_request(index, ids, 12)
            outputs = engine.run_to_completion()
        except NotImplementedError:
            pytest.skip("a stage in app/ is not implemented: this check needs dev/verify.sh")
        for index, ids in enumerate(prompts):
            expected, gaps = oracle_greedy(model, ids, 12)
            first = next((i for i, (a, b) in enumerate(zip(outputs[index], expected)) if a != b), None)
            assert first is None or gaps[first] <= NEAR_TIE, (
                f"with the parts up to {LIVE_STAGES[upto - 1]}, request {index} differs from the "
                f"oracle at token {first}, where the oracle's gap is {gaps[first]:.2f}")
