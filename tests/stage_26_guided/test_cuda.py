"""Stage 26 - guided JSON decoding inside the engine.

The spec is in app/s26_guided.py. The automaton must agree with the json
module of Python. Every JSON-mode answer must parse, also when max_tokens is
small. And the masks must come from a cache, not from a new build at every
step.
"""

import json
from itertools import product

import pytest

from app.s23_graphs import GraphedModelRunner
from app.s26_guided import START, GuidedEngine, distance, is_valid_json, run

CASES = ['{}', '{"a":1}', '{"a":[1,2,{"b":null}],"c":"x\\"y"}',
         '{"a":-0.5e+3}', '{"k":"\\u00e9"}', '{"a":1e5}', '{"a":-0}',
         '{"a":tru}', '{"a":01}', '{"a":1,}', '{"a" 1}', '{"a":.5}',
         '{"a":"\\x"}', '{"a":[1 2]}']


@pytest.mark.parametrize("text", CASES)
def test_the_automaton_agrees_with_json(text):
    state = run(START, text)
    accepted = state is not None and state.mode == "done"
    assert accepted == is_valid_json(text), text


def test_an_array_at_the_top_is_refused():
    """JSON mode asks for an object, as the OpenAI API does."""
    assert run(START, "[1]") is None


@pytest.mark.parametrize("prefix", ['{', '{"ab', '{"a":', '{"a":[1,', '{"a":tr',
                                    '{"a":"x\\u00'])
def test_distance_is_the_fewest_characters(prefix):
    """No shorter completion exists: try every string up to that length."""
    fewest = distance(run(START, prefix))
    alphabet = '0}]":e'
    shorter_exists = any(is_valid_json(prefix + "".join(tail))
                         for length in range(fewest)
                         for tail in product(alphabet, repeat=length))
    assert not shorter_exists
    assert fewest > 0


@pytest.fixture(scope="module")
def engine(tmodel):
    runner = GraphedModelRunner(tmodel, 800, max_model_len=1024,
                                buckets=(1, 2, 4, 8))
    runner.capture()
    yield GuidedEngine(tmodel, 800, runner=runner, num_speculative=4)
    runner.kv_caches.clear()


def test_every_answer_parses(nvcc, engine):
    """Small budgets on purpose: the object must close in time."""
    prompts = ["Describe a cat as JSON.", "A book with title and year, as JSON.",
               "The weather in Paris as JSON.", "Three fruits in JSON.", "hello"]
    budgets = [80, 60, 40, 25, 15]
    for index, prompt in enumerate(prompts * 2):
        engine.add_request(("parse", index), prompt,
                           budgets[index % len(budgets)], json_mode=True)
    outputs = engine.run_to_completion()
    for rid, token_ids in outputs.items():
        if rid[0] == "parse":
            text = engine.tokenizer.decode(token_ids, skip_special_tokens=True)
            json.loads(text)


def test_masks_come_from_the_cache(nvcc, engine):
    """A second round of the same prompts builds almost no new mask."""
    def five_dogs(round_name):
        for index in range(5):
            engine.add_request((round_name, index), "Describe a dog as JSON.",
                               40, json_mode=True)
        engine.run_to_completion()

    five_dogs("warm")
    built_before = engine.masker.misses
    five_dogs("again")
    print(f"\n  {len(engine.masker.cache)} masks cached, "
          f"{engine.masker.hits} hits")
    assert engine.masker.misses - built_before <= 2


def test_a_plain_request_is_not_masked(nvcc, engine):
    engine.add_request("plain", "Count: 1, 2, 3,", 12, ignore_eos=True)
    text = engine.tokenizer.decode(engine.run_to_completion()["plain"])
    assert not text.strip().startswith("{")
