"""Checks on the harness itself, not on any stage.

These exist because the two tracks are wired together by CONVENTION -- a stage
belongs to the JAX track if stages.yaml gives it a `jax_file`, and a test file
belongs to the JAX track if it is named test_jax.py. Two places, one rule, and
nothing but this file to stop them drifting apart.

They are cheap and framework-free, so they run on both tracks.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def stages():
    data = yaml.safe_load((ROOT / "stages.yaml").read_text())
    return [s for arc in data["arcs"] for s in arc["stages"]]


def _test_dir(stage):
    return ROOT / "tests" / f"stage_{stage['id'].replace('-', '_')}"


@pytest.mark.parametrize("stage", stages(), ids=lambda s: s["id"])
def test_every_stage_has_a_test_directory(stage):
    d = _test_dir(stage)
    assert d.is_dir(), f"{stage['id']} has no {d.relative_to(ROOT)}"


@pytest.mark.parametrize("stage", stages(), ids=lambda s: s["id"])
def test_jax_twins_line_up(stage):
    """A stage declaring a jax_file must have a test_jax.py, and vice versa.

    Get this wrong in the direction of a missing test_jax.py and the failure is
    silent and confusing: conftest treats the stage as framework-free, runs the
    TORCH checks on the JAX track, and they pass, because they were never
    testing the JAX file at all.
    """
    twin = _test_dir(stage) / "test_jax.py"
    declared = "jax_file" in stage

    if declared:
        assert twin.exists(), (
            f"{stage['id']} declares jax_file={stage['jax_file']} but there is "
            f"no {twin.relative_to(ROOT)}. On --backend jax this stage would "
            "silently run the torch checks instead."
        )
        assert (ROOT / stage["jax_file"]).exists(), (
            f"{stage['id']} declares jax_file={stage['jax_file']}, which does "
            "not exist"
        )
    else:
        assert not twin.exists(), (
            f"{twin.relative_to(ROOT)} exists but {stage['id']} declares no "
            "jax_file in stages.yaml, so ./vc will never point anyone at it"
        )


@pytest.mark.parametrize("stage", stages(), ids=lambda s: s["id"])
def test_stage_files_exist(stage):
    assert (ROOT / stage["file"]).exists(), f"missing {stage['file']}"
