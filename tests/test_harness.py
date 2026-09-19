"""Checks on the harness itself, not on any stage.

These checks exist because a CONVENTION connects the two tracks. A stage
belongs to the JAX track when stages.yaml gives it a `jax_file`. A test file
belongs to the JAX track when its name is test_jax.py.

That is one rule in two places, and only this file stops the two from
separating.

They are cheap and framework-free, so they run on both tracks.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def stages():
    data = yaml.safe_load((ROOT / "stages.yaml").read_text())
    return [stage for arc in data["arcs"] for stage in arc["stages"]]


def _test_dir(stage):
    return ROOT / "tests" / f"stage_{stage['id'].replace('-', '_')}"


@pytest.mark.parametrize("stage", stages(), ids=lambda stage: stage["id"])
def test_every_stage_has_a_test_directory(stage):
    directory = _test_dir(stage)
    assert directory.is_dir(), (
        f"{stage['id']} has no {directory.relative_to(ROOT)}")


@pytest.mark.parametrize("stage", stages(), ids=lambda stage: stage["id"])
def test_jax_twins_line_up(stage):
    """A stage with a jax_file must have a test_jax.py, and the opposite.

    A missing test_jax.py gives a failure with no signal. conftest then
    treats the stage as free of a framework and runs the TORCH checks on the
    JAX track. They pass, because they never test the JAX file.
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


@pytest.mark.parametrize("stage", stages(), ids=lambda stage: stage["id"])
def test_stage_files_exist(stage):
    assert (ROOT / stage["file"]).exists(), f"missing {stage['file']}"


@pytest.mark.parametrize("stage", stages(), ids=lambda stage: stage["id"])
def test_every_declared_file_exists(stage):
    """A CUDA stage is two files: the .cu and the .py that builds it. `files`
    lists them, and `./vc guide` and `./vc peek` both read that list. So a
    typo there sends the learner to a file that does not exist."""
    for path in stage.get("files", []):
        assert (ROOT / path).exists(), (
            f"{stage['id']} lists a missing file: {path}")
    if "files" in stage:
        assert stage["file"] in stage["files"], (
            f"{stage['id']}: `file` must also appear in `files`, because the "
            "two are read by different commands")


@pytest.mark.parametrize("stage", stages(), ids=lambda stage: stage["id"])
def test_torch_only_stages_name_their_checks_test_cuda(stage):
    """conftest decides a stage is framework-free when it has no test_jax.py,
    and runs it on BOTH tracks. That rule is wrong for a CUDA stage, which
    has no JAX twin and no business on the JAX ladder either.

    So a stage marked `tracks: [torch]` names its checks test_cuda.py, which
    conftest deselects on the JAX track. Get this wrong and the failure is
    silent in the worst direction: a JAX learner is quietly asked to write
    __shfl_xor_sync.
    """
    directory = _test_dir(stage)
    torch_only = stage.get("tracks") == ["torch"]
    cuda_checks = (directory / "test_cuda.py").exists()

    if torch_only:
        assert cuda_checks, (
            f"{stage['id']} is torch-only but its checks are not in "
            f"{(directory / 'test_cuda.py').relative_to(ROOT)}, so --backend jax "
            "would run them")
        assert not (directory / "test_stage.py").exists(), (
            f"{directory.relative_to(ROOT)} has both test_stage.py and test_cuda.py; "
            "the test_stage.py half would run on the JAX track")
    else:
        assert not cuda_checks, (
            f"{(directory / 'test_cuda.py').relative_to(ROOT)} exists but "
            f"{stage['id']} is not marked `tracks: [torch]` in stages.yaml")


@pytest.mark.parametrize("stage", stages(), ids=lambda stage: stage["id"])
def test_track_restrictions_are_spelled_correctly(stage):
    """`tracks` keeps a stage off the other ladders. A misspelt backend there
    removes the stage from every ladder, silently."""
    for name in stage.get("tracks", []):
        assert name in ("torch", "jax"), (
            f"{stage['id']} lists an unknown track {name!r}")
    if "tracks" in stage:
        assert "jax_file" not in stage, (
            f"{stage['id']} is restricted to {stage['tracks']} but also "
            "declares a jax_file")
