"""Checks that the docs say what stages.yaml says.

The README, docs/OVERVIEW.md and docs/MAP.md repeat facts about the ladder:
the stage counts, the stage numbers of each box of the map, the Part of each
arc. Each fact lives in stages.yaml. When the ladder changes and a doc does
not, a learner reads a number that is not true. These checks catch that.

They are cheap and framework-free, so they run on both tracks.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def ladder():
    return yaml.safe_load((ROOT / "stages.yaml").read_text())


def all_stages():
    return [stage for arc in ladder()["arcs"] for stage in arc["stages"]]


def label(stage):
    return stage["id"].split("-")[0]


def test_overview_counts_match_the_ladder():
    stages = all_stages()
    extensions = [stage for stage in stages if stage.get("extension")]
    on_jax = [stage for stage in stages if "jax" in stage.get("tracks", ("torch", "jax"))]
    in_jax = [stage for stage in on_jax if "jax_file" in stage]
    shared = len(on_jax) - len(in_jax)
    overview = (ROOT / "docs" / "OVERVIEW.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert (f"PyTorch Track ({len(stages) - len(extensions)} core stages, and "
            f"{len(extensions)} optional extensions)") in overview
    assert f"JAX Track ({len(on_jax)} stages)" in overview
    assert f"{len(in_jax)} stages in JAX" in overview
    assert f"the {shared} framework-free stages" in overview
    assert f"all {len(stages)} stages" in readme
    core = len(stages) - len(extensions)
    for text in (readme, overview):     # one count, in the words of the first screen
        assert (f"**{len(stages)} stages:** {core} core stages and {len(extensions)} optional "
                "extensions") in text, "the stage count of the README and OVERVIEW.md differs"


def stage_numbers_in(text):
    """-> each stage number that docs/MAP.md names: the numbers at the end of
    a line of the layer diagram, the numbers in *( )* in the life of a
    request, and the table cells that hold only stage numbers."""
    groups = re.findall(r"│\s+([0-9bc, ]+)$", text, re.MULTILINE)
    groups += re.findall(r"\*\(([0-9bc, ]+)\)\*", text)
    for row in re.findall(r"^\|(.*)\|$", text, re.MULTILINE):
        groups += [cell for cell in row.split("|")
                   if re.fullmatch(r"\s*\d\d[bc]?(,\s*\d\d[bc]?)*\s*", cell)]
    return {number.strip() for group in groups for number in group.split(",")
            if number.strip()}


def test_map_names_only_real_stages():
    real = {label(stage) for stage in all_stages()}
    named = stage_numbers_in((ROOT / "docs" / "MAP.md").read_text())
    assert named, "the parser found no stage numbers in docs/MAP.md"
    assert not named - real, f"docs/MAP.md names stages that do not exist: {named - real}"


@pytest.mark.parametrize("arc", ladder()["arcs"], ids=lambda arc: arc["id"])
def test_arc_n_is_part_n(arc):
    """Arc A3 is course Part 3. The runner uses this to send a stuck learner
    to the notebooks of their Part. A9 is after the notebooks."""
    number = int(arc["id"].lstrip("A"))
    parts = sorted((ROOT / "course").glob(f"Part{number}_*"))
    if number == 9:
        assert not parts
        return
    assert len(parts) == 1, f"arc {arc['id']} has no course Part {number}"


@pytest.mark.parametrize("stage", all_stages(), ids=lambda stage: stage["id"])
def test_every_stage_says_where_it_is_why_and_what_is_given(stage):
    assert stage.get("layer", "").strip(), "no `layer:`: where on docs/MAP.md?"
    assert stage.get("because", "").strip(), "no `because:`: which bill asks for it?"
    assert stage.get("given"), "no `given:`: what does the learner have before they start?"


# ---------------------------------------------------------------- pictures
# docs/PICTURES.md has one ASCII picture for each stage. `./vc guide`, `./vc
# picture` and the TUI show it, so a missing or broken picture is a hole in
# the guide. The rules are at the top of docs/PICTURES.md.

PICTURE_WIDTH = 72


def pictures():
    from runner import pictures as module
    return module.load()


@pytest.mark.parametrize("stage", all_stages(), ids=lambda stage: stage["id"])
def test_every_stage_has_a_picture(stage):
    picture = pictures().get(stage["id"])
    assert picture, f"docs/PICTURES.md has no '## {stage['id']}' entry"
    assert picture.caption and picture.art.strip()


def test_pictures_name_only_real_stages():
    real = {stage["id"] for stage in all_stages()} | {"map"}
    keys = {key.removesuffix(" jax") for key in pictures()}
    assert not keys - real, f"pictures of stages that do not exist: {keys - real}"
    for key in pictures():
        if key.endswith(" jax"):
            stage = next(s for s in all_stages() if s["id"] == key.removesuffix(" jax"))
            assert "jax_file" in stage, f"'{key}': the stage is not on the JAX track"


@pytest.mark.parametrize("key", sorted(pictures()))
def test_picture_is_plain_ascii_and_fits(key):
    for number, line in enumerate(pictures()[key].art.splitlines(), 1):
        assert line.isascii(), f"{key}, line {number} is not plain ASCII: {line!r}"
        assert len(line) <= PICTURE_WIDTH, (
            f"{key}, line {number} has {len(line)} columns, more than "
            f"{PICTURE_WIDTH}: {line!r}")


def test_the_picture_map_names_only_real_stages():
    from runner.pictures import LABEL
    real = {label(stage) for stage in all_stages()}
    named = set(LABEL.findall(pictures()["map"].art))
    assert not named - real, f"the map names stages that do not exist: {named - real}"
    assert not real - named, f"the map leaves out these stages: {real - named}"
