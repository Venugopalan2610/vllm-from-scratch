"""Read docs/PICTURES.md: one ASCII picture for each stage.

The format of one entry:

    ## 07-paged-attention            (a stage id, or "<id> jax", or "map")

    The caption: one line.

    ```text
    the picture
    ```

A "<id> jax" entry is the picture of the JAX track. A stage with no JAX
entry shows the same picture on both tracks.
"""

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICTURES = ROOT / "docs" / "PICTURES.md"

ENTRY = re.compile(
    r"^## (?P<key>[\w-]+(?: jax)?)\n+(?P<caption>(?!```)[^\n]+)\n+```text\n(?P<art>.*?)\n```",
    re.M | re.S)
# A stage label in the map: "08", "08b", "24b". Not inside a word.
LABEL = re.compile(r"(?<![\w.])(\d\d[a-z]?)(?![\w.])")


@dataclass
class Picture:
    key: str
    caption: str
    art: str


def load():
    """-> {key: Picture}. The key is a stage id, "<id> jax" or "map"."""
    if not PICTURES.exists():
        return {}
    return {match["key"]: Picture(match["key"], match["caption"].strip(),
                                  match["art"])
            for match in ENTRY.finditer(PICTURES.read_text())}


def for_stage(stage_id, backend="torch", pictures=None):
    """The picture of a stage on a track, or None."""
    pictures = load() if pictures is None else pictures
    if backend == "jax" and f"{stage_id} jax" in pictures:
        return pictures[f"{stage_id} jax"]
    return pictures.get(stage_id)


def color_map(art, done, current, paint):
    """The map, with each stage label colored: green when it is done,
    yellow and bold when it is the current stage, dim when it is ahead.

    done, current: stage labels, like {"01", "02"} and "03"."""
    def recolor(match):
        label = match.group(1)
        if label in done:
            return paint(label, "green")
        if label == current:
            return paint(label, "yellow", "bold")
        return paint(label, "dim")
    return LABEL.sub(recolor, art)
