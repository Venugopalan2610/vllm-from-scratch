"""Find each glossary term that a notebook uses before the course defines it.

The course defines a term at its first use, in one of three ways:

  - in bold, with a definition next to it:  **KV cache**
  - as a link to the glossary:  [KV cache](../../GLOSSARY.md#kv-cache)
  - in a markdown heading:  ### Top-k: keep the k best

The script reads the notebooks in the reading order, which is in their
names: Part3_PagedAttention/2_gather/part3_gat_1_attentionFromScratch.ipynb
is Part 3, section 2, notebook 1. It reads only the markdown cells, and it
ignores code blocks and text in backticks.

    .venv/bin/python dev/jargon.py          report, exit 1 if a term is early
    .venv/bin/python dev/jargon.py --fix    link each early use to the glossary
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.glossary import GLOSSARY, load_glossary  # noqa: E402

COURSE = ROOT / "course"
# Part<number>_<name>/<number>_<section>/part<number>_<abbreviation>_<number>_<name>.ipynb
NOTEBOOK_PATH = re.compile(r"Part(\d+)_[^/]+/(\d+)_[^/]+/part\d+_[a-z]+_(\d+)_[^/]+\.ipynb$")
CODE = re.compile(r"```.*?```|`[^`\n]*`|^ {4,}\S.*$", re.DOTALL | re.MULTILINE)
BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
GLOSSARY_LINK = re.compile(r"\[([^\]]+)\]\([^)]*GLOSSARY\.md[^)]*\)")
HEADING = re.compile(r"^#+ (.*)$", re.MULTILINE)
MIN_NAME_LENGTH = 2          # "Q", "K" and "V" match too much ordinary text


@dataclass
class EarlyUse:
    notebook: Path
    cell_index: int
    start: int
    end: int
    term: object

    def context(self, source, width=60):
        middle = (self.start + self.end) // 2
        snippet = source[max(0, middle - width // 2):middle + width // 2]
        return " ".join(snippet.split())


def notebooks():
    """-> each notebook that a student reads: all of them but the solutions."""
    return [path for path in COURSE.rglob("*.ipynb")
            if "solutions" not in path.parts and ".ipynb_checkpoints" not in path.parts]


def order_key(path):
    """(part, section, notebook), from the numbers in the path. None if the
    path does not have the numbers."""
    match = NOTEBOOK_PATH.search(path.relative_to(COURSE).as_posix())
    return tuple(int(number) for number in match.groups()) if match else None


def reading_order():
    """-> the notebook paths, in the order that a student reads them."""
    return sorted(notebooks(), key=order_key)


def masked(source):
    """The source, with spaces in place of each code span. The length stays
    the same, so a position in the result is the same position in the source."""
    return CODE.sub(lambda match: " " * len(match.group()), source)


def markdown_cells(notebook):
    """-> (cell index, source) of each markdown cell, without the header cell."""
    return [(index, "".join(cell["source"])) for index, cell in enumerate(notebook["cells"])
            if index > 0 and cell["cell_type"] == "markdown"]


def name_pattern(name):
    """A whole-word pattern. A name with capitals inside it, like "KV" or
    "TTFT", must match its case. Other names match any case."""
    has_inner_capital = any(character.isupper() for character in name[1:])
    flags = 0 if has_inner_capital else re.IGNORECASE
    return re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", flags)


def defining_spans(text):
    """-> the (start, end) of each bold span, each glossary link and each
    heading. A heading introduces its topic, so it counts as a definition."""
    return [match.span(1) for pattern in (BOLD, GLOSSARY_LINK, HEADING)
            for match in pattern.finditer(text)]


def first_match(term, text):
    """-> the first match of any name of the term in text, or None."""
    matches = [match for name in term.names if len(name) >= MIN_NAME_LENGTH
               for match in name_pattern(name).finditer(text)]
    return min(matches, key=lambda match: match.start()) if matches else None


def is_defined_at(position, spans):
    return any(start <= position < end for start, end in spans)


def first_use_in(term, cells):
    """-> (cell index, match, True if the use is a definition) of the first
    use, or None."""
    for cell_index, source in cells:
        text = masked(source)
        match = first_match(term, text)
        if match:
            return cell_index, match, is_defined_at(match.start(), defining_spans(text))
    return None


def early_uses(terms, notebooks):
    """-> an EarlyUse for each term that a notebook uses before a definition."""
    defined, early = set(), []
    for notebook_path in notebooks:
        cells = markdown_cells(json.loads(notebook_path.read_text()))
        for term in terms:
            if term.name in defined:
                continue
            use = first_use_in(term, cells)
            if use is None:
                continue
            cell_index, match, is_defined = use
            if not is_defined:
                early.append(EarlyUse(notebook_path, cell_index, match.start(),
                                      match.end(), term))
            defined.add(term.name)
    return early


def glossary_link(notebook_path, text, term):
    relative = os.path.relpath(GLOSSARY, notebook_path.parent)
    return f"[{text}]({relative}#{term.anchor})"


def add_links(early):
    """Wrap each early use in a link to its glossary entry."""
    for notebook_path in sorted({use.notebook for use in early}):
        notebook = json.loads(notebook_path.read_text())
        uses = [use for use in early if use.notebook == notebook_path]
        # From the end of each cell to its start, so earlier positions stay valid.
        for use in sorted(uses, key=lambda use: (use.cell_index, use.start), reverse=True):
            cell = notebook["cells"][use.cell_index]
            source = "".join(cell["source"])
            link = glossary_link(notebook_path, source[use.start:use.end], use.term)
            source = source[:use.start] + link + source[use.end:]
            cell["source"] = source.splitlines(keepends=True)
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"linked {len(uses)} terms in {notebook_path.relative_to(COURSE)}")


def report(early):
    current = None
    for use in early:
        if use.notebook != current:
            print(f"\n{use.notebook.relative_to(COURSE)}")
            current = use.notebook
        notebook = json.loads(use.notebook.read_text())
        source = "".join(notebook["cells"][use.cell_index]["source"])
        print(f"  {use.term.name:<24} ...{use.context(source)}...")


def main(argv):
    terms = [term for term in load_glossary().values() if term.checked]
    unnumbered = [path for path in notebooks() if order_key(path) is None]
    if unnumbered:
        print("These notebooks have no numbers in their path, so they have no "
              "place in the reading order:")
        for path in unnumbered:
            print(f"  {path.relative_to(COURSE)}")
        return 1
    ordered = reading_order()
    early = early_uses(terms, ordered)
    if "--fix" in argv:
        add_links(early)
        return 0
    report(early)
    print(f"\n{len(early)} terms used before a definition, "
          f"{len(terms)} terms checked, {len(ordered)} notebooks.")
    return 1 if early else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
