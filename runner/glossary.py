"""Read course/GLOSSARY.md: the terms of the course, their other names, and
their definitions.

The format of one entry:

    ### KV cache

    *Also:* KV, key-value cache

    The keys and values of all earlier tokens. ...

    **Like:** memoization of the attention inputs.

    **Check:** no                  (optional: the checker ignores the term)

    **Taught in:** Part 1 `3_kvCache/`. Stage 02.
"""

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "course" / "GLOSSARY.md"

ALIASES_PREFIX = "*Also:*"
FIELD = re.compile(r"^\*\*(?P<name>[^*]+):\*\*\s*(?P<text>.*)$")
# "Stage 02", "stage 18b", "Stages 06 to 08c"
STAGE_REFERENCE = re.compile(r"\b[Ss]tages? (\d\d[a-z]?)(?: to (\d\d[a-z]?))?")


@dataclass
class Term:
    name: str
    aliases: list
    definition: str
    fields: dict            # "Like", "Taught in", "Check" -> text

    @property
    def anchor(self):
        """The GitHub anchor of the heading of this term."""
        slug = re.sub(r"[^\w\- ]", "", self.name.lower())
        return slug.replace(" ", "-")

    @property
    def checked(self):
        """False if the checker must ignore this term."""
        return self.fields.get("Check", "").strip().lower() != "no"

    @property
    def names(self):
        return [self.name] + self.aliases

    def stages(self, ladder):
        """-> the stage labels that the Taught in field names, like "08b".
        ladder: all stage labels, in order, to expand "Stages 06 to 08c"."""
        labels = []
        for first, last in STAGE_REFERENCE.findall(self.fields.get("Taught in", "")):
            if last:
                labels += ladder[ladder.index(first):ladder.index(last) + 1]
            else:
                labels.append(first)
        return labels

    def summary(self):
        """The first sentence of the definition."""
        match = re.match(r"(.+?\.)(\s|$)", self.definition)
        return match.group(1) if match else self.definition


def _paragraphs(lines):
    """Divide lines into paragraphs at blank lines. -> list of joined strings."""
    paragraphs, current = [], []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _parse_entry(name, lines):
    aliases, definition, fields = [], [], {}
    for paragraph in _paragraphs(lines):
        field = FIELD.match(paragraph)
        if paragraph.startswith(ALIASES_PREFIX):
            text = paragraph[len(ALIASES_PREFIX):]
            aliases = [alias.strip() for alias in text.split(",") if alias.strip()]
        elif field:
            fields[field["name"]] = field["text"]
        else:
            definition.append(paragraph)
    return Term(name, aliases, " ".join(definition), fields)


def load_glossary(path=GLOSSARY):
    """-> {term name: Term}, in the order of the file."""
    terms, name, lines = {}, None, []
    for line in Path(path).read_text().splitlines() + ["## end"]:
        if line.startswith(("## ", "### ", "---")):
            if name:
                terms[name] = _parse_entry(name, lines)
            name, lines = None, []
            if line.startswith("### "):
                name = line[4:].strip()
        elif name:
            lines.append(line)
    return terms


def terms_for_stage(label, ladder, path=GLOSSARY):
    """-> the terms that the glossary says this stage teaches, in file order."""
    return [term for term in load_glossary(path).values()
            if label in term.stages(ladder)]
