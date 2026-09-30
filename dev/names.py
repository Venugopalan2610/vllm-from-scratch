"""Find each vague name in the code that a learner reads.

A name must say what it is, of what, and its unit: `step_measured_ms`, not
`ms`. A ratio names its top and its bottom: `step_measured_over_floor`, not
`ratio`. A vague name costs the reader one more thing to decode before the
idea. docs/METHOD.md has the rule.

The script reads the code cells of every notebook in course/ (the solutions
too: a learner reads them) and the stage files in app/. It parses the code,
so it finds the names that are assigned, unpacked, looped over or taken as a
parameter, not words in comments.

    .venv/bin/python dev/names.py          report, exit 1 if a name is vague
    .venv/bin/python dev/names.py --app    the stage files in app/ too

The stage files are not checked by default. Their parameters are the contract
that the checks, the solutions branch and the JAX track call by name, and
after a learner finishes a stage, the file holds the learner's own code.

Accepted: q, k and v in attention (the names of the idea, in the glossary),
and i, j and _ as a loop index.
"""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VAGUE = {
    "ratio", "ratios", "out", "outs", "output", "outputs", "result", "results", "res",
    "tmp", "temp", "val", "vals", "diff", "delta", "total", "count", "scale", "factor",
    "frac", "share", "speedup", "num", "data", "arr", "lst", "obj", "item", "items",
    "ms", "t0", "t1", "t2", "a", "b", "c", "d", "e", "f", "g", "h", "l", "m", "n", "p",
    "r", "s", "t", "u", "w", "x", "y", "z",
}
LOOP_INDEX = {"i", "j", "_"}

# A file here is not checked, with the reason. Keep this list short.
NOT_CHECKED = {
    "course/Part1_TheNaiveLoop/3_kvCache/part1_kv_3_CCwriteBothLoops_challenge.ipynb":
        "it holds the answers of the author, who works through the course as a "
        "learner (commit 7f051a5). The author decides: rename, or restore the blanks.",
}


def targets(node):
    """-> the plain names that `node` binds, as (name, line)."""
    if isinstance(node, ast.Name):
        yield node.id, node.lineno
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from targets(element)
    elif isinstance(node, ast.Starred):
        yield from targets(node.value)


def bound_names(tree):
    """-> (name, line, how) for each name that the code binds."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name, line in targets(target):
                    yield name, line, "assigned"
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            for name, line in targets(node.target):
                yield name, line, "assigned"
        elif isinstance(node, (ast.For, ast.comprehension)):
            for name, line in targets(node.target):
                if name not in LOOP_INDEX:
                    yield name, getattr(node.target, "lineno", 0), "a loop variable"
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            for argument in node.args.args + node.args.kwonlyargs:
                if argument.arg not in ("self", "cls"):
                    yield argument.arg, argument.lineno, "a parameter"
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for name, line in targets(node.optional_vars):
                yield name, line, "assigned"


ASSIGNED = re.compile(r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=(?!=)")
LOOPED = re.compile(r"\bfor\s+\(?([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\)?\s+in\b")
PARAMETERS = re.compile(r"^\s*def\s+\w+\((.*)\)")


def bound_by_pattern(lines):
    """The fallback of bound_names() for code that does not parse."""
    for number, line in enumerate(lines, 1):
        code = line.split("#")[0]
        for pattern, how in ((ASSIGNED, "assigned"), (LOOPED, "a loop variable")):
            for match in pattern.finditer(code):
                for name in match.group(1).split(","):
                    name = name.strip()
                    if how == "assigned" or name not in LOOP_INDEX:
                        yield name, number, how
        match = PARAMETERS.match(code)
        if match:
            for parameter in match.group(1).split(","):
                name = parameter.split("=")[0].split(":")[0].strip().lstrip("*")
                if name and name not in ("self", "cls"):
                    yield name, number, "a parameter"


def sources():
    """-> (path, cell index or None, code) for everything a learner reads."""
    for path in sorted((ROOT / "course").rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts or str(path.relative_to(ROOT)) in NOT_CHECKED:
            continue
        cells = json.loads(path.read_text())["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                yield path, index, "".join(cell["source"])
    for path in (sorted((ROOT / "app").glob("*.py")) if "--app" in sys.argv else []):
        yield path, None, path.read_text()


def vague_names():
    """-> [(path, cell, line, name, how)]."""
    found = []
    for path, cell, code in sources():
        lines = [line for line in code.splitlines() if not line.lstrip().startswith(("%", "!"))]
        try:
            found_here = bound_names(ast.parse("\n".join(lines)))
        except SyntaxError:
            # A challenge cell with blanks (`x = `) is not valid Python. Find
            # its names by pattern, so that no cell escapes the check.
            found_here = bound_by_pattern(lines)
        seen = set()
        for name, line, how in found_here:
            if name in VAGUE and (name, line) not in seen:
                seen.add((name, line))
                found.append((path.relative_to(ROOT), cell, line, name, how))
    return found


def main():
    found = vague_names()
    for path, cell, line, name, how in found:
        where = f"cell {cell}, line {line}" if cell is not None else f"line {line}"
        print(f"{path}  {where}:  `{name}` is {how}")
    files = len({path for path, *_ in found})
    print(f"\n{len(found)} vague names in {files} files.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
