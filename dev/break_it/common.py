"""Shared parts of the notebooks "break it on purpose", Parts 0 and 2 to 8.

Each Part has a module next to this one (part0.py, part2.py, ...) that defines
the drills and their solutions. `dev/build_break_it.py` builds the notebooks.
Part 1 has its own builder, dev/build_incidents.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')}


def code(text=''):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.strip('\n')}


def header(part, part_name, lecture):
    return md(f'''
|<h2>Course:</h2>|<h1><a href="https://derivingsystems.com/course.html" target="_blank">Build your own vLLM: inference engines from the memory system up</a></h1>|
|-|:-:|
|<h2>Part {part}:</h2>|<h1>{part_name}<h1>|
|<h2>Section:</h2>|<h1>Incidents<h1>|
|<h2>Lecture:</h2>|<h1><b>{lecture}<b></h1>|

<br>

<h5><b>Course repo:</b> <a href="https://github.com/Venugopalan2610/vllm-from-scratch" target="_blank">github.com/Venugopalan2610/vllm-from-scratch</a></h5>
<h5><b>The derivations:</b> <a href="https://derivingsystems.com" target="_blank">derivingsystems.com</a></h5>
<i>The notebooks build the intuition. The ladder in app/ makes you build the thing.</i>
''')


def setup(folder, extra_imports=''):
    return code(f'''
# find the repo root. The directory you start from does not matter.
import sys
from pathlib import Path
ROOT = next(folder for folder in [Path.cwd(), *Path.cwd().parents]
            if (folder/'cudalib').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'{folder}'))

import math, time
import torch
import lab
{extra_imports}
''')


def save(path, cells):
    nb = {'cells': [dict(cell) for cell in cells],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    for cell in nb['cells']:
        lines = cell['source'].split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')


PREDICT = md('''
**Your prediction** (write it before you run the cell)

- Crash?
- When?
- What?
- Which guard?

**What happened, and where you were wrong:**
''')


def routine(what_the_reference_is):
    return f'''
The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** When does the output first differ from the reference: which
  token, which request, which size?
- **What?** What does the wrong result look like?
- **Which guard?** Which check would catch it? If none of the stages has
  one: what check would?

{what_the_reference_is}
'''


def mystery_answer(names):
    lines = ['**Your diagnosis**', '']
    lines += [f'- `{name}`: the fault, and the experiment that proves it:' for name in names]
    return md('\n'.join(lines))


def fingerprints(rows):
    lines = ['# Your fingerprint table', '',
             'Fill in this table from what you saw, not from what you predicted.', '',
             '| Fault | Crash? | When it shows | What it looks like | The guard |',
             '|---|---|---|---|---|']
    lines += [f'| {row} | | | | |' for row in rows]
    return md('\n'.join(lines))


def build(spec, with_solutions):
    title = ('CodeChallenge: break it on purpose' if with_solutions
             else 'CodeChallenge HELPER: break it on purpose')
    cells = [header(spec.PART, spec.NAME, title), setup(spec.FOLDER, spec.IMPORTS),
             md(spec.INTRO), code(spec.LOAD)]
    for key, text, drill in spec.DRILLS:
        if with_solutions:
            cells += [md(text), code(drill), md(spec.SOLUTIONS[key])]
        else:
            cells += [md(text), PREDICT, code(drill)]
    cells += [md(spec.MYSTERY), code(spec.MYSTERY_CODE)]
    if with_solutions:
        cells += [md(spec.MYSTERY_SOL), md(spec.FINGERPRINTS_SOL)]
    else:
        cells += [code(), code(), mystery_answer(spec.MYSTERY_NAMES),
                  fingerprints(spec.FINGERPRINT_ROWS)]
    return cells


def write(spec):
    folder = ROOT / spec.FOLDER
    save(folder / f'part{spec.PART}_inc_2_CCbreakItOnPurpose_helper.ipynb', build(spec, False))
    if getattr(spec, 'SOLUTIONS', None):
        save(folder / 'solutions' / f'part{spec.PART}_inc_2_CCbreakItOnPurpose.ipynb', build(spec, True))
