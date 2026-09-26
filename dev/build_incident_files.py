"""Build the incident-file notebooks of every Part from its tickets.yaml.

Each Part has a folder `course/Part*/N_incidents/` with a `tickets.yaml`. This
script writes two notebooks from it: the challenge (`..._helper.ipynb`) and
the solution (`solutions/...ipynb`). The /incident skill in
`.claude/skills/incident/` reads the same file, so a ticket is written once.

    python dev/build_incident_files.py            # every Part
    python dev/build_incident_files.py 2 5        # only Parts 2 and 5
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


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


ANSWER = md('''
**Your answer**

- Root cause:
- The number that proves it:
- The fix:
- The guard (the check that catches it next time):
''')


def colleague(part):
    return f'''
**The on-call colleague.** In Claude Code, type `/incident {part}.1` (or any
other ticket number) to work a ticket as a conversation. The colleague has
access to the system. Ask for a log, a measurement or an experiment, and it
answers with what the system shows. When you write your four lines, it tells
you which lines are weak, and it asks a question about each one. It does not
tell you the cause until you ask for the solution.
'''


def probes_md(probes):
    lines = ['**What you could ask the system**', '',
             'The on-call colleague answers these questions. Each answer is a '
             'piece of evidence that the ticket does not show.', '']
    for probe in probes:
        lines.append(f'- *{str(probe["ask"]).strip()}*')
        lines.append('  ' + '\n  '.join(str(probe['answer']).strip().split('\n')))
    return '\n'.join(lines)


def build(path):
    data = yaml.safe_load(path.read_text())
    part, name = data['part'], data['part_name']
    folder = ROOT / data['folder']
    intro = data['intro'].strip('\n') + '\n\n' + colleague(part).strip('\n')

    helper = [header(part, name, 'CodeChallenge HELPER: the incident file'),
              md(intro), md(data['reference'])]
    solution = [header(part, name, 'CodeChallenge: the incident file'),
                md(intro), md(data['reference'])]
    for ticket in data['tickets']:
        helper += [md(ticket['ticket']), code('# your computation'), ANSWER]
        solution += [md(ticket['ticket']), md(ticket['solution'])]
        if ticket.get('code'):
            solution.append(code(ticket['code']))
        if ticket.get('probes'):
            solution.append(md(probes_md(ticket['probes'])))
    helper.append(md(data['outro']))
    solution.append(md(data['patterns']))

    save(folder / f'part{part}_inc_1_CCtheIncidentFile_helper.ipynb', helper)
    save(folder / 'solutions' / f'part{part}_inc_1_CCtheIncidentFile.ipynb', solution)
    return part, len(data['tickets'])


def main(parts):
    for path in sorted(ROOT.glob('course/Part*/*_incidents/tickets.yaml')):
        part = yaml.safe_load(path.read_text())['part']
        if parts and str(part) not in parts:
            continue
        part, count = build(path)
        print(f'Part {part}: {count} tickets')


if __name__ == '__main__':
    main(sys.argv[1:])
