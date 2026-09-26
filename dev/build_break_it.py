"""Build the notebooks "break it on purpose" of Parts 0 and 2 to 8.

    python dev/build_break_it.py          # every Part
    python dev/build_break_it.py 3 5      # only Parts 3 and 5

Part 1 has its own builder: dev/build_incidents.py.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from break_it import common  # noqa: E402

PARTS = ['0', '2', '3', '4', '5', '6', '7', '8']

for part in sys.argv[1:] or PARTS:
    try:
        spec = importlib.import_module(f'break_it.part{part}')
    except ModuleNotFoundError:
        continue
    common.write(spec)
    print(f'Part {part}: {len(spec.DRILLS)} drills'
          + ('' if getattr(spec, 'SOLUTIONS', None) else ', no solutions yet'))
