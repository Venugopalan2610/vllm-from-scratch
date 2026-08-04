#!/usr/bin/env bash
# Install the reference solutions over app/, run the given stages' tests, then
# restore the stubs. Used to prove every test suite is actually passable.
#   dev/verify.sh            all stages that have tests
#   dev/verify.sh 3 4 5      just those
set -u
cd "$(dirname "$0")/.."
cp .solutions/*.py app/ 2>/dev/null
if [ $# -eq 0 ]; then
  .venv/bin/python -m pytest tests/ -q --timeout=900
else
  dirs=""
  for s in "$@"; do
    d=$(ls -d tests/stage_$(printf '%02d' "$s")_* 2>/dev/null)
    dirs="$dirs $d"
  done
  .venv/bin/python -m pytest $dirs -q --timeout=900
fi
rc=$?
git checkout -- app/ 2>/dev/null
exit $rc
