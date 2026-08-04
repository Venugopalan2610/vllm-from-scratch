#!/usr/bin/env bash
# Install reference solutions over app/, run tests, then restore the stubs.
#   dev/verify.sh          all stages with tests
#   dev/verify.sh 3 4 5    just those
set -u
cd "$(dirname "$0")/.."

# Guard: never back up an app/ that already holds solutions, or the restore
# would write solutions back over the stubs (this bit me once already).
if grep -lq "Reference solution" app/*.py 2>/dev/null; then
  echo "ERROR: app/ contains reference solutions, not stubs:"
  grep -l "Reference solution" app/*.py
  echo "Restore them with: git checkout -- app/"
  exit 2
fi

BAK=$(mktemp -d)
cp app/*.py "$BAK"/ 2>/dev/null
restore() { cp "$BAK"/*.py app/ 2>/dev/null; rm -rf "$BAK"; }
trap restore EXIT INT TERM

cp .solutions/*.py app/ 2>/dev/null
if [ $# -eq 0 ]; then
  .venv/bin/python -m pytest tests/ -q --timeout=900
else
  dirs=""
  for s in "$@"; do
    dirs="$dirs $(ls -d tests/stage_$(printf '%02d' "$s")_* 2>/dev/null)"
  done
  .venv/bin/python -m pytest $dirs -q --timeout=900
fi
