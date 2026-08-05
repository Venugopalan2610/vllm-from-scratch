#!/usr/bin/env bash
# Prove every check is passable: install the reference solutions over app/,
# run the suite, restore the stubs.
#
# Solutions live on the `solutions` branch, not on master. This pulls them
# from there (or from a local .solutions/ if you have one checked out).
#
#   dev/verify.sh          all stages
#   dev/verify.sh 7 8      just those
set -u
cd "$(dirname "$0")/.."

if grep -lq "Reference solution" app/*.py 2>/dev/null; then
  echo "ERROR: app/ already contains reference solutions, not stubs:"
  grep -l "Reference solution" app/*.py
  echo "Restore them with: git checkout -- app/"
  exit 2
fi

SOL=$(mktemp -d)
if [ -d .solutions ]; then
  cp .solutions/*.py "$SOL"/
else
  REF=solutions
  git rev-parse --verify --quiet "$REF" >/dev/null || REF=origin/solutions
  git rev-parse --verify --quiet "$REF" >/dev/null || {
    echo "Fetching the solutions branch..."
    git fetch origin solutions:refs/remotes/origin/solutions -q || {
      echo "ERROR: could not fetch the solutions branch."; exit 2; }
    REF=origin/solutions
  }
  for f in $(git ls-tree --name-only "$REF":.solutions); do
    git show "$REF:.solutions/$f" > "$SOL/$f"
  done
fi

BAK=$(mktemp -d)
cp app/*.py "$BAK"/ 2>/dev/null
restore() { rm -f app/*.py; cp "$BAK"/*.py app/ 2>/dev/null; rm -rf "$BAK" "$SOL"; }
trap restore EXIT INT TERM

cp "$SOL"/*.py app/
if [ $# -eq 0 ]; then
  .venv/bin/python -m pytest tests/ -q --timeout=900
else
  dirs=""
  for s in "$@"; do
    dirs="$dirs $(ls -d tests/stage_$(printf '%02d' "$s")_* 2>/dev/null)"
  done
  .venv/bin/python -m pytest $dirs -q --timeout=900
fi
