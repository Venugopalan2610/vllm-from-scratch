#!/usr/bin/env bash
# Prove that every check can pass: put the reference solutions in app/, run
# the suite, and put the stubs back.
#
# The solutions are on the `solutions` branch, not on master. This script
# gets them from there, or from a local .solutions/ if you have one.
#
#   dev/verify.sh              all stages, torch track.
#   dev/verify.sh 7 8          only those stages.
#   dev/verify.sh --jax        the JAX track, with the shared stages.
#   dev/verify.sh --both       both tracks, in ONE pytest run.
#   dev/verify.sh --jax 7 8    combine the arguments freely.
#
# --both loads four models onto one GPU: torch bf16 and fp32, jax bf16 and
# fp32. On a 12GB card that is tight. Run the two tracks as two commands.
# That is safer, and CI does it that way.
set -u
cd "$(dirname "$0")/.."

BACKEND=torch
stage_args=()
for argument in "$@"; do
  case "$argument" in
    --jax)   BACKEND=jax ;;
    --torch) BACKEND=torch ;;
    --both)  BACKEND=both ;;
    *)       stage_args+=("$argument") ;;
  esac
done
set -- ${stage_args[@]+"${stage_args[@]}"}

if grep -lq "Reference solution" app/*.py app/cuda/*.cu 2>/dev/null; then
  echo "ERROR: app/ already contains reference solutions, not stubs:"
  grep -l "Reference solution" app/*.py app/cuda/*.cu
  echo "Restore them with: git checkout -- app/"
  exit 2
fi

SOLUTIONS=$(mktemp -d)
if [ -d .solutions ]; then
  cp .solutions/*.py .solutions/*.cu "$SOLUTIONS"/ 2>/dev/null
else
  branch=solutions
  git rev-parse --verify --quiet "$branch" >/dev/null || branch=origin/solutions
  git rev-parse --verify --quiet "$branch" >/dev/null || {
    echo "Fetching the solutions branch..."
    git fetch origin solutions:refs/remotes/origin/solutions -q || {
      echo "ERROR: could not fetch the solutions branch."; exit 2; }
    branch=origin/solutions
  }
  for file in $(git ls-tree --name-only "$branch":.solutions); do
    git show "$branch:.solutions/$file" > "$SOLUTIONS/$file"
  done
fi

# A CUDA stage has two files: a .cu kernel and a .py wrapper. Back up and
# restore both. If not, a run leaves the reference kernels in app/.
STUBS=$(mktemp -d)
mkdir -p "$STUBS/cuda"
cp app/*.py "$STUBS"/ 2>/dev/null
cp app/cuda/*.cu "$STUBS/cuda"/ 2>/dev/null
restore() {
  rm -f app/*.py app/cuda/*.cu
  cp "$STUBS"/*.py app/ 2>/dev/null
  cp "$STUBS"/cuda/*.cu app/cuda/ 2>/dev/null
  rm -rf "$STUBS" "$SOLUTIONS"
}
trap restore EXIT INT TERM

cp "$SOLUTIONS"/*.py app/
cp "$SOLUTIONS"/*.cu app/cuda/ 2>/dev/null
echo "==> backend: $BACKEND"
if [ $# -eq 0 ]; then
  .venv/bin/python -m pytest tests/ -q --timeout=900 --backend "$BACKEND"
else
  test_dirs=""
  for stage in "$@"; do
    # This accepts 7, 07, 8b and 08b. A sub-stage has a letter, so pad the
    # digits and keep the letter where it was.
    digits=$(printf '%s' "$stage" | tr -cd '[:digit:]')
    number=$((10#$digits))              # "08" is not octal here
    letter=$(printf '%s' "$stage" | tr -cd '[:alpha:]')
    pattern="tests/stage_$(printf '%02d' "$number")${letter}_*"
    test_dirs="$test_dirs $(ls -d $pattern 2>/dev/null)"
  done
  .venv/bin/python -m pytest $test_dirs -q --timeout=900 --backend "$BACKEND"
fi
