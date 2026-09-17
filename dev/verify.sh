#!/usr/bin/env bash
# Prove every check is passable: install the reference solutions over app/,
# run the suite, restore the stubs.
#
# Solutions live on the `solutions` branch, not on master. This pulls them
# from there (or from a local .solutions/ if you have one checked out).
#
#   dev/verify.sh              all stages, torch track
#   dev/verify.sh 7 8          just those
#   dev/verify.sh --jax        the JAX track (shared stages included)
#   dev/verify.sh --both       both tracks, in ONE pytest run
#   dev/verify.sh --jax 7 8    combine freely
#
# --both loads four models onto one GPU (torch bf16 + fp32, jax bf16 + fp32).
# On a 12GB card that is tight; running the tracks as two invocations is safer
# and is what CI does.
set -u
cd "$(dirname "$0")/.."

BACKEND=torch
args=()
for a in "$@"; do
  case "$a" in
    --jax)   BACKEND=jax ;;
    --torch) BACKEND=torch ;;
    --both)  BACKEND=both ;;
    *)       args+=("$a") ;;
  esac
done
set -- ${args[@]+"${args[@]}"}

if grep -lq "Reference solution" app/*.py app/cuda/*.cu 2>/dev/null; then
  echo "ERROR: app/ already contains reference solutions, not stubs:"
  grep -l "Reference solution" app/*.py app/cuda/*.cu
  echo "Restore them with: git checkout -- app/"
  exit 2
fi

SOL=$(mktemp -d)
if [ -d .solutions ]; then
  cp .solutions/*.py .solutions/*.cu "$SOL"/ 2>/dev/null
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

# The CUDA stages are two files each: a .cu kernel and a .py wrapper. Back up
# and restore both, or a run leaves the reference kernels sitting in app/.
BAK=$(mktemp -d)
mkdir -p "$BAK/cuda"
cp app/*.py "$BAK"/ 2>/dev/null
cp app/cuda/*.cu "$BAK/cuda"/ 2>/dev/null
restore() {
  rm -f app/*.py app/cuda/*.cu
  cp "$BAK"/*.py app/ 2>/dev/null
  cp "$BAK"/cuda/*.cu app/cuda/ 2>/dev/null
  rm -rf "$BAK" "$SOL"
}
trap restore EXIT INT TERM

cp "$SOL"/*.py app/
cp "$SOL"/*.cu app/cuda/ 2>/dev/null
echo "==> backend: $BACKEND"
if [ $# -eq 0 ]; then
  .venv/bin/python -m pytest tests/ -q --timeout=900 --backend "$BACKEND"
else
  dirs=""
  for s in "$@"; do
    # accepts 7, 07, 8b and 08b. A sub-stage carries a letter, so pad the
    # digits and keep the letter where it was.
    n=$(printf '%s' "$s" | tr -cd '[:digit:]')
    n=$((10#$n))                       # "08" is not octal here
    a=$(printf '%s' "$s" | tr -cd '[:alpha:]')
    pat="tests/stage_$(printf '%02d' "$n")${a}_*"
    dirs="$dirs $(ls -d $pat 2>/dev/null)"
  done
  .venv/bin/python -m pytest $dirs -q --timeout=900 --backend "$BACKEND"
fi
