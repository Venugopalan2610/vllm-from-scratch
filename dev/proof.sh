#!/usr/bin/env bash
# Make the proof log in docs/proof/: every check of the torch track against
# the reference solutions, the four lines of each stage, and the live engine
# with every reference part, measured on this card.
#
#   dev/proof.sh
#
# It works in a temporary worktree at HEAD, so app/ in this checkout (your
# own code) does not change. The solutions come from .solutions/ if you have
# it, or from the `solutions` branch on origin. It needs a GPU and nvcc. It
# took about 15 minutes on an RTX 4080 Laptop GPU.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT="$ROOT/docs/proof"
WORK=$(mktemp -d)/proof
mkdir -p "$OUT"

git worktree add -q --detach "$WORK" HEAD
trap 'git -C "$ROOT" worktree remove --force "$WORK"' EXIT
ln -s "$ROOT/.venv" "$WORK/.venv"
[ -d "$ROOT/.cudacache" ] && ln -s "$ROOT/.cudacache" "$WORK/.cudacache"

if [ -d .solutions ]; then
  cp .solutions/*.py "$WORK/app/"
  cp .solutions/*.cu "$WORK/app/cuda/" 2>/dev/null
else
  git fetch -q origin solutions:refs/remotes/origin/solutions
  for file in $(git ls-tree --name-only origin/solutions:.solutions); do
    case "$file" in
      *.cu) git show "origin/solutions:.solutions/$file" > "$WORK/app/cuda/$file" ;;
      *)    git show "origin/solutions:.solutions/$file" > "$WORK/app/$file" ;;
    esac
  done
fi

PYTHON="$ROOT/.venv/bin/python"
cd "$WORK"
{
  echo "date:    $(date -Iseconds)"
  echo "commit:  $(git rev-parse --short HEAD)"
  echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "nvcc:    $(nvcc --version 2>/dev/null | tail -1)"
  "$PYTHON" -c "import torch, transformers; print('torch:  ', torch.__version__, '(CUDA', torch.version.cuda + ')'); print('transformers:', transformers.__version__)"
} > "$OUT/machine.txt"

echo "==> every check of the torch track (as dev/verify.sh)"
"$PYTHON" -m pytest tests/ -q --timeout=900 --backend torch -p no:cacheprovider > "$OUT/checks.log" 2>&1
tail -1 "$OUT/checks.log"

echo "==> the four lines of each stage"
"$PYTHON" -m pytest tests/ -q -s -k four_lines --backend torch -p no:cacheprovider 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | tr '\r' '\n' | grep -v 'Loading weights\|Warning' > "$OUT/four_lines.log"
tail -1 "$OUT/four_lines.log"

echo "==> ./vc run, with every reference part from 06 to 19"
"$PYTHON" runner/live_run.py --stages 06,07,08,08b,08c,09,10,11,12,13,14,15,16,17,18,18b,19 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | tr '\r' '\n' | grep -v 'Loading weights\|Warning' > "$OUT/vc_run.log"

echo "==> ./vc run, with only the given parts"
"$PYTHON" runner/live_run.py --stages none 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | tr '\r' '\n' | grep -v 'Loading weights\|Warning' > "$OUT/vc_run_given.log"
echo "the log is in docs/proof/"
