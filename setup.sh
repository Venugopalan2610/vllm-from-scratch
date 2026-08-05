#!/usr/bin/env bash
# One-time setup for a fresh clone. Idempotent -- safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || {
  echo "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}

echo "==> python 3.12 venv"
[ -d .venv ] || uv venv --python 3.12

echo "==> dependencies (torch is ~2.5GB, this takes a few minutes)"
VIRTUAL_ENV=.venv uv pip install -q \
  torch transformers accelerate safetensors numpy \
  pytest pytest-timeout pytest-asyncio pyyaml rich \
  fastapi uvicorn httpx sse-starlette huggingface_hub

echo "==> model (Qwen3-0.6B, ~1.2GB)"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("ok")
PY

echo "==> checking CUDA"
.venv/bin/python - <<'PY'
import torch
print(f"   torch {torch.__version__} | cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   {torch.cuda.get_device_name(0)}")
else:
    print("   WARNING: no CUDA. Stages 6, 9, 10, 11, 13, 14, 16, 17, 19, 20")
    print("   still work; the GPU stages will skip.")
PY

echo
echo "Done.  Start with:   ./vc"
