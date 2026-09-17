#!/usr/bin/env bash
# One-time setup for a fresh clone. Idempotent -- safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

# One list, both paths below, so local and hosted cannot drift apart.
#
# transformers carries a floor because Qwen3 needs 4.51+, and a hosted
# notebook may well ship something older. Everything else is happy with
# whatever version it finds.
# ninja is not optional: torch.utils.cpp_extension shells out to it to build
# the CUDA stages, and without it stages 08, 08b and 08c cannot compile.
DEPS=(torch "transformers>=4.51" accelerate safetensors numpy ninja
      pytest pytest-timeout pytest-asyncio pyyaml rich
      fastapi uvicorn httpx sse-starlette huggingface_hub)

# The JAX track is opt-in: `./setup.sh --jax`. It is another ~400MB, and half
# the ladder (allocator, scheduler, detokenizer, ...) is framework-free and
# needs none of it. cuda13 matches the torch wheel this repo pins; on an older
# driver swap it for jax[cuda12].
JAX_DEPS=("jax[cuda13]")
WANT_JAX=0
for a in "$@"; do [ "$a" = "--jax" ] && WANT_JAX=1; done

# Colab already has torch, CUDA and Triton installed and working
# together, which is the fiddly part. Pulling another 2.5GB of PyTorch
# into a private venv would spend several minutes arriving at the same
# place, so there we build the venv ON TOP of the system packages and
# install only what is genuinely missing. `./vc` finds .venv/bin either
# way, which is the point: the loop is identical on both.
if [ "${1:-}" = "--colab" ] || [ -n "${COLAB_RELEASE_TAG:-}" ]; then
  echo "==> hosted notebook detected: venv over the system packages"
  [ -d .venv ] || python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q "${DEPS[@]}"
  # Colab already has a working jax; do not fight it for the CUDA plugin.
  [ "$WANT_JAX" = 1 ] && .venv/bin/pip install -q jax
else
  command -v uv >/dev/null || {
    echo "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  }

  echo "==> python 3.12 venv"
  [ -d .venv ] || uv venv --python 3.12

  echo "==> dependencies (torch is ~2.5GB, this takes a few minutes)"
  VIRTUAL_ENV=.venv uv pip install -q "${DEPS[@]}"

  if [ "$WANT_JAX" = 1 ]; then
    echo "==> jax (the second track, ~400MB)"
    VIRTUAL_ENV=.venv uv pip install -q "${JAX_DEPS[@]}"
  fi
fi

echo "==> model (Qwen3-0.6B, ~1.2GB)"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("ok")
PY

echo "==> checking CUDA"
.venv/bin/python - <<'PY'
import shutil
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(f"   torch {torch.__version__} | cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"   {torch.cuda.get_device_name(0)}  sm_{major}{minor}")
else:
    print("   WARNING: no CUDA. Stages 6, 9, 10, 11, 13, 14, 16, 17, 19, 20")
    print("   still work; the GPU stages will skip.")

# Stages 08, 08b and 08c compile a .cu of your own, so they need a real
# toolkit. Triton used to ship its own compiler and this repo needed none;
# writing CUDA means nvcc.
nvcc = shutil.which("nvcc") or (CUDA_HOME and f"{CUDA_HOME}/bin/nvcc")
if nvcc and shutil.os.path.exists(nvcc):
    print(f"   nvcc {nvcc}")
else:
    print("   WARNING: no nvcc. Stages 08, 08b and 08c will SKIP.")
    print("   Install the CUDA toolkit for your distribution, or use the")
    print("   Colab notebook, which already has one at /usr/local/cuda.")
PY

if [ "$WANT_JAX" = 1 ]; then
  echo "==> checking JAX"
  .venv/bin/python - <<'PY'
import jvllm                     # pins the allocator before jax touches the GPU
import jax
from jvllm import compat
print(f"   jax {jax.__version__} | {jax.default_backend()} | {jax.devices()}")
kind = compat.pallas_backend()
print(f"   pallas backend: {kind or 'NONE -- stage 08 will skip'}")
PY
fi

echo
echo "Done.  Start with:   ./vc"
if [ "$WANT_JAX" = 1 ]; then
  echo "The JAX ladder:      ./vc backend jax"
fi
