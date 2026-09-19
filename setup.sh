#!/usr/bin/env bash
# One-time setup for a fresh clone. You can run it again safely.
set -euo pipefail
cd "$(dirname "$0")"

# One list for both paths below, so that local and hosted setups stay the
# same.
#
# transformers has a minimum version because Qwen3 needs 4.51+, and a hosted
# notebook can have an older one. The other packages accept the version that
# they find.
# ninja is necessary: torch.utils.cpp_extension runs it to build the CUDA
# stages. Without it, stages 08, 08b, 08c, 18b and 24b cannot compile.
DEPS=(torch "transformers>=4.51" accelerate safetensors numpy ninja
      pytest pytest-timeout pytest-asyncio pyyaml rich
      fastapi uvicorn httpx sse-starlette huggingface_hub
      matplotlib matplotlib-inline jupyterlab nbformat nbclient ipykernel)

# The JAX track is optional: `./setup.sh --jax`. It is about 400MB more, and
# half the ladder (allocator, scheduler, detokenizer, ...) uses no framework
# and needs none of it. cuda13 matches the torch wheel of this repo. On an
# older driver, use jax[cuda12].
JAX_DEPS=("jax[cuda13]")
WANT_JAX=0
for argument in "$@"; do [ "$argument" = "--jax" ] && WANT_JAX=1; done

# Colab already has torch and CUDA installed and working together, which
# is the difficult part. Another 2.5GB of PyTorch in a private venv costs
# several minutes and gives the same result.
#
# So on Colab this script makes the venv ON TOP of the system packages, and
# it installs only what is missing. `./vc` finds .venv/bin in both cases.
# That is the point: the loop is the same on both.
if [ "${1:-}" = "--colab" ] || [ -n "${COLAB_RELEASE_TAG:-}" ]; then
  echo "==> hosted notebook detected: venv over the system packages"
  [ -d .venv ] || python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q "${DEPS[@]}"
  # Colab already has a jax that works. Do not replace its CUDA plugin.
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
    print("   still work. The GPU stages skip.")

# Stages 08, 08b, 08c, 18b and 24b compile a .cu file of your own, so they
# need a real toolkit. You write CUDA here, and CUDA needs nvcc.
nvcc = shutil.which("nvcc") or (CUDA_HOME and f"{CUDA_HOME}/bin/nvcc")
if nvcc and shutil.os.path.exists(nvcc):
    print(f"   nvcc {nvcc}")
else:
    print("   WARNING: no nvcc. Stages 08, 08b, 08c, 18b, 24b and the capstone SKIP.")
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
print(f"   pallas backend: {kind or 'NONE: stage 08 skips'}")
PY
fi

echo
echo "Done.  Start with:   ./vc"
if [ "$WANT_JAX" = 1 ]; then
  echo "The JAX ladder:      ./vc backend jax"
fi
echo "New to ML or GPUs?   course/Part0_FromAProgramToAModel/ first"
echo
echo "Do not stop. Continue. Be better than before."
