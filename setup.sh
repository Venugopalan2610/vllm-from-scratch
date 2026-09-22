#!/usr/bin/env bash
# One-time setup for a fresh clone. You can run it again safely.
set -euo pipefail
cd "$(dirname "$0")"

# One list for both paths below, so that local and hosted setups stay the
# same.
#
# THE VERSIONS ARE PINNED ON PURPOSE. The checks of stages 01, 02, 17 and 25
# compare your tokens with `model.generate` of transformers, and that API and
# its cache classes change between releases. An unpinned course breaks for a
# learner who starts it six months later, and the failure looks like their
# code. These are the versions that the check counts in README.md come from.
#
#   ./setup.sh --latest     take the newest of everything instead
#
# ninja is necessary: torch.utils.cpp_extension runs it to build the CUDA
# stages. Without it, stages 08, 08b, 08c, 18b and 24b cannot compile.
PINNED=(torch==2.13.0 transformers==5.14.1 accelerate==1.14.0
        safetensors==0.8.0 numpy==2.5.1 ninja==1.13.2 pytest==9.1.1
        matplotlib==3.11.2 nbformat==5.11.1 nbclient==0.11.0 rich==15.0.0)
UNPINNED=(pytest-timeout==2.4.0 pytest-asyncio==1.4.0 pyyaml==6.0.3
          fastapi==0.115.12 uvicorn==0.34.3 httpx==0.28.1
          sse-starlette==2.3.6 huggingface_hub==1.26.0
          matplotlib-inline==0.1.7 jupyterlab==4.4.3 ipykernel==6.30.0)
DEPS=("${PINNED[@]}" "${UNPINNED[@]}")
for argument in "$@"; do
  if [ "$argument" = "--latest" ]; then
    DEPS=(torch "transformers>=4.51" accelerate safetensors numpy ninja
          "${UNPINNED[@]}" pytest matplotlib nbformat nbclient rich)
    echo "==> --latest: the versions are not pinned. A check that fails may be"
    echo "    a change in a library, not your code."
  fi
done

# The JAX track is optional: `./setup.sh --jax`. It is about 400MB more, and
# half the ladder (allocator, scheduler, detokenizer, ...) uses no framework
# and needs none of it. cuda13 matches the torch wheel of this repo. On an
# older driver, use jax[cuda12].
JAX_DEPS=("jax[cuda13]==0.11.0")
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

# The notebooks use Qwen3-1.7B. The checks use Qwen3-0.6B: many of them hold
# the model in bf16 and in fp32 at the same time, and 1.7B does not fit twice
# on a 12GB card.
echo "==> models (Qwen3-1.7B ~3.4GB for the notebooks, Qwen3-0.6B ~1.2GB for the checks)"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for name in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"):
    snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt"])
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
echo "==> what this machine can run"
.venv/bin/python runner/doctor.py || true

echo
echo "Done.  Start with:   ./vc"
if [ "$WANT_JAX" = 1 ]; then
  echo "The JAX ladder:      ./vc backend jax"
fi
echo "New to ML or GPUs?   course/Part0_FromAProgramToAModel/ first"
echo
echo "Do not stop. Continue. Be better than before."
