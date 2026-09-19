"""jvllm - the JAX side of the course.

The repo gives you this package, like tests/helpers.py. You do not edit it. It
exists so that the JAX track starts where the torch track starts: with a model
that works, a tokenizer, and nothing else.

The torch track gets that from HuggingFace `transformers`, which ships a Qwen3
implementation. There is no Flax Qwen3, so `jvllm.model` is one: ~200 lines of
jnp that loads the same safetensors file and produces the same logits.

    from jvllm import load_model

    model = load_model()                  # or load_model("Qwen/Qwen3-0.6B")
    ids = model.encode("The capital of France is")
    logits, cache = model.forward(ids[None], positions, cache, cache_len)

Read `jvllm/model.py` before stage 01. The KV cache API in that file is the
whole reason that the JAX stages separate from the torch stages. XLA needs
static shapes. So you PREALLOCATE the cache and write into it. It never grows.
"""

import os

# JAX takes 75% of VRAM on first use by default. The torch track runs on the
# same card, and often in the same pytest session. So that default is a certain
# OOM.
#
# Set this before any import of jax. The backend reads it lazily, at the first
# device use, so an early import of jvllm is sufficient.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Do NOT also set XLA_PYTHON_CLIENT_ALLOCATOR=platform. That turns off the
# caching allocator, so every intermediate buffer becomes a real cudaMalloc.
#
# A decode step that allocates a new KV cache each time then spends more wall
# clock in the allocator than on the GPU. This repo measured 10.4 ms/token in
# the place of 5.0. `preallocate=false` alone shares the card with torch.

# Stage 20 shards across DEVICES, and one GPU is one device. XLA gives you as
# many CPU devices as you ask for.
#
# That is how the tensor-parallel stage gets a real mesh, real shardings and a
# real psum, and not a Python loop that imitates the ranks.
#
# Append to the flags. Never overwrite them. A learner can have their own
# XLA_FLAGS.
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (
        _flags + " --xla_force_host_platform_device_count=8").strip()

from jvllm.compat import enable_pallas_triton, pallas_backend  # noqa: E402
from jvllm.model import Qwen3, Qwen3Config, load_model  # noqa: E402

__all__ = [
    "Qwen3",
    "Qwen3Config",
    "load_model",
    "enable_pallas_triton",
    "pallas_backend",
]
