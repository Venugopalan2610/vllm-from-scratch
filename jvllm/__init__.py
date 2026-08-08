"""jvllm - the JAX side of the course.

This package is PROVIDED, like tests/helpers.py. You do not edit it. It exists
so the JAX track starts where the torch track starts: with a working model and
a tokenizer, and nothing else.

The torch track gets that from HuggingFace `transformers`, which ships a Qwen3
implementation. There is no Flax Qwen3, so `jvllm.model` is one: ~200 lines of
jnp that loads the same safetensors file and produces the same logits.

    from jvllm import load_model

    model = load_model()                  # or load_model("Qwen/Qwen3-0.6B")
    ids = model.encode("The capital of France is")
    logits, cache = model.forward(ids[None], positions, cache, cache_len)

Read `jvllm/model.py` before stage 01. The KV cache API you are handed there is
the whole reason the JAX stages diverge from the torch ones: XLA needs static
shapes, so the cache is PREALLOCATED and written into, never grown.
"""

import os

# JAX grabs 75% of VRAM on first use by default. The torch track is running on
# the same card -- often in the same pytest session -- so that is a guaranteed
# OOM. Set before anything imports jax; the backend reads it lazily, at first
# device use, so importing jvllm early is enough.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Do NOT also set XLA_PYTHON_CLIENT_ALLOCATOR=platform to be extra polite. That
# turns off the caching allocator, so every intermediate buffer becomes a real
# cudaMalloc -- and a decode step that allocates a fresh KV cache each time then
# spends more wall clock in the allocator than in the GPU. Measured on this
# repo: 10.4 ms/token instead of 5.0. `preallocate=false` alone is enough to
# share the card with torch.

# Stage 20 shards across DEVICES, and a single GPU is one device. XLA can hand
# out as many CPU devices as you ask for, which is how the tensor-parallel
# stage gets a real mesh, real shardings and a real psum instead of a Python
# loop pretending to be ranks. Appended, never clobbered -- a learner may have
# their own XLA_FLAGS.
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
