import os
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = os.environ.get("VC_MODEL", "Qwen/Qwen3-0.6B")

# JAX preallocates 75% of VRAM the moment it touches the GPU, and torch is on
# the same card in the same session. Set before anything imports jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Stage 20 needs a multi-device mesh; one GPU is one device, so the CPU backend
# supplies the ranks. Must be set before jax initialises its backends.
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (
        _flags + " --xla_force_host_platform_device_count=8").strip()


# ---------------------------------------------------------------- backends
#
# Two tracks share this tree. A stage with a JAX twin has both a test_stage.py
# (torch) and a test_jax.py; a stage that is pure logic -- the allocator, the
# scheduler, the detokenizer -- has only test_stage.py and belongs to BOTH
# tracks, because there is nothing framework-shaped in it to port.

def pytest_addoption(parser):
    parser.addoption(
        "--backend", action="store", default=None,
        choices=["torch", "jax", "both"],
        help="which track to check. default: $VC_BACKEND, else torch.",
    )


def backend(config):
    return (config.getoption("--backend")
            or os.environ.get("VC_BACKEND") or "torch")


def pytest_configure(config):
    config.addinivalue_line("markers", "jax: a check on the JAX track")


def pytest_collection_modifyitems(config, items):
    want = backend(config)
    if want == "both":
        return
    keep, dropped = [], []
    for item in items:
        path = Path(str(item.fspath))
        is_jax = path.name == "test_jax.py"
        stage_has_jax_twin = (path.parent / "test_jax.py").exists()
        if want == "jax":
            take = is_jax or not stage_has_jax_twin
        else:
            take = not is_jax
        (keep if take else dropped).append(item)
    items[:] = keep
    if dropped:
        config.hook.pytest_deselected(items=dropped)


@pytest.fixture(scope="session")
def dev():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"


# ---------------------------------------------------------------- jax side

@pytest.fixture(scope="session")
def jdev():
    """A JAX GPU device, or skip. Importing jvllm pins the allocator first."""
    pytest.importorskip("jvllm", reason="JAX not installed -- ./setup.sh --jax")
    import jax

    if jax.default_backend() == "cpu":
        pytest.skip("JAX sees no GPU")
    return jax.devices()[0]


@pytest.fixture(scope="session")
def jmodel(jdev):
    """Qwen3 in JAX, bf16. The counterpart of the `hf` fixture.

    Use for PERFORMANCE tests. This is how you would really serve.
    """
    from jvllm import load_model

    return load_model(MODEL)


@pytest.fixture(scope="session")
def jmodel_exact(jdev):
    """Qwen3 in JAX, fp32. The counterpart of `hf_exact`, for the same reason.

    In bf16, two correct implementations of greedy decode can produce
    different SENTENCES: they reduce in different orders, logits move by ~1e-2,
    and argmax flips wherever the top two candidates are near-tied. Correctness
    is checked in fp32 where the margin swamps the noise.
    """
    import jax.numpy as jnp

    from jvllm import load_model

    return load_model(MODEL, dtype=jnp.float32)


@pytest.fixture(scope="session")
def jpallas(jdev):
    """Skip unless a Pallas GPU backend can actually compile here."""
    from jvllm import compat

    compat.silence_pallas_deprecations()
    kind = compat.pallas_backend()
    if kind is None:
        pytest.skip("no usable Pallas GPU backend on this device")
    return kind


def _load(dev, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
    return model, tok


@pytest.fixture(scope="session")
def hf(dev):
    """bf16 model. Use for PERFORMANCE tests. This is how you'd really serve."""
    return _load(dev, torch.bfloat16)


@pytest.fixture(scope="session")
def hf_exact(dev):
    """fp32 model. Use for CORRECTNESS tests.

    Why two models: in bf16, a cached decode and an uncached recompute do not
    produce bit-identical logits. Different reduction orders give ~1e-2 logit
    differences, and when the top-2 candidates are nearly tied, argmax flips
    and the two paths diverge into completely different sentences.

    That is not a bug in your code -- it is a real property of low-precision
    inference, and it is why production LLM serving is not bit-reproducible
    across batch sizes or cache configurations. But it makes a terrible
    equivalence test, so correctness is checked in fp32 where the margin
    swamps the noise.
    """
    return _load(dev, torch.float32)


@pytest.fixture(scope="session")
def prompts():
    return [
        "The capital of France is",
        "def fibonacci(n):",
        "In 1969, humans first",
    ]


class Timer:
    """Report a measurement into the stage log so you can watch numbers move."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.t) * 1000

    def report(self, tokens=None):
        extra = f"  ({tokens / (self.ms / 1000):.1f} tok/s)" if tokens else ""
        line = f"  \033[36m{self.label}\033[0m: {self.ms:.1f} ms{extra}"
        print(line)
        _record(self.label, self.ms, tokens)


def _record(label, ms, tokens):
    import json

    f = ROOT / ".measurements.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[label] = {"ms": round(ms, 2), "tok_s": round(tokens / (ms / 1000), 1) if tokens else None}
    f.write_text(json.dumps(d, indent=2))


def measurement(label):
    import json

    f = ROOT / ".measurements.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get(label)


@pytest.fixture
def timer():
    return Timer
