import os
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.helpers import MODEL, record_measurement  # noqa: E402  (needs ROOT)

# JAX takes 75% of VRAM when it first uses the GPU, and torch uses the same
# card in the same session. Set this before an import of jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Stage 20 needs a mesh of several devices, and one GPU is one device. So the
# CPU backend gives the ranks. Set this before jax starts its backends.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_force_host_platform_device_count=8").strip()


# ---------------------------------------------------------------- backends
#
# Two tracks share this tree. The file name tells the track of a check:
#
#   test_jax.py    the JAX twin of a stage that has one.
#
#   test_cuda.py   a stage on the torch track ONLY. It is a CUDA kernel, and
#                  JAX has no true equivalent of a warp shuffle.
#                  stages.yaml marks these stages `tracks: [torch]`.
#
#   test_stage.py  every other stage. A stage with no test_jax.py next to it
#                  uses no framework, for example the allocator, the
#                  scheduler or the detokenizer. It belongs to BOTH tracks,
#                  because it has nothing to port.
#
# tests/test_harness.py compares this rule with stages.yaml. The rule is in
# two places, and nothing else finds a difference between them.

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
        is_cuda = path.name == "test_cuda.py"
        stage_has_jax_twin = (path.parent / "test_jax.py").exists()
        if want == "jax":
            take = is_jax or not (stage_has_jax_twin or is_cuda)
        else:
            take = not is_jax
        (keep if take else dropped).append(item)
    items[:] = keep
    if dropped:
        config.hook.pytest_deselected(items=dropped)


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"


# --------------------------------------------------------------- cuda side

@pytest.fixture(scope="session")
def nvcc(device):
    """Skip unless a CUDA toolkit can compile here.

    The twin of `jpallas`. The CUDA stages compile a .cu file with
    torch.utils.cpp_extension, which needs nvcc and ninja. Without them the
    checks skip and do not fail, as the Pallas checks do.
    """
    from cudalib import probe

    if not probe.have_nvcc():
        pytest.skip("no nvcc -- these stages need a CUDA toolkit")
    return True


@pytest.fixture(scope="session")
def peak_gbs(device):
    """The streaming bandwidth of this card in GB/s, measured one time for
    the session. A check compares with this number, never with a datasheet."""
    import cudalib

    return cudalib.peak_bandwidth(fresh=True)


# ---------------------------------------------------------------- jax side

@pytest.fixture(scope="session")
def jax_device():
    """A JAX GPU device, or a skip. An import of jvllm pins the allocator."""
    pytest.importorskip("jvllm", reason="JAX not installed -- ./setup.sh --jax")
    import jax

    if jax.default_backend() == "cpu":
        pytest.skip("JAX sees no GPU")
    return jax.devices()[0]


@pytest.fixture(scope="session")
def jmodel(jax_device):
    """Qwen3 in JAX, bf16. The JAX twin of the `hf` fixture.

    Use it for PERFORMANCE checks. A real server uses this precision.
    """
    from jvllm import load_model

    return load_model(MODEL)


@pytest.fixture(scope="session")
def jmodel_exact(jax_device):
    """Qwen3 in JAX, fp32. The JAX twin of `hf_exact`, for the same reason.

    In bf16, two correct greedy decodes can give different SENTENCES. They
    add in different orders, the logits move by about 1e-2, and argmax
    changes where the top two tokens are almost equal. So the correctness
    checks use fp32, where the margin is much larger than the noise.
    """
    import jax.numpy as jnp

    from jvllm import load_model

    return load_model(MODEL, dtype=jnp.float32)


@pytest.fixture(scope="session")
def jpallas(jax_device):
    """Skip unless a Pallas GPU backend can compile here."""
    from jvllm import compat

    compat.silence_pallas_deprecations()
    kind = compat.pallas_backend()
    if kind is None:
        pytest.skip("no usable Pallas GPU backend on this device")
    return kind


def _load(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    return model.to(device).eval(), tokenizer


@pytest.fixture(scope="session")
def hf(device):
    """The bf16 model. Use it for PERFORMANCE checks. A real server uses this
    precision."""
    return _load(device, torch.bfloat16)


@pytest.fixture(scope="session")
def hf_exact(device):
    """The fp32 model. Use it for CORRECTNESS checks.

    The reason for two models: in bf16, a cached decode and a recompute with
    no cache do not give the same logits, bit for bit. A different order of
    addition moves a logit by about 1e-2. When the top two tokens are almost
    equal, argmax changes, and the two paths give different sentences.

    That is not a bug in your code. It is a property of low-precision
    inference. It is also why a production server does not give the same
    bits across batch sizes or cache settings. But it makes a bad equality
    check. So the correctness checks use fp32, where the margin is much
    larger than the noise.
    """
    return _load(device, torch.float32)


@pytest.fixture(scope="session")
def tmodel(hf):
    """The capstone model (tvllm) in bf16, on the weights of `hf`.

    It shares the embedding, the norms and the lm_head with `hf`, and it adds
    fused copies of the QKV and gate/up matrices. So it costs much less memory
    than a second load. Use it for PERFORMANCE checks."""
    from tvllm import Model

    model, tokenizer = hf
    return Model(model, tokenizer, MODEL)


@pytest.fixture(scope="session")
def tmodel_exact(hf_exact):
    """The capstone model in fp32. Use it for CORRECTNESS checks, for the
    reason that `hf_exact` gives."""
    from tvllm import Model

    model, tokenizer = hf_exact
    return Model(model, tokenizer, MODEL)


@pytest.fixture(scope="session")
def prompts():
    return [
        "The capital of France is",
        "def fibonacci(n):",
        "In 1969, humans first",
    ]


class Timer:
    """Put a measurement into the stage log, so that you can see the numbers
    change."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.start) * 1000

    def report(self, tokens=None):
        rate = f"  ({tokens / (self.ms / 1000):.1f} tok/s)" if tokens else ""
        print(f"  \033[36m{self.label}\033[0m: {self.ms:.1f} ms{rate}")
        record_measurement(self.label, self.ms, tokens)


@pytest.fixture
def timer():
    return Timer


def pytest_runtest_teardown(item):
    """The capstone checks make KV pools and graph pools of several hundred
    MB. Give the memory back after each check, because the next check shares
    the card with four models."""
    if "stage_2" in str(item.fspath) and "stage_20" not in str(item.fspath):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
