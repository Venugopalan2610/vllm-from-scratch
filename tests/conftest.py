import os
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = os.environ.get("VC_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def dev():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"


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
