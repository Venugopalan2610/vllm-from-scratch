"""Provided plumbing for the CUDA stages, like jvllm/ is for the JAX track.

    from cudalib import build
    mod = build("s08_paged_attn", "app/cuda/s08_paged_attn.cu")

You do not edit anything in here. The kernels are yours; the compiler
invocation, the arch flags and the bandwidth arithmetic are not the lesson.
"""

from cudalib.build import CACHE, ROOT, build, module_path
from cudalib.probe import (
    achieved_bandwidth,
    bench_ms,
    have_nvcc,
    kernel_stats,
    ncu_metrics,
    occupancy,
    peak_bandwidth,
    racecheck,
)

__all__ = [
    "ROOT", "CACHE", "build", "module_path",
    "have_nvcc", "peak_bandwidth", "achieved_bandwidth", "bench_ms",
    "kernel_stats", "occupancy", "ncu_metrics", "racecheck",
]
