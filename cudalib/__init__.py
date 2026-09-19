"""Provided plumbing for the CUDA stages, like jvllm/ is for the JAX track.

    from cudalib import build
    mod = build("s08_paged_attn", "app/cuda/s08_paged_attn.cu")

You do not edit anything here. The kernels are yours. The compiler command,
the arch flags and the bandwidth arithmetic are not the lesson.
"""

from cudalib.build import CACHE, ROOT, build, build_source, module_path
from cudalib.probe import (
    achieved_bandwidth,
    bench_ms,
    compare_ms,
    copy_bandwidth,
    have_nvcc,
    kernel_stats,
    matmul_flops,
    ncu_metrics,
    occupancy,
    peak_bandwidth,
    racecheck,
    run_ncu,
    read_bandwidth,
)

__all__ = [
    "ROOT", "CACHE", "build", "build_source", "module_path",
    "have_nvcc", "peak_bandwidth", "read_bandwidth", "matmul_flops",
    "achieved_bandwidth", "bench_ms", "compare_ms", "copy_bandwidth",
    "kernel_stats", "occupancy", "ncu_metrics", "racecheck", "run_ncu",
]
