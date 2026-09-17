"""Provided: what the machine can do, and what your kernel actually did.

A kernel time in milliseconds means nothing on its own. Decode attention is
bandwidth-bound, so the only honest question is: of the bytes per second this
card can deliver, what fraction did you get? Everything here exists to answer
that in one line.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENTS = ROOT / ".measurements.json"


def have_nvcc():
    """Can we compile CUDA here at all? The stages skip rather than fail."""
    from torch.utils.cpp_extension import CUDA_HOME

    if shutil.which("nvcc"):
        return True
    return bool(CUDA_HOME) and (Path(CUDA_HOME) / "bin" / "nvcc").exists()


def bench_ms(fn, iters=50, warmup=10, best_of=1):
    """Milliseconds per call, timed with CUDA events.

    Events time the GPU, not the Python that enqueues the work. At 50us a
    kernel that difference is most of the measurement.

    `best_of` returns the fastest of several rounds. A laptop GPU boosts and
    then throttles to about two thirds of its clock, so two kernels timed one
    after the other are not timed on the same machine: whichever runs second
    looks worse, by more than the difference you are trying to measure. The
    fastest round is the one least polluted by heat. Interleave the two
    callables and take the best of each, and the comparison holds still.
    """
    import torch

    best = float("inf")
    for _ in range(best_of):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def peak_bandwidth(fresh=False):
    """Measured streaming bandwidth of this card, GB/s.

    Not the number on the box. A 256MB copy reads 256MB and writes 256MB, and
    it is the most bytes per second you will ever see out of this memory
    system. Your kernel is measured against this, not against a datasheet.

    `fresh=True` re-measures instead of reading the cache. Use it whenever
    the number is about to be compared with a kernel: a laptop GPU boosts,
    then throttles to two thirds of its clock, and a peak measured cold
    against a kernel measured hot is a comparison between two machines.
    """
    if not fresh:
        cached = _cached("peak_bandwidth_gbs")
        if cached is not None:
            return cached

    import torch

    n = 256 * 1024 * 1024 // 2                    # 256MB of bf16
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    ms = bench_ms(lambda: b.copy_(a))
    gbs = (2 * a.numel() * 2) / (ms * 1e-3) / 1e9  # read + write
    del a, b
    torch.cuda.empty_cache()
    _cache("peak_bandwidth_gbs", round(gbs, 1))
    return gbs


def achieved_bandwidth(fn, bytes_moved, iters=50, fresh=False):
    """(GB/s, fraction of peak) for a callable that moves bytes_moved bytes.

    `bytes_moved` is the traffic the ALGORITHM needs, not the traffic the
    hardware did. A fraction over 1.0 is not an error and not magic: it means
    L2 served some of the reads, which is what happens when several query
    heads share a KV head.
    """
    ms = bench_ms(fn, iters=iters)
    gbs = bytes_moved / (ms * 1e-3) / 1e9
    return gbs, gbs / peak_bandwidth(fresh=fresh)


def kernel_stats(name):
    """Registers, shared memory and spills per kernel, read out of the .so.

        {"paged_attn_v1": {"reg": 40, "shared": 1088, "local": 0, ...}}

    cuobjdump reports what ptxas actually allocated. Registers per thread cap
    how many warps an SM can hold at once, which is occupancy, which is how
    much latency the SM can hide. Spills (local != 0) mean ptxas ran out and
    started using memory as registers, and that is always a loss.

    "shared" counts STATIC shared memory only. A kernel that declares
    `extern __shared__` and gets its size from the launch reports 0 here,
    because that size is not in the binary.
    """
    from cudalib.build import module_path

    so = module_path(name)
    if not so.exists() or not shutil.which("cuobjdump"):
        return {}
    out = subprocess.run(["cuobjdump", "-res-usage", str(so)],
                         capture_output=True, text=True).stdout

    stats = {}
    fn = None
    for line in out.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            fn = _demangle(m.group(1))
            continue
        if fn and "REG:" in line:
            stats[fn] = {k.lower(): int(v) for k, v in
                         re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", line)}
            fn = None
    return stats


def occupancy(name, kernel, threads_per_block):
    """Warps resident per SM against the hardware maximum, from the register
    count. The simple model: registers are the usual limit on these cards."""
    import torch

    st = kernel_stats(name)
    hit = next((v for k, v in st.items() if kernel in k), None)
    if not hit or not hit.get("reg"):
        return None
    p = torch.cuda.get_device_properties(0)
    regs_per_sm = getattr(p, "regs_per_multiprocessor", 65536)
    warps_per_block = (threads_per_block + 31) // 32
    regs_per_warp = hit["reg"] * 32
    blocks = min(regs_per_sm // (regs_per_warp * warps_per_block),
                 p.max_threads_per_multi_processor // threads_per_block)
    active = blocks * warps_per_block
    return active / (p.max_threads_per_multi_processor // 32)


def racecheck(script, timeout=600):
    """Shared-memory hazards in whatever kernels `script` runs.

    Returns a list of report lines, [] for a clean run, or None when
    compute-sanitizer is not installed. Unlike a counter read, this needs no
    special driver permission: it works by instrumenting the kernel, not by
    reading the hardware.

    A race here is the worst kind of bug to find by hand. It depends on which
    warp got ahead, so it corrupts a few percent of the output on some inputs
    and vanishes when you add a print. This tool is deterministic about it,
    and it names the two source lines.
    """
    import sys

    if not shutil.which("compute-sanitizer"):
        return None
    try:
        r = subprocess.run(
            ["compute-sanitizer", "--tool", "racecheck",
             "--racecheck-report", "analysis", sys.executable, str(script)],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    if "RACECHECK SUMMARY" not in out:
        return None
    return [l.strip("= ") for l in out.splitlines()
            if "Race reported" in l or "access at" in l]


def ncu_metrics(script, metrics, kernel=None, timeout=300):
    """Real hardware counters for one kernel, or None if ncu cannot run.

    On a consumer Linux box this usually returns None: reading performance
    counters needs NVreg_RestrictProfilingToAdminUsers=0, which is a driver
    module parameter and a reboot. `./vc ncu` prints the fix. No check ever
    gates on this.
    """
    import sys

    if not shutil.which("ncu"):
        return None
    cmd = ["ncu", "--csv", "--target-processes", "all",
           "--metrics", ",".join(metrics)]
    if kernel:
        cmd += ["--kernel-name", kernel]
    cmd += [sys.executable, str(script)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if "ERR_NVGPUCTRPERM" in r.stdout + r.stderr:
        return None

    rows = [l for l in r.stdout.splitlines() if l.count(",") >= 4]
    if len(rows) < 2:
        return None
    import csv

    out = {}
    for row in csv.DictReader(rows):
        name = row.get("Metric Name")
        if name:
            out[name] = row.get("Metric Value")
    return out or None


def _demangle(sym):
    if not shutil.which("cu++filt"):
        return sym
    r = subprocess.run(["cu++filt", "-p", sym], capture_output=True, text=True)
    name = (r.stdout.strip() or sym)
    name = re.sub(r"^.*?(\w+)\(.*$", r"\1", name)
    return name


def _cached(key):
    if not MEASUREMENTS.exists():
        return None
    try:
        return json.loads(MEASUREMENTS.read_text()).get(key)
    except json.JSONDecodeError:
        return None


def _cache(key, value):
    d = {}
    if MEASUREMENTS.exists():
        try:
            d = json.loads(MEASUREMENTS.read_text())
        except json.JSONDecodeError:
            pass
    d[key] = value
    MEASUREMENTS.write_text(json.dumps(d, indent=2))
