"""Provided: what the machine can do, and what your kernel did.

A kernel time in milliseconds has no meaning alone. Decode attention is
bandwidth-bound, so the true question is: of the bytes per second that this
card can deliver, what fraction did you get? Everything here answers that in
one line.
"""

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENTS = ROOT / ".measurements.json"
BYTES_PER_BF16 = 2
PEAK_WORKING_SET_MB = 256


def have_nvcc():
    """Can this machine compile CUDA? The stages skip, not fail, if not."""
    from torch.utils.cpp_extension import CUDA_HOME

    if shutil.which("nvcc"):
        return True
    return bool(CUDA_HOME) and (Path(CUDA_HOME) / "bin" / "nvcc").exists()


# ---------------------------------------------------------------- timing


def _round_ms(function, iters, warmup):
    """One round: warm up, then the mean milliseconds of `iters` calls."""
    import torch

    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_ms(fn, iters=50, warmup=10, best_of=1):
    """Milliseconds for each call, timed with CUDA events.

    Events time the GPU, not the Python that puts the work in the queue. For
    a kernel of 50 us, that difference is most of the measurement.

    `best_of` returns the fastest of several rounds. The fastest round is
    the one that heat changed least. To compare two kernels, use
    compare_ms, not two calls of this function.
    """
    return min(_round_ms(fn, iters, warmup) for _ in range(best_of))


def compare_ms(first, second, rounds=5, iters=50, warmup=10):
    """-> (best ms of first, best ms of second), with the rounds interleaved.

    A laptop GPU boosts, and then it throttles to about two thirds of its
    clock. So two kernels timed one after the other are not timed on the
    same machine. The second one looks worse, by more than the difference
    that you want to measure. Interleave the rounds and keep the best of each, and
    the comparison is stable.
    """
    first_best = second_best = float("inf")
    for _ in range(rounds):
        first_best = min(first_best, _round_ms(first, iters, warmup))
        second_best = min(second_best, _round_ms(second, iters, warmup))
    return first_best, second_best


# ---------------------------------------------------------------- the card


def _measure_then_free(measure):
    """Run measure(), then give its GPU memory back. The tensors of measure()
    are free when it returns, so empty_cache() can release them."""
    import torch

    result = measure()
    torch.cuda.empty_cache()
    return result


def _copy_ms(num_values, iters, warmup):
    import torch

    source = torch.empty(num_values, dtype=torch.bfloat16, device="cuda")
    target = torch.empty_like(source)
    return bench_ms(lambda: target.copy_(source), iters=iters, warmup=warmup)


def copy_bandwidth(working_set_mb, iters=50, warmup=10):
    """GB/s of a device-to-device copy with this working set. Half of the
    working set is the source, and half is the destination. The count
    includes the read and the write."""
    num_values = working_set_mb * 2**20 // 2 // BYTES_PER_BF16
    copy_ms = _measure_then_free(lambda: _copy_ms(num_values, iters, warmup))
    num_bytes = 2 * num_values * BYTES_PER_BF16
    return num_bytes / (copy_ms * 1e-3) / 1e9


def peak_bandwidth(fresh=False):
    """The measured streaming bandwidth of this card, in GB/s.

    Not the number on the box. A 256 MB copy reads and writes, and it is the
    highest bytes per second that this memory system gives. The checks
    measure your kernel against this number, not against a datasheet.

    `fresh=True` measures again, and does not read the cache. Use it each
    time that you compare the number with a kernel. A laptop GPU boosts,
    then throttles to two thirds of its clock. A peak measured cold against
    a kernel measured hot compares two different machines.
    """
    if not fresh:
        cached = read_measurement("peak_bandwidth_gbs")
        if cached is not None:
            return cached
    gbs = copy_bandwidth(PEAK_WORKING_SET_MB)
    write_measurement("peak_bandwidth_gbs", round(gbs, 1))
    return gbs


def read_bandwidth(num_bytes=512 * 2**20):
    """The read-only streaming bandwidth of this card, in bytes/s.

    peak_bandwidth() times a copy, which reads and writes. A decode step
    almost only reads, and a read-only stream is 4% to 10% faster than a
    copy. So use this one for the floor of a decode step. Measure it just
    before the work that you compare with it, for the reason in bench_ms."""
    read_ms = _measure_then_free(lambda: _read_ms(num_bytes))
    return num_bytes / (read_ms * 1e-3)


def _read_ms(num_bytes):
    import torch

    values = torch.empty(num_bytes // BYTES_PER_BF16, dtype=torch.bfloat16,
                         device="cuda").normal_()
    return bench_ms(lambda: values.sum(dtype=torch.float32), iters=20,
                    warmup=3, best_of=3)


def matmul_flops(size=4096, heat_seconds=0.0):
    """The FLOP/s of a dense bf16 matmul on this card.

    heat_seconds > 0 runs the matmul that long first. A laptop GPU boosts,
    then throttles under sustained load, and serving is a sustained load.
    """
    matmul_ms = _measure_then_free(lambda: _matmul_ms(size, heat_seconds))
    return 2 * size ** 3 / (matmul_ms * 1e-3)


def _matmul_ms(size, heat_seconds):
    import torch

    left = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    right = torch.randn_like(left)
    end = time.perf_counter() + heat_seconds
    while time.perf_counter() < end:
        left @ right
    return bench_ms(lambda: left @ right, iters=20, warmup=3, best_of=3)


def achieved_bandwidth(fn, bytes_moved, iters=50, fresh=False):
    """(GB/s, fraction of peak) for a callable that moves bytes_moved bytes.

    `bytes_moved` is the traffic that the ALGORITHM needs, not the traffic
    of the hardware. A fraction over 1.0 is not an error and not magic: the
    L2 cache served some reads. That occurs when several query heads share
    a KV head.
    """
    gbs = bytes_moved / (bench_ms(fn, iters=iters) * 1e-3) / 1e9
    return gbs, gbs / peak_bandwidth(fresh=fresh)


# ---------------------------------------------------------------- kernels


def kernel_stats(name):
    """The registers, shared memory and spills of each kernel, from the .so.

        {"paged_attn_v1": {"reg": 40, "shared": 1088, "local": 0, ...}}

    cuobjdump reports what ptxas allocated. The registers of each thread
    limit how many warps an SM can hold at one time. That is the occupancy,
    and it sets how much latency the SM can hide. A spill (local != 0) means
    that ptxas ran out of registers and used memory. That is always a loss.

    "shared" counts STATIC shared memory only. A kernel that declares
    `extern __shared__` and gets its size from the launch reports 0 here,
    because that size is not in the binary.
    """
    from cudalib.build import module_path

    library = module_path(name)
    if not library.exists() or not shutil.which("cuobjdump"):
        return {}
    listing = subprocess.run(["cuobjdump", "-res-usage", str(library)],
                             capture_output=True, text=True).stdout

    stats = {}
    kernel = None
    for line in listing.splitlines():
        function_line = re.match(r"\s*Function (\S+):", line)
        if function_line:
            kernel = _demangle(function_line.group(1))
        elif kernel and "REG:" in line:
            stats[kernel] = {key.lower(): int(value) for key, value in
                             re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)",
                                        line)}
            kernel = None
    return stats


def occupancy(name, kernel, threads_per_block):
    """The warps that can stay on each SM, against the hardware maximum,
    from the register count. The simple model: on these cards the registers
    are usually the limit."""
    import torch

    usage = next((resources for kernel_name, resources
                  in kernel_stats(name).items() if kernel in kernel_name), None)
    if not usage or not usage.get("reg"):
        return None
    properties = torch.cuda.get_device_properties(0)
    registers_per_sm = getattr(properties, "regs_per_multiprocessor", 65536)
    warps_per_block = (threads_per_block + 31) // 32
    registers_per_warp = usage["reg"] * 32
    blocks_per_sm = min(
        registers_per_sm // (registers_per_warp * warps_per_block),
        properties.max_threads_per_multi_processor // threads_per_block)
    max_warps = properties.max_threads_per_multi_processor // 32
    return blocks_per_sm * warps_per_block / max_warps


def racecheck(script, timeout=600):
    """The shared-memory hazards of the kernels that `script` runs.

    -> a list of report lines, [] for a clean run, or None when
    compute-sanitizer is not installed. Unlike a counter read, this needs no
    special driver permission: it instruments the kernel, and it does not
    read the hardware.

    A race is the worst kind of bug to find by hand. It depends on which
    warp got ahead. So it corrupts a few percent of the output on some
    inputs, and it goes away when you add a print. This tool finds it every
    time, and it names the two source lines.
    """
    if not shutil.which("compute-sanitizer"):
        return None
    try:
        result = subprocess.run(
            ["compute-sanitizer", "--tool", "racecheck",
             "--racecheck-report", "analysis", sys.executable, str(script)],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    report = result.stdout + result.stderr
    if "RACECHECK SUMMARY" not in report:
        return None
    return [line.strip("= ") for line in report.splitlines()
            if "Race reported" in line or "access at" in line]


# ---------------------------------------------------------------- ncu


NO_COUNTER_PERMISSION = "ERR_NVGPUCTRPERM"


def run_ncu(script, metrics, kernel=None, timeout=300):
    """Run Nsight Compute on `script`.
    -> ({metric: (value, unit)} or None, all the output of ncu)."""
    if not shutil.which("ncu"):
        return None, "ncu is not installed"
    command = ["ncu", "--csv", "--target-processes", "all",
               "--metrics", ",".join(metrics)]
    if kernel:
        command += ["--kernel-name", kernel]
    command += [sys.executable, str(script)]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, "ncu took too long"
    output = result.stdout + result.stderr
    if NO_COUNTER_PERMISSION in output:
        return None, output
    return _parse_ncu_csv(result.stdout), output


def _parse_ncu_csv(stdout):
    """The first value of each metric in the CSV of ncu, or None."""
    rows = [line for line in stdout.splitlines() if line.count(",") >= 4]
    if len(rows) < 2:
        return None
    values = {}
    for row in csv.DictReader(rows):
        name = row.get("Metric Name")
        if name and name not in values:
            values[name] = (row.get("Metric Value"), row.get("Metric Unit", ""))
    return values or None


def ncu_metrics(script, metrics, kernel=None, timeout=300):
    """The real hardware counters of one kernel, or None if ncu cannot run.

    On a consumer Linux machine this usually returns None. A read of the
    performance counters needs NVreg_RestrictProfilingToAdminUsers=0, which
    is a driver module parameter and a reboot. `./vc ncu` prints the fix. No
    check ever gates on this.
    """
    values, _ = run_ncu(script, metrics, kernel, timeout)
    if not values:
        return None
    return {name: value for name, (value, _unit) in values.items()}


def _demangle(symbol):
    if not shutil.which("cu++filt"):
        return symbol
    result = subprocess.run(["cu++filt", "-p", symbol], capture_output=True,
                            text=True)
    name = result.stdout.strip() or symbol
    return re.sub(r"^.*?(\w+)\(.*$", r"\1", name)


# ---------------------------------------------------------------- cache


def _read_all_measurements():
    if not MEASUREMENTS.exists():
        return {}
    try:
        return json.loads(MEASUREMENTS.read_text())
    except json.JSONDecodeError:
        return {}


def read_measurement(key):
    return _read_all_measurements().get(key)


def write_measurement(key, value):
    measurements = _read_all_measurements()
    measurements[key] = value
    MEASUREMENTS.write_text(json.dumps(measurements, indent=2))
