"""./vc ncu [stage]

Run Nsight Compute on the kernel of a stage. Print the counters that decide if
the kernel is fast:

  - the bytes for each second that the memory system delivered,
  - the sectors that it moved for each request,
  - the warps that were resident.

A wall clock tells you that a kernel is slow. These tell you which of the
three possible reasons it is.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"

B, D, Y, X = "\033[1m", "\033[2m", "\033[33m", "\033[0m"

# One line each, because a metric you cannot explain is a metric you will
# misread.
METRICS = {
    "dram__bytes.sum.per_second":
        "bytes/s out of DRAM. Compare with ./vc info.",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second":
        "32-byte sectors read per second.",
    "smsp__average_data_bytes_per_sector_mem_global_op_ld.pct":
        "of each sector you fetched, how much you used. Coalescing, measured.",
    "sm__warps_active.avg.pct_of_peak_sustained_active":
        "achieved occupancy: warps resident against what an SM can hold.",
    "smsp__inst_executed.avg.pct_of_peak_sustained_active":
        "instruction issue. Low with high DRAM means memory-bound, as decode "
        "should be.",
}

STAGES = {
    "8": ("s08_paged_attn", "paged_attn", "app.s08_paged_cuda",
          "paged_attention_cuda"),
    "8b": ("s08b_paged_attn_vec", "paged_attn_vec", "app.s08b_cuda_memory",
           "paged_attention_vec"),
    "8c": ("s08c_paged_attn_split", "paged_attn_split", "app.s08c_cuda_warps",
           "paged_attention_split"),
}

DRIVER = '''
import sys, torch
sys.path.insert(0, {root!r})
from {module} import {fn} as f
from tests.helpers import build_paged, rand_kv

S, H, KVH, D, L = {shape}
k, v = rand_kv(S, KVH, L, D, "cuda", dtype=torch.float16)
q = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
kc, vc, bt, ctx = build_paged(k, v, 16)
for _ in range(3):
    f(q, kc, vc, bt, ctx)
torch.cuda.synchronize()
'''


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    stage = args[0].lstrip("0") if args else "8"
    if stage not in STAGES:
        print(f"{Y}./vc ncu takes a CUDA stage: {', '.join(STAGES)}{X}")
        return 1
    _, kernel, module, fn = STAGES[stage]

    if not shutil.which("ncu"):
        print(f"{Y}Nsight Compute (ncu) is not installed.{X}")
        print(f"{D}It ships with the CUDA toolkit, usually at "
              f"/usr/local/cuda/bin/ncu.{X}")
        return 1

    small = args[1] if len(args) > 1 else "64"
    shape = f"({small}, 16, 8, 128, 1024)"
    driver = ROOT / ".cudacache" / "ncu_driver.py"
    driver.parent.mkdir(exist_ok=True)
    driver.write_text(DRIVER.format(root=str(ROOT), module=module, fn=fn,
                                    shape=shape))

    print(f"\n{B}stage {stage}{X}  {D}{kernel}, {small} sequences, "
          f"1024 tokens of context{X}")
    print(f"{D}Profile run...{X}")
    r = subprocess.run(
        ["ncu", "--csv", "--target-processes", "all",
         "--kernel-name", f"regex:{kernel}",
         "--metrics", ",".join(METRICS),
         str(PY), str(driver)],
        capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr

    if "ERR_NVGPUCTRPERM" in out:
        print(f"\n{Y}The driver will not let a normal user read the "
              f"performance counters.{X}")
        print(f"""{D}
This is a driver setting, not a permission on a file. To change it:

    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\
        | sudo tee /etc/modprobe.d/nvidia-profiling.conf
    sudo update-initramfs -u
    # then reboot

Until then, the stage checks still run. They skip only the counter check.
See https://developer.nvidia.com/ERR_NVGPUCTRPERM{X}""")
        return 1

    if "NotImplementedError" in out or "stage 08" in out:
        print(f"\n{Y}That stage is not implemented yet.{X}")
        print(f"{D}There is no kernel to profile until ./vc test {stage} "
              f"passes.{X}")
        return 1

    rows = [l for l in r.stdout.splitlines() if l.count(",") >= 4]
    if len(rows) < 2:
        print(f"{Y}ncu returned nothing to read.{X}")
        print(out[-1500:])
        return 1

    import csv

    seen = {}
    for row in csv.DictReader(rows):
        name = row.get("Metric Name")
        if name and name not in seen:
            seen[name] = (row.get("Metric Value"), row.get("Metric Unit", ""))

    print()
    for metric, blurb in METRICS.items():
        value, unit = seen.get(metric, ("-", ""))
        short = metric.split(".")[0].split("__")[-1]
        print(f"  {B}{short:<38}{X} {value:>14} {unit}")
        print(f"  {D}{blurb}{X}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
