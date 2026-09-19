"""./vc ncu [stage] [num_seqs]

Run Nsight Compute on the kernel of a stage. Print the counters that decide
if the kernel is fast:

  - the bytes for each second that the memory system delivered,
  - the sectors that it moved for each request,
  - the warps that were resident.

A wall clock tells you that a kernel is slow. These counters tell you which
of the three possible reasons it is.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cudalib  # noqa: E402
from cudalib.probe import NO_COUNTER_PERMISSION  # noqa: E402
from runner.style import paint  # noqa: E402

# One line each, because you misread a metric that you cannot explain.
METRICS = {
    "dram__bytes.sum.per_second":
        "bytes/s out of DRAM. Compare with ./vc info.",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second":
        "32-byte sectors read each second.",
    "smsp__average_data_bytes_per_sector_mem_global_op_ld.pct":
        "of each sector that you fetched, how much you used. Coalescing, "
        "measured.",
    "sm__warps_active.avg.pct_of_peak_sustained_active":
        "achieved occupancy: resident warps against what an SM can hold.",
    "smsp__inst_executed.avg.pct_of_peak_sustained_active":
        "instruction issue. Low with a high DRAM rate means memory-bound, as "
        "decode must be.",
}

# stage -> (kernel name, module, function)
STAGES = {
    "8": ("paged_attn", "app.s08_paged_cuda", "paged_attention_cuda"),
    "8b": ("paged_attn_vec", "app.s08b_cuda_memory", "paged_attention_vec"),
    "8c": ("paged_attn_split", "app.s08c_cuda_warps", "paged_attention_split"),
}

DRIVER = '''
import sys, torch
sys.path.insert(0, {root!r})
from {module} import {function} as paged_attention
from tests.helpers import paged_problem

query, _, _, paged_cache = paged_problem({num_seqs}, 16, 8, 128, 1024, 16,
                                         "cuda", torch.float16)
for _ in range(3):
    paged_attention(query, *paged_cache)
torch.cuda.synchronize()
'''

PERMISSION_FIX = """
This is a setting of the driver, not a permission on a file. To change it:

    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\
        | sudo tee /etc/modprobe.d/nvidia-profiling.conf
    sudo update-initramfs -u
    # then reboot

Until then, the stage checks still run. They skip only the counter check.
See https://developer.nvidia.com/ERR_NVGPUCTRPERM"""


def parse_args(argv):
    positional = [argument for argument in argv if not argument.startswith("-")]
    stage = positional[0].lstrip("0") if positional else "8"
    num_seqs = positional[1] if len(positional) > 1 else "64"
    return stage, num_seqs


def write_driver(stage, num_seqs):
    _, module, function = STAGES[stage]
    driver = ROOT / ".cudacache" / "ncu_driver.py"
    driver.parent.mkdir(exist_ok=True)
    driver.write_text(DRIVER.format(root=str(ROOT), module=module,
                                    function=function, num_seqs=num_seqs))
    return driver


def explain_failure(stage, output):
    if NO_COUNTER_PERMISSION in output:
        print("\n" + paint("The driver does not let a normal user read the "
                           "performance counters.", "yellow"))
        print(paint(PERMISSION_FIX, "dim"))
    elif "NotImplementedError" in output or "stage 08" in output:
        print("\n" + paint("That stage is not implemented yet.", "yellow"))
        print(paint(f"There is no kernel to profile until ./vc test {stage} "
                    "passes.", "dim"))
    else:
        print(paint("ncu returned nothing to read.", "yellow"))
        print(output[-1500:])


def print_counters(values):
    print()
    for metric, meaning in METRICS.items():
        value, unit = values.get(metric, ("-", ""))
        short_name = metric.split(".")[0].split("__")[-1]
        print("  " + paint(f"{short_name:<38}", "bold") + f" {value:>14} {unit}")
        print("  " + paint(meaning, "dim") + "\n")


def main(argv):
    stage, num_seqs = parse_args(argv)
    if stage not in STAGES:
        print(paint(f"./vc ncu takes a CUDA stage: {', '.join(STAGES)}",
                    "yellow"))
        return 1
    if not shutil.which("ncu"):
        print(paint("Nsight Compute (ncu) is not installed.", "yellow"))
        print(paint("It comes with the CUDA toolkit, usually at "
                    "/usr/local/cuda/bin/ncu.", "dim"))
        return 1

    kernel = STAGES[stage][0]
    print("\n" + paint(f"stage {stage}", "bold") + "  "
          + paint(f"{kernel}, {num_seqs} sequences, 1024 tokens of context",
                  "dim"))
    print(paint("Profile run...", "dim"))
    values, output = cudalib.run_ncu(write_driver(stage, num_seqs), METRICS,
                                     kernel=f"regex:{kernel}")
    if not values:
        explain_failure(stage, output)
        return 1
    print_counters(values)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
