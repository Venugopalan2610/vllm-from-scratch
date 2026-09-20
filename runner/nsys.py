"""./vc nsys [stage or engine]

Run NVIDIA Nsight Systems on the engine or a specific stage kernel.
Captures an end-to-end timeline trace showing:
  - GPU kernel execution timeline across CUDA streams
  - Host-to-Device (HtoD) and Device-to-Host (DtoH) transfers
  - CPU launch overhead and execution bubbles between steps
  - OS runtime thread scheduling and CUDA driver API calls
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.style import paint

from contextlib import contextmanager

OUTPUT_DIR = ROOT / ".cudacache"
PYTHON = ROOT / ".venv" / "bin" / "python"

ENGINE_DRIVER = '''
import sys, torch, time
sys.path.insert(0, {root!r})
from tvllm import load_model
from app.s22_engine import LLMEngine
from tests.helpers import CAPSTONE_PROMPTS

# Load model via capstone loader
model = load_model()
num_blocks = 128
engine = LLMEngine(model, num_blocks=num_blocks, max_num_seqs=16, token_budget=256)

# Warmup pass
for i in range(2):
    engine.add_request(f"warm{{i}}", "The capital of France is", 8)
engine.run_to_completion()
torch.cuda.synchronize()

# Profiled execution: simulate continuous batching under concurrency
torch.cuda.nvtx.range_push("profiled_inference_loop")
for i in range(8):
    engine.add_request(f"req{{i}}", CAPSTONE_PROMPTS[i % len(CAPSTONE_PROMPTS)], 16)

step_count = 0
while engine.has_work() and step_count < 20:
    torch.cuda.nvtx.range_push(f"engine_step_{{step_count}}")
    engine.step()
    torch.cuda.nvtx.range_pop()
    step_count += 1

torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
'''


@contextmanager
def active_solutions():
    """Ensure working implementation is in app/ during the profile run,
    then restore stubs."""
    solutions_dir = ROOT / ".solutions"
    app_dir = ROOT / "app"
    cuda_dir = app_dir / "cuda"
    stubs_backup = {}
    cuda_backup = {}

    has_stubs = any("raise NotImplementedError" in f.read_text() for f in app_dir.glob("s*.py"))
    if has_stubs and solutions_dir.exists():
        for f in app_dir.glob("*.py"):
            stubs_backup[f.name] = f.read_text()
        for f in cuda_dir.glob("*.cu"):
            cuda_backup[f.name] = f.read_text()

        for sol in solutions_dir.glob("*.py"):
            (app_dir / sol.name).write_text(sol.read_text())
        for sol in solutions_dir.glob("*.cu"):
            (cuda_dir / sol.name).write_text(sol.read_text())

    try:
        yield
    finally:
        for name, text in stubs_backup.items():
            (app_dir / name).write_text(text)
        for name, text in cuda_backup.items():
            (cuda_dir / name).write_text(text)


def check_nsys():
    path = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
    if not Path(path).exists():
        print(paint("NVIDIA Nsight Systems ('nsys') not found.", "yellow"))
        print("Install the CUDA Toolkit or Nsight Systems from developer.nvidia.com.")
        return None
    return str(path)


def run_profile():
    nsys_bin = check_nsys()
    if not nsys_bin:
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)
    driver_script = OUTPUT_DIR / "nsys_driver.py"
    driver_script.write_text(ENGINE_DRIVER.format(root=str(ROOT)))

    report_base = OUTPUT_DIR / "engine_profile"

    cmd = [
        nsys_bin,
        "profile",
        "-t", "cuda,nvtx,osrt",
        "-s", "none",
        "-o", str(report_base),
        "--force-overwrite=true",
        "--stats=true",
        str(PYTHON),
        str(driver_script)
    ]

    print("\n" + paint("Running NVIDIA Nsight Systems (nsys) profile on continuous batching engine...", "bold", "cyan"))
    print(paint("Tracing CUDA API, NVTX ranges, kernel launches, and GPU memory...", "dim"))

    with active_solutions():
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(paint("nsys execution failed:", "yellow"))
        print(result.stderr[-1000:])
        return result.returncode

    # Print relevant sections of the stats report
    lines = result.stdout.splitlines()
    in_report = False
    for line in lines:
        if "Executing 'cuda_gpu_kern_sum'" in line or "Executing 'cuda_api_sum'" in line:
            in_report = True
            print("\n" + paint(line, "bold"))
        elif in_report and line.startswith("["):
            in_report = False
        elif in_report:
            print(line)

    rep_file = OUTPUT_DIR / "engine_profile.nsys-rep"
    print("\n" + paint("=" * 70, "dim"))
    print(paint("Nsight Systems trace saved to:", "green", "bold") + f" {rep_file}")
    print(paint("View the full interactive timeline by opening it in the Nsight Systems GUI:", "dim"))
    print(paint(f"  nsys-ui {rep_file}", "cyan"))
    print(paint("=" * 70, "dim") + "\n")
    return 0


def main():
    return run_profile()


if __name__ == "__main__":
    sys.exit(main())
