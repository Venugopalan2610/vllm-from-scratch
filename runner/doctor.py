"""./vc doctor

Say what this machine can run, and what it cannot. Run it before you start,
and when a stage behaves in a way that you cannot explain.

It checks the hardware, the toolchain and the disk. It changes nothing.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import capability  # noqa: E402
from runner.style import paint  # noqa: E402

MODEL_GB = 5.0          # Qwen3-1.7B and Qwen3-0.6B in bf16, plus the tokenizers
WORK_GB = 4.0           # the venv, the build cache and room for the KV pool
NEEDED_VRAM_GB = 6.0    # the capstone at its default pool size


def line(ok, text, detail=""):
    mark = paint("  ok  ", "green") if ok else paint(" MISS ", "yellow")
    print(f"{mark} {text}")
    if detail:
        print(paint(f"        {detail}", "dim"))


def check_hardware():
    properties = capability.device()
    print("\n" + paint("Hardware", "bold"))
    if properties is None:
        line(False, "no GPU that torch can see",
             "The pure-logic stages still run: 06, 10, 11, 13, 14, 15, 16, "
             "17, 19, 20.")
    else:
        major, minor = capability.capability(properties)
        line(True, f"{properties.name}, sm_{major}{minor}, "
                   f"{properties.total_memory / 1e9:.1f} GB VRAM")
        free = capability.free_vram_gb()
        line(free >= NEEDED_VRAM_GB, f"{free:.1f} GB of VRAM free now",
             "" if free >= NEEDED_VRAM_GB else
             f"The capstone wants about {NEEDED_VRAM_GB:.0f} GB. Close the "
             "other programs that use the GPU.")
    for name, stages, met, why in capability.report():
        line(met, f"{name}   " + paint(f"stages {stages}", "dim"),
             "" if met else why)


def check_toolchain():
    print("\n" + paint("Toolchain", "bold"))
    line(True, f"torch {capability.torch_version()}")
    ninja = shutil.which("ninja") or (ROOT / ".venv" / "bin" / "ninja").exists()
    line(bool(ninja), "ninja (torch builds the .cu files with it)",
         "" if ninja else "Run ./setup.sh again.")
    driver = capability.nvidia_smi("driver_version")
    if driver:
        line(True, f"driver {driver}")
    throttles = capability.active_throttles()
    line(not throttles, "the GPU runs at its normal clock" if not throttles
         else "the GPU holds its clock down: " + ", ".join(throttles),
         "" if not throttles else "A hot or power-limited GPU changes every "
         "measurement. Let it cool, then measure again.")


def check_disk():
    print("\n" + paint("Disk", "bold"))
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    line(free_gb >= MODEL_GB + WORK_GB,
         f"{free_gb:.0f} GB free in the repo",
         "" if free_gb >= MODEL_GB + WORK_GB else
         f"The models need about {MODEL_GB} GB, and the build cache and the "
         f"venv about {WORK_GB} GB.")
    cache = Path.home() / ".cache" / "huggingface"
    line(cache.exists(), f"HuggingFace cache at {cache}",
         "" if cache.exists() else "The first run downloads the models, "
         "about 5 GB.")


def main():
    print("\n" + paint("./vc doctor", "bold", "cyan")
          + paint("   what this machine can run", "dim"))
    check_hardware()
    check_toolchain()
    check_disk()
    print("\n  " + paint("A MISS is not a failure. A stage that needs the "
                         "missing part skips.", "dim"))
    print("  " + paint("Do not stop. Continue. Be better than before.", "dim") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
