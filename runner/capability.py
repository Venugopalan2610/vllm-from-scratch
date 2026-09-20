"""What this machine can run, and what each part of the ladder needs.

One source for a question that the README, setup.sh, ./vc info and ./vc
doctor all ask. A wrong answer here sends a learner to a GPU that cannot
finish the course, so each requirement names the hardware feature and the
compute capability that gives it.

    bf16 tensor cores      sm_80 (Ampere)     the model dtype of the course
    FP8 e4m3               sm_89 (Ada)        stage 24b, the FP8 KV cache
    a CUDA toolkit (nvcc)  any                the five CUDA stages
"""

import shutil
import subprocess

BF16_CAPABILITY = (8, 0)
FP8_CAPABILITY = (8, 9)

# (what it needs, the stages that need it, why)
REQUIREMENTS = [
    ("a GPU", "01-05, 07-09, 12-14, 17-18, 21-28",
     "the model runs on the GPU. The pure-logic stages do not need one."),
    ("bf16 tensor cores (sm_80+)", "01-28",
     "the course loads the model in bf16. A card below sm_80 runs it in "
     "software, and every speed gate then measures the wrong thing."),
    ("a CUDA toolkit (nvcc)", "08, 08b, 08c, 18b, 24b",
     "you write and compile .cu files of your own."),
    ("FP8 e4m3 (sm_89+)", "24b",
     "the FP8 KV cache needs the hardware format."),
]


def device():
    """-> the torch device properties, or None if there is no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0)
    except Exception:
        return None


def capability(properties=None):
    """-> (major, minor) of the GPU, or None."""
    properties = properties or device()
    return (properties.major, properties.minor) if properties else None


def has_bf16(properties=None):
    return (capability(properties) or (0, 0)) >= BF16_CAPABILITY


def has_fp8(properties=None):
    return (capability(properties) or (0, 0)) >= FP8_CAPABILITY


def has_nvcc():
    return bool(shutil.which("nvcc"))


def free_vram_gb(properties=None):
    try:
        import torch

        return torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        return 0.0


def torch_version():
    try:
        import torch

        return torch.__version__
    except Exception:
        return "not installed"


def report():
    """-> a list of (requirement, stages, met, why) for this machine."""
    properties = device()
    met = {
        "a GPU": properties is not None,
        "bf16 tensor cores (sm_80+)": has_bf16(properties),
        "a CUDA toolkit (nvcc)": has_nvcc(),
        "FP8 e4m3 (sm_89+)": has_fp8(properties),
    }
    return [(name, stages, met[name], why) for name, stages, why in REQUIREMENTS]


THROTTLE_REASONS = {
    "sw_power_cap": "the power limit",
    "hw_thermal_slowdown": "heat, in hardware",
    "sw_thermal_slowdown": "heat, in software",
    "hw_slowdown": "a hardware slowdown",
}


def active_throttles():
    """-> the reasons that the GPU gives for a lower clock, in plain words.

    A hot or power-limited GPU changes every measurement, so a benchmark
    must say when this is true. The idle reason is normal and never listed.
    """
    fields = ",".join(f"clocks_throttle_reasons.{name}" for name in THROTTLE_REASONS)
    answer = nvidia_smi(fields)
    if not answer:
        return []
    states = [state.strip() for state in answer.split(",")]
    return [why for (name, why), state in zip(THROTTLE_REASONS.items(), states)
            if state == "Active"]


def nvidia_smi(query, default=""):
    """One field from nvidia-smi, or the default. Never raises."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 else default
    except Exception:
        return default
