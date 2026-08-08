"""Making Pallas run on the GPU you actually own.

Pallas has two GPU backends. Mosaic GPU is the default from JAX 0.9 on, and it
targets Hopper and Blackwell -- it wants tensor-core layouts and TMA that an
sm_89 consumer card does not have. The older Triton backend lowers to the same
Triton IR that stage 08's torch twin uses, and runs on anything from Turing up.

Two things get in the way on a laptop GPU:

  1. Mosaic is picked by default, and then fails to infer layouts with a
     message ("Failed to infer a possible set of layouts") that says nothing
     about your GPU being too old.
  2. The Triton backend gates on a hardcoded ALLOWLIST of device kinds --
     A100, H100, L4, RTX 4090, and so on. An RTX 4080 Laptop GPU is not on it,
     and you get "No supported GPU devices found" on a card that supports it
     perfectly well.

(2) is a lookup table, and JAX exposes the table. `enable_pallas_triton()`
reads the compute capability off the device you have and registers it. Nothing
here patches JAX behaviour; it only fills in a row that upstream has not
enumerated yet.
"""

import warnings


def _device():
    import jax

    try:
        return jax.devices()[0]
    except RuntimeError:
        return None


def enable_pallas_triton():
    """Register this GPU with Pallas's Triton backend. Idempotent.

    Returns True if the Triton backend is usable after this call.
    """
    try:
        from jax._src.pallas.triton import gpu_info as gi
    except ImportError:
        return False        # no Triton backend in this JAX build

    dev = _device()
    if dev is None or dev.platform not in ("gpu", "cuda", "rocm"):
        return False

    kind = dev.device_kind
    try:
        if gi.gpu_version_from_device_kind(kind) is not None or kind in gi.registry:
            return True     # upstream already knows this card
    except Exception:
        pass

    cc = getattr(dev, "compute_capability", None)
    if cc is None:
        return False

    # jax reports "8.9"; the Triton backend wants arch_name "8.9" and an
    # integer compute_capability 89.
    arch = str(cc)
    try:
        cc_int = int(arch.replace(".", ""))
    except ValueError:
        return False

    gi.registry[kind] = lambda: gi.GpuInfo(
        gpu_version=None, arch_name=arch, compute_capability=cc_int
    )
    return True


def pallas_backend():
    """'triton', 'mosaic', or None if Pallas cannot run here.

    Mosaic needs sm_90+. Below that we register and use Triton.
    """
    dev = _device()
    if dev is None:
        return None
    if dev.platform == "cpu":
        return None
    if enable_pallas_triton():
        return "triton"
    cc = str(getattr(dev, "compute_capability", "0"))
    try:
        if int(cc.replace(".", "")) >= 90:
            return "mosaic"
    except ValueError:
        pass
    return None


def compiler_params(num_warps=4, num_stages=3):
    """CompilerParams for `pl.pallas_call`, or None to take the default.

    Pass the result straight through:

        pl.pallas_call(..., compiler_params=jvllm.compat.compiler_params())
    """
    if pallas_backend() != "triton":
        return None
    from jax.experimental.pallas import triton as plt

    return plt.CompilerParams(num_warps=num_warps, num_stages=num_stages)


def silence_pallas_deprecations():
    """JAX 0.11 warns on every Triton-backend pallas_call.

    The warning is aimed at library authors with a Hopper CI fleet. On a
    consumer card the Triton backend is the only one that works, so the warning
    is noise on a path you cannot leave.
    """
    warnings.filterwarnings(
        "ignore", message=".*Pallas Triton backend is deprecated.*"
    )
    try:
        from jax._src import deprecations

        if not getattr(deprecations.warn, "_jvllm_quiet", False):
            deprecations.warn = _quiet_warn(deprecations.warn)
    except Exception:
        pass


def _quiet_warn(orig):
    def warn(deprecation_id, message, stacklevel=2):
        if deprecation_id in ("jax-pallas-triton", "jax-pallas-call-mgpu"):
            return
        return orig(deprecation_id, message, stacklevel)

    warn._jvllm_quiet = True
    return warn
