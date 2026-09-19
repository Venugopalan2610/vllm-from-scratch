"""How to make Pallas run on the GPU that you own.

Pallas has two GPU backends. Mosaic GPU is the default from JAX 0.9 on. It
targets Hopper and Blackwell, and it wants tensor-core layouts and TMA that
an sm_89 consumer card does not have. The older Triton backend lowers to
Triton IR, and it runs on all cards from Turing up.

Two things get in the way on a consumer GPU:

  1. JAX selects Mosaic by default. Mosaic then fails to infer layouts, with
     a message ("Failed to infer a possible set of layouts") that says
     nothing about an old GPU.
  2. The Triton backend accepts only the device kinds on a fixed list: A100,
     H100, L4, RTX 4090, and more. Many laptop parts are not on it. You then
     get "No supported GPU devices found" on a card that can run it.

(2) is a lookup table, and JAX exposes the table. `enable_pallas_triton()`
reads the compute capability of your device and registers it. Nothing here
changes the behavior of JAX. It only adds a row that upstream did not list
yet.
"""

import warnings

HOPPER_COMPUTE_CAPABILITY = 90
GPU_PLATFORMS = ("gpu", "cuda", "rocm")
QUIET_DEPRECATIONS = ("jax-pallas-triton", "jax-pallas-call-mgpu")


def _device():
    import jax

    try:
        return jax.devices()[0]
    except RuntimeError:
        return None


def _compute_capability(device):
    """"8.9" -> ("8.9", 89), or None if the device does not say."""
    capability = getattr(device, "compute_capability", None)
    if capability is None:
        return None
    arch_name = str(capability)
    try:
        return arch_name, int(arch_name.replace(".", ""))
    except ValueError:
        return None


def enable_pallas_triton():
    """Register this GPU with the Triton backend of Pallas. A second call
    does nothing new.

    -> True if the Triton backend can run after this call.
    """
    try:
        from jax._src.pallas.triton import gpu_info
    except ImportError:
        return False        # this JAX build has no Triton backend

    device = _device()
    if device is None or device.platform not in GPU_PLATFORMS:
        return False
    kind = device.device_kind
    try:
        if (gpu_info.gpu_version_from_device_kind(kind) is not None
                or kind in gpu_info.registry):
            return True     # upstream already knows this card
    except Exception:
        pass

    capability = _compute_capability(device)
    if capability is None:
        return False
    # jax reports "8.9". The Triton backend wants arch_name "8.9" and an
    # integer compute_capability of 89.
    arch_name, capability_number = capability
    gpu_info.registry[kind] = lambda: gpu_info.GpuInfo(
        gpu_version=None, arch_name=arch_name,
        compute_capability=capability_number)
    return True


def pallas_backend():
    """'triton', 'mosaic', or None if Pallas cannot run here.

    Mosaic needs sm_90 or newer. Below that, register and use Triton.
    """
    device = _device()
    if device is None or device.platform == "cpu":
        return None
    if enable_pallas_triton():
        return "triton"
    capability = _compute_capability(device)
    if capability and capability[1] >= HOPPER_COMPUTE_CAPABILITY:
        return "mosaic"
    return None


def compiler_params(num_warps=4, num_stages=3):
    """CompilerParams for `pl.pallas_call`, or None for the default.

    Give the result directly:

        pl.pallas_call(..., compiler_params=jvllm.compat.compiler_params())
    """
    if pallas_backend() != "triton":
        return None
    from jax.experimental.pallas import triton as pallas_triton

    return pallas_triton.CompilerParams(num_warps=num_warps,
                                        num_stages=num_stages)


def silence_pallas_deprecations():
    """JAX 0.11 warns on every Triton-backend pallas_call.

    The warning is for library authors with a Hopper CI fleet. On a consumer
    card the Triton backend is the only one that works, so the warning is
    noise on a path that you cannot leave.
    """
    warnings.filterwarnings(
        "ignore", message=".*Pallas Triton backend is deprecated.*")
    try:
        from jax._src import deprecations

        if not getattr(deprecations.warn, "_jvllm_quiet", False):
            deprecations.warn = _quiet_warn(deprecations.warn)
    except Exception:
        pass


def _quiet_warn(original_warn):
    def warn(deprecation_id, message, stacklevel=2):
        if deprecation_id in QUIET_DEPRECATIONS:
            return None
        return original_warn(deprecation_id, message, stacklevel)

    warn._jvllm_quiet = True
    return warn
