"""Stage 03 - prefill and decode are two different machines.

`./vc lore 3` for the insight. `./vc test 3` to check yourself.
"""

import torch


def model_bytes(model) -> int:
    """Total bytes of the model's parameters.

    This is what must cross the memory bus on every forward pass.
    Sum numel() * element_size() over model.parameters().
    """
    raise NotImplementedError("stage 03: implement model_bytes")


@torch.inference_mode()
def time_prefill(model, n_tokens: int, iters: int = 10) -> float:
    """Milliseconds for ONE forward pass over a prompt of n_tokens.

    Feed a (1, n_tokens) batch of token ids. Warm up first, then put
    torch.cuda.synchronize() on both sides of the timed region -- CUDA is
    async, and without it you measure only the kernel LAUNCH, not the work.
    """
    raise NotImplementedError("stage 03: implement time_prefill")


@torch.inference_mode()
def time_decode(model, ctx_len: int, steps: int = 40) -> float:
    """Milliseconds per token for single-token decode with ctx_len of history.

    Prefill ctx_len tokens to build a cache, warm up, then time `steps`
    single-token forwards, feeding the cache forward each time.

    Do NOT copy the cache inside the timed loop. (I made exactly that mistake
    while writing this stage: a deepcopy of 28 layers of KV dominated the
    measurement and made decode look 20% slower than it really is.)
    """
    raise NotImplementedError("stage 03: implement time_decode")


def achieved_gbs(nbytes: int, ms_per_token: float) -> float:
    """Effective GB/s implied by reading `nbytes` in `ms_per_token`.

        bytes / seconds / 1e9

    Compare against the ~380 GB/s that ./vc info measured. Decode should reach
    a large fraction of peak, because it is doing almost nothing but reading.
    """
    raise NotImplementedError("stage 03: implement achieved_gbs")
