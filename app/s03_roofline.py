"""Stage 03 - prefill and decode are two different machines.

`./vc lore 3` for the insight. `./vc test 3` to check yourself.
"""

import torch


def model_bytes(model) -> int:
    """The bytes of all the parameters of the model.

    Every forward pass reads these bytes across the memory bus.
    Add numel() * element_size() for each tensor in model.parameters().
    """
    raise NotImplementedError("stage 03: implement model_bytes")


@torch.inference_mode()
def time_prefill(model, num_tokens: int, iters: int = 10) -> float:
    """Milliseconds for ONE forward pass over a prompt of num_tokens.

    Give a (1, num_tokens) batch of token ids. Warm up first. Then put
    torch.cuda.synchronize() before and after the timed part. CUDA is
    asynchronous. Without the synchronize, you measure only the kernel
    launch, not the work.
    """
    raise NotImplementedError("stage 03: implement time_prefill")


@torch.inference_mode()
def time_decode(model, context_len: int, steps: int = 40) -> float:
    """Milliseconds for each token of single-token decode, at a context of
    context_len.

    Prefill context_len tokens to make a cache. Warm up. Then time `steps`
    single-token forward passes, and give each pass the cache of the pass
    before it.

    Do NOT copy the cache inside the timed loop. A deepcopy of 28 layers of
    KV takes most of the time, and decode then looks 20% slower than it is.
    """
    raise NotImplementedError("stage 03: implement time_decode")


def achieved_gbs(num_bytes: int, ms_per_token: float) -> float:
    """The GB/s that a read of num_bytes in ms_per_token gives.

        bytes / seconds / 1e9

    Compare it with the peak that ./vc info measured. Decode reaches a large
    fraction of the peak, because it does almost nothing but read.
    """
    raise NotImplementedError("stage 03: implement achieved_gbs")
