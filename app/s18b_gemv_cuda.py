"""Stage 18b - the int8 GEMV, and the Linear that uses it.

`./vc lore 18b`. `./vc test 18b`.

Stage 18 proved that weight-only int8 keeps the accuracy. It did not take the
speed, because PyTorch must materialise a bf16 weight before it can multiply.
The kernel is in app/cuda/s18b_gemv_int8.cu.

This wrapper holds one number you have to MEASURE rather than guess:
GEMV_MAX_ROWS, the batch size above which cuBLAS beats your kernel. The
checks print the crossover table. Read it before you set it.
"""

import torch
import torch.nn as nn

from app.s18_quantization import quantize_int8_per_channel
from cudalib import build

SOURCE = "app/cuda/s18b_gemv_int8.cu"


def _ext():
    return build("s18b_gemv_int8", SOURCE)


def gemv_int8(x, w_q, scales):
    """y = x @ dequantize(w_q, scales).T, without ever forming the dequantized
    weight.

        x       (B, K) or (K,)
        w_q     (N, K) int8
        scales  (N,)
        ->      (B, N) or (N,)
    """
    raise NotImplementedError("stage 18b: call your gemv kernel")


class QuantizedLinearCUDA(nn.Module):
    """Stage 18's QuantizedLinear with your kernel underneath.

    Required, same as stage 18:
        from_linear(nn.Linear) -> QuantizedLinearCUDA     classmethod
        forward(x)                                        matches the original
        nbytes() -> int                                   storage cost

    forward() has one extra job: send small batches to the kernel and
    everything else to a GEMM. A kernel that is 3x faster at one row and 6x
    slower at four is not a drop-in replacement, it is a decode path.
    """

    GEMV_MAX_ROWS = 1

    @classmethod
    def from_linear(cls, lin):
        raise NotImplementedError("stage 18b: implement from_linear")

    def forward(self, x):
        raise NotImplementedError("stage 18b: implement forward")

    def nbytes(self):
        raise NotImplementedError("stage 18b: implement nbytes")
