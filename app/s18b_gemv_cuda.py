"""Stage 18b - the int8 GEMV, and the Linear that uses it.

`./vc lore 18b`. `./vc test 18b`.

Stage 18 proved that weight-only int8 keeps the accuracy. It did not get the
speed, because PyTorch must make a bf16 weight in memory before it can
multiply. The kernel is in app/cuda/s18b_gemv_int8.cu.

This wrapper holds one number that you must MEASURE, not guess:
GEMV_MAX_ROWS, the batch size above which cuBLAS is faster than your kernel.
The checks print the crossover table. Read it before you set the number.
"""

import torch.nn as nn

from app.s18_quantization import quantize_int8_per_channel
from cudalib import build

SOURCE = "app/cuda/s18b_gemv_int8.cu"


def _extension():
    return build("s18b_gemv_int8", SOURCE)


def gemv_int8(inputs, int8_weight, scales):
    """inputs @ dequantize(int8_weight, scales).T. Do not make the dequantized
    weight.

        inputs       (B, K) or (K,)
        int8_weight  (N, K) int8
        scales       (N,)
        ->           (B, N) or (N,)
    """
    raise NotImplementedError("stage 18b: call your gemv kernel")


class QuantizedLinearCUDA(nn.Module):
    """The QuantizedLinear of stage 18, with your kernel under it.

    Required, as in stage 18:
        from_linear(nn.Linear) -> QuantizedLinearCUDA     classmethod
        forward(inputs)                                   matches the original
        nbytes() -> int                                   the storage cost

    forward() has one more job: send small batches to the kernel, and all
    other batches to a GEMM. A kernel that is 3x faster at one row and 6x
    slower at four rows cannot replace the GEMM everywhere. It is a decode
    path.
    """

    GEMV_MAX_ROWS = 1

    @classmethod
    def from_linear(cls, linear):
        raise NotImplementedError("stage 18b: implement from_linear")

    def forward(self, inputs):
        raise NotImplementedError("stage 18b: implement forward")

    def nbytes(self):
        raise NotImplementedError("stage 18b: implement nbytes")
