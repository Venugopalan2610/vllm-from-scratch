"""Reference solution, stage 18b - int8 weight-only GEMV."""

import torch.nn as nn

from app.s18_quantization import quantize_int8_per_channel
from cudalib import build

SOURCE = "app/cuda/s18b_gemv_int8.cu"


def _extension():
    return build("s18b_gemv_int8", SOURCE)


def gemv_int8(inputs, int8_weight, scales):
    return _extension().gemv_int8(inputs.contiguous(), int8_weight, scales)


class QuantizedLinearCUDA(nn.Module):
    """The QuantizedLinear of stage 18, with your kernel under it.

    A measurement sets GEMV_MAX_ROWS. It is not a guess. Alone, with 4 rows
    for each warp, the kernel is faster than cuBLAS up to 8 rows. In the real
    model the crossover is at about 4 rows, because the small matrices of the
    model give cuBLAS less to lose. Stage 24 measures it there.
    """

    GEMV_MAX_ROWS = 4

    def __init__(self, int8_weight, scales, bias, in_features, out_features):
        super().__init__()
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_linear(cls, linear):
        int8_weight, scales = quantize_int8_per_channel(
            linear.weight.data.float())
        bias = None if linear.bias is None else linear.bias.data.clone()
        return cls(int8_weight.contiguous(), scales.float(), bias,
                   linear.in_features, linear.out_features)

    def _matmul(self, rows):
        if rows.shape[0] <= self.GEMV_MAX_ROWS:
            return gemv_int8(rows, self.int8_weight, self.scales)
        weight = (self.int8_weight.to(rows.dtype)
                  * self.scales.to(rows.dtype)[:, None])
        return rows @ weight.t()

    def forward(self, inputs):
        outputs = self._matmul(inputs.reshape(-1, inputs.shape[-1]))
        if self.bias is not None:
            outputs = outputs + self.bias.to(outputs.dtype)
        return outputs.reshape(*inputs.shape[:-1], self.out_features)

    def nbytes(self):
        return self.int8_weight.numel() + self.scales.numel() * 4
