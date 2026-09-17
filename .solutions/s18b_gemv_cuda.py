"""Reference solution, stage 18b - int8 weight-only GEMV."""

import torch
import torch.nn as nn

from app.s18_quantization import quantize_int8_per_channel
from cudalib import build

SOURCE = "app/cuda/s18b_gemv_int8.cu"


def _ext():
    return build("s18b_gemv_int8", SOURCE)


def gemv_int8(x, w_q, scales):
    return _ext().gemv_int8(x.contiguous(), w_q, scales)


class QuantizedLinearCUDA(nn.Module):
    """QuantizedLinear from stage 18, with your kernel underneath.

    GEMV_MAX_ROWS is measured, not chosen. See the crossover table that
    test_where_the_crossover_is prints: one row and the kernel wins by 3x,
    four rows and cuBLAS wins by 6x.
    """

    GEMV_MAX_ROWS = 1

    def __init__(self, q, scales, bias, in_features, out_features):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_linear(cls, lin):
        q, s = quantize_int8_per_channel(lin.weight.data.float())
        bias = None if lin.bias is None else lin.bias.data.clone()
        return cls(q.contiguous(), s.float(), bias,
                   lin.in_features, lin.out_features)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if flat.shape[0] <= self.GEMV_MAX_ROWS:
            y = gemv_int8(flat, self.q, self.scales)
        else:
            w = self.q.to(x.dtype) * self.scales.to(x.dtype)[:, None]
            y = flat @ w.t()
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shape[:-1], self.out_features)

    def nbytes(self):
        return self.q.numel() + self.scales.numel() * 4
