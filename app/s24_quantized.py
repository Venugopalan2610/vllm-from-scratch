"""Stage 24 - quantized weights in the engine.

`./vc lore 24` for the insight. `./vc test 24` to check yourself.

Stage 18 quantized weights to int8 and kept the accuracy. Stage 18b wrote the
GEMV that reads them. Neither one ran inside the engine. Here your int8
weights go into the capstone model, and the decode step reads half the bytes.

tvllm accepts a weight that is a tensor OR anything callable (see
`tvllm.model.linear`). So you replace each matmul weight with an object whose
__call__ does the matmul, and the model does not know.

WHAT YOU ARE BUILDING

    MAX_ROWS                                   the crossover, MEASURED
    Int8Linear(bf16_weight, max_rows=MAX_ROWS) a callable weight for tvllm
        __call__(inputs)       (rows, in) -> (rows, out)
        dequantized_weight()   the weight that the int8 path multiplies by
        nbytes()               the bytes that a small-batch step reads
    quantize_model(model, max_rows=MAX_ROWS) -> number of weights replaced
        every qkv, o, gate_up and down weight, and the lm_head
    perplexity(model, token_ids) -> float

DISPATCH ON THE BATCH, BECAUSE YOUR KERNEL HAS A CROSSOVER

At a few rows, your stage 18b GEMV reads half the bytes of bf16 and wins. At
many rows, cuBLAS in bf16 wins, because the rows share the weight read and
the arithmetic takes over. So Int8Linear keeps BOTH weights:

    rows <= max_rows   ->  your int8 GEMV
    rows >  max_rows   ->  F.linear on the bf16 weight

That buys latency, not capacity: the bf16 copy stays in memory. To also save
the memory, you need an int8 kernel that is fast at every batch size, like
Marlin in vLLM. LORE.md section 11 lists that as a stretch goal.

MEASURE MAX_ROWS INSIDE THE MODEL

Do not measure it on one matrix in a loop. One matrix of a small model fits
in the L2 cache, so cuBLAS reads it from L2, and your crossover is wrong. Time
the whole graphed decode step (stage 23) at 1, 2, 4 and 8 sequences, with
max_rows large and with max_rows 0. A check prints that table.

TRAPS

  - quantize_int8_per_channel (stage 18) gives one scale for each OUTPUT row.
    Quantize in float32, and keep the scales in float32.
  - The lm_head is a quarter of the bytes of a small model. Measure its effect
    on perplexity, then decide. Per-channel int8 barely moves it.
  - The dispatch happens inside a captured graph. Each bucket has one row
    count, so each bucket takes one path, always. That is correct.
"""

import math

import torch
import torch.nn.functional as F

from app.s18_quantization import quantize_int8_per_channel
from app.s18b_gemv_cuda import gemv_int8
from tvllm import DenseReference, LayerWeights

# Measured in the graphed decode step of the real model: the int8 GEMV wins
# up to 4 rows, and cuBLAS in bf16 wins above that.
MAX_ROWS = 4


class Int8Linear:
    """A callable weight for tvllm. Your int8 GEMV at a small batch, and the
    bf16 weight above the crossover. This buys latency, not capacity: both
    copies stay in memory."""

    def __init__(self, bf16_weight, max_rows=MAX_ROWS):
        raise NotImplementedError("stage 24: implement Int8Linear.__init__")

    def __call__(self, inputs):
        raise NotImplementedError("stage 24: implement Int8Linear.__call__")

    def dequantized_weight(self):
        """The weight that the int8 path really multiplies by."""
        raise NotImplementedError("stage 24: implement Int8Linear.dequantized_weight")

    def nbytes(self):
        """The bytes that a small-batch step reads."""
        raise NotImplementedError("stage 24: implement Int8Linear.nbytes")


def quantize_model(model, max_rows=MAX_ROWS):
    """Replace every matmul weight of a tvllm model, the lm_head too.
    -> the number of weights replaced."""
    raise NotImplementedError("stage 24: implement quantize_model")


@torch.no_grad()
def perplexity(model, token_ids):
    """exp(mean cross-entropy) of a tvllm model on one sequence."""
    raise NotImplementedError("stage 24: implement perplexity")
