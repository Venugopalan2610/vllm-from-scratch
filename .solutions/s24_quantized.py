"""Reference solution, stage 24 - quantized weights in the engine."""

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
        int8_weight, scales = quantize_int8_per_channel(bf16_weight.float())
        self.int8_weight = int8_weight.contiguous()
        self.scales = scales.float().contiguous()
        self.bf16_weight = bf16_weight
        self.max_rows = max_rows

    def __call__(self, inputs):
        if inputs.shape[0] <= self.max_rows:
            return gemv_int8(inputs, self.int8_weight, self.scales)
        return F.linear(inputs, self.bf16_weight)

    def dequantized_weight(self):
        """The weight that the int8 path really multiplies by."""
        return (self.int8_weight.to(self.bf16_weight.dtype)
                * self.scales.to(self.bf16_weight.dtype)[:, None])

    def nbytes(self):
        """The bytes that a small-batch step reads."""
        return self.int8_weight.numel() + self.scales.numel() * 4


def quantize_model(model, max_rows=MAX_ROWS):
    """Replace every matmul weight of a tvllm model, the lm_head too.
    -> the number of weights replaced."""
    for layer in model.layers:
        for name in LayerWeights.MATMULS:
            setattr(layer, name, Int8Linear(getattr(layer, name), max_rows))
    model.lm_head = Int8Linear(model.lm_head, max_rows)
    return len(model.layers) * len(LayerWeights.MATMULS) + 1


@torch.no_grad()
def continuation_logits(model, token_ids, prompt_len):
    """The float32 logits that predict each output token of a tvllm model."""
    tokens = torch.tensor(token_ids, device=model.device)
    positions = torch.arange(len(token_ids), device=model.device)
    output_rows = torch.arange(prompt_len - 1, len(token_ids) - 1, device=model.device)
    logits = model.forward(tokens, positions,
                           DenseReference(model.config.num_layers), output_rows)
    return logits.float()
