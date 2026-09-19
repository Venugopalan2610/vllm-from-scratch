"""Reference solution, stage 18 - weight-only quantization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

INT8_MAX = 127.0
FP8_E4M3_MAX = 448.0
TINY = 1e-8


def quantize_int8_per_channel(weight):
    """weight (out, in) -> (int8 weight, scales (out,)). Symmetric, one scale
    for each output row."""
    scales = weight.abs().amax(dim=1).clamp(min=TINY) / INT8_MAX
    int8_weight = torch.round(weight / scales.unsqueeze(1))
    int8_weight = int8_weight.clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return int8_weight, scales.to(torch.float32)


def dequantize_int8(int8_weight, scales):
    return int8_weight.to(torch.float32) * scales.unsqueeze(1)


class QuantizedLinear(nn.Module):
    """int8 weights, dequantized before the matmul. It reads half the
    bytes."""

    def __init__(self, int8_weight, scales, bias=None,
                 out_dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.out_dtype = out_dtype

    @classmethod
    def from_linear(cls, linear):
        int8_weight, scales = quantize_int8_per_channel(
            linear.weight.data.float())
        bias = None if linear.bias is None else linear.bias.data.clone()
        return cls(int8_weight, scales, bias, out_dtype=linear.weight.dtype)

    def forward(self, inputs):
        weight = (self.int8_weight.to(inputs.dtype)
                  * self.scales.to(inputs.dtype).unsqueeze(1))
        outputs = F.linear(inputs, weight)
        if self.bias is not None:
            outputs = outputs + self.bias.to(outputs.dtype)
        return outputs

    def nbytes(self):
        return self.int8_weight.numel() + self.scales.numel() * 4


def quantize_fp8(tensor):
    """FP8 (e4m3) with one scale for the tensor. Ada (sm_89) has the dtype in
    hardware."""
    amax = tensor.abs().amax().clamp(min=TINY).float()
    scale = amax / FP8_E4M3_MAX
    fp8_tensor = (tensor.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return fp8_tensor.to(torch.float8_e4m3fn), scale


def dequantize_fp8(fp8_tensor, scale, dtype=torch.bfloat16):
    return (fp8_tensor.to(torch.float32) * scale).to(dtype)


def quantize_model_(model, skip=("lm_head",)):
    """Replace each nn.Linear with a QuantizedLinear, in place.
    -> the number replaced."""
    targets = [(parent, child_name)
               for parent_name, parent in model.named_modules()
               for child_name, child in parent.named_children()
               if isinstance(child, nn.Linear)
               and not any(part in f"{parent_name}.{child_name}"
                           for part in skip)]
    for parent, child_name in targets:
        linear = getattr(parent, child_name)
        setattr(parent, child_name, QuantizedLinear.from_linear(linear))
    return len(targets)


@torch.inference_mode()
def perplexity(model, tokenizer, text, max_len=512):
    token_ids = tokenizer(text, return_tensors="pt").input_ids[:, :max_len]
    token_ids = token_ids.to(next(model.parameters()).device)
    logits = model(token_ids).logits[:, :-1].float()
    targets = token_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           targets.reshape(-1))
    return float(torch.exp(loss))
