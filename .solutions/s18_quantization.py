"""Reference solution, stage 18 - weight-only quantization."""

import torch
import torch.nn as nn


def quantize_int8_per_channel(W):
    """W (out, in) -> (int8 tensor, scales (out,)). Symmetric, per output row."""
    scales = W.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(W / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q, scales.to(torch.float32)


def dequantize_int8(q, scales):
    return q.to(torch.float32) * scales.unsqueeze(1)


class QuantizedLinear(nn.Module):
    """INT8 weights, dequantized in the epilogue. Halves the bytes read."""

    def __init__(self, q, scales, bias=None, out_dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias if bias is not None else None)
        self.out_dtype = out_dtype

    @classmethod
    def from_linear(cls, lin):
        q, s = quantize_int8_per_channel(lin.weight.data.float())
        b = lin.bias.data.clone() if lin.bias is not None else None
        return cls(q, s, b, out_dtype=lin.weight.dtype)

    def forward(self, x):
        w = (self.q.to(x.dtype) * self.scales.to(x.dtype).unsqueeze(1))
        out = torch.nn.functional.linear(x, w, None)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def nbytes(self):
        return self.q.numel() + self.scales.numel() * 4


def quantize_fp8(t):
    """Per-tensor FP8 (e4m3). Ada (sm_89) supports the dtype natively."""
    amax = t.abs().amax().clamp(min=1e-8).float()
    scale = amax / 448.0                       # e4m3 max representable
    q = (t.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def dequantize_fp8(q, scale, dtype=torch.bfloat16):
    return (q.to(torch.float32) * scale).to(dtype)


def quantize_model_(model, skip=("lm_head",)):
    """Replace every nn.Linear with a QuantizedLinear, in place. Returns count."""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, QuantizedLinear.from_linear(child))
                n += 1
    return n


@torch.inference_mode()
def perplexity(model, tokenizer, text, max_len=512):
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :max_len]
    ids = ids.to(next(model.parameters()).device)
    out = model(ids)
    logits = out.logits[:, :-1].float()
    targets = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    return float(torch.exp(loss))
