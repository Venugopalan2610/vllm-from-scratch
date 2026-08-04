"""Stage 18 - weight-only quantization.

`./vc lore 18` for the insight. `./vc test 18` to check yourself.

Decode reads every weight to produce every token (stage 01's fact), so decode
time is proportional to WEIGHT BYTES. Halve the bytes, nearly halve the time.
That is the entire argument, and it is why quantization is a serving technique
and not just a memory trick.

Weight-only, per-output-channel, symmetric INT8:

    scale_i = max(|W[i, :]|) / 127          one scale per output row
    q[i, :] = round(W[i, :] / scale_i)

Per-CHANNEL rather than per-tensor matters: one outlier row would otherwise
crush the resolution of every other row.

Activations stay in bf16. Only the weights are stored small, and they are
dequantized in the kernel epilogue. That is what "weight-only" means, and it is
why accuracy holds up so well -- activation outliers, which are what actually
break naive quantization, never get quantized at all.

Then do the same to the KV cache, which is the other big reader.
"""

import torch
import torch.nn as nn


def quantize_int8_per_channel(W):
    """W (out, in) -> (int8 tensor, float32 scales (out,)).

    Clamp the scale away from zero, or an all-zero row gives you inf.
    """
    raise NotImplementedError("stage 18: implement quantize_int8_per_channel")


def dequantize_int8(q, scales):
    """q.float() * scales[:, None]"""
    raise NotImplementedError("stage 18: implement dequantize_int8")


class QuantizedLinear(nn.Module):
    """Drop-in nn.Linear replacement holding INT8 weights.

    Required:
        from_linear(nn.Linear) -> QuantizedLinear     classmethod
        forward(x)                                    matches the original
        nbytes() -> int                               storage cost
    """

    @classmethod
    def from_linear(cls, lin):
        raise NotImplementedError("stage 18: implement QuantizedLinear.from_linear")


def quantize_fp8(t):
    """Per-tensor FP8 (e4m3): returns (float8_e4m3fn tensor, float scale).

    Your GPU is sm_89, so torch.float8_e4m3fn is native. The max representable
    magnitude is 448, so scale = amax / 448.

    Unlike INT8, FP8 keeps an exponent, so it handles the wide dynamic range of
    KV cache entries much better than an integer format at the same width.
    """
    raise NotImplementedError("stage 18: implement quantize_fp8")


def dequantize_fp8(q, scale, dtype=torch.bfloat16):
    raise NotImplementedError("stage 18: implement dequantize_fp8")


def quantize_model_(model, skip=("lm_head",)):
    """Replace every nn.Linear with a QuantizedLinear, in place. Return the count.

    Skip lm_head: it is huge, it feeds directly into the sampler, and
    quantizing it costs more accuracy than it saves bytes.
    """
    raise NotImplementedError("stage 18: implement quantize_model_")


@torch.inference_mode()
def perplexity(model, tokenizer, text, max_len=512):
    """exp(mean cross-entropy of next-token prediction).

    Your accuracy guard. Compute logits, shift by one, cross-entropy against
    the true next tokens, exponentiate. Any quantization scheme must move this
    number only slightly, or you have traded correctness for speed.
    """
    raise NotImplementedError("stage 18: implement perplexity")
