"""Stage 18 - weight-only quantization.

`./vc lore 18` for the insight. `./vc test 18` to check yourself.

Decode reads every weight to make every token (the fact of stage 01), so
the decode time is proportional to the WEIGHT BYTES. Make the bytes half as
many, and the time is almost half. That is the whole argument. It is why
quantization is a serving method, not only a memory method.

Weight-only, symmetric int8, with one scale for each output channel:

    scale_i = max(|W[i, :]|) / 127          one scale for each output row
    q[i, :] = round(W[i, :] / scale_i)

One scale for each CHANNEL, not for the tensor, is important: if not, one
row with outliers destroys the resolution of every other row.

The activations stay in bf16. Only the weights become small, and the kernel
dequantizes them after the dot product. That is the meaning of
"weight-only".

It is also why the accuracy stays good. Activation outliers break simple
quantization, and this method never quantizes an activation.

Then do the same to the KV cache, which is the other large reader.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


def quantize_int8_per_channel(weight):
    """weight (out, in) -> (int8 weight, float32 scales (out,)).

    Clamp the scale above zero. If not, a row of zeros gives inf.
    """
    raise NotImplementedError("stage 18: implement quantize_int8_per_channel")


def dequantize_int8(int8_weight, scales):
    """int8_weight.float() * scales[:, None]"""
    raise NotImplementedError("stage 18: implement dequantize_int8")


class QuantizedLinear(nn.Module):
    """An nn.Linear with int8 weights. It replaces nn.Linear with no other
    change. Keep the weight in a buffer named int8_weight.

    Required:
        from_linear(nn.Linear) -> QuantizedLinear     classmethod
        forward(inputs)                               matches the original
        nbytes() -> int                               storage cost
    """

    @classmethod
    def from_linear(cls, linear):
        raise NotImplementedError("stage 18: implement QuantizedLinear.from_linear")


def quantize_fp8(tensor):
    """FP8 (e4m3) with one scale for the tensor. -> (float8_e4m3fn tensor,
    float scale).

    Ada and newer GPUs have torch.float8_e4m3fn in hardware. The largest
    magnitude is 448, so scale = amax / 448.

    FP8 keeps an exponent, and int8 does not. So at the same width, FP8 holds
    the wide range of KV cache values much better than an integer format.
    """
    raise NotImplementedError("stage 18: implement quantize_fp8")


def dequantize_fp8(fp8_tensor, scale, dtype=torch.bfloat16):
    raise NotImplementedError("stage 18: implement dequantize_fp8")


def quantize_model_(model, skip=("lm_head",)):
    """Replace each nn.Linear with a QuantizedLinear, in place. Return the
    number replaced.

    Skip lm_head: it is large, its output goes directly to the sampler, and
    its quantization costs more accuracy than the bytes it saves.
    """
    raise NotImplementedError("stage 18: implement quantize_model_")


@torch.inference_mode()
def continuation_logits(model, token_ids, prompt_len):
    """The float32 logits that predict each output token. -> (OSL, vocab).

    token_ids is a prompt of prompt_len tokens, then its continuation of OSL
    tokens. Row i of the result is the prediction of token_ids[prompt_len + i].
    It comes from the position before that token, so the rows start at
    prompt_len - 1.

    This is the quality that a user sees: the tokens that the engine makes
    after the prompt, not a score over the prompt.
    """
    raise NotImplementedError("stage 18: implement continuation_logits")


@dataclass
class Fidelity:
    top1_agreement: float       # the fraction of positions with the same top token
    mean_kl: float              # the mean KL(reference || candidate), in nats


def fidelity(reference_logits, candidate_logits):
    """How close the candidate model stays to the reference, position by
    position. Both arguments are (positions, vocab). -> Fidelity.

    top1_agreement: the fraction of positions where the two models choose the
        same most likely token. A quantized engine must make the same choices.
    mean_kl: the mean of KL(P || Q), with P from the reference and Q from the
        candidate. It also finds a change that does not change the top token
        yet. Compute it from log_softmax in float32, not from probabilities:
        a probability of 1e-30 is 0 in bf16, and log(0) is -inf.

    Both models read the SAME tokens, so the difficulty of the text and its
    length cancel. That is why this metric is stable, and perplexity is not.
    """
    raise NotImplementedError("stage 18: implement fidelity")
