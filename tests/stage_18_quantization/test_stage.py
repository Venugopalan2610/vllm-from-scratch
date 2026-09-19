"""Stage 18 - weight-only quantization.

The spec is in app/s18_quantization.py. The accuracy guard runs the real
model on a serving-shaped workload: prompts of ISL tokens, then OSL generated
tokens. It compares the choices of the int8 model with those of bf16 at each
output position.
"""

import torch
import torch.nn as nn

import math

from app.s18_quantization import (
    QuantizedLinear,
    continuation_logits,
    dequantize_fp8,
    dequantize_int8,
    fidelity,
    quantize_fp8,
    quantize_int8_per_channel,
    quantize_model_,
)
from tests.helpers import ISL, MODEL, OSL, hf_workload

# Measured on Qwen3-0.6B over the ISL/OSL workload. The reference solution:
# the same top token at 96.5% of the positions, and 0.004 nats of KL. One
# scale for each TENSOR, the classic mistake: 92% and 0.037 nats. The
# top-token rate hardly moves, and KL moves 10x. So KL is the gate.
#
# Two bf16 runs can differ only in the order of the additions, for example
# with and without a KV cache. They already disagree at about 2% of the
# positions. So a top-1 floor near 100% fails on noise.
TOP1_FLOOR = 0.90
KL_CEILING = 0.015


def relative_error(approximation, exact):
    """The mean absolute error over the mean absolute value."""
    return ((approximation.float() - exact.float()).abs().mean()
            / exact.float().abs().mean()).item()


def test_int8_roundtrip_is_close(device):
    torch.manual_seed(0)
    weight = torch.randn(64, 128, device=device)
    int8_weight, scales = quantize_int8_per_channel(weight)
    assert int8_weight.dtype == torch.int8
    assert scales.shape == (64,)
    error = relative_error(dequantize_int8(int8_weight, scales), weight)
    print(f"\n  int8 per-channel mean relative error: {error:.4f}")
    assert error < 0.01


def test_per_channel_beats_per_tensor_with_an_outlier_row(device):
    """One very large row must not destroy the resolution of all the
    others."""
    torch.manual_seed(0)
    weight = torch.randn(16, 128, device=device)
    weight[0] *= 500.0                               # the outlier channel

    int8_weight, scales = quantize_int8_per_channel(weight)
    per_channel_error = (dequantize_int8(int8_weight, scales)[1:]
                         - weight[1:]).abs().mean()
    tensor_scale = weight.abs().max() / 127.0        # one scale for the tensor
    per_tensor = torch.round(weight / tensor_scale).clamp(-127, 127) * tensor_scale
    per_tensor_error = (per_tensor[1:] - weight[1:]).abs().mean()

    print(f"\n  error on the normal rows: per channel {per_channel_error:.5f}, "
          f"per tensor {per_tensor_error:.5f}")
    assert per_channel_error < per_tensor_error / 5


def test_zero_row_does_not_produce_nan(device):
    weight = torch.zeros(4, 32, device=device)
    weight[1] = torch.randn(32, device=device)
    assert torch.isfinite(dequantize_int8(*quantize_int8_per_channel(
        weight))).all(), "clamp the scale above zero"


def test_quantized_linear_matches_the_original(device):
    torch.manual_seed(0)
    linear = nn.Linear(256, 128, device=device, dtype=torch.bfloat16)
    inputs = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
    expected = linear(inputs).float()
    output = QuantizedLinear.from_linear(linear)(inputs).float()
    error = ((output - expected).norm() / expected.norm()).item()
    print(f"\n  QuantizedLinear relative error: {error:.4f}")
    assert error < 0.02


def test_quantized_linear_keeps_the_bias(device):
    linear = nn.Linear(32, 16, bias=True, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.bias.fill_(5.0)
        linear.weight.zero_()
    output = QuantizedLinear.from_linear(linear)(
        torch.randn(2, 32, device=device, dtype=torch.bfloat16)).float()
    assert torch.allclose(output, torch.full_like(output, 5.0), atol=1e-2)


def test_int8_halves_the_storage(device):
    linear = nn.Linear(1024, 1024, bias=False, device=device,
                       dtype=torch.bfloat16)
    int8_bytes = QuantizedLinear.from_linear(linear).nbytes()
    bf16_bytes = linear.weight.numel() * 2
    print(f"\n  bf16: {bf16_bytes / 1e6:.2f} MB -> int8: {int8_bytes / 1e6:.2f} MB")
    assert int8_bytes < bf16_bytes * 0.55


# ---- the FP8 KV cache -----------------------------------------------

def test_fp8_roundtrip(device):
    torch.manual_seed(0)
    keys = torch.randn(4, 8, 128, 128, device=device, dtype=torch.bfloat16)
    fp8_keys, scale = quantize_fp8(keys)
    assert fp8_keys.dtype == torch.float8_e4m3fn
    error = relative_error(dequantize_fp8(fp8_keys, scale), keys)
    print(f"\n  fp8 KV mean relative error: {error:.4f}")
    print(f"  bytes: {keys.numel() * 2} -> {fp8_keys.numel()}  (2x smaller)")
    assert error < 0.05
    assert fp8_keys.numel() == keys.numel()


def test_fp8_handles_wide_dynamic_range(device):
    """KV values cover many orders of magnitude. An exponent field is worth
    its bits."""
    values = torch.tensor([[1e-3, 1.0, 50.0, 0.02]], device=device,
                          dtype=torch.bfloat16)
    restored = dequantize_fp8(*quantize_fp8(values)).float()
    worst = ((restored - values.float()).abs() / values.float().abs()).max().item()
    print(f"\n  worst relative error across 5 orders of magnitude: {worst:.4f}")
    assert worst < 0.15


# ---- the accuracy guard ---------------------------------------------

def test_a_model_agrees_with_itself(device):
    logits = torch.randn(6, 50, device=device)
    result = fidelity(logits, logits)
    assert result.top1_agreement == 1.0
    assert abs(result.mean_kl) < 1e-6


def test_fidelity_counts_choices_and_divergence(device):
    """Two positions. At the first the choice changes, at the second the
    distributions are the same. KL((0.5, 0.5) || (0.25, 0.75)) = ln(4/3) / 2."""
    reference = torch.tensor([[0.0, 0.0], [2.0, 0.0]], device=device)
    candidate = torch.tensor([[0.0, math.log(3.0)], [2.0, 0.0]], device=device)
    result = fidelity(reference, candidate)
    assert result.top1_agreement == 0.5
    assert abs(result.mean_kl - math.log(4 / 3) / 2 / 2) < 1e-5


def test_continuation_logits_predict_the_output_tokens(hf):
    model, tokenizer = hf
    token_ids = tokenizer("The capital of France is Paris, and").input_ids
    prompt_len = 5
    logits = continuation_logits(model, token_ids, prompt_len)
    assert logits.shape == (len(token_ids) - prompt_len, model.config.vocab_size)
    with torch.no_grad():
        full = model(torch.tensor([token_ids], device=model.device)).logits[0]
    torch.testing.assert_close(logits[0], full[prompt_len - 1].float())


def test_quantizing_the_whole_model_keeps_its_choices(device):
    """The important check. The real model, a serving-shaped workload, and
    the choices of the model at each generated position."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device).eval()
    sequences = hf_workload(model, tokenizer)

    before = torch.cat([continuation_logits(model, ids, ISL) for ids in sequences])
    num_replaced = quantize_model_(model)
    after = torch.cat([continuation_logits(model, ids, ISL) for ids in sequences])
    result = fidelity(before, after)

    print(f"\n  quantized {num_replaced} Linear layers")
    print(f"  {len(sequences)} prompts, ISL {ISL}, OSL {OSL}: the same top token "
          f"at {100 * result.top1_agreement:.1f}% of the output positions, "
          f"mean KL {result.mean_kl:.4f} nats")
    assert num_replaced > 100, (
        f"replaced only {num_replaced} layers. Did the walk find all of them?")
    assert result.top1_agreement >= TOP1_FLOOR and result.mean_kl <= KL_CEILING, (
        "the int8 model makes different choices. Look at the scales for "
        "each CHANNEL, and make sure that the activations stay in bf16.")
    print("\n  \033[2mHalf the weight bytes, and almost the same choices. The")
    print("  decode time is proportional to the weight bytes, so this is")
    print("  almost a free 2x. That is why every production deployment does")
    print("  it.\033[0m")
