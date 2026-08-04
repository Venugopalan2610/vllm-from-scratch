"""Stage 18 - Weight-only quantization.

Spec in app/s18_quantization.py. The accuracy guard is a real perplexity
measurement on the real model, not a tolerance on random tensors.
"""

import pytest
import torch
import torch.nn as nn

from app.s18_quantization import (
    QuantizedLinear,
    dequantize_fp8,
    dequantize_int8,
    perplexity,
    quantize_fp8,
    quantize_int8_per_channel,
    quantize_model_,
)

TEXT = (
    "The history of computing began with mechanical calculators. "
    "Charles Babbage designed the Analytical Engine in 1837, and Ada Lovelace "
    "wrote what is considered the first computer program for it. "
) * 6


def test_int8_roundtrip_is_close(dev):
    torch.manual_seed(0)
    W = torch.randn(64, 128, device=dev)
    q, s = quantize_int8_per_channel(W)
    assert q.dtype == torch.int8
    assert s.shape == (64,)
    rel = ((dequantize_int8(q, s) - W).abs().mean() / W.abs().mean()).item()
    print(f"\n  int8 per-channel mean relative error: {rel:.4f}")
    assert rel < 0.01


def test_per_channel_beats_per_tensor_with_an_outlier_row(dev):
    """One huge row must not destroy the resolution of all the others."""
    torch.manual_seed(0)
    W = torch.randn(16, 128, device=dev)
    W[0] *= 500.0                                   # outlier channel

    q, s = quantize_int8_per_channel(W)
    per_ch = (dequantize_int8(q, s)[1:] - W[1:]).abs().mean()

    scale = W.abs().max() / 127.0                   # naive per-tensor
    per_t = ((torch.round(W / scale).clamp(-127, 127) * scale)[1:]
             - W[1:]).abs().mean()

    print(f"\n  error on the normal rows: per-channel {per_ch:.5f} vs "
          f"per-tensor {per_t:.5f}")
    assert per_ch < per_t / 5


def test_zero_row_does_not_produce_nan(dev):
    W = torch.zeros(4, 32, device=dev)
    W[1] = torch.randn(32, device=dev)
    q, s = quantize_int8_per_channel(W)
    out = dequantize_int8(q, s)
    assert torch.isfinite(out).all(), "clamp the scale away from zero"


def test_quantized_linear_matches_the_original(dev):
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, device=dev, dtype=torch.bfloat16)
    x = torch.randn(8, 256, device=dev, dtype=torch.bfloat16)
    ql = QuantizedLinear.from_linear(lin)
    want, got = lin(x), ql(x)
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    print(f"\n  QuantizedLinear relative error: {rel:.4f}")
    assert rel < 0.02


def test_quantized_linear_keeps_the_bias(dev):
    lin = nn.Linear(32, 16, bias=True, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.bias.fill_(5.0)
        lin.weight.zero_()
    ql = QuantizedLinear.from_linear(lin)
    out = ql(torch.randn(2, 32, device=dev, dtype=torch.bfloat16))
    assert torch.allclose(out.float(), torch.full_like(out.float(), 5.0), atol=1e-2)


def test_int8_halves_the_storage(dev):
    lin = nn.Linear(1024, 1024, bias=False, device=dev, dtype=torch.bfloat16)
    ql = QuantizedLinear.from_linear(lin)
    fp16_bytes = lin.weight.numel() * 2
    print(f"\n  bf16: {fp16_bytes/1e6:.2f} MB -> int8: {ql.nbytes()/1e6:.2f} MB")
    assert ql.nbytes() < fp16_bytes * 0.55


# ---- FP8 KV cache ---------------------------------------------------

def test_fp8_roundtrip(dev):
    torch.manual_seed(0)
    k = torch.randn(4, 8, 128, 128, device=dev, dtype=torch.bfloat16)
    q, s = quantize_fp8(k)
    assert q.dtype == torch.float8_e4m3fn
    dk = dequantize_fp8(q, s)
    rel = ((dk.float() - k.float()).abs().mean()
           / k.float().abs().mean()).item()
    print(f"\n  fp8 KV mean relative error: {rel:.4f}")
    print(f"  bytes: {k.numel() * 2} -> {q.numel()}  (2x smaller)")
    assert rel < 0.05
    assert q.numel() == k.numel()


def test_fp8_handles_wide_dynamic_range(dev):
    """KV entries span orders of magnitude; an exponent field earns its keep."""
    t = torch.tensor([[1e-3, 1.0, 50.0, 0.02]], device=dev, dtype=torch.bfloat16)
    q, s = quantize_fp8(t)
    back = dequantize_fp8(q, s).float()
    rel = ((back - t.float()).abs() / t.float().abs()).max().item()
    print(f"\n  worst relative error across 5 orders of magnitude: {rel:.4f}")
    assert rel < 0.15


# ---- the accuracy guard ---------------------------------------------

def test_perplexity_is_sane(hf):
    model, tok = hf
    ppl = perplexity(model, tok, TEXT)
    print(f"\n  bf16 perplexity on the sample text: {ppl:.3f}")
    assert 1.0 < ppl < 100.0, f"perplexity {ppl} is not plausible"


def test_quantizing_the_whole_model_barely_moves_perplexity(dev):
    """The test that actually matters. Real model, real text, real quality."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(dev).eval()

    before = perplexity(model, tok, TEXT)
    n = quantize_model_(model)
    after = perplexity(model, tok, TEXT)
    delta = (after - before) / before

    print(f"\n  quantized {n} Linear layers")
    print(f"  perplexity  bf16 {before:.4f}  ->  int8 {after:.4f}  "
          f"({delta * 100:+.2f}%)")

    assert n > 100, f"only replaced {n} layers -- did the walk find them all?"
    assert abs(delta) < 0.05, (
        f"perplexity moved {delta*100:+.1f}%. Check per-CHANNEL scales, and "
        "that activations stay in bf16."
    )
    print("\n  \033[2mHalf the weight bytes for under 1% quality. Decode time is")
    print("  proportional to weight bytes, so this is close to a free 2x --")
    print("  which is why every production deployment does it.\033[0m")

    del model
    torch.cuda.empty_cache()
