"""Stage 18b - int8 weight-only GEMV, dequantized after the dot product.

The spec is in app/cuda/s18b_gemv_int8.cu.

Stage 18 is the oracle of the arithmetic: dequantize_int8 and then a matmul
is what this kernel must agree with. The gates: the fused version is faster
than the unfused one, and it is faster than bf16 cuBLAS at batch 1.
"""

import pytest
import torch
import torch.nn as nn

from app.s18_quantization import dequantize_int8, quantize_int8_per_channel
from app.s18b_gemv_cuda import QuantizedLinearCUDA, gemv_int8


def dequantized_matmul(inputs, int8_weight, scales):
    """The stage 18 oracle, in fp32."""
    return inputs.float() @ dequantize_int8(int8_weight, scales).t()


def random_int8_weight(out_features, in_features, device, seed=0):
    """-> (the float weight, its int8 weight, its scales)."""
    torch.manual_seed(seed)
    weight = torch.randn(out_features, in_features, device=device)
    return (weight, *quantize_int8_per_channel(weight))


def random_inputs(batch_size, in_features, device, dtype=torch.float32):
    return torch.randn(batch_size, in_features, device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16,
                                   torch.bfloat16])
@pytest.mark.parametrize("batch_size,out_features,in_features",
                         [(1, 512, 1024), (4, 4096, 2048), (8, 1024, 768)])
def test_matches_dequantize_then_matmul(nvcc, device, dtype, batch_size,
                                        out_features, in_features):
    """The oracle is stage 18: the same numbers, with no bf16 weight."""
    _, int8_weight, scales = random_int8_weight(out_features, in_features,
                                                device)
    inputs = random_inputs(batch_size, in_features, device, dtype)
    expected = dequantized_matmul(inputs, int8_weight, scales).to(dtype).float()
    output = gemv_int8(inputs, int8_weight, scales).float()
    error = ((output - expected).abs().max() / expected.abs().max()).item()
    assert error < (1e-3 if dtype == torch.float32 else 2e-2), error


@pytest.mark.parametrize("batch_size,out_features,in_features",
                         [(1, 151, 1000), (2, 33, 17), (3, 7, 5)])
def test_shapes_that_do_not_divide_by_anything(nvcc, device, batch_size,
                                               out_features, in_features):
    """If in_features is not a multiple of 16, there are no 16-byte loads. If
    out_features is not a multiple of the rows of a block, there is a partial
    block. Both must still be correct. The partial block must not stop at a
    barrier that its empty warps skipped."""
    _, int8_weight, scales = random_int8_weight(out_features, in_features,
                                                device, seed=1)
    inputs = random_inputs(batch_size, in_features, device)
    torch.testing.assert_close(gemv_int8(inputs, int8_weight, scales),
                               dequantized_matmul(inputs, int8_weight, scales),
                               rtol=1e-3, atol=1e-3)


def test_accepts_a_bare_vector(nvcc, device):
    """Decode gives you one row. The caller must not have to make it
    (1, in_features)."""
    _, int8_weight, scales = random_int8_weight(512, 1024, device)
    vector = torch.randn(1024, device=device)
    output = gemv_int8(vector, int8_weight, scales)
    assert output.shape == (512,)
    torch.testing.assert_close(
        output, dequantized_matmul(vector.unsqueeze(0), int8_weight,
                                   scales).squeeze(0),
        rtol=1e-3, atol=1e-3)


def test_accumulates_in_fp32(nvcc, device):
    """A bf16 sum of 4096 int8 * bf16 products loses the low bits."""
    in_features = 4096
    _, int8_weight, scales = random_int8_weight(2048, in_features, device,
                                                seed=2)
    inputs = random_inputs(1, in_features, device, torch.bfloat16)
    expected = dequantized_matmul(inputs, int8_weight, scales)
    output = gemv_int8(inputs, int8_weight, scales).float()
    error = ((output - expected).abs().mean() / expected.abs().mean()).item()
    print(f"\n  mean relative error over in_features={in_features}: {error:.5f}")
    assert error < 5e-3


def test_the_fusion_is_the_point(nvcc, device):
    """The gate that justifies the stage.

    A dequantize of the weights and then a matmul does the same arithmetic.
    It is slower, because it writes out_features * in_features bf16 values
    to memory and reads them back. That is more traffic than the int8
    weights cost at the start.
    """
    import cudalib

    _, int8_weight, scales = random_int8_weight(11008, 4096, device)
    bf16_scales = scales.to(torch.bfloat16)
    inputs = random_inputs(1, 4096, device, torch.bfloat16)
    unfused_ms, fused_ms = cudalib.compare_ms(
        lambda: inputs @ (int8_weight.to(torch.bfloat16)
                          * bf16_scales[:, None]).t(),
        lambda: gemv_int8(inputs, int8_weight, scales))

    print(f"\n  dequantize, then matmul   {unfused_ms:>8.3f} ms")
    print(f"  fused                     {fused_ms:>8.3f} ms   "
          f"\033[1m{unfused_ms / fused_ms:.1f}x\033[0m")
    assert unfused_ms / fused_ms > 3.0, (
        f"only {unfused_ms / fused_ms:.1f}x. If the dequantize is fused, the "
        "bf16 weight is never written.")


def _cublas_and_gemv_ms(weight, int8_weight, scales, batch_size, device):
    import cudalib

    bf16_weight = weight.to(torch.bfloat16)
    inputs = random_inputs(batch_size, weight.shape[1], device, torch.bfloat16)
    return cudalib.compare_ms(lambda: inputs @ bf16_weight.t(),
                              lambda: gemv_int8(inputs, int8_weight, scales))


def test_it_beats_bf16_at_batch_one(nvcc, device):
    """The gate that justifies quantization.

    Half the weight bytes must be most of half the time. cuBLAS reads a bf16
    matrix. You read an int8 one.
    """
    rows = []
    for out_features, in_features in ((4096, 4096), (11008, 4096)):
        weight, int8_weight, scales = random_int8_weight(out_features,
                                                         in_features, device)
        rows.append((out_features, in_features,
                     *_cublas_and_gemv_ms(weight, int8_weight, scales, 1,
                                          device)))

    print(f"\n  {'N':>6} {'K':>6} {'bf16 cuBLAS':>13} {'your int8':>11} "
          f"{'gain':>7}")
    for out_features, in_features, cublas_ms, gemv_ms in rows:
        print(f"  {out_features:>6} {in_features:>6} {cublas_ms:>11.3f}ms "
              f"{gemv_ms:>9.3f}ms {cublas_ms / gemv_ms:>6.2f}x")
    worst = min(cublas_ms / gemv_ms for _, _, cublas_ms, gemv_ms in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x over bf16. Half the bytes must be most of half "
        "the time. Make sure that the weights go directly to registers and "
        "never through shared memory.")


def test_where_the_crossover_is(nvcc, device):
    """Not a gate. The measurement that sets GEMV_MAX_ROWS.

    A GEMV is not a small GEMM. Your kernel has the shape for a few
    activation rows. cuBLAS reads the weights one time for EVERY batch size
    and keeps the tensor cores busy. After a few rows, cuBLAS is faster, int8
    or not.

    This is stage 03 again. Prefill and decode are different machines, and
    this is the boundary between them, in milliseconds.
    """
    weight, int8_weight, scales = random_int8_weight(11008, 4096, device)
    print(f"\n  {'rows':>5} {'bf16 cuBLAS':>13} {'your int8':>11} {'gain':>7}")
    crossover = None
    for batch_size in (1, 2, 4, 8):
        cublas_ms, gemv_ms = _cublas_and_gemv_ms(weight, int8_weight, scales,
                                                 batch_size, device)
        cublas_faster = cublas_ms <= gemv_ms
        if cublas_faster and crossover is None:
            crossover = batch_size
        marker = "   <- cuBLAS is faster from here" if cublas_faster else ""
        print(f"  {batch_size:>5} {cublas_ms:>11.3f}ms {gemv_ms:>9.3f}ms "
              f"{cublas_ms / gemv_ms:>6.2f}x{marker}")
    print(f"\n  \033[2mQuantizedLinearCUDA.GEMV_MAX_ROWS is "
          f"{QuantizedLinearCUDA.GEMV_MAX_ROWS}. The crossover measured here "
          f"is {crossover}.\033[0m")


def test_the_linear_dispatches_on_batch_size(nvcc, device):
    """A large batch must not go through the GEMV path.

    Both paths must agree, so that a request does not get different numbers
    when a different number of requests is in progress with it.
    """
    linear = nn.Linear(512, 256).to(device)
    quantized = QuantizedLinearCUDA.from_linear(linear).to(device)
    small = torch.randn(1, 512, device=device)
    large = torch.randn(64, 512, device=device)

    assert quantized(small).shape == (1, 256)
    assert quantized(large).shape == (64, 256)
    # The same row, through both paths.
    torch.testing.assert_close(quantized(large[:1]), quantized(large)[:1],
                               rtol=2e-2, atol=2e-2)
    # Near the layer with no quantization.
    exact = linear(small)
    error = ((quantized(small) - exact).abs().mean()
             / exact.abs().mean()).item()
    print(f"\n  mean relative error against the fp32 Linear: {error:.4f}")
    assert error < 0.05


def test_it_actually_stores_int8(nvcc, device):
    """A layer that quantizes and keeps a bf16 copy saved nothing, and
    nbytes() must tell the truth about it."""
    linear = nn.Linear(4096, 4096, bias=False).to(device)
    quantized = QuantizedLinearCUDA.from_linear(linear).to(device)
    dense_bytes = linear.weight.numel() * linear.weight.element_size()
    print(f"\n  fp32 Linear   {dense_bytes / 1e6:>7.2f} MB")
    print(f"  int8 version  {quantized.nbytes() / 1e6:>7.2f} MB   "
          f"\033[1m{dense_bytes / quantized.nbytes():.1f}x smaller\033[0m")
    assert quantized.nbytes() < dense_bytes / 3.5
