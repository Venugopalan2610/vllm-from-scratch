"""Stage 18b - INT8 weight-only GEMV with a fused dequantize.

Spec in app/cuda/s18b_gemv_int8.cu.

Stage 18 is the oracle for the arithmetic: dequantize_int8 followed by a
matmul is what this kernel has to agree with. The gates are that the fused
version beats the unfused one, and that it beats bf16 cuBLAS at batch 1.
"""

import pytest
import torch
import torch.nn as nn

from app.s18_quantization import dequantize_int8, quantize_int8_per_channel
from app.s18b_gemv_cuda import QuantizedLinearCUDA, gemv_int8


def _ref(x, q, s):
    return (x.float() @ dequantize_int8(q, s).t()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16,
                                   torch.bfloat16])
@pytest.mark.parametrize("B,N,K", [(1, 512, 1024), (4, 4096, 2048),
                                   (8, 1024, 768)])
def test_matches_dequantize_then_matmul(nvcc, dev, dtype, B, N, K):
    """The oracle is stage 18: same numbers, without the bf16 weight."""
    torch.manual_seed(0)
    W = torch.randn(N, K, device=dev)
    q, s = quantize_int8_per_channel(W)
    x = torch.randn(B, K, device=dev, dtype=dtype)

    want = _ref(x, q, s)
    got = gemv_int8(x, q, s)
    rel = ((got.float() - want.float()).abs().max()
           / want.float().abs().max()).item()
    assert rel < (1e-3 if dtype == torch.float32 else 2e-2), rel


@pytest.mark.parametrize("B,N,K", [(1, 151, 1000), (2, 33, 17), (3, 7, 5)])
def test_shapes_that_do_not_divide_by_anything(nvcc, dev, B, N, K):
    """K not a multiple of 16 means no 16-byte loads; N not a multiple of the
    warps per block means a partial block. Both have to still be right, and
    the partial block must not hang on a barrier its dead warps skipped."""
    torch.manual_seed(1)
    W = torch.randn(N, K, device=dev)
    q, s = quantize_int8_per_channel(W)
    x = torch.randn(B, K, device=dev)
    torch.testing.assert_close(gemv_int8(x, q, s), _ref(x, q, s),
                               rtol=1e-3, atol=1e-3)


def test_accepts_a_bare_vector(nvcc, dev):
    """Decode hands you one row. It should not have to be shaped (1, K)."""
    W = torch.randn(512, 1024, device=dev)
    q, s = quantize_int8_per_channel(W)
    x = torch.randn(1024, device=dev)
    out = gemv_int8(x, q, s)
    assert out.shape == (512,)
    torch.testing.assert_close(out, _ref(x.unsqueeze(0), q, s).squeeze(0),
                               rtol=1e-3, atol=1e-3)


def test_accumulates_in_fp32(nvcc, dev):
    """4096 int8-times-bf16 products summed in bf16 lose the low bits."""
    torch.manual_seed(2)
    N, K = 2048, 4096
    W = torch.randn(N, K, device=dev)
    q, s = quantize_int8_per_channel(W)
    x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
    want = (x.float() @ dequantize_int8(q, s).t())
    got = gemv_int8(x, q, s).float()
    rel = ((got - want).abs().mean() / want.abs().mean()).item()
    print(f"\n  mean relative error over K={K}: {rel:.5f}")
    assert rel < 5e-3


def test_the_fusion_is_the_point(nvcc, dev):
    """The gate that justifies the stage.

    Dequantizing the weights and then calling matmul does the same
    arithmetic. It is slower because it writes N*K bf16 values to memory and
    reads them back, which is more traffic than the int8 weights cost in the
    first place.
    """
    import cudalib

    N, K = 11008, 4096
    W = torch.randn(N, K, device=dev)
    q, s = quantize_int8_per_channel(W)
    sb = s.to(torch.bfloat16)
    x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)

    unfused = cudalib.bench_ms(
        lambda: x @ (q.to(torch.bfloat16) * sb[:, None]).t())
    fused = cudalib.bench_ms(lambda: gemv_int8(x, q, s))

    print(f"\n  dequantize, then matmul   {unfused:>8.3f} ms")
    print(f"  fused epilogue            {fused:>8.3f} ms   "
          f"\033[1m{unfused / fused:.1f}x\033[0m")
    assert unfused / fused > 3.0, (
        f"only {unfused / fused:.1f}x. If the dequantize is fused, the bf16 "
        "weight is never written at all.")


def test_it_beats_bf16_at_batch_one(nvcc, dev):
    """The gate that justifies quantization.

    Half the weight bytes should be most of half the time. cuBLAS is reading
    a bf16 matrix; you are reading an int8 one.
    """
    import cudalib

    rows = []
    for N, K in ((4096, 4096), (11008, 4096)):
        W = torch.randn(N, K, device=dev)
        q, s = quantize_int8_per_channel(W)
        Wb = W.to(torch.bfloat16)
        x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        a = cudalib.bench_ms(lambda: x @ Wb.t())
        b = cudalib.bench_ms(lambda: gemv_int8(x, q, s))
        rows.append((N, K, a, b))

    print(f"\n  {'N':>6} {'K':>6} {'bf16 cuBLAS':>13} {'your int8':>11} {'gain':>7}")
    for N, K, a, b in rows:
        print(f"  {N:>6} {K:>6} {a:>11.3f}ms {b:>9.3f}ms {a / b:>6.2f}x")

    worst = min(a / b for _, _, a, b in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x over bf16. Half the bytes should be most of half "
        "the time; check that the weights go straight to registers and never "
        "through shared memory.")


def test_where_the_crossover_is(nvcc, dev):
    """Not a gate. The measurement that sets GEMV_MAX_ROWS.

    A GEMV is not a small GEMM. Your kernel gives every output row one warp,
    which is the right shape for one activation row and the wrong shape for
    sixteen: cuBLAS reads the weights once for ANY batch size and keeps the
    tensor cores fed, and past a row or two that wins, int8 or not.

    This is stage 03 again. Prefill and decode are different machines, and
    this is the boundary between them, in milliseconds.
    """
    import cudalib

    N, K = 11008, 4096
    W = torch.randn(N, K, device=dev)
    q, s = quantize_int8_per_channel(W)
    Wb = W.to(torch.bfloat16)

    print(f"\n  {'rows':>5} {'bf16 cuBLAS':>13} {'your int8':>11} {'gain':>7}")
    crossover = None
    for B in (1, 2, 4, 8):
        x = torch.randn(B, K, device=dev, dtype=torch.bfloat16)
        a = cudalib.bench_ms(lambda: x @ Wb.t())
        b = cudalib.bench_ms(lambda: gemv_int8(x, q, s))
        flag = "" if a > b else "   <- cuBLAS wins from here"
        if a <= b and crossover is None:
            crossover = B
        print(f"  {B:>5} {a:>11.3f}ms {b:>9.3f}ms {a / b:>6.2f}x{flag}")
    print(f"\n  \033[2mQuantizedLinearCUDA.GEMV_MAX_ROWS is "
          f"{QuantizedLinearCUDA.GEMV_MAX_ROWS}. The crossover measured here "
          f"is {crossover}.\033[0m")


def test_the_linear_dispatches_on_batch_size(nvcc, dev):
    """A big batch must not go through the GEMV path, whatever it costs.

    Both paths must agree, so a request does not get different numbers
    depending on how many other requests happened to be in flight with it.
    """
    lin = nn.Linear(512, 256).to(dev)
    qlin = QuantizedLinearCUDA.from_linear(lin).to(dev)

    small = torch.randn(1, 512, device=dev)
    big = torch.randn(64, 512, device=dev)

    ys = qlin(small)
    yb = qlin(big)
    assert ys.shape == (1, 256) and yb.shape == (64, 256)

    # the same row, through both paths
    row = big[:1]
    torch.testing.assert_close(qlin(row), qlin(big)[:1], rtol=2e-2, atol=2e-2)

    # and close to the unquantized layer
    rel = ((qlin(small) - lin(small)).abs().mean()
           / lin(small).abs().mean()).item()
    print(f"\n  mean relative error vs the fp32 Linear: {rel:.4f}")
    assert rel < 0.05


def test_it_actually_stores_int8(nvcc, dev):
    """A layer that quantizes and then keeps a bf16 copy around has saved
    nothing, and nbytes() has to tell the truth about that."""
    lin = nn.Linear(4096, 4096, bias=False).to(dev)
    qlin = QuantizedLinearCUDA.from_linear(lin).to(dev)
    dense = lin.weight.numel() * lin.weight.element_size()
    print(f"\n  fp32 Linear   {dense / 1e6:>7.2f} MB")
    print(f"  int8 version  {qlin.nbytes() / 1e6:>7.2f} MB   "
          f"\033[1m{dense / qlin.nbytes():.1f}x smaller\033[0m")
    assert qlin.nbytes() < dense / 3.5
