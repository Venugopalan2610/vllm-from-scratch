"""Stage 18 (JAX) - Weight-only quantization over a pytree.

Spec in app/j18_quantization.py. The check that matters most is not the round
trip error -- it is that the dequantize stays in the epilogue.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j18_quantization import (dequantize_fp8, dequantize_int8,
                                  dequantize_tree, perplexity,
                                  quantize_fp8, quantize_int8_per_channel,
                                  quantize_tree, quantized_matmul, tree_bytes)

TEXT = (
    "The history of computing began with mechanical calculators, and every "
    "step since has been a fight against the cost of moving data rather than "
    "the cost of arithmetic. "
) * 8


def test_int8_round_trip(jdev):
    W = jax.random.normal(jax.random.key(0), (64, 128), jnp.float32)
    q, s = quantize_int8_per_channel(W)

    assert q.dtype == jnp.int8, f"quantized weights are {q.dtype}, not int8"
    assert s.shape == (64,), f"scales are {s.shape}, expected one per output row"
    assert int(jnp.max(jnp.abs(q))) <= 127

    err = float(jnp.max(jnp.abs(dequantize_int8(q, s) - W)))
    row_scale = float(jnp.max(s))
    assert err < row_scale, f"round-trip error {err:.4f} exceeds one step {row_scale:.4f}"


def test_scales_are_per_channel_not_per_tensor(jdev):
    """A row that is 1000x smaller must not be crushed to zero."""
    W = jnp.concatenate([
        jnp.ones((1, 32), jnp.float32) * 100.0,
        jnp.ones((1, 32), jnp.float32) * 0.1,
    ])
    q, s = quantize_int8_per_channel(W)
    back = dequantize_int8(q, s)
    np.testing.assert_allclose(np.asarray(back[1]), np.asarray(W[1]), rtol=0.02)
    assert float(s[0]) > float(s[1]) * 100, (
        "both rows got the same scale -- that is per-tensor, not per-channel"
    )


def test_works_on_stacked_layer_weights(jdev):
    """jvllm stacks layers, so weights arrive as (layers, out, in)."""
    W = jax.random.normal(jax.random.key(1), (4, 16, 32), jnp.float32)
    q, s = quantize_int8_per_channel(W)
    assert s.shape == (4, 16), f"scales {s.shape}, expected (layers, out)"
    err = float(jnp.max(jnp.abs(dequantize_int8(q, s) - W)))
    assert err < float(jnp.max(s))


def test_quantized_matmul_matches_dequantize_then_matmul(jdev):
    x = jax.random.normal(jax.random.key(2), (8, 128), jnp.float32)
    W = jax.random.normal(jax.random.key(3), (64, 128), jnp.float32)
    q, s = quantize_int8_per_channel(W)

    got = quantized_matmul(x, q, s)
    want = x @ dequantize_int8(q, s).T
    # Loose on purpose: XLA runs fp32 matmuls on the tensor cores in tf32 by
    # default, so two mathematically identical orderings differ by ~1e-3
    # relative. That is not your bug, and chasing it is a whole afternoon.
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=5e-3, atol=1e-2)


def test_the_dequant_stays_in_the_epilogue(jdev):
    """The check the whole stage turns on.

    Fusing the scale into the weight builds a full-size dequantized matrix --
    the exact tensor you were trying not to read. Compiled, that shows up as
    temporary memory roughly the size of the weight.
    """
    N = 2048
    x = jnp.zeros((8, N), jnp.float32)
    q = jnp.zeros((N, N), jnp.int8)
    s = jnp.ones((N,), jnp.float32)

    def naive(x, q, s):
        return x @ dequantize_int8(q, s).T

    fused = jax.jit(quantized_matmul).lower(x, q, s).compile()
    slow = jax.jit(naive).lower(x, q, s).compile()

    try:
        f_tmp = fused.memory_analysis().temp_size_in_bytes
        s_tmp = slow.memory_analysis().temp_size_in_bytes
    except Exception:                                   # pragma: no cover
        pytest.skip("this jaxlib does not expose memory_analysis")

    weight_bytes = N * N * 4                            # if it were dequantized
    print(f"\n  fused epilogue: {f_tmp / 1e6:8.2f} MB of temporaries")
    print(f"  dequant first:  {s_tmp / 1e6:8.2f} MB")
    print(f"  the fp32 weight would be {weight_bytes / 1e6:.2f} MB")
    assert f_tmp < weight_bytes / 2, (
        f"{f_tmp / 1e6:.1f} MB of temporaries for a matmul against an int8 "
        "weight -- the dequantized weight is being materialized. Move the "
        "scale after the matmul: (x @ q.T) * scales."
    )


def test_fp8_round_trip(jdev):
    t = jax.random.normal(jax.random.key(4), (128, 64), jnp.float32) * 3.0
    q, scale = quantize_fp8(t)
    assert q.dtype == jnp.float8_e4m3fn, f"got {q.dtype}, expected e4m3"
    back = dequantize_fp8(q, scale, jnp.float32)
    rel = float(jnp.mean(jnp.abs(back - t)) / jnp.mean(jnp.abs(t)))
    print(f"\n  fp8 e4m3 mean relative error: {rel * 100:.2f}%")
    assert rel < 0.06, f"{rel:.3f} relative error is too much even for e4m3"


def test_tree_bytes_and_the_saving(jmodel):
    """Quantizing is a tree_map, and the accounting has to be honest."""
    full = tree_bytes(jmodel.params)
    qtree = quantize_tree(jmodel.params)
    quant = tree_bytes(qtree)

    ratio = quant / full
    print(f"\n  bf16 params:      {full / 1e9:.3f} GB")
    print(f"  int8 + scales:    {quant / 1e9:.3f} GB   ({ratio * 100:.0f}%)")
    assert quant < full, "quantizing made the tree bigger"
    print("  \033[2mNot 50%: embed and lm_head are skipped, and on a 0.6B model")
    print("  those are a large share of the parameters. On a 7B model the same")
    print("  code gets much closer to half.\033[0m")


def test_norms_are_left_alone(jmodel):
    qtree = quantize_tree(jmodel.params)
    assert qtree["final_norm"].dtype == jmodel.params["final_norm"].dtype, (
        "the final norm was quantized -- it is one vector, saves nothing, and "
        "is where the model is most sensitive"
    )
    for name in ("attn_norm", "mlp_norm", "q_norm", "k_norm"):
        leaf = qtree["layers"][name]
        assert not isinstance(leaf, dict), f"{name} should not be quantized"


def test_perplexity_barely_moves(jmodel):
    """The guard. int8 weights must not cost you the model."""
    from jvllm import Qwen3

    base = perplexity(jmodel, TEXT)
    qtree = quantize_tree(jmodel.params)
    deq = dequantize_tree(qtree, jmodel.dtype)
    qmodel = Qwen3(deq, jmodel.config, jmodel.tokenizer, jmodel.eos_ids)
    after = perplexity(qmodel, TEXT)

    delta = (after - base) / base
    print(f"\n  bf16 perplexity: {base:8.3f}")
    print(f"  int8 perplexity: {after:8.3f}   ({delta * 100:+.1f}%)")
    assert np.isfinite(base) and np.isfinite(after)
    assert delta < 0.15, (
        f"perplexity moved {delta * 100:.1f}%. Are you quantizing the norms, "
        "or using a per-tensor scale instead of per-channel?"
    )
