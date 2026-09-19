"""Stage 18 (JAX) - weight-only quantization over a pytree.

The spec is in app/j18_quantization.py. The most important check is not the
round-trip error. It is that the dequantize stays after the matmul.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app.j18_quantization import (
    dequantize_fp8,
    dequantize_int8,
    dequantize_tree,
    perplexity,
    quantize_fp8,
    quantize_int8_per_channel,
    quantize_tree,
    quantized_matmul,
    tree_bytes,
)

TEXT = (
    "The history of computing began with mechanical calculators, and every "
    "step since has been a fight against the cost of moving data rather than "
    "the cost of arithmetic. "
) * 8


def _random(seed, shape, scale=1.0):
    return jax.random.normal(jax.random.key(seed), shape, jnp.float32) * scale


def _max_round_trip_error(weight):
    int8_weight, scales = quantize_int8_per_channel(weight)
    error = float(jnp.max(jnp.abs(dequantize_int8(int8_weight, scales)
                                  - weight)))
    return error, int8_weight, scales


def test_int8_round_trip(jax_device):
    error, int8_weight, scales = _max_round_trip_error(_random(0, (64, 128)))
    assert int8_weight.dtype == jnp.int8, (
        f"the quantized weights are {int8_weight.dtype}, not int8")
    assert scales.shape == (64,), (
        f"the scales are {scales.shape}, expected one for each output row")
    assert int(jnp.max(jnp.abs(int8_weight))) <= 127
    one_step = float(jnp.max(scales))
    assert error < one_step, (
        f"the round-trip error {error:.4f} is more than one step {one_step:.4f}")


def test_scales_are_per_channel_not_per_tensor(jax_device):
    """A row that is 1000 times smaller must not become zero."""
    weight = jnp.concatenate([jnp.ones((1, 32), jnp.float32) * 100.0,
                              jnp.ones((1, 32), jnp.float32) * 0.1])
    int8_weight, scales = quantize_int8_per_channel(weight)
    restored = dequantize_int8(int8_weight, scales)
    np.testing.assert_allclose(np.asarray(restored[1]), np.asarray(weight[1]),
                               rtol=0.02)
    assert float(scales[0]) > float(scales[1]) * 100, (
        "both rows got the same scale. That is one scale for the tensor, not "
        "one for each channel.")


def test_works_on_stacked_layer_weights(jax_device):
    """jvllm stacks the layers, so a weight is (layers, out, in)."""
    error, _, scales = _max_round_trip_error(_random(1, (4, 16, 32)))
    assert scales.shape == (4, 16), (
        f"the scales are {scales.shape}, expected (layers, out)")
    assert error < float(jnp.max(scales))


def test_quantized_matmul_matches_dequantize_then_matmul(jax_device):
    inputs = _random(2, (8, 128))
    int8_weight, scales = quantize_int8_per_channel(_random(3, (64, 128)))
    expected = inputs @ dequantize_int8(int8_weight, scales).T
    # Loose on purpose. By default, XLA runs fp32 matmuls on the tensor cores
    # in tf32. So two orders that are the same in mathematics differ by about
    # 1e-3 relative. That is not your bug, and a search for it takes hours.
    np.testing.assert_allclose(
        np.asarray(quantized_matmul(inputs, int8_weight, scales)),
        np.asarray(expected), rtol=5e-3, atol=1e-2)


def test_the_dequant_stays_in_the_epilogue(jax_device):
    """The check that the whole stage depends on.

    If you put the scale into the weight, you make a full-size dequantized
    matrix: the exact tensor that you did not want to read. After
    compilation, that shows as temporary memory of about the size of the
    weight.
    """
    size = 2048
    inputs = jnp.zeros((8, size), jnp.float32)
    int8_weight = jnp.zeros((size, size), jnp.int8)
    scales = jnp.ones((size,), jnp.float32)

    def dequantize_first(inputs, int8_weight, scales):
        return inputs @ dequantize_int8(int8_weight, scales).T

    def temporary_bytes(function):
        compiled = jax.jit(function).lower(inputs, int8_weight,
                                           scales).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    try:
        fused_bytes = temporary_bytes(quantized_matmul)
        unfused_bytes = temporary_bytes(dequantize_first)
    except Exception:                                   # pragma: no cover
        pytest.skip("this jaxlib does not give memory_analysis")

    weight_bytes = size * size * 4                      # if it were dequantized
    print(f"\n  scale after the matmul: {fused_bytes / 1e6:8.2f} MB of "
          "temporaries")
    print(f"  dequantize first:       {unfused_bytes / 1e6:8.2f} MB")
    print(f"  the fp32 weight is {weight_bytes / 1e6:.2f} MB")
    assert fused_bytes < weight_bytes / 2, (
        f"{fused_bytes / 1e6:.1f} MB of temporaries for a matmul with an int8 "
        "weight. The dequantized weight is in memory. Move the scale after "
        "the matmul: (x @ q.T) * scales.")


def test_fp8_round_trip(jax_device):
    values = _random(4, (128, 64), scale=3.0)
    fp8_values, scale = quantize_fp8(values)
    assert fp8_values.dtype == jnp.float8_e4m3fn, (
        f"got {fp8_values.dtype}, expected e4m3")
    restored = dequantize_fp8(fp8_values, scale, jnp.float32)
    error = float(jnp.mean(jnp.abs(restored - values))
                  / jnp.mean(jnp.abs(values)))
    print(f"\n  fp8 e4m3 mean relative error: {error * 100:.2f}%")
    assert error < 0.06, f"a relative error of {error:.3f} is too much for e4m3"


def test_tree_bytes_and_the_saving(jmodel):
    """You quantize with a tree_map, and the byte count must be true."""
    full_bytes = tree_bytes(jmodel.params)
    quantized_bytes = tree_bytes(quantize_tree(jmodel.params))
    fraction = quantized_bytes / full_bytes
    print(f"\n  bf16 params:      {full_bytes / 1e9:.3f} GB")
    print(f"  int8 + scales:    {quantized_bytes / 1e9:.3f} GB   "
          f"({fraction * 100:.0f}%)")
    assert quantized_bytes < full_bytes, "the quantization made the tree larger"
    print("  \033[2mNot 50%: the embedding and lm_head are skipped, and on a")
    print("  0.6B model they are a large part of the parameters. On a 7B")
    print("  model the same code gets much nearer to half.\033[0m")


def test_norms_are_left_alone(jmodel):
    quantized = quantize_tree(jmodel.params)
    assert quantized["final_norm"].dtype == jmodel.params["final_norm"].dtype, (
        "the final norm was quantized. It is one vector, it saves nothing, "
        "and the model is most sensitive there.")
    for name in ("attn_norm", "mlp_norm", "q_norm", "k_norm"):
        assert not isinstance(quantized["layers"][name], dict), (
            f"{name} must not be quantized")


def test_perplexity_barely_moves(jmodel):
    """The guard. int8 weights must not cost you the model."""
    from jvllm import Qwen3

    bf16_perplexity = perplexity(jmodel, TEXT)
    restored_params = dequantize_tree(quantize_tree(jmodel.params),
                                      jmodel.dtype)
    int8_model = Qwen3(restored_params, jmodel.config, jmodel.tokenizer,
                       jmodel.eos_ids)
    int8_perplexity = perplexity(int8_model, TEXT)
    change = (int8_perplexity - bf16_perplexity) / bf16_perplexity

    print(f"\n  bf16 perplexity: {bf16_perplexity:8.3f}")
    print(f"  int8 perplexity: {int8_perplexity:8.3f}   ({change * 100:+.1f}%)")
    assert np.isfinite(bf16_perplexity) and np.isfinite(int8_perplexity)
    assert change < 0.15, (
        f"the perplexity moved {change * 100:.1f}%. Do you quantize the norms, "
        "or use one scale for the tensor, not one for each channel?")
