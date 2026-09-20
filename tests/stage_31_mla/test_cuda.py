"""Tests for Stage 31 - DeepSeek Multi-Head Latent Attention (MLA)."""

import math
import pytest
import torch

from app.s31_mla import (
    MLAConfig,
    compress_kv,
    standard_unabsorbed_attention,
    absorbed_decode_attention,
    mla_kv_compression_ratio,
)


def test_compress_kv_shapes():
    config = MLAConfig(
        hidden_size=256,
        num_heads=8,
        head_dim=32,
        kv_lora_rank=64,
        qk_rope_head_dim=16,
    )
    batch = 3
    seq_len = 24
    x = torch.randn(batch, seq_len, config.hidden_size)
    W_DKV = torch.randn(config.kv_lora_rank, config.hidden_size)
    W_KR = torch.randn(config.qk_rope_head_dim, config.hidden_size)

    c_kv, k_rope = compress_kv(x, W_DKV, W_KR)
    assert c_kv.shape == (batch, seq_len, config.kv_lora_rank)
    assert k_rope.shape == (batch, seq_len, config.qk_rope_head_dim)


def test_mla_decode_absorption_numerical_equivalence():
    """Verify that absorbing W_UK and W_UV produces mathematically identical
    attention output to decompressing full K and V, while avoiding materializing
    the uncompressed cache."""
    config = MLAConfig(
        hidden_size=256,
        num_heads=8,
        head_dim=32,
        kv_lora_rank=64,
        qk_rope_head_dim=16,
    )
    torch.manual_seed(42)

    for batch, seq_len in [(1, 16), (4, 64)]:
        q_nope = torch.randn(batch, config.num_heads, config.head_dim)
        q_rope = torch.randn(batch, config.num_heads, config.qk_rope_head_dim)
        c_kv = torch.randn(batch, seq_len, config.kv_lora_rank)
        k_rope = torch.randn(batch, seq_len, config.qk_rope_head_dim)

        W_UK = torch.randn(config.num_heads * config.head_dim, config.kv_lora_rank)
        W_UV = torch.randn(config.num_heads * config.head_dim, config.kv_lora_rank)

        out_unabsorbed = standard_unabsorbed_attention(
            q_nope, q_rope, c_kv, k_rope, W_UK, W_UV, config
        )
        out_absorbed = absorbed_decode_attention(
            q_nope, q_rope, c_kv, k_rope, W_UK, W_UV, config
        )

        assert out_unabsorbed.shape == (batch, config.num_heads, config.head_dim)
        assert out_absorbed.shape == (batch, config.num_heads, config.head_dim)
        assert torch.allclose(out_unabsorbed, out_absorbed, atol=1e-4, rtol=1e-4)


def test_mla_kv_compression_ratio():
    """DeepSeek-style MLA compresses H*D down to d_c + d_R, yielding >80% savings."""
    config = MLAConfig(
        hidden_size=256,
        num_heads=8,
        head_dim=32,          # 2 * 8 * 32 = 512 elements per token in standard MHA
        kv_lora_rank=64,
        qk_rope_head_dim=16,  # 64 + 16 = 80 elements per token in MLA
    )
    ratio = mla_kv_compression_ratio(config)
    # (1 - 80/512) * 100 = 84.375%
    expected = (1.0 - 80.0 / 512.0) * 100.0
    assert math.isclose(ratio, expected, rel_tol=1e-5)
    assert ratio > 80.0
