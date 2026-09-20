"""Tests for Stage 30 - Multi-LoRA Serving & Batched GEMV (BGMV)."""

import pytest
import torch
from app.s30_multi_lora import BatchedLoRAManager


def test_base_model_only_matches_linear():
    torch.manual_seed(42)
    in_dim, out_dim = 64, 128
    batch_size = 4

    manager = BatchedLoRAManager(in_dim, out_dim)
    W0 = torch.randn(out_dim, in_dim)
    x = torch.randn(batch_size, in_dim)

    # All base model requests
    out = manager.forward(x, W0, [None] * batch_size)
    expected = torch.matmul(x, W0.t())

    assert torch.allclose(out, expected, atol=1e-5)


def test_single_adapter_matches_full_lora():
    torch.manual_seed(42)
    in_dim, out_dim, r = 64, 128, 8
    alpha = 16.0

    manager = BatchedLoRAManager(in_dim, out_dim)
    A = torch.randn(r, in_dim)
    B = torch.randn(out_dim, r)
    manager.register_adapter("coder", r, alpha, A, B)

    W0 = torch.randn(out_dim, in_dim)
    x = torch.randn(2, in_dim)

    out = manager.forward(x, W0, ["coder", "coder"])

    # Full equivalent weight
    W_full = W0 + (alpha / r) * torch.matmul(B, A)
    expected = torch.matmul(x, W_full.t())

    assert torch.allclose(out, expected, atol=1e-5)


def test_heterogeneous_batch_applies_correct_adapters():
    torch.manual_seed(42)
    in_dim, out_dim, r = 64, 128, 8

    manager = BatchedLoRAManager(in_dim, out_dim)
    A1, B1 = torch.randn(r, in_dim), torch.randn(out_dim, r)
    A2, B2 = torch.randn(r, in_dim), torch.randn(out_dim, r)
    manager.register_adapter("lora_a", r, 16.0, A1, B1)
    manager.register_adapter("lora_b", r, 8.0, A2, B2)

    W0 = torch.randn(out_dim, in_dim)
    x = torch.randn(3, in_dim)

    # Row 0: base model, Row 1: lora_a, Row 2: lora_b
    out = manager.forward(x, W0, [None, "lora_a", "lora_b"])

    # Verify Row 0 (base)
    exp0 = torch.matmul(x[0:1], W0.t())
    assert torch.allclose(out[0:1], exp0, atol=1e-5)

    # Verify Row 1 (lora_a)
    W_a = W0 + (16.0 / r) * torch.matmul(B1, A1)
    exp1 = torch.matmul(x[1:2], W_a.t())
    assert torch.allclose(out[1:2], exp1, atol=1e-5)

    # Verify Row 2 (lora_b)
    W_b = W0 + (8.0 / r) * torch.matmul(B2, A2)
    exp2 = torch.matmul(x[2:3], W_b.t())
    assert torch.allclose(out[2:3], exp2, atol=1e-5)


def test_adapter_scaling_factor():
    in_dim, out_dim, r = 32, 32, 4
    manager = BatchedLoRAManager(in_dim, out_dim)

    A = torch.ones(r, in_dim)
    B = torch.ones(out_dim, r)

    manager.register_adapter("test1", r=4, alpha=4.0, A=A, B=B)   # scale = 1.0
    manager.register_adapter("test2", r=4, alpha=16.0, A=A, B=B)  # scale = 4.0

    W0 = torch.zeros(out_dim, in_dim)
    x = torch.ones(2, in_dim)

    out = manager.forward(x, W0, ["test1", "test2"])

    # test2 delta must be exactly 4x test1 delta
    assert torch.allclose(out[1], out[0] * 4.0, atol=1e-5)
