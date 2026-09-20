"""Stage 31 - DeepSeek Multi-Head Latent Attention (MLA).

Reference implementation with Decode Weight Absorption.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLAConfig:
    hidden_size: int = 256
    num_heads: int = 8
    head_dim: int = 32           # H * D = 256
    kv_lora_rank: int = 64       # d_c = 64 (vs 256 standard)
    qk_rope_head_dim: int = 16   # d_R = 16


def compress_kv(x: torch.Tensor, W_DKV: torch.Tensor,
                W_KR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project hidden state x into compressed latent KV and decoupled RoPE key.

    Args:
        x: (batch, seq_len, hidden_size)
        W_DKV: (kv_lora_rank, hidden_size)
        W_KR: (qk_rope_head_dim, hidden_size)

    Returns:
        c_kv: (batch, seq_len, kv_lora_rank)
        k_rope: (batch, seq_len, qk_rope_head_dim)
    """
    c_kv = F.linear(x, W_DKV)
    k_rope = F.linear(x, W_KR)
    return c_kv, k_rope


def standard_unabsorbed_attention(
    q_nope: torch.Tensor,   # (batch, num_heads, head_dim)
    q_rope: torch.Tensor,   # (batch, num_heads, qk_rope_head_dim)
    c_kv: torch.Tensor,     # (batch, seq_len, kv_lora_rank)
    k_rope: torch.Tensor,   # (batch, seq_len, qk_rope_head_dim)
    W_UK: torch.Tensor,     # (num_heads * head_dim, kv_lora_rank)
    W_UV: torch.Tensor,     # (num_heads * head_dim, kv_lora_rank)
    config: MLAConfig
) -> torch.Tensor:
    """Baseline MLA: explicitly decompresses K and V from latent c_kv, then attends."""
    H = config.num_heads
    D = config.head_dim
    scale = 1.0 / math.sqrt(D + config.qk_rope_head_dim)

    # 1. Decompress K and V
    W_UK_h = W_UK.view(H, D, config.kv_lora_rank)
    W_UV_h = W_UV.view(H, D, config.kv_lora_rank)
    K_nope = torch.einsum("bsc,hdc->bhsd", c_kv, W_UK_h)
    V = torch.einsum("bsc,hdc->bhsd", c_kv, W_UV_h)

    # 2. Attention scores (NoPE key + RoPE key)
    k_rope_exp = k_rope.unsqueeze(1).expand(-1, H, -1, -1)
    score_nope = torch.einsum("bhd,bhsd->bhs", q_nope, K_nope)
    score_rope = torch.einsum("bhr,bhsr->bhs", q_rope, k_rope_exp)
    score = (score_nope + score_rope) * scale

    attn = torch.softmax(score, dim=-1)
    out = torch.einsum("bhs,bhsd->bhd", attn, V)
    return out


def absorbed_decode_attention(
    q_nope: torch.Tensor,   # (batch, num_heads, head_dim)
    q_rope: torch.Tensor,   # (batch, num_heads, qk_rope_head_dim)
    c_kv: torch.Tensor,     # (batch, seq_len, kv_lora_rank)
    k_rope: torch.Tensor,   # (batch, seq_len, qk_rope_head_dim)
    W_UK: torch.Tensor,     # (num_heads * head_dim, kv_lora_rank)
    W_UV: torch.Tensor,     # (num_heads * head_dim, kv_lora_rank)
    config: MLAConfig
) -> torch.Tensor:
    """Decode-time MLA: absorbs W_UK into Query and W_UV into Output.
    Directly attends over compressed c_kv without expanding K and V in VRAM."""
    H = config.num_heads
    D = config.head_dim
    scale = 1.0 / math.sqrt(D + config.qk_rope_head_dim)

    # 1. Absorb W_UK into query: (b, h, d) @ (h, d, c) -> (b, h, c)
    W_UK_h = W_UK.view(H, D, config.kv_lora_rank)
    q_abs = torch.einsum("bhd,hdc->bhc", q_nope, W_UK_h)

    # 2. Compute attention directly against compressed latent c_kv
    score_nope = torch.einsum("bhc,bsc->bhs", q_abs, c_kv)
    score_rope = torch.einsum("bhr,bsr->bhs", q_rope, k_rope)
    score = (score_nope + score_rope) * scale

    attn = torch.softmax(score, dim=-1)

    # 3. Aggregate latent values, then post-multiply by absorbed W_UV
    W_UV_h = W_UV.view(H, D, config.kv_lora_rank)
    c_ctx = torch.einsum("bhs,bsc->bhc", attn, c_kv)
    out = torch.einsum("bhc,hdc->bhd", c_ctx, W_UV_h)
    return out


def mla_kv_compression_ratio(config: MLAConfig) -> float:
    """Compute percentage of KV cache memory saved by MLA vs standard MHA/GQA."""
    standard_elements = 2 * config.num_heads * config.head_dim
    mla_elements = config.kv_lora_rank + config.qk_rope_head_dim
    return (1.0 - mla_elements / standard_elements) * 100.0
