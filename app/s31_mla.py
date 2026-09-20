"""Stage 31 - DeepSeek Multi-Head Latent Attention (MLA).

`./vc lore 31` for the insight. `./vc test 31` to check yourself.

At 128k context, storing full Keys and Values (H_kv * D elements per token)
consumes hundreds of gigabytes of VRAM.
DeepSeek-V2 and V3 introduced Multi-Head Latent Attention (MLA):
1. KV Compression:
   Keys and Values are compressed into a single low-rank latent vector:
       c_t = x_t @ W_DKV.T   (shape: d_c, where d_c << H * D)
2. Decoupled RoPE:
   Rotary embeddings cannot be easily applied to compressed latents. MLA extracts
   a small decoupled key:
       k_t^R = x_t @ W_KR.T   (shape: d_R)
   The stored KV cache per token is ONLY (c_t, k_t^R).

3. The Decode Absorption Trick:
   In autoregressive decode mode, instead of uncompressing c_t into K and V:
       K = c_t @ W_UK.T,   V = c_t @ W_UV.T
   the projection weights W_UK and W_UV are mathematically absorbed into
   the Query and Output projections:
       Q_absorbed = Q @ W_UK
       Score = (Q_absorbed @ c_t.T + Q_R @ k_t^R.T) / sqrt(D + d_R)
       Context_latent = Softmax(Score) @ c_t
       Output = Context_latent @ W_UV
   Attention runs directly against the compressed cache, slashing KV cache
   bytes per token by up to 80% with zero loss in precision!
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
    raise NotImplementedError("stage 31: implement compress_kv")


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
    raise NotImplementedError("stage 31: implement standard_unabsorbed_attention")


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
    raise NotImplementedError("stage 31: implement absorbed_decode_attention")


def mla_kv_compression_ratio(config: MLAConfig) -> float:
    """Compute percentage of KV cache memory saved by MLA vs standard MHA/GQA."""
    raise NotImplementedError("stage 31: implement mla_kv_compression_ratio")
