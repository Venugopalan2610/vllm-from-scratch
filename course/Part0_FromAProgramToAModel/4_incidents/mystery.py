"""Three attention functions. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part0_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 10.

Each function takes q of shape (batch, heads, L, head_dim) and k, v of shape
(batch, heads, S, head_dim). The queries are the last L of the S positions.
"""

import math

import torch

from lab import causal_bias


def attention_a(q, k, v):
    # looks only at the last 64 keys
    WINDOW = 64
    k, v = k[..., -WINDOW:, :], v[..., -WINDOW:, :]
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.float() + causal_bias(scores.float())
    return torch.softmax(scores, dim=-1).to(q.dtype) @ v


def attention_b(q, k, v):
    # each query may also see the ONE token after it
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.float()
    L, S = scores.shape[-2], scores.shape[-1]
    query_positions = torch.arange(S - L, S, device=q.device)[:, None]
    key_positions = torch.arange(S, device=q.device)[None, :]
    scores = scores.masked_fill(key_positions > query_positions + 1, float('-inf'))
    return torch.softmax(scores, dim=-1).to(q.dtype) @ v


def attention_c(q, k, v):
    # the softmax runs over the queries, not over the keys
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.float() + causal_bias(scores.float())
    return torch.softmax(scores, dim=-2).nan_to_num(0.0).to(q.dtype) @ v
