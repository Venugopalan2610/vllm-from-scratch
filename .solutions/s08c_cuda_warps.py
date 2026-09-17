"""Reference solution, stage 08c - host side for the split-K kernel."""

import math

import torch

from cudalib import build

SOURCE = "app/cuda/s08c_paged_attn_split.cu"


def _ext():
    return build("s08c_paged_attn_split", SOURCE)


def paged_attention_split(query, key_cache, value_cache, block_tables,
                          context_lens, scale=None, splits=0):
    """splits=0 lets the kernel choose from the grid size and the context."""
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    return _ext().paged_attn(query.contiguous(), key_cache, value_cache,
                             block_tables, context_lens, float(scale),
                             int(splits))


def splits_for(num_seqs, num_heads, max_context):
    return _ext().splits_for(num_seqs, num_heads, max_context)
