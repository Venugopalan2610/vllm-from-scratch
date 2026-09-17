"""Reference solution, stage 08b - host side, unchanged from stage 08."""

import math

import torch

from cudalib import build

SOURCE = "app/cuda/s08b_paged_attn_vec.cu"


def _ext():
    return build("s08b_paged_attn_vec", SOURCE)


def paged_attention_vec(query, key_cache, value_cache, block_tables,
                        context_lens, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    return _ext().paged_attn(query.contiguous(), key_cache, value_cache,
                             block_tables, context_lens, float(scale))
