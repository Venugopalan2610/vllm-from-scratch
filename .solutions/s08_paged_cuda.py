"""Reference solution, stage 08 - the host side of the CUDA paged kernel."""

import math

import torch

from cudalib import build

SOURCE = "app/cuda/s08_paged_attn.cu"


def _ext():
    # Lazy and memoized. The first call compiles; later calls are a dict hit.
    return build("s08_paged_attn", SOURCE)


def write_kv_cuda(key_cache, value_cache, key, value, slot_indices):
    _ext().write_kv(key_cache, value_cache,
                    key.contiguous(), value.contiguous(), slot_indices)


def paged_attention_cuda(query, key_cache, value_cache, block_tables,
                         context_lens, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    return _ext().paged_attn(query.contiguous(), key_cache, value_cache,
                             block_tables, context_lens, float(scale))
