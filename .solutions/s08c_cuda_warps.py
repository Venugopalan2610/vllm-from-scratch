"""Reference solution, stage 08c - the host side of the split-K kernel."""

from app.s08_paged_cuda import default_scale
from cudalib import build

SOURCE = "app/cuda/s08c_paged_attn_split.cu"


def _extension():
    return build("s08c_paged_attn_split", SOURCE)


def paged_attention_split(query, key_cache, value_cache, block_tables,
                          context_lens, scale=None, splits=0):
    """splits=0: the kernel selects the splits from the grid and the
    context."""
    scale = scale or default_scale(query)
    return _extension().paged_attn(query.contiguous(), key_cache, value_cache,
                                   block_tables, context_lens, float(scale),
                                   int(splits))


def splits_for(num_seqs, num_heads, max_context):
    return _extension().splits_for(num_seqs, num_heads, max_context)
