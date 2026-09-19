"""Reference solution, stage 08b - the host side is the same as stage 08."""

from app.s08_paged_cuda import default_scale
from cudalib import build

SOURCE = "app/cuda/s08b_paged_attn_vec.cu"


def _extension():
    return build("s08b_paged_attn_vec", SOURCE)


def paged_attention_vec(query, key_cache, value_cache, block_tables,
                        context_lens, scale=None):
    scale = scale or default_scale(query)
    return _extension().paged_attn(query.contiguous(), key_cache, value_cache,
                                   block_tables, context_lens, float(scale))
