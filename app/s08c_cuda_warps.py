"""Stage 08c - the host side for the split-K kernel.

`./vc lore 8c`. `./vc test 8c`.

The kernel is in app/cuda/s08c_paged_attn_split.cu.

One new argument: `splits`. Zero means the kernel decides from the grid size
and the context length. The checks force specific values too, because a
merge that is only correct for the split count your heuristic happens to
pick is not correct.
"""

import math

import torch

from cudalib import build

SOURCE = "app/cuda/s08c_paged_attn_split.cu"


def _ext():
    return build("s08c_paged_attn_split", SOURCE)


def paged_attention_split(query, key_cache, value_cache, block_tables,
                          context_lens, scale=None, splits=0):
    """Same numbers as stage 08b, for any value of `splits`."""
    raise NotImplementedError("stage 08c: call your paged_attn kernel")


def splits_for(num_seqs, num_heads, max_context):
    """How many splits your heuristic picks for this shape. The checks read
    this to explain the speedup, and to catch a heuristic that never splits."""
    raise NotImplementedError("stage 08c: call your splits_for")
