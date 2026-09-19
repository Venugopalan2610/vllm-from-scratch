"""Stage 08c - the host side for the split-K kernel.

`./vc lore 8c`. `./vc test 8c`.

The kernel is in app/cuda/s08c_paged_attn_split.cu.

One new argument: `splits`. Zero means that the kernel selects the number
from the grid size and the context length. The checks also force specific
values. A merge that is correct only for the number that your rule selects
is not correct.
"""

import math

from cudalib import build

SOURCE = "app/cuda/s08c_paged_attn_split.cu"


def _extension():
    return build("s08c_paged_attn_split", SOURCE)


def paged_attention_split(query, key_cache, value_cache, block_tables,
                          context_lens, scale=None, splits=0):
    """The same numbers as stage 08b, for every value of `splits`."""
    raise NotImplementedError("stage 08c: call your paged_attn kernel")


def splits_for(num_seqs, num_heads, max_context):
    """The number of splits that your rule selects for this shape. The checks
    read it to explain the speedup, and to find a rule that never splits."""
    raise NotImplementedError("stage 08c: call your splits_for")
