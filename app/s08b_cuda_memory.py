"""Stage 08b - the host side. It has the same shape as the host side of
stage 08.

`./vc lore 8b`. `./vc test 8b`.

The kernel is in app/cuda/s08b_paged_attn_vec.cu. Copy your stage 08 kernel
into it, and change the access pattern. The header of that file has the
spec.

This wrapper does not change, and that is the point of the stage. The
function signature, the tolerances and the oracle are all the same. Only the
bytes that each transaction moves are different.
"""

import math

from cudalib import build

SOURCE = "app/cuda/s08b_paged_attn_vec.cu"


def _extension():
    return build("s08b_paged_attn_vec", SOURCE)


def paged_attention_vec(query, key_cache, value_cache, block_tables,
                        context_lens, scale=None):
    """The same signature, the same numbers and the same oracle as stage 08.

    The checks compare you with stage 07 for correctness and with stage 08
    for speed. They also measure the bandwidth that you got against the
    bandwidth of this card.
    """
    raise NotImplementedError("stage 08b: call your paged_attn kernel")
