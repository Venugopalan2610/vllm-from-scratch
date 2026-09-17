"""Stage 08b - the host side. Identical in shape to stage 08's.

`./vc lore 8b`. `./vc test 8b`.

The kernel is in app/cuda/s08b_paged_attn_vec.cu. Copy your stage 08 kernel
into it and change the access pattern; the spec is in that file's header.

Nothing about this wrapper changes, and that is the point of the stage. The
function signature, the tolerances and the oracle are all the same. Only the
bytes moved per transaction are different.
"""

import math

import torch

from cudalib import build

SOURCE = "app/cuda/s08b_paged_attn_vec.cu"


def _ext():
    return build("s08b_paged_attn_vec", SOURCE)


def paged_attention_vec(query, key_cache, value_cache, block_tables,
                        context_lens, scale=None):
    """Same signature, same numbers, same oracle as stage 08.

    The checks compare you against stage 07 for correctness and against
    stage 08 for speed, and they measure the bandwidth you actually achieved
    against what this card can deliver.
    """
    raise NotImplementedError("stage 08b: call your paged_attn kernel")
