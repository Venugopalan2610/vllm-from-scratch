"""Stage 08 - the host side of your first CUDA kernel.

`./vc lore 8` for the insight. `./vc test 8` to check yourself.

The kernel lives in app/cuda/s08_paged_attn.cu and that is where the stage
is. This file is the seam between Python and it, and the seam has a job:
everything the kernel is allowed to ASSUME has to be made true here.

    contiguous?      a kernel indexes with plain arithmetic, so a transposed
                     or sliced tensor silently reads the wrong addresses
    dtype?           int32 for the block table and the context lengths
    on the GPU?      .is_cuda, checked in the .cu with TORCH_CHECK

cudalib.build() compiles the .cu on first call and caches it, keyed on the
source text. Edit the kernel, run the checks, and the rebuild happens for
you. The first build takes 20-40 seconds; after that it is a dict lookup.

Call build() INSIDE the function, never at import time. A kernel that does
not compile should fail the check that uses it, not stop pytest from
collecting the stage.
"""

import math

import torch

from cudalib import build

SOURCE = "app/cuda/s08_paged_attn.cu"


def _ext():
    """The compiled extension. Memoized, so calling it per invocation is free."""
    return build("s08_paged_attn", SOURCE)


def write_kv_cuda(key_cache, value_cache, key, value, slot_indices):
    """Scatter this step's K/V into the paged cache, on the GPU.

        key, value    (num_tokens, num_kv_heads, head_dim)
        slot_indices  (num_tokens,) flat slot from BlockTable.slot_index()

    Same semantics as stage 07's write_kv, which is your oracle.
    """
    raise NotImplementedError("stage 08: call your write_kv kernel")


def paged_attention_cuda(query, key_cache, value_cache, block_tables,
                         context_lens, scale=None):
    """Same signature and the same numbers as stage 07's paged_attention.

        query        (num_seqs, num_heads, head_dim)
        key_cache    (num_blocks, num_kv_heads, block_size, head_dim)
        value_cache  same
        block_tables (num_seqs, max_blocks_per_seq)
        context_lens (num_seqs,)
        -> (num_seqs, num_heads, head_dim)

    scale defaults to 1/sqrt(head_dim).

    Stage 07 is your oracle. Do not move on until they agree to 1e-3.
    """
    raise NotImplementedError("stage 08: call your paged_attn kernel")
