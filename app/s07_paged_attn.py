"""Stage 07 - attention that reads through the page table.

`./vc lore 7` for the insight. `./vc test 7` to check yourself.

Stage 06 scattered the KV cache into blocks. Attention can no longer stride a
contiguous tensor -- it has to GATHER through the block table. Do it in plain
PyTorch here and get it exactly right; this becomes the oracle that stage 08's
Triton kernel is checked against, so correctness matters more than speed.

Shapes, following vLLM's own convention:

    query        (num_seqs, num_heads, head_dim)          one decode token/seq
    key_cache    (num_blocks, num_kv_heads, block_size, head_dim)
    value_cache  (num_blocks, num_kv_heads, block_size, head_dim)
    block_tables (num_seqs, max_blocks_per_seq)  int32, row i = seq i's blocks
    context_lens (num_seqs,)                     int32, tokens valid in seq i
    output       (num_seqs, num_heads, head_dim)
"""

import torch


def write_kv(key_cache, value_cache, key, value, slot_indices):
    """Scatter this step's K/V into the paged cache.

        key, value    (num_tokens, num_kv_heads, head_dim)
        slot_indices  (num_tokens,) flat slot from BlockTable.slot_index()

    A flat slot decomposes as:
        block = slot // block_size
        off   = slot %  block_size

    This is the "slot mapping" you will see all over real serving code.
    """
    raise NotImplementedError("stage 07: implement write_kv")


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    """Decode attention over a paged KV cache.

    For each sequence i:
        gather its context_lens[i] keys/values by walking block_tables[i]
        scores = (query_i @ K^T) * scale
        mask out positions >= context_lens[i]
        out_i  = softmax(scores) @ V

    scale defaults to 1/sqrt(head_dim).

    GQA: num_heads is usually a multiple of num_kv_heads. Query head h reads
    KV head `h // (num_heads // num_kv_heads)`. Getting this ratio backwards
    produces plausible-looking output that is quietly wrong, so the tests check
    a GQA config specifically.

    Correctness first. It is allowed to be slower than stage 02 -- measure how
    much, then win it back in stage 08.
    """
    raise NotImplementedError("stage 07: implement paged_attention")


def reference_attention(query, keys, values, scale=None):
    """Dense reference: no paging, no blocks. The oracle.

        query   (num_seqs, num_heads, head_dim)
        keys    (num_seqs, num_kv_heads, seq_len, head_dim)
        values  same

    Implement this too -- it is how you prove paged_attention is right, and
    writing both makes the gather logic obvious.
    """
    raise NotImplementedError("stage 07: implement reference_attention")
