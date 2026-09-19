"""Stage 07 - attention that reads through the page table.

`./vc lore 7` for the insight. `./vc test 7` to check yourself.

Stage 06 put the KV cache into blocks. Attention cannot stride through a
contiguous tensor now. It must GATHER through the block table.

Do it in plain PyTorch here, and make it exactly correct. It becomes the
oracle of your stage 08 CUDA kernels, so correctness is more important than
speed.

The shapes, with the convention of vLLM:

    query        (num_seqs, num_heads, head_dim)     one decode token for each seq
    key_cache    (num_blocks, num_kv_heads, block_size, head_dim)
    value_cache  (num_blocks, num_kv_heads, block_size, head_dim)
    block_tables (num_seqs, max_blocks_per_seq)  int32, row i = the blocks of seq i
    context_lens (num_seqs,)                     int32, the valid tokens of seq i
    output       (num_seqs, num_heads, head_dim)
"""

import torch


def write_kv(key_cache, value_cache, key, value, slot_indices):
    """Scatter the K and V of this step into the paged cache.

        key, value    (num_tokens, num_kv_heads, head_dim)
        slot_indices  (num_tokens,) the flat slot from BlockTable.slot_index()

    A flat slot has two parts:
        block_id = slot // block_size
        offset   = slot %  block_size

    This is the "slot mapping" that all real serving code uses.
    """
    raise NotImplementedError("stage 07: implement write_kv")


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    """Decode attention over a paged KV cache.

    For each sequence i:
        gather its context_lens[i] keys and values through block_tables[i]
        scores = (query_i @ K^T) * scale
        mask the positions >= context_lens[i]
        output_i = softmax(scores) @ V

    The default scale is 1/sqrt(head_dim).

    GQA: num_heads is usually a multiple of num_kv_heads. Query head h reads
    KV head `h // (num_heads // num_kv_heads)`. If this ratio is the wrong
    way, the output looks correct and is wrong. So a check uses a GQA
    configuration.

    Correctness first. It can be slower than stage 02. Measure how much, then
    get it back in stage 08.
    """
    raise NotImplementedError("stage 07: implement paged_attention")


def reference_attention(query, keys, values, scale=None):
    """The dense reference: no pages, no blocks. The oracle.

        query   (num_seqs, num_heads, head_dim)
        keys    (num_seqs, num_kv_heads, seq_len, head_dim)
        values  the same shape

    Write this function too. It proves that paged_attention is correct, and
    with both functions the gather logic is clear.
    """
    raise NotImplementedError("stage 07: implement reference_attention")
