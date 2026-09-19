"""Reference solution, stage 07."""

import math

import torch


def write_kv(key_cache, value_cache, key, value, slot_indices):
    """key and value: (tokens, kv_heads, head_dim). A cache:
    (blocks, kv_heads, block_size, head_dim)."""
    block_size = key_cache.shape[2]
    block_ids = slot_indices // block_size
    offsets = slot_indices % block_size
    key_cache[block_ids, :, offsets, :] = key
    value_cache[block_ids, :, offsets, :] = value


def gather_context(cache, block_ids, context_len):
    """(blocks, kv_heads, block_size, D) -> (kv_heads, context_len, D)."""
    num_kv_heads, head_dim = cache.shape[1], cache.shape[3]
    rows = cache[block_ids].permute(1, 0, 2, 3).reshape(num_kv_heads, -1,
                                                         head_dim)
    return rows[:, :context_len]


def attend(query, keys, values, scale):
    """query (heads, queries, D). keys and values (heads, context, D).
    The arithmetic is in fp32."""
    scores = torch.matmul(query.float(), keys.float().transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, values.float())


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    num_seqs, num_heads, head_dim = query.shape
    block_size = key_cache.shape[2]
    heads_per_kv_head = num_heads // key_cache.shape[1]
    scale = scale or 1.0 / math.sqrt(head_dim)

    output = torch.empty_like(query)
    for seq in range(num_seqs):
        context_len = int(context_lens[seq])
        num_blocks = math.ceil(context_len / block_size)
        block_ids = block_tables[seq, :num_blocks].long()
        keys = gather_context(key_cache, block_ids, context_len)
        values = gather_context(value_cache, block_ids, context_len)
        keys = keys.repeat_interleave(heads_per_kv_head, dim=0)
        values = values.repeat_interleave(heads_per_kv_head, dim=0)
        seq_output = attend(query[seq].unsqueeze(1), keys, values, scale)
        output[seq] = seq_output.squeeze(1).to(query.dtype)
    return output


def reference_attention(query, keys, values, scale=None):
    """query (seqs, heads, D). keys and values (seqs, kv_heads, context, D)."""
    head_dim = query.shape[2]
    heads_per_kv_head = query.shape[1] // keys.shape[1]
    scale = scale or 1.0 / math.sqrt(head_dim)
    keys = keys.repeat_interleave(heads_per_kv_head, dim=1)
    values = values.repeat_interleave(heads_per_kv_head, dim=1)
    output = attend(query.unsqueeze(2), keys, values, scale)
    return output.squeeze(2).to(query.dtype)
