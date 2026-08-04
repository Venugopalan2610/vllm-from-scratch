"""Reference solution, stage 07."""

import math

import torch


def write_kv(key_cache, value_cache, key, value, slot_indices):
    block_size = key_cache.shape[2]
    blocks = slot_indices // block_size
    offs = slot_indices % block_size
    # key: (T, kv_heads, head_dim) -> cache: (blocks, kv_heads, block_size, hd)
    key_cache[blocks, :, offs, :] = key
    value_cache[blocks, :, offs, :] = value


def paged_attention(query, key_cache, value_cache, block_tables,
                    context_lens, scale=None):
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    group = num_heads // num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(query)
    for i in range(num_seqs):
        L = int(context_lens[i])
        n_blocks = (L + block_size - 1) // block_size
        blk = block_tables[i, :n_blocks].long()

        # (n_blocks, kv_heads, block_size, hd) -> (kv_heads, n_blocks*bs, hd)
        k = key_cache[blk].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
        v = value_cache[blk].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
        k, v = k[:, :L], v[:, :L]

        if group > 1:
            k = k.repeat_interleave(group, dim=0)
            v = v.repeat_interleave(group, dim=0)

        q = query[i].unsqueeze(1)                    # (heads, 1, hd)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out[i] = torch.matmul(probs, v.float()).squeeze(1).to(query.dtype)
    return out


def reference_attention(query, keys, values, scale=None):
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = keys.shape[1]
    group = num_heads // num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    k, v = keys, values
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    q = query.unsqueeze(2)                            # (S, H, 1, hd)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).squeeze(2).to(query.dtype)
