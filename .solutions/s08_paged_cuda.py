"""Reference solution, stage 08 - the host side of the CUDA paged kernel."""

import math

from cudalib import build

SOURCE = "app/cuda/s08_paged_attn.cu"


def _extension():
    # Lazy and memoized. The first call compiles. A later call is a dict hit.
    return build("s08_paged_attn", SOURCE)


def default_scale(query):
    return 1.0 / math.sqrt(query.shape[-1])


def write_kv_cuda(key_cache, value_cache, key, value, slot_indices):
    _extension().write_kv(key_cache, value_cache, key.contiguous(),
                          value.contiguous(), slot_indices)


def paged_attention_cuda(query, key_cache, value_cache, block_tables,
                         context_lens, scale=None):
    scale = scale or default_scale(query)
    return _extension().paged_attn(query.contiguous(), key_cache, value_cache,
                                   block_tables, context_lens, float(scale))


# ---------------------------------------------------------------- the four lines


def four_lines(step_at, n, facts):
    """The four moves of docs/METHOD.md. Reference solution."""
    import cudalib

    attention_call_floor_ms = (n * facts.context_len * 2 * facts.num_kv_heads * facts.head_dim * facts.value_bytes) / facts.bandwidth_bytes_per_s * 1e3
    attention_call_measured_ms = cudalib.bench_ms(step_at(n))
    attention_call_measured_over_floor = attention_call_measured_ms / attention_call_floor_ms
    attention_call_ms_2n_over_n = cudalib.bench_ms(step_at(2 * n)) / attention_call_measured_ms
    return {"attention_call_floor_ms": attention_call_floor_ms, "attention_call_measured_ms": attention_call_measured_ms,
            "attention_call_measured_over_floor": attention_call_measured_over_floor, "attention_call_ms_2n_over_n": attention_call_ms_2n_over_n}
