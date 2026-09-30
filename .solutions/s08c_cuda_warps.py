"""Reference solution, stage 08c - the host side of the split-K kernel."""

from app.s08_paged_cuda import default_scale
from cudalib import build

SOURCE = "app/cuda/s08c_paged_attn_split.cu"


def _extension():
    return build("s08c_paged_attn_split", SOURCE)


def paged_attention_split(query, key_cache, value_cache, block_tables,
                          context_lens, scale=None, splits=0):
    """splits=0: the kernel selects the splits from the grid and the
    context."""
    scale = scale or default_scale(query)
    return _extension().paged_attn(query.contiguous(), key_cache, value_cache,
                                   block_tables, context_lens, float(scale),
                                   int(splits))


def splits_for(num_seqs, num_heads, max_context):
    return _extension().splits_for(num_seqs, num_heads, max_context)


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
