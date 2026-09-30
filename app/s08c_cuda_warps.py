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


# ---------------------------------------------------------------- the four lines


def four_lines(step_at, n, facts):
    """The four moves of docs/METHOD.md: predict, measure, divide, double.
    The subject: one call of your split-K paged attention, for n
    sequences.

    GIVEN
        step_at(n)
            -> a function with no arguments. It runs one
            call of your split-K paged attention, for n
            sequences.
        n
            the size to measure at
        facts
            facts.bandwidth_bytes_per_s   the read bandwidth of this card
            facts.context_len    the tokens in the context of each sequence
            facts.num_kv_heads   the KV heads
            facts.head_dim       the numbers in one head
            facts.value_bytes    the bytes of one number (fp16: 2)
        cudalib.bench_ms(function)
            -> the milliseconds of one call: warmed up, waited for, repeated

    ASKED: one line for each number
        attention_call_floor_ms
            predict: the same bytes as in stage 07
        attention_call_measured_ms
            measure: the time of step_at(n)
        attention_call_measured_over_floor
            divide
        attention_call_ms_2n_over_n
            double: the time at 2n over the time at n
    """
    import cudalib

    attention_call_floor_ms = ...
    attention_call_measured_ms = ...
    attention_call_measured_over_floor = ...
    attention_call_ms_2n_over_n = ...
    return {
        "attention_call_floor_ms": attention_call_floor_ms,
        "attention_call_measured_ms": attention_call_measured_ms,
        "attention_call_measured_over_floor": attention_call_measured_over_floor,
        "attention_call_ms_2n_over_n": attention_call_ms_2n_over_n,
    }
