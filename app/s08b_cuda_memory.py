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


# ---------------------------------------------------------------- the four lines


def four_lines(step_at, n, facts):
    """The four moves of docs/METHOD.md: predict, measure, divide, double.
    The subject: one call of your coalesced paged attention, for n
    sequences.

    GIVEN
        step_at(n)
            -> a function with no arguments. It runs one
            call of your coalesced paged attention, for n
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
