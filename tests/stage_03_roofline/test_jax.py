"""Stage 03 (JAX) - prefill and decode: two different machines.

The spec:

    app/j03_roofline.py must define

        model_bytes(model) -> int
        time_prefill(model, num_tokens, iters=10) -> float  # ms for each FORWARD PASS
        time_decode(model, context_len, steps=40) -> float  # ms for each TOKEN
        achieved_gbs(num_bytes, ms_per_token) -> float

Everything here depends on real timings. If you did not block on the
device, the first check finds it.
"""

import jax
import pytest

from app.j03_roofline import achieved_gbs, model_bytes, time_decode, time_prefill


def test_model_bytes(jmodel):
    measured = model_bytes(jmodel)
    expected = sum(int(leaf.size) * leaf.dtype.itemsize
                   for leaf in jax.tree.leaves(jmodel.params))
    assert measured == expected
    print(f"\n  the model is {measured / 1e9:.2f} GB in bf16")


def test_achieved_gbs_math():
    # 14 GB read in 37 ms -> about 378 GB/s
    assert achieved_gbs(14e9, 37.0) == pytest.approx(378, rel=0.02)


def test_timings_are_not_measuring_python(jmodel, peak_gbs):
    """The async trap, found directly.

    A decode step must read every weight of the model. That puts a HARD
    FLOOR under ms/token: nothing is faster than model_bytes /
    peak_bandwidth, and JAX does not change physics. A number below the
    floor means that the timed loop returned before the GPU finished. You
    measured the dispatch, not the compute.
    """
    decode_ms = time_decode(jmodel, context_len=128, steps=20)
    floor_ms = model_bytes(jmodel) / (peak_gbs * 1e9) * 1000
    print(f"\n  measured:       {decode_ms:6.3f} ms/token")
    print(f"  roofline floor: {floor_ms:6.3f} ms/token")
    assert decode_ms > floor_ms * 0.9, (
        f"{decode_ms:.3f} ms/token is below the {floor_ms:.3f} ms roofline "
        "floor. That is not possible on this hardware. You time the enqueue, "
        "not the execution. Call jax.block_until_ready() on the last output "
        "before you stop the clock.")


def test_prefill_amortizes_and_then_stops(jmodel):
    """The prefill cost for each token decreases as the prompt grows, up to
    a point.

    A short prompt cannot fill the GPU. A longer one can, so the large GEMMs
    amortize, and us/token decreases. Prefill is compute-bound and wants
    large chunks, the opposite of decode. That is why stage 11 has a
    setting.

    Then it increases again, and the reason is important. The attention of
    jvllm is the dense reference: it makes the full (T, S) score matrix in
    memory. That is O(T^2) memory traffic. So after a few hundred tokens you
    pay for the score matrix, not for the weights. FlashAttention exists to
    remove this term, and in stage 08 you write the online softmax that
    removes it.
    """
    us_per_token = {}
    for num_tokens in (128, 512, 2048):
        us_per_token[num_tokens] = (time_prefill(jmodel, num_tokens)
                                    / num_tokens * 1000)
        print(f"\n  prefill {num_tokens:>5} tok: "
              f"{us_per_token[num_tokens]:6.1f} us/token")
    assert us_per_token[512] < us_per_token[128], (
        "the cost for each token must decrease from 128 to 512 tokens. Larger "
        "GEMMs use the GPU better.")
    print(f"\n  \033[2mThe increase at 2048 ({us_per_token[2048]:.0f} us/token, "
          "near the")
    print(f"  {us_per_token[128]:.0f} of a 128-token prompt) is the score "
          "matrix, not the weights.")
    print("  It is quadratic, and in stage 08 you stop making it.\033[0m")


def test_decode_is_memory_bound(jmodel, peak_gbs):
    """Decode must get a large fraction of the peak memory bandwidth.

    It does almost no arithmetic. So if it does not fill the bus, the time
    goes somewhere else, and the point is to find where.
    """
    num_bytes = model_bytes(jmodel)
    decode_ms = time_decode(jmodel, context_len=128)
    gbs = achieved_gbs(num_bytes, decode_ms)
    floor_ms = num_bytes / (peak_gbs * 1e9) * 1000

    print(f"\n  decode:          {decode_ms:6.2f} ms/token")
    print(f"  roofline floor:  {floor_ms:6.2f} ms/token")
    print(f"  achieved:        {gbs:6.0f} GB/s  "
          f"({gbs / peak_gbs * 100:.0f}% of the measured peak)")
    print(f"  overhead factor: {decode_ms / floor_ms:6.2f}x")
    assert gbs > 0.20 * peak_gbs, (
        f"only {gbs:.0f} GB/s of {peak_gbs:.0f}. The weight reads must be most "
        "of decode. Do you compile inside the loop, or make the cache again "
        "at every step?")
    print(f"\n  \033[2mThe gap between {floor_ms:.2f} and {decode_ms:.2f} ms is "
          "the dispatch")
    print("  and the Python of each step. XLA already fused the step into one")
    print("  graph, so stage 12 works on the other side: it makes sure that")
    print("  the graph is USED AGAIN, not made again.\033[0m")


def test_decode_barely_cares_about_context_length(jmodel):
    """Two times the context does NOT give two times the decode time.

    The WEIGHT reads are most of the work, and that cost does not change.
    That is
    why long context hurts memory CAPACITY much more than speed. It is also
    why stages 06 to 09 are about fitting KV in VRAM, not about slow
    attention.

    The special JAX detail, and it is worth some thought: jvllm ALLOCATES
    the cache BEFORE the run and writes it functionally. Every step reads the whole buffer
    to attend over it, and writes a whole new buffer to update it, full or
    not. So you pay for the limit that you selected, not for the context
    that you hold. On this track the number increases faster with max_len
    than on the torch track with real context.

    It is still far from linear, and that is the point. But the rest is
    real, and no compiler flag fixes it. Buffer donation does not help,
    because the copy occurs for each LAYER inside the scan. The fix is a
    buffer in small parts, so that a step touches only the blocks that it
    needs. That is PagedAttention, stages 06 to 09.
    """
    short_ms = time_decode(jmodel, context_len=128)
    long_ms = time_decode(jmodel, context_len=1024)
    print(f"\n   128 ctx: {short_ms:.2f} ms/token")
    print(f"  1024 ctx: {long_ms:.2f} ms/token   ({long_ms / short_ms:.2f}x)")
    assert long_ms < short_ms * 1.8, (
        f"8x the context cost {long_ms / short_ms:.2f}x the time. Some "
        "increase is expected here (see the docstring), but that is too much. "
        "Do you make the cache again, or zero it, inside the loop?")
    print("  \033[2m8x the context, much less than 2x the time. The WEIGHT")
    print("  reads are still most of the work, and that cost does not")
    print("  change.\033[0m")


def test_prefill_and_decode_are_different_machines(jmodel):
    """The main number of this stage."""
    prefill_us = time_prefill(jmodel, 2048) / 2048 * 1000
    decode_us = time_decode(jmodel, context_len=2048) * 1000
    ratio = decode_us / prefill_us

    print(f"\n  prefill: {prefill_us:8.1f} us/token   (compute-bound, large GEMMs)")
    print(f"  decode:  {decode_us:8.1f} us/token   (memory-bound, one token)")
    print(f"\n  \033[1mDecode costs {ratio:.0f}x more for each token than "
          "prefill.\033[0m")
    assert ratio > 20, (
        f"only {ratio:.0f}x. Expected a large gap. These are two completely "
        "different workloads.")
    print("  \033[2mThe same weights, the same model, the same GPU. The only")
    print("  difference is the number of tokens in each pass. That is the")
    print("  whole stage.\033[0m")
