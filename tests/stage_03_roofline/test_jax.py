"""Stage 03 (JAX) - Prefill vs decode: two different machines.

The spec:

    app/j03_roofline.py must define

        model_bytes(model) -> int
        time_prefill(model, n_tokens, iters=10) -> float   # ms per FORWARD PASS
        time_decode(model, ctx_len, steps=40) -> float     # ms per TOKEN
        achieved_gbs(nbytes, ms_per_token) -> float

Everything here rests on your timings being real. If you did not block on the
device, the first check will catch you.
"""

import jax
import jax.numpy as jnp
import pytest

from app.j03_roofline import achieved_gbs, model_bytes, time_decode, time_prefill

PEAK_GBS = 380.0  # your card, per ./vc info


def test_model_bytes(jmodel):
    got = model_bytes(jmodel)
    want = sum(int(x.size) * x.dtype.itemsize
               for x in jax.tree.leaves(jmodel.params))
    assert got == want
    print(f"\n  model is {got / 1e9:.2f} GB in bf16")


def test_achieved_gbs_math():
    # 14 GB read in 37 ms -> ~378 GB/s
    assert achieved_gbs(14e9, 37.0) == pytest.approx(378, rel=0.02)


def test_timings_are_not_measuring_python(jmodel):
    """The async trap, caught directly.

    A decode step must read every weight in the model. That puts a HARD FLOOR
    under ms/token: you cannot go faster than model_bytes / peak_bandwidth, and
    nothing about JAX changes physics. A number below the floor means the timed
    loop returned before the GPU did -- you measured dispatch, not compute.
    """
    ms = time_decode(jmodel, ctx_len=128, steps=20)
    floor_ms = model_bytes(jmodel) / (PEAK_GBS * 1e9) * 1000

    print(f"\n  measured:       {ms:6.3f} ms/token")
    print(f"  roofline floor: {floor_ms:6.3f} ms/token")
    assert ms > floor_ms * 0.9, (
        f"{ms:.3f} ms/token is below the {floor_ms:.3f} ms roofline floor. "
        "That is not possible on this hardware -- you are timing enqueue, not "
        "execution. jax.block_until_ready() the last output before stopping "
        "the clock."
    )


def test_prefill_amortizes_and_then_stops(jmodel):
    """Prefill cost per token falls as the prompt grows -- up to a point.

    A short prompt cannot fill the GPU; a longer one can, so the big GEMMs
    amortize and us/token drops. Prefill is compute-bound and wants big chunks,
    the opposite of what decode wants, which is why stage 11 has a knob at all.

    Then it turns back UP, and the reason matters. jvllm's attention is the
    dense reference: it materializes the full (T, S) score matrix. That is
    O(T^2) memory traffic, so past a few hundred tokens the score matrix, not
    the weights, is what you are paying for. FlashAttention exists precisely to
    delete this term, and stage 08 is where you write the online softmax that
    does it.
    """
    per_token = {}
    for n in (128, 512, 2048):
        per_token[n] = time_prefill(jmodel, n) / n * 1000  # us/token
        print(f"\n  prefill {n:>5} tok: {per_token[n]:6.1f} us/token")

    assert per_token[512] < per_token[128], (
        "cost per token should drop from 128 to 512 tokens -- bigger GEMMs use "
        "the GPU better"
    )
    print(f"\n  \033[2mThe turn-up at 2048 ({per_token[2048]:.0f} us/token, back")
    print(f"  near the {per_token[128]:.0f} of a 128-token prompt) is the score")
    print("  matrix, not the weights. It is quadratic, and stage 08 is where")
    print("  you stop materialising it.\033[0m")


def test_decode_is_memory_bound(jmodel):
    """Decode should achieve a large fraction of peak memory bandwidth.

    It does almost no arithmetic, so if it is not saturating the bus, the time
    is going somewhere else -- and finding out where is the point.
    """
    nbytes = model_bytes(jmodel)
    ms = time_decode(jmodel, ctx_len=128)
    gbs = achieved_gbs(nbytes, ms)
    floor_ms = nbytes / (PEAK_GBS * 1e9) * 1000

    print(f"\n  decode:          {ms:6.2f} ms/token")
    print(f"  roofline floor:  {floor_ms:6.2f} ms/token")
    print(f"  achieved:        {gbs:6.0f} GB/s  ({gbs / PEAK_GBS * 100:.0f}% of peak)")
    print(f"  overhead factor: {ms / floor_ms:6.2f}x")

    assert gbs > 0.20 * PEAK_GBS, (
        f"only {gbs:.0f} GB/s of {PEAK_GBS:.0f}. Decode should be dominated by "
        "reading weights. Are you recompiling inside the loop, or rebuilding "
        "the cache every step?"
    )
    print(f"\n  \033[2mThe gap between {floor_ms:.2f} and {ms:.2f} ms is dispatch")
    print("  and per-step Python. XLA already fused the step into one graph,")
    print("  so stage 12 attacks this from the other side: it makes sure the")
    print("  graph is REUSED rather than rebuilt.\033[0m")


def test_decode_barely_cares_about_context_length(jmodel):
    """Doubling the context does NOT double decode time.

    Reading the WEIGHTS dominates, and that cost is fixed. This is why long
    context hurts memory CAPACITY far more than speed -- and why stages 06-09
    are a fight about fitting KV in VRAM, not about attention being slow.

    The JAX-specific wrinkle, and it is worth sitting with: the cache is
    PREALLOCATED and rewritten functionally. Every step reads the whole buffer
    to attend over it and writes a whole new one to update it, whether or not
    the buffer is full. So you pay for the ceiling you chose, not the context
    you actually hold -- and this track's number climbs faster with max_len
    than the torch track's does with real context.

    It is still nothing like linear, which is the point. But the residue is
    real, and it is not something a compiler flag fixes: buffer donation does
    not help, because the copy happens per LAYER inside the scan. What fixes it
    is making the buffer granular so a step only touches the blocks it needs --
    which is PagedAttention, stages 06 through 09.
    """
    short = time_decode(jmodel, ctx_len=128)
    long = time_decode(jmodel, ctx_len=1024)
    print(f"\n   128 ctx: {short:.2f} ms/token")
    print(f"  1024 ctx: {long:.2f} ms/token   ({long / short:.2f}x)")
    assert long < short * 1.8, (
        f"8x the context cost {long / short:.2f}x the time. Some growth is "
        "expected here (see the docstring), but that is too much -- is the "
        "cache being rebuilt or re-zeroed inside the loop?"
    )
    print("  \033[2m8x the context, well under 2x the time. Reading the WEIGHTS")
    print("  still dominates, and that cost is fixed.\033[0m")


def test_prefill_and_decode_are_different_machines(jmodel):
    """The headline number for this stage."""
    prefill_us = time_prefill(jmodel, 2048) / 2048 * 1000
    decode_us = time_decode(jmodel, ctx_len=2048) * 1000

    ratio = decode_us / prefill_us
    print(f"\n  prefill: {prefill_us:8.1f} us/token   (compute-bound, big GEMMs)")
    print(f"  decode:  {decode_us:8.1f} us/token   (memory-bound, one token)")
    print(f"\n  \033[1mDecode costs {ratio:.0f}x more per token than prefill.\033[0m")
    assert ratio > 20, (
        f"only {ratio:.0f}x -- expected a large gap. These are supposed to be "
        "two completely different workloads."
    )
    print("  \033[2mSame weights, same model, same GPU. The only difference is")
    print("  how many tokens you feed per pass. That is the whole stage.\033[0m")
