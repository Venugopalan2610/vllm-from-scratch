"""Stage 03 - Prefill vs decode: two different machines.

The spec:

    app/s03_roofline.py must define

        model_bytes(model) -> int
        time_prefill(model, n_tokens, iters=10) -> float   # ms per FORWARD PASS
        time_decode(model, ctx_len, steps=40) -> float     # ms per TOKEN
        achieved_gbs(nbytes, ms_per_token) -> float

This stage produces no new capability. It produces a NUMBER you will spend the
next seventeen stages moving. Do not skip it.
"""

import pytest
import torch

from app.s03_roofline import achieved_gbs, model_bytes, time_decode, time_prefill

PEAK_GBS = 380.0  # your card, per ./vc info


def test_model_bytes(hf):
    model, _ = hf
    got = model_bytes(model)
    want = sum(p.numel() * p.element_size() for p in model.parameters())
    assert got == want
    print(f"\n  model is {got / 1e9:.2f} GB in bf16")


def test_achieved_gbs_math():
    # 14 GB read in 37 ms -> ~378 GB/s
    assert achieved_gbs(14e9, 37.0) == pytest.approx(378, rel=0.02)


def test_prefill_amortizes(hf):
    """Prefill cost per token FALLS as the prompt grows.

    A short prompt can't fill the GPU; a long one can. Prefill is compute-bound,
    so it wants big chunks. This is the opposite of decode's preference, and it
    is why stage 11 (chunked prefill) has a tuning knob at all.
    """
    model, _ = hf
    per_token = {}
    for n in (128, 512, 2048):
        per_token[n] = time_prefill(model, n) / n * 1000  # us/token
        print(f"\n  prefill {n:>5} tok: {per_token[n]:6.1f} us/token")

    assert per_token[2048] < per_token[128], (
        "cost per token should drop as the prompt grows -- "
        "bigger GEMMs use the GPU better"
    )


def test_decode_is_memory_bound(hf):
    """Decode should achieve a large fraction of peak memory bandwidth.

    It is doing almost no arithmetic, so if it isn't saturating the bus, the
    time is going somewhere else (Python, kernel launches) -- which is itself
    the finding.
    """
    model, _ = hf
    nbytes = model_bytes(model)
    ms = time_decode(model, ctx_len=128)
    gbs = achieved_gbs(nbytes, ms)
    floor_ms = nbytes / (PEAK_GBS * 1e9) * 1000

    print(f"\n  decode:          {ms:6.2f} ms/token")
    print(f"  roofline floor:  {floor_ms:6.2f} ms/token")
    print(f"  achieved:        {gbs:6.0f} GB/s  ({gbs / PEAK_GBS * 100:.0f}% of peak)")
    print(f"  overhead factor: {ms / floor_ms:6.2f}x")

    assert gbs > 0.20 * PEAK_GBS, (
        f"only {gbs:.0f} GB/s of {PEAK_GBS:.0f}. Decode should be dominated by "
        "reading weights. Are you timing without cuda.synchronize(), or copying "
        "the cache inside the loop?"
    )
    print(f"\n  \033[2mThe gap between {floor_ms:.2f} and {ms:.2f} ms is Python and")
    print(f"  kernel-launch overhead. You will delete it in stage 12.\033[0m")


def test_decode_barely_cares_about_context_length(hf):
    """Doubling the context does NOT double decode time.

    Attention over the cache is a small part of the work; reading the WEIGHTS
    dominates, and that cost is fixed. This is why long context hurts memory
    capacity far more than it hurts speed -- and why the fight in stages 06-09
    is about fitting KV in VRAM, not about attention being slow.
    """
    model, _ = hf
    short = time_decode(model, ctx_len=128)
    long = time_decode(model, ctx_len=1024)
    print(f"\n   128 ctx: {short:.2f} ms/token")
    print(f"  1024 ctx: {long:.2f} ms/token   ({long / short:.2f}x)")
    assert long < short * 1.5, (
        f"8x the context cost {long / short:.2f}x the time -- expected near 1x"
    )


def test_prefill_and_decode_are_different_machines(hf):
    """The headline number for this stage."""
    model, _ = hf
    prefill_us = time_prefill(model, 2048) / 2048 * 1000
    decode_us = time_decode(model, ctx_len=2048) * 1000

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
