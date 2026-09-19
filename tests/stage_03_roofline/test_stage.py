"""Stage 03 - prefill and decode: two different machines.

The spec:

    app/s03_roofline.py must define

        model_bytes(model) -> int
        time_prefill(model, num_tokens, iters=10) -> float  # ms for each FORWARD PASS
        time_decode(model, context_len, steps=40) -> float  # ms for each TOKEN
        achieved_gbs(num_bytes, ms_per_token) -> float

This stage adds no new capability. It makes a NUMBER, and the next seventeen
stages change that number. Do not skip it.
"""

import pytest

from app.s03_roofline import achieved_gbs, model_bytes, time_decode, time_prefill


def test_model_bytes(hf):
    model, _ = hf
    measured = model_bytes(model)
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert measured == expected
    print(f"\n  the model is {measured / 1e9:.2f} GB in bf16")


def test_achieved_gbs_math():
    # 14 GB read in 37 ms -> about 378 GB/s
    assert achieved_gbs(14e9, 37.0) == pytest.approx(378, rel=0.02)


def test_prefill_amortizes(hf):
    """The prefill cost for each token DECREASES as the prompt grows.

    A short prompt cannot fill the GPU. A long one can. Prefill is
    compute-bound, so it wants large chunks. Decode wants the opposite, and
    that is why stage 11 (chunked prefill) has a setting to tune.
    """
    model, _ = hf
    us_per_token = {}
    for num_tokens in (128, 512, 2048):
        us_per_token[num_tokens] = (time_prefill(model, num_tokens)
                                    / num_tokens * 1000)
        print(f"\n  prefill {num_tokens:>5} tok: "
              f"{us_per_token[num_tokens]:6.1f} us/token")
    assert us_per_token[2048] < us_per_token[128], (
        "the cost for each token must decrease as the prompt grows. Larger "
        "GEMMs use the GPU better.")


def test_decode_is_memory_bound(hf, peak_gbs):
    """Decode must get a large fraction of the peak memory bandwidth.

    It does almost no arithmetic. So if it does not fill the bus, the time
    goes somewhere else (Python, kernel launches). That is the finding.
    """
    model, _ = hf
    num_bytes = model_bytes(model)
    decode_ms = time_decode(model, context_len=128)
    gbs = achieved_gbs(num_bytes, decode_ms)
    floor_ms = num_bytes / (peak_gbs * 1e9) * 1000

    print(f"\n  decode:          {decode_ms:6.2f} ms/token")
    print(f"  roofline floor:  {floor_ms:6.2f} ms/token")
    print(f"  achieved:        {gbs:6.0f} GB/s  "
          f"({gbs / peak_gbs * 100:.0f}% of the measured peak)")
    print(f"  overhead factor: {decode_ms / floor_ms:6.2f}x")
    assert gbs > 0.20 * peak_gbs, (
        f"only {gbs:.0f} GB/s of {peak_gbs:.0f}. The weight reads must be "
        "most of decode. Do you time without cuda.synchronize(), or copy the "
        "cache inside the loop?")
    print(f"\n  \033[2mThe gap between {floor_ms:.2f} and {decode_ms:.2f} ms is "
          "the overhead of")
    print("  Python and kernel launches. You remove it in stage 12.\033[0m")


def test_decode_barely_cares_about_context_length(hf):
    """Two times the context does NOT give two times the decode time.

    Attention over the cache is a small part of the work. The WEIGHT reads
    are most of it, and that cost does not change. That is why long context hurts
    memory capacity much more than speed. It is also why stages 06 to 09 are
    about fitting KV in VRAM, not about slow attention.
    """
    model, _ = hf
    short_ms = time_decode(model, context_len=128)
    long_ms = time_decode(model, context_len=1024)
    print(f"\n   128 ctx: {short_ms:.2f} ms/token")
    print(f"  1024 ctx: {long_ms:.2f} ms/token   ({long_ms / short_ms:.2f}x)")
    assert long_ms < short_ms * 1.5, (
        f"8x the context cost {long_ms / short_ms:.2f}x the time. "
        "Expected about 1x.")


def test_prefill_and_decode_are_different_machines(hf):
    """The main number of this stage."""
    model, _ = hf
    prefill_us = time_prefill(model, 2048) / 2048 * 1000
    decode_us = time_decode(model, context_len=2048) * 1000
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
