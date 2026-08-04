"""Stage 11 - Chunked prefill and mixed batches.

Spec in app/s11_chunked.py. The headline test measures inter-token latency for
sequences that are already streaming when a huge prompt arrives.
"""

import pytest

from app.s06_blocks import BlockAllocator
from app.s11_chunked import ChunkedScheduler, UnchunkedScheduler


def make(cls, budget=512, blocks=4000, max_num_seqs=8):
    return cls(BlockAllocator(blocks, 16), max_num_seqs=max_num_seqs,
               token_budget=budget)


def test_never_exceeds_the_token_budget():
    s = make(ChunkedScheduler, budget=128)
    s.add_request("big", 5000, 3)
    for i in range(4):
        s.add_request(f"d{i}", 16, 50)
    while s.has_work():
        rec = s.step()
        assert rec["tokens_used"] <= 128, (
            f"step used {rec['tokens_used']} tokens, budget is 128"
        )


def test_long_prompt_is_split_into_chunks():
    s = make(ChunkedScheduler, budget=512)
    s.add_request("big", 5000, 1)
    chunks = []
    while s.has_work():
        for rid, n in s.step()["prefill"]:
            if rid == "big":
                chunks.append(n)
    assert len(chunks) >= 5000 // 512, (
        f"a 5000-token prompt under a 512 budget needs at least 9 chunks, "
        f"got {len(chunks)}"
    )
    assert sum(chunks) == 5000, (
        f"chunks sum to {sum(chunks)}, prompt was 5000 -- tokens lost or double "
        "counted"
    )


def test_short_prompt_is_not_chunked():
    s = make(ChunkedScheduler, budget=512)
    s.add_request("small", 100, 1)
    rec = s.step()
    assert rec["prefill"] == [("small", 100)]


def test_everything_still_completes_correctly():
    s = make(ChunkedScheduler, budget=256)
    want = {"a": 20, "b": 5, "big": 12}
    s.add_request("a", 40, 20)
    s.add_request("b", 2000, 5)
    s.add_request("big", 6000, 12)
    fin = s.run_to_completion()
    assert {f.id for f in fin} == set(want)
    for f in fin:
        assert f.num_generated == want[f.id]
        assert f.num_computed == f.prompt_len
    assert s.alloc.num_free == 4000, "blocks leaked"


def test_prefill_and_decode_share_a_step():
    """The 'mixed batch' part. Both kinds of work in one forward pass."""
    s = make(ChunkedScheduler, budget=512)
    for i in range(3):
        s.add_request(f"d{i}", 16, 100)
    for _ in range(3):
        s.step()
    s.add_request("big", 4000, 5)

    mixed = 0
    for _ in range(20):
        rec = s.step()
        if rec["prefill"] and rec["decoded"]:
            mixed += 1
    assert mixed > 0, (
        "no step contained both a prefill chunk and decode tokens -- the "
        "budget should be shared, not handed entirely to one kind of work"
    )


def _itl_trace(cls, budget=512):
    """Run 4 streaming decoders, inject a huge prompt mid-stream, and record
    the gap between consecutive tokens for the decoders. 'Time' is measured in
    tokens of work, which is what a step actually costs."""
    s = make(cls, budget=budget)
    for i in range(4):
        s.add_request(f"d{i}", 32, 400)
    t, last, gaps = 0.0, {}, []
    for n in range(20000):
        if not s.has_work():
            break
        if n == 30:
            s.add_request("BIG", 8000, 5)
        rec = s.step()
        t += rec["tokens_used"]
        for seq in rec["decoded"]:
            if seq.id.startswith("d"):
                if seq.id in last:
                    gaps.append(t - last[seq.id])
                last[seq.id] = t
    gaps.sort()
    return s.steps, gaps


def test_chunking_bounds_the_latency_spike():
    """The headline. A long prefill must not freeze everyone else's stream."""
    un_steps, un = _itl_trace(UnchunkedScheduler)
    ch_steps, ch = _itl_trace(ChunkedScheduler, budget=512)

    def stats(g):
        return g[len(g) // 2], g[int(len(g) * 0.99) - 1], max(g)

    un_med, un_p99, un_max = stats(un)
    ch_med, ch_p99, ch_max = stats(ch)

    print(f"\n  4 sequences streaming; an 8000-token prompt arrives mid-flight.")
    print(f"  Latency in tokens-of-work (what a step actually costs).\n")
    print(f"  {'':>10} {'steps':>7} {'median':>9} {'p99':>9} {'MAX':>9}")
    print(f"  {'unchunked':>10} {un_steps:>7} {un_med:>9.0f} {un_p99:>9.0f} {un_max:>9.0f}")
    print(f"  {'chunked':>10} {ch_steps:>7} {ch_med:>9.0f} {ch_p99:>9.0f} {ch_max:>9.0f}")
    print(f"\n  \033[1mWorst-case stall: {un_max:.0f} -> {ch_max:.0f} "
          f"({un_max / ch_max:.0f}x better)\033[0m")

    assert ch_max <= 512, (
        f"chunked worst case was {ch_max}, should be bounded by the budget"
    )
    assert ch_max < un_max / 5, (
        f"chunking barely helped: {un_max} -> {ch_max}"
    )
    assert ch_med == un_med, (
        "median latency should be unchanged -- chunking fixes the TAIL, it is "
        "not supposed to make the common case faster"
    )
    print("\n  \033[2mMedian is identical. Chunking does not make anything faster;")
    print("  it stops one request from monopolising a step. That is why it is a")
    print("  latency knob and not a throughput knob.\033[0m")


def test_budget_size_is_the_latency_throughput_dial():
    """Smaller budget -> smaller spikes, more steps. Show the trade."""
    rows = []
    for b in (128, 512, 2048):
        steps, gaps = _itl_trace(ChunkedScheduler, budget=b)
        rows.append((b, steps, max(gaps)))

    print(f"\n  {'budget':>8} {'steps':>7} {'worst ITL':>11}")
    for b, st, mx in rows:
        print(f"  {b:>8} {st:>7} {mx:>11.0f}")

    assert rows[0][2] < rows[-1][2], "a smaller budget must bound the spike more tightly"
    print("\n  \033[2mThis is the dial. Turn it down for chat, up for batch jobs.\033[0m")


def test_prefill_first_policy_is_worse_for_streaming():
    """Both policies are legitimate; the tests just check you can tell them apart."""
    a = BlockAllocator(4000, 16)
    s = ChunkedScheduler(a, max_num_seqs=8, token_budget=512, prefill_first=True)
    for i in range(3):
        s.add_request(f"d{i}", 32, 20)
    s.add_request("BIG", 4000, 2)
    for _ in range(3):
        s.step()
    rec = s.step()
    # with prefill_first and a 512 budget, the chunk eats the whole budget
    assert rec["prefill"], "prefill_first should schedule prefill chunks first"
