"""Stage 11 - chunked prefill and mixed batches.

The spec is in app/s11_chunked.py. The main check measures the latency
between tokens of the sequences that already stream when a very large prompt
arrives.
"""

from app.s06_blocks import BlockAllocator
from app.s11_chunked import ChunkedScheduler, UnchunkedScheduler


def make_scheduler(scheduler_class, token_budget=512, num_blocks=4000,
                   max_num_seqs=8):
    return scheduler_class(BlockAllocator(num_blocks, 16),
                           max_num_seqs=max_num_seqs, token_budget=token_budget)


def test_never_exceeds_the_token_budget():
    scheduler = make_scheduler(ChunkedScheduler, token_budget=128)
    scheduler.add_request("big", 5000, 3)
    for index in range(4):
        scheduler.add_request(f"decoder{index}", 16, 50)
    while scheduler.has_work():
        tokens_used = scheduler.step()["tokens_used"]
        assert tokens_used <= 128, (
            f"a step used {tokens_used} tokens, the budget is 128")


def test_long_prompt_is_split_into_chunks():
    scheduler = make_scheduler(ChunkedScheduler, token_budget=512)
    scheduler.add_request("big", 5000, 1)
    chunk_lens = []
    while scheduler.has_work():
        chunk_lens += [chunk_len for rid, chunk_len
                       in scheduler.step()["prefill"] if rid == "big"]
    assert len(chunk_lens) >= 5000 // 512, (
        "a prompt of 5000 tokens under a budget of 512 needs at least 9 "
        f"chunks, got {len(chunk_lens)}")
    assert sum(chunk_lens) == 5000, (
        f"the chunks add up to {sum(chunk_lens)}, and the prompt was 5000. "
        "Tokens were lost or counted two times.")


def test_short_prompt_is_not_chunked():
    scheduler = make_scheduler(ChunkedScheduler, token_budget=512)
    scheduler.add_request("small", 100, 1)
    assert scheduler.step()["prefill"] == [("small", 100)]


def test_everything_still_completes_correctly():
    scheduler = make_scheduler(ChunkedScheduler, token_budget=256)
    budgets = {"a": 20, "b": 5, "big": 12}
    scheduler.add_request("a", 40, 20)
    scheduler.add_request("b", 2000, 5)
    scheduler.add_request("big", 6000, 12)
    finished = scheduler.run_to_completion()
    assert {seq.id for seq in finished} == set(budgets)
    for seq in finished:
        assert seq.num_generated == budgets[seq.id]
        assert seq.num_computed == seq.prompt_len
    assert scheduler.allocator.num_free == 4000, "blocks were lost"


def test_prefill_and_decode_share_a_step():
    """The "mixed batch" part: both kinds of work in one forward pass."""
    scheduler = make_scheduler(ChunkedScheduler, token_budget=512)
    for index in range(3):
        scheduler.add_request(f"decoder{index}", 16, 100)
    for _ in range(3):
        scheduler.step()
    scheduler.add_request("big", 4000, 5)

    mixed_steps = 0
    for _ in range(20):
        record = scheduler.step()
        if record["prefill"] and record["decoded"]:
            mixed_steps += 1
    assert mixed_steps > 0, (
        "no step had both a prefill chunk and decode tokens. The budget must "
        "be shared, not given to one kind of work.")


def _latency_gaps(scheduler_class, token_budget=512):
    """Run 4 streaming decoders, add a very large prompt during the stream,
    and record the gap between adjacent tokens of the decoders. The time is
    in tokens of work, which is the cost of a step. -> (steps, sorted gaps).
    """
    scheduler = make_scheduler(scheduler_class, token_budget=token_budget)
    for index in range(4):
        scheduler.add_request(f"decoder{index}", 32, 400)
    work_done, last_token_time, gaps = 0.0, {}, []
    for step in range(20000):
        if not scheduler.has_work():
            break
        if step == 30:
            scheduler.add_request("BIG", 8000, 5)
        record = scheduler.step()
        work_done += record["tokens_used"]
        for seq in record["decoded"]:
            if not seq.id.startswith("decoder"):
                continue
            if seq.id in last_token_time:
                gaps.append(work_done - last_token_time[seq.id])
            last_token_time[seq.id] = work_done
    return scheduler.steps, sorted(gaps)


def _median_p99_max(sorted_gaps):
    return (sorted_gaps[len(sorted_gaps) // 2],
            sorted_gaps[int(len(sorted_gaps) * 0.99) - 1], max(sorted_gaps))


def test_chunking_bounds_the_latency_spike():
    """The main check. A long prefill must not stop the streams of the
    others."""
    unchunked_steps, unchunked_gaps = _latency_gaps(UnchunkedScheduler)
    chunked_steps, chunked_gaps = _latency_gaps(ChunkedScheduler)
    unchunked_median, unchunked_p99, unchunked_max = _median_p99_max(
        unchunked_gaps)
    chunked_median, chunked_p99, chunked_max = _median_p99_max(chunked_gaps)

    print("\n  4 sequences stream. A prompt of 8000 tokens arrives during the "
          "stream.")
    print("  The latency is in tokens of work, the cost of a step.\n")
    print(f"  {'':>10} {'steps':>7} {'median':>9} {'p99':>9} {'MAX':>9}")
    print(f"  {'unchunked':>10} {unchunked_steps:>7} {unchunked_median:>9.0f} "
          f"{unchunked_p99:>9.0f} {unchunked_max:>9.0f}")
    print(f"  {'chunked':>10} {chunked_steps:>7} {chunked_median:>9.0f} "
          f"{chunked_p99:>9.0f} {chunked_max:>9.0f}")
    print(f"\n  \033[1mWorst stop: {unchunked_max:.0f} -> {chunked_max:.0f} "
          f"({unchunked_max / chunked_max:.0f}x better)\033[0m")

    assert chunked_max <= 512, (
        f"the chunked worst case was {chunked_max}. The budget must limit it.")
    assert chunked_max < unchunked_max / 5, (
        f"chunking helped very little: {unchunked_max} -> {chunked_max}")
    assert chunked_median == unchunked_median, (
        "the median latency must not change. Chunking fixes the TAIL. It must "
        "not make the usual case faster.")
    print("\n  \033[2mThe median is the same. Chunking makes nothing faster.")
    print("  It stops one request from taking a whole step. That is why it is")
    print("  a latency setting and not a throughput setting.\033[0m")


def test_budget_size_is_the_latency_throughput_dial():
    """A smaller budget -> smaller peaks, more steps. Show the exchange."""
    rows = []
    for token_budget in (128, 512, 2048):
        steps, gaps = _latency_gaps(ChunkedScheduler, token_budget)
        rows.append((token_budget, steps, max(gaps)))

    print(f"\n  {'budget':>8} {'steps':>7} {'worst ITL':>11}")
    for token_budget, steps, worst_gap in rows:
        print(f"  {token_budget:>8} {steps:>7} {worst_gap:>11.0f}")
    assert rows[0][2] < rows[-1][2], (
        "a smaller budget must limit the peak more")
    print("\n  \033[2mThis is the setting. Make it small for chat, and large")
    print("  for batch jobs.\033[0m")


def test_prefill_first_policy_is_worse_for_streaming():
    """Both policies are correct. The check only asks you to make them
    different."""
    scheduler = ChunkedScheduler(BlockAllocator(4000, 16), max_num_seqs=8,
                                 token_budget=512, prefill_first=True)
    for index in range(3):
        scheduler.add_request(f"decoder{index}", 32, 20)
    scheduler.add_request("BIG", 4000, 2)
    for _ in range(3):
        scheduler.step()
    # With prefill_first and a budget of 512, the chunk takes the whole budget.
    assert scheduler.step()["prefill"], (
        "prefill_first must schedule the prefill chunks first")
