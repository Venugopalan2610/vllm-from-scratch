"""Stage 10 - admission, queues, and preemption.

The spec is in app/s10_scheduler.py. There is no GPU. This is scheduling
logic, and it needs the hardest checks that you can write. Its failures are
leaks, starvation and livelock. In production all three look the same: "the
server became slow, and then it stopped".
"""

import random

import pytest

from app.s06_blocks import BlockAllocator
from app.s10_scheduler import Scheduler


def make_scheduler(num_blocks=64, block_size=16, max_num_seqs=8):
    return Scheduler(BlockAllocator(num_blocks, block_size), max_num_seqs)


def add_random_load(scheduler, seed, num_requests=20):
    """Random prompt lengths and budgets. -> {rid: max_tokens}."""
    rng = random.Random(seed)
    budgets = {}
    for rid in range(num_requests):
        prompt_len, max_tokens = rng.choice([16, 32, 64]), rng.choice([8, 20, 50])
        budgets[rid] = max_tokens
        scheduler.add_request(rid, prompt_len, max_tokens)
    return budgets


def test_respects_max_num_seqs():
    scheduler = make_scheduler(num_blocks=1000, max_num_seqs=3)
    for rid in range(10):
        scheduler.add_request(rid, prompt_len=16, max_tokens=5)
    while scheduler.has_work():
        scheduler.step()
        assert len(scheduler.running) <= 3, (
            f"the batch grew to {len(scheduler.running)}, the limit is 3")


def test_does_not_admit_what_it_cannot_fit():
    """Admission must look at the KV budget, not only at the number of
    sequences."""
    scheduler = make_scheduler(num_blocks=4, block_size=16, max_num_seqs=8)
    for rid in range(4):
        scheduler.add_request(rid, prompt_len=64, max_tokens=1)   # 4 blocks each
    scheduler.step()
    assert len(scheduler.running) == 1, (
        f"admitted {len(scheduler.running)} sequences into a pool that holds "
        "one")


def test_head_of_line_blocking_is_preserved():
    """The scheduler must not skip a large prompt at the head for a small
    prompt behind it.

    That looks like a throughput gain. It makes the long prompts wait with
    no limit. Use first come, first served at the head of the queue.
    """
    scheduler = make_scheduler(num_blocks=4, block_size=16, max_num_seqs=8)
    scheduler.add_request("big", prompt_len=64, max_tokens=2)    # all 4 blocks
    scheduler.add_request("small", prompt_len=16, max_tokens=2)  # fits next to it
    scheduler.step()
    running_ids = [seq.id for seq in scheduler.running]
    assert running_ids == ["big"], (
        f"expected only 'big' admitted, got {running_ids}")


def test_everything_finishes_with_the_right_token_count():
    scheduler = make_scheduler(num_blocks=200, block_size=16, max_num_seqs=8)
    budgets = add_random_load(scheduler, seed=0)
    finished = scheduler.run_to_completion()
    assert len(finished) == 20
    for seq in finished:
        assert seq.num_generated == budgets[seq.id]


def test_no_block_leaks():
    scheduler = make_scheduler(num_blocks=64, block_size=16)
    for rid in range(30):
        scheduler.add_request(rid, prompt_len=32, max_tokens=10)
    scheduler.run_to_completion()
    assert scheduler.allocator.num_free == 64, (
        f"lost {64 - scheduler.allocator.num_free} blocks. A path frees "
        "neither at the finish nor at a preemption.")


@pytest.mark.parametrize("num_blocks", [12, 16, 24])
def test_no_starvation_under_severe_pressure(num_blocks):
    """A pool much too small for the load must still finish all of it.

    This is the livelock check. A scheduler that always preempts the same
    victim, or admits again in the wrong order, runs for ever with no
    progress.
    """
    scheduler = make_scheduler(num_blocks=num_blocks, block_size=16,
                               max_num_seqs=8)
    budgets = add_random_load(scheduler, seed=1)
    finished = scheduler.run_to_completion(max_steps=50000)
    assert len(finished) == 20, (
        f"only {len(finished)}/20 finished with a pool of {num_blocks} blocks. "
        "That is livelock or starvation.")
    for seq in finished:
        assert seq.num_generated == budgets[seq.id], (
            f"seq {seq.id} made {seq.num_generated} tokens, expected "
            f"{budgets[seq.id]}. A preempted sequence must start again "
            "cleanly and still emit exactly max_tokens.")
    assert scheduler.allocator.num_free == num_blocks


def test_preemption_actually_happens_when_memory_is_tight():
    scheduler = make_scheduler(num_blocks=12, block_size=16, max_num_seqs=8)
    for rid in range(12):
        scheduler.add_request(rid, prompt_len=32, max_tokens=40)
    scheduler.run_to_completion()
    assert scheduler.preemptions > 0, (
        "a pool of 12 blocks with 12 growing sequences must preempt at some "
        "point")
    assert any(seq.preempted_count > 0 for seq in scheduler.finished)


def test_preemption_is_recompute_not_swap():
    """A preempted sequence gives its blocks back AT ONCE.

    That is the reason to use recompute: the memory is available in the
    next step, with no PCIe round trip.
    """
    scheduler = make_scheduler(num_blocks=8, block_size=16, max_num_seqs=8)
    for rid in range(6):
        scheduler.add_request(rid, prompt_len=16, max_tokens=60)

    preempted_any = False
    for _ in range(2000):
        if not scheduler.has_work():
            break
        for victim in scheduler.step()["preempted"]:
            preempted_any = True
            assert victim.blocks == [], (
                f"preempted seq {victim.id} still holds {len(victim.blocks)} "
                "blocks")
            assert victim.num_generated == 0, (
                "recompute: a preempted sequence starts again from its "
                "prompt, so num_generated is 0 again")
            assert not victim.prefilled
    assert preempted_any


def test_degrades_gracefully_rather_than_collapsing():
    """Overload must cost throughput, not correctness or completion."""
    rows = []
    for num_blocks in (200, 24, 12):
        scheduler = make_scheduler(num_blocks=num_blocks, block_size=16,
                                   max_num_seqs=8)
        add_random_load(scheduler, seed=0)
        scheduler.run_to_completion(max_steps=50000)
        rows.append((num_blocks, scheduler.steps, scheduler.preemptions,
                     len(scheduler.finished)))

    print(f"\n  {'pool':>6} {'steps':>7} {'preemptions':>12} {'finished':>9}")
    for num_blocks, steps, preemptions, num_finished in rows:
        print(f"  {num_blocks:>6} {steps:>7} {preemptions:>12} "
              f"{num_finished:>8}/20")

    roomy, tight, tightest = rows
    assert all(row[3] == 20 for row in rows)
    assert roomy[2] == 0, "a large pool must never preempt"
    assert tightest[2] > tight[2], "less memory must cause more preemptions"
    assert tightest[1] < roomy[1] * 20, (
        "the throughput collapsed, it did not degrade: thrashing")
    print("\n  \033[2mMore pressure costs more steps and more preemptions,")
    print("  but every request still completes. That is the property that")
    print("  you want: degrade, do not collapse.\033[0m")
