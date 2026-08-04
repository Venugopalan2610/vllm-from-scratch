"""Stage 10 - Admission, queues, and preemption.

Spec in app/s10_scheduler.py. No GPU: this is scheduling logic, and it is worth
testing to destruction because the failure modes (leaks, starvation, livelock)
all look like "the server got slow and then died" in production.
"""

import random

import pytest

from app.s06_blocks import BlockAllocator
from app.s10_scheduler import Scheduler


def sched(num_blocks=64, block_size=16, max_num_seqs=8):
    return Scheduler(BlockAllocator(num_blocks, block_size), max_num_seqs)


def test_respects_max_num_seqs():
    s = sched(num_blocks=1000, max_num_seqs=3)
    for i in range(10):
        s.add_request(i, prompt_len=16, max_tokens=5)
    while s.has_work():
        s.step()
        assert len(s.running) <= 3, f"batch grew to {len(s.running)}, cap is 3"


def test_does_not_admit_what_it_cannot_fit():
    """Admission must check the KV budget, not just the sequence count."""
    s = sched(num_blocks=4, block_size=16, max_num_seqs=8)
    for i in range(4):
        s.add_request(i, prompt_len=64, max_tokens=1)   # 4 blocks each
    s.step()
    assert len(s.running) == 1, (
        f"admitted {len(s.running)} sequences into a pool that fits one"
    )


def test_head_of_line_blocking_is_preserved():
    """A big prompt at the head must not be skipped for a small one behind it.

    Skipping looks like a throughput win and is actually unbounded starvation
    of long prompts. FCFS at the head of the queue.
    """
    s = sched(num_blocks=4, block_size=16, max_num_seqs=8)
    s.add_request("big", prompt_len=64, max_tokens=2)     # needs all 4 blocks
    s.add_request("small", prompt_len=16, max_tokens=2)   # would fit alongside
    s.step()
    ids = [q.id for q in s.running]
    assert ids == ["big"], f"expected only 'big' admitted, got {ids}"


def test_everything_finishes_with_the_right_token_count():
    s = sched(num_blocks=200, block_size=16, max_num_seqs=8)
    want = {}
    rng = random.Random(0)
    for i in range(20):
        pl, mt = rng.choice([16, 32, 64]), rng.choice([8, 20, 50])
        want[i] = mt
        s.add_request(i, pl, mt)
    fin = s.run_to_completion()
    assert len(fin) == 20
    for f in fin:
        assert f.num_generated == want[f.id]


def test_no_block_leaks():
    s = sched(num_blocks=64, block_size=16)
    for i in range(30):
        s.add_request(i, prompt_len=32, max_tokens=10)
    s.run_to_completion()
    assert s.alloc.num_free == 64, (
        f"leaked {64 - s.alloc.num_free} blocks -- some path frees neither on "
        "finish nor on preemption"
    )


@pytest.mark.parametrize("num_blocks", [12, 16, 24])
def test_no_starvation_under_severe_pressure(num_blocks):
    """A pool far too small to hold the workload must still drain completely.

    This is the livelock test. A scheduler that always preempts the same victim,
    or readmits in the wrong order, will spin forever making no progress.
    """
    s = sched(num_blocks=num_blocks, block_size=16, max_num_seqs=8)
    want = {}
    rng = random.Random(1)
    for i in range(20):
        pl, mt = rng.choice([16, 32, 64]), rng.choice([8, 20, 50])
        want[i] = mt
        s.add_request(i, pl, mt)

    fin = s.run_to_completion(max_steps=50000)
    assert len(fin) == 20, (
        f"only {len(fin)}/20 finished with a {num_blocks}-block pool -- "
        "livelock or starvation"
    )
    for f in fin:
        assert f.num_generated == want[f.id], (
            f"seq {f.id} produced {f.num_generated}, wanted {want[f.id]}. "
            "A preempted sequence must restart cleanly and still emit exactly "
            "max_tokens."
        )
    assert s.alloc.num_free == num_blocks


def test_preemption_actually_happens_when_memory_is_tight():
    s = sched(num_blocks=12, block_size=16, max_num_seqs=8)
    for i in range(12):
        s.add_request(i, prompt_len=32, max_tokens=40)
    s.run_to_completion()
    assert s.preemptions > 0, (
        "a 12-block pool with 12 growing sequences must preempt at some point"
    )
    assert any(f.preempted_count > 0 for f in s.finished)


def test_preemption_is_recompute_not_swap():
    """A preempted sequence gives its blocks back IMMEDIATELY.

    That is the whole reason to prefer recompute: the memory is available on
    the very next step, with no PCIe round trip.
    """
    s = sched(num_blocks=8, block_size=16, max_num_seqs=8)
    for i in range(6):
        s.add_request(i, prompt_len=16, max_tokens=60)

    saw_preemption = False
    for _ in range(2000):
        if not s.has_work():
            break
        rec = s.step()
        if rec["preempted"]:
            saw_preemption = True
            for v in rec["preempted"]:
                assert v.blocks == [], (
                    f"preempted seq {v.id} still holds {len(v.blocks)} blocks"
                )
                assert v.num_generated == 0, (
                    "recompute semantics: a preempted sequence restarts from "
                    "its prompt, so num_generated resets to 0"
                )
                assert not v.prefilled
    assert saw_preemption


def test_degrades_gracefully_rather_than_collapsing():
    """Overload should cost you throughput, not correctness or termination."""
    rows = []
    for nb in (200, 24, 12):
        s = sched(num_blocks=nb, block_size=16, max_num_seqs=8)
        rng = random.Random(0)
        for i in range(20):
            s.add_request(i, rng.choice([16, 32, 64]), rng.choice([8, 20, 50]))
        s.run_to_completion(max_steps=50000)
        rows.append((nb, s.steps, s.preemptions, len(s.finished)))

    print(f"\n  {'pool':>6} {'steps':>7} {'preemptions':>12} {'finished':>9}")
    for nb, st, pr, fin in rows:
        print(f"  {nb:>6} {st:>7} {pr:>12} {fin:>8}/20")

    assert all(fin == 20 for _, _, _, fin in rows)
    assert rows[0][2] == 0, "a roomy pool should never need to preempt"
    assert rows[-1][2] > rows[1][2], "tighter memory should preempt more"
    assert rows[-1][1] < rows[0][1] * 20, (
        "throughput collapsed rather than degraded -- thrashing"
    )
    print("\n  \033[2mMore pressure costs more steps and more preemptions, but")
    print("  every request still completes. That is the property you want:")
    print("  degrade, do not collapse.\033[0m")
