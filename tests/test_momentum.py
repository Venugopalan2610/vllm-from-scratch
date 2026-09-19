"""Checks on runner/momentum.py: what ./vc says to keep you going.

The messages must be true. "Better than before" must mean a higher score,
and the way out must appear when a learner is stuck, not before.
"""

from datetime import date

from runner import momentum


def new_progress():
    return {"backend": "torch", "tracks": {"torch": [], "jax": []}}


def run(progress, passed, total=10, stage_id="06-block-allocator"):
    return momentum.record_run(progress, stage_id, passed, total)


def test_the_first_run_is_not_called_better():
    result = run(new_progress(), 3)
    assert result.is_first
    assert "First run" in momentum.after_run(result)[0][0]


def test_a_higher_score_is_better_than_before():
    progress = new_progress()
    run(progress, 3)
    result = run(progress, 5)
    assert result.is_better and not result.is_stuck
    assert "Better than before: 5/10. Your best was 3/10." in momentum.after_run(result)[0][0]


def test_a_lower_score_keeps_the_best():
    progress = new_progress()
    run(progress, 5)
    result = run(progress, 2)
    assert not result.is_better
    assert momentum.stage_record(progress, "06-block-allocator").best == 5
    assert "Your best is 5/10" in momentum.after_run(result)[0][0]


def test_stuck_after_enough_runs_with_no_new_pass():
    progress = new_progress()
    run(progress, 4)
    results = [run(progress, 4) for _ in range(momentum.STUCK_RUNS)]
    assert not any(result.is_stuck for result in results[:-1])
    assert results[-1].is_stuck
    text = " ".join(line for line, _ in momentum.when_stuck(results[-1], "06", "A2"))
    assert momentum.MOTTO in text
    assert "./vc peek 06" in text and "Part3_PagedAttention" in text


def test_a_new_best_resets_the_stuck_count():
    progress = new_progress()
    for _ in range(momentum.STUCK_RUNS + 1):
        run(progress, 4)
    assert not run(progress, 5).is_stuck


def test_each_track_has_its_own_history():
    progress = new_progress()
    run(progress, 7)
    progress["backend"] = "jax"
    assert momentum.stage_record(progress, "06-block-allocator").best == -1


def test_days_away():
    progress = {"last_seen": "2026-01-01"}
    assert momentum.days_away(progress, today=date(2026, 1, 11)) == 10
    assert momentum.days_away({}) is None


def test_each_arc_has_its_notebooks():
    for arc_number in range(8):
        assert momentum.notebooks_for(f"A{arc_number}") is not None
