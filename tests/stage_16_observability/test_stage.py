"""Stage 16 - the important metrics.

The spec is in app/s16_metrics.py. If you cannot see the KV utilization and
the preemption count, you cannot tune anything, and you find the wrong cause
for every performance problem.
"""

import pytest

from app.s16_metrics import MetricsCollector, percentile, poisson_arrivals


def record_request(metrics, rid, arrival, token_times):
    """One request from arrival to finish. It finishes at its last token."""
    metrics.on_arrival(rid, arrival)
    for token_time in token_times:
        metrics.on_token(rid, token_time)
    metrics.on_finish(rid, token_times[-1])


def gaps_between(timestamps):
    return [later - earlier for earlier, later
            in zip(timestamps, timestamps[1:])]


def test_percentile_known_values():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0.0) == 1
    assert percentile(values, 0.5) == 3
    assert percentile(values, 1.0) == 5
    assert percentile([1, 2], 0.5) == pytest.approx(1.5)
    assert percentile([10], 0.99) == 10


def test_ttft_is_arrival_to_first_token():
    metrics = MetricsCollector()
    metrics.on_arrival("a", 100.0)
    metrics.on_schedule("a", 100.5)
    metrics.on_token("a", 101.0)
    metrics.on_token("a", 101.1)
    metrics.on_finish("a", 101.2)
    record = metrics.records["a"]
    assert record.ttft == pytest.approx(1.0), "TTFT is arrival -> FIRST token"
    assert record.queue_time == pytest.approx(0.5)


def test_itl_is_between_consecutive_tokens():
    metrics = MetricsCollector()
    record_request(metrics, "a", 0.0, [1.0, 1.2, 1.5, 1.6])
    itls = metrics.records["a"].itls
    assert itls == pytest.approx([0.2, 0.3, 0.1])
    assert len(itls) == 3, "n tokens give n-1 gaps between tokens, not n"


def test_snapshot_aggregates():
    metrics = MetricsCollector()
    for index, first_token in enumerate((1.0, 2.0, 3.0)):
        record_request(metrics, f"r{index}", 0.0,
                       [first_token, first_token + 0.1])
    snapshot = metrics.snapshot()
    assert snapshot["num_requests"] == 3
    assert snapshot["num_finished"] == 3
    assert snapshot["total_tokens"] == 6
    assert snapshot["ttft_p50"] == pytest.approx(2.0)
    assert snapshot["throughput_tok_s"] == pytest.approx(6 / 3.1, rel=1e-6)


def test_kv_utilization_is_tracked():
    metrics = MetricsCollector()
    for used_blocks in (10, 50, 90):
        metrics.on_kv_utilization(used_blocks, 100)
    snapshot = metrics.snapshot()
    assert snapshot["kv_utilization_mean"] == pytest.approx(0.5)
    assert snapshot["kv_utilization_max"] == pytest.approx(0.9)


def test_preemptions_are_counted():
    metrics = MetricsCollector()
    metrics.on_preemption()
    metrics.on_preemption(3)
    assert metrics.snapshot()["preemptions"] == 4


def test_goodput_requires_both_slos():
    """Throughput counts tokens. Goodput counts the requests that met their
    targets."""
    metrics = MetricsCollector()
    record_request(metrics, "fast", 0.0, [0.1, 0.15, 0.2])     # meets both
    record_request(metrics, "slow_ttft", 0.0, [5.0, 5.05, 5.1])
    record_request(metrics, "irregular", 0.0, [0.1, 3.0, 3.05])  # a long gap
    assert metrics.goodput(ttft_slo=1.0, itl_slo=0.5) == 1
    assert metrics.goodput(ttft_slo=10.0, itl_slo=10.0) == 3


def test_poisson_arrivals_have_the_right_rate():
    arrivals = poisson_arrivals(rate_per_s=10.0, num_arrivals=20000, seed=0)
    assert len(arrivals) == 20000
    gaps = gaps_between(arrivals)
    assert all(gap > 0 for gap in gaps), "the timestamps must increase"
    mean_gap = sum(gaps) / len(gaps)
    assert mean_gap == pytest.approx(0.1, rel=0.05), (
        f"a mean gap of {mean_gap:.4f}s gives a rate of {1 / mean_gap:.1f}/s, "
        "expected 10/s")


def test_poisson_is_bursty_not_uniform():
    """Real traffic comes in bursts. A uniform generator hides your queue."""
    gaps = gaps_between(poisson_arrivals(rate_per_s=10.0, num_arrivals=5000,
                                         seed=1))
    assert max(gaps) > 5 * (sum(gaps) / len(gaps)), (
        "no long gaps. This looks uniform, not Poisson.")


def test_the_pareto_curve(capsys):
    """Information only: the exchange of throughput and latency as the batch
    grows.

    No milliseconds here, because they belong to one card. The step cost is
    in units of one weight read, and the one number about the card is its
    ridge (FLOP per byte, stage 03). Each request adds 1/ridge of a weight
    read of arithmetic. The shape is the same on every card. Only the knee
    moves.
    """
    ridge = 150                    # try the value that ./vc info prints
    rows = []
    for batch_size in (1, 4, 16, 64, 128, 256, 512):
        step_cost = 1 + batch_size / ridge      # in units of one weight read
        rows.append((batch_size, batch_size / step_cost, step_cost))

    print(f"\n  ridge {ridge}. Throughput in units of batch 1, ITL in units "
          "of one weight read.")
    print(f"  {'batch':>6} {'throughput':>11} {'ITL':>7}")
    for batch_size, throughput, itl in rows:
        print(f"  {batch_size:>6} {throughput:>10.1f}x {itl:>6.2f}x")

    assert rows[-1][1] > rows[0][1], "the throughput must increase with batch"
    assert rows[-1][2] > rows[0][2], "the latency must increase with batch"
    # After the ridge, two times the batch gives less than 1.4x throughput.
    past_ridge = [row for row in rows if row[0] >= ridge]
    assert past_ridge[1][1] / past_ridge[0][1] < 1.4
    print("\n  \033[2mSelect the knee, not the peak. After the knee, you buy")
    print("  throughput with the latency of other users.\033[0m")
