"""Stage 16 - The metrics that matter.

Spec in app/s16_metrics.py. If you cannot see KV utilization and preemption
count, you cannot tune anything, and you will misdiagnose every performance
problem you hit.
"""

import pytest

from app.s16_metrics import MetricsCollector, percentile, poisson_arrivals


def test_percentile_known_values():
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 0.0) == 1
    assert percentile(xs, 0.5) == 3
    assert percentile(xs, 1.0) == 5
    assert percentile([1, 2], 0.5) == pytest.approx(1.5)
    assert percentile([10], 0.99) == 10


def test_ttft_is_arrival_to_first_token():
    m = MetricsCollector()
    m.on_arrival("a", 100.0)
    m.on_schedule("a", 100.5)
    m.on_token("a", 101.0)
    m.on_token("a", 101.1)
    m.on_finish("a", 101.2)
    r = m.records["a"]
    assert r.ttft == pytest.approx(1.0), "TTFT is arrival -> FIRST token"
    assert r.queue_time == pytest.approx(0.5)


def test_itl_is_between_consecutive_tokens():
    m = MetricsCollector()
    m.on_arrival("a", 0.0)
    for t in (1.0, 1.2, 1.5, 1.6):
        m.on_token("a", t)
    itls = m.records["a"].itls
    assert itls == pytest.approx([0.2, 0.3, 0.1])
    assert len(itls) == 3, "n tokens give n-1 inter-token gaps, not n"


def test_snapshot_aggregates():
    m = MetricsCollector()
    for i, (arr, first) in enumerate([(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]):
        rid = f"r{i}"
        m.on_arrival(rid, arr)
        m.on_token(rid, first)
        m.on_token(rid, first + 0.1)
        m.on_finish(rid, first + 0.1)
    s = m.snapshot()
    assert s["num_requests"] == 3
    assert s["num_finished"] == 3
    assert s["total_tokens"] == 6
    assert s["ttft_p50"] == pytest.approx(2.0)
    assert s["throughput_tok_s"] == pytest.approx(6 / 3.1, rel=1e-6)


def test_kv_utilization_is_tracked():
    m = MetricsCollector()
    for used in (10, 50, 90):
        m.on_kv_utilization(used, 100)
    s = m.snapshot()
    assert s["kv_utilization_mean"] == pytest.approx(0.5)
    assert s["kv_utilization_max"] == pytest.approx(0.9)


def test_preemptions_are_counted():
    m = MetricsCollector()
    m.on_preemption()
    m.on_preemption(3)
    assert m.snapshot()["preemptions"] == 4


def test_goodput_requires_both_slos():
    """Throughput counts tokens. Goodput counts requests you didn't fail."""
    m = MetricsCollector()
    # fast request: meets both
    m.on_arrival("fast", 0.0)
    for t in (0.1, 0.15, 0.2):
        m.on_token("fast", t)
    m.on_finish("fast", 0.2)
    # slow TTFT
    m.on_arrival("slow_ttft", 0.0)
    for t in (5.0, 5.05, 5.1):
        m.on_token("slow_ttft", t)
    m.on_finish("slow_ttft", 5.1)
    # fast start, janky stream
    m.on_arrival("janky", 0.0)
    for t in (0.1, 3.0, 3.05):
        m.on_token("janky", t)
    m.on_finish("janky", 3.05)

    assert m.goodput(ttft_slo=1.0, itl_slo=0.5) == 1
    assert m.goodput(ttft_slo=10.0, itl_slo=10.0) == 3


def test_poisson_arrivals_have_the_right_rate():
    ts = poisson_arrivals(rate_per_s=10.0, n=20000, seed=0)
    assert len(ts) == 20000
    assert all(b > a for a, b in zip(ts, ts[1:])), "timestamps must increase"
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    mean_gap = sum(gaps) / len(gaps)
    assert mean_gap == pytest.approx(0.1, rel=0.05), (
        f"mean gap {mean_gap:.4f}s implies rate {1 / mean_gap:.1f}/s, wanted 10/s"
    )


def test_poisson_is_bursty_not_uniform():
    """Real traffic clumps. A uniform generator would hide your queueing."""
    ts = poisson_arrivals(rate_per_s=10.0, n=5000, seed=1)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert max(gaps) > 5 * (sum(gaps) / len(gaps)), (
        "no long gaps -- this looks uniform, not Poisson"
    )


def test_the_pareto_curve(capsys):
    """Informational: simulate the throughput/latency trade as batch grows.

    This is the shape you get from sweeping max_num_seqs, and it is the plot
    you should look at before tuning anything.
    """
    rows = []
    for batch in (1, 4, 16, 64, 128, 256):
        step_ms = 37.0 + 0.25 * batch      # stage 03's model: read + compute
        tput = batch / (step_ms / 1000)
        itl = step_ms
        rows.append((batch, tput, itl))

    print(f"\n  {'batch':>6} {'throughput':>13} {'ITL':>10}")
    for b, t, i in rows:
        print(f"  {b:>6} {t:>9.0f} tok/s {i:>8.1f} ms")

    assert rows[-1][1] > rows[0][1], "throughput should rise with batch"
    assert rows[-1][2] > rows[0][2], "latency should worsen with batch"
    print("\n  \033[2mPick the knee, not the peak. Past it you are buying")
    print("  throughput with other people's latency.\033[0m")
