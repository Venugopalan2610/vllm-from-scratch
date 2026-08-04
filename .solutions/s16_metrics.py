"""Reference solution, stage 16 - observability."""

import math
import random


def percentile(values, q):
    """Linear-interpolated percentile, matching numpy's default."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


class RequestRecord:
    def __init__(self, rid, arrival):
        self.rid = rid
        self.arrival = arrival
        self.scheduled = None
        self.token_times = []
        self.finish = None

    @property
    def ttft(self):
        if not self.token_times:
            return None
        return self.token_times[0] - self.arrival

    @property
    def queue_time(self):
        if self.scheduled is None:
            return None
        return self.scheduled - self.arrival

    @property
    def itls(self):
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

    @property
    def num_tokens(self):
        return len(self.token_times)


class MetricsCollector:
    def __init__(self):
        self.records = {}
        self.finished = []
        self.preemptions = 0
        self.kv_samples = []
        self.start = None
        self.end = None

    def on_arrival(self, rid, t):
        self.records[rid] = RequestRecord(rid, t)
        self.start = t if self.start is None else min(self.start, t)

    def on_schedule(self, rid, t):
        self.records[rid].scheduled = t

    def on_token(self, rid, t):
        self.records[rid].token_times.append(t)

    def on_finish(self, rid, t):
        r = self.records[rid]
        r.finish = t
        self.finished.append(r)
        self.end = t if self.end is None else max(self.end, t)

    def on_preemption(self, n=1):
        self.preemptions += n

    def on_kv_utilization(self, used_blocks, total_blocks):
        self.kv_samples.append(used_blocks / total_blocks if total_blocks else 0.0)

    def snapshot(self):
        ttfts = [r.ttft for r in self.records.values() if r.ttft is not None]
        itls = [x for r in self.records.values() for x in r.itls]
        queues = [r.queue_time for r in self.records.values()
                  if r.queue_time is not None]
        total_tokens = sum(r.num_tokens for r in self.records.values())
        elapsed = (self.end - self.start) if (self.start is not None
                                             and self.end is not None) else 0.0
        return {
            "num_requests": len(self.records),
            "num_finished": len(self.finished),
            "total_tokens": total_tokens,
            "elapsed_s": elapsed,
            "throughput_tok_s": (total_tokens / elapsed) if elapsed > 0 else 0.0,
            "ttft_p50": percentile(ttfts, 0.50),
            "ttft_p99": percentile(ttfts, 0.99),
            "itl_p50": percentile(itls, 0.50),
            "itl_p99": percentile(itls, 0.99),
            "queue_p50": percentile(queues, 0.50),
            "preemptions": self.preemptions,
            "kv_utilization_mean": (sum(self.kv_samples) / len(self.kv_samples)
                                    if self.kv_samples else 0.0),
            "kv_utilization_max": max(self.kv_samples) if self.kv_samples else 0.0,
        }

    def goodput(self, ttft_slo, itl_slo):
        """Requests that met BOTH latency targets. The number that matters."""
        ok = 0
        for r in self.records.values():
            if r.ttft is None or r.finish is None:
                continue
            if r.ttft <= ttft_slo and percentile(r.itls, 0.99) <= itl_slo:
                ok += 1
        return ok


def poisson_arrivals(rate_per_s, n, seed=0, start=0.0):
    """Arrival timestamps for a Poisson process: exponential gaps."""
    rng = random.Random(seed)
    t = start
    out = []
    for _ in range(n):
        t += rng.expovariate(rate_per_s)
        out.append(t)
    return out
