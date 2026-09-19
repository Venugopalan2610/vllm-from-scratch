"""Reference solution, stage 16 - observability."""

import math
import random


def percentile(values, q):
    """The percentile with linear interpolation, as numpy does by default."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    below, above = math.floor(position), math.ceil(position)
    fraction = position - below
    return float(ordered[below] + (ordered[above] - ordered[below]) * fraction)


def mean(values):
    return sum(values) / len(values) if values else 0.0


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
        return [later - earlier for earlier, later
                in zip(self.token_times, self.token_times[1:])]

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

    def on_arrival(self, rid, timestamp):
        self.records[rid] = RequestRecord(rid, timestamp)
        self.start = timestamp if self.start is None else min(self.start,
                                                              timestamp)

    def on_schedule(self, rid, timestamp):
        self.records[rid].scheduled = timestamp

    def on_token(self, rid, timestamp):
        self.records[rid].token_times.append(timestamp)

    def on_finish(self, rid, timestamp):
        record = self.records[rid]
        record.finish = timestamp
        self.finished.append(record)
        self.end = timestamp if self.end is None else max(self.end, timestamp)

    def on_preemption(self, count=1):
        self.preemptions += count

    def on_kv_utilization(self, used_blocks, total_blocks):
        self.kv_samples.append(used_blocks / total_blocks if total_blocks
                               else 0.0)

    @property
    def elapsed(self):
        if self.start is None or self.end is None:
            return 0.0
        return self.end - self.start

    def _present(self, field):
        values = [getattr(record, field) for record in self.records.values()]
        return [value for value in values if value is not None]

    def snapshot(self):
        ttfts = self._present("ttft")
        itls = [itl for record in self.records.values() for itl in record.itls]
        queue_times = self._present("queue_time")
        total_tokens = sum(record.num_tokens
                           for record in self.records.values())
        return {
            "num_requests": len(self.records),
            "num_finished": len(self.finished),
            "total_tokens": total_tokens,
            "elapsed_s": self.elapsed,
            "throughput_tok_s": (total_tokens / self.elapsed
                                 if self.elapsed > 0 else 0.0),
            "ttft_p50": percentile(ttfts, 0.50),
            "ttft_p99": percentile(ttfts, 0.99),
            "itl_p50": percentile(itls, 0.50),
            "itl_p99": percentile(itls, 0.99),
            "queue_p50": percentile(queue_times, 0.50),
            "preemptions": self.preemptions,
            "kv_utilization_mean": mean(self.kv_samples),
            "kv_utilization_max": max(self.kv_samples, default=0.0),
        }

    def goodput(self, ttft_slo, itl_slo):
        """The requests that met BOTH latency targets. This number counts."""
        return sum(1 for record in self.records.values()
                   if record.ttft is not None and record.finish is not None
                   and record.ttft <= ttft_slo
                   and percentile(record.itls, 0.99) <= itl_slo)


def poisson_arrivals(rate_per_s, num_arrivals, seed=0, start=0.0):
    """The arrival times of a Poisson process: exponential gaps."""
    rng = random.Random(seed)
    arrivals = []
    timestamp = start
    for _ in range(num_arrivals):
        timestamp += rng.expovariate(rate_per_s)
        arrivals.append(timestamp)
    return arrivals
