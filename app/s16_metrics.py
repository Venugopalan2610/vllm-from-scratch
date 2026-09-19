"""Stage 16 - the important metrics.

`./vc lore 16` for the insight. `./vc test 16` to check yourself.

Six numbers. If you cannot see them, you cannot tune anything, and you find
the wrong cause for every performance problem:

    TTFT            arrival -> first token    (queue wait + prefill)
    ITL / TPOT      the gap between output tokens (smooth streaming)
    throughput      tokens/sec over all requests
    queue time      how long requests waited before the scheduler took them
    KV utilization  how full the block pool is
    preemptions     how frequently you had to preempt a request

People forget the last two, and those two explain the other four. More
preemptions with a high KV utilization means that max_num_seqs is too high,
and the system thrashes. From the outside, that looks the same as "the model
became slow".
"""

import math
import random


def percentile(values, q):
    """The percentile with linear interpolation, as numpy does by default.

        pos = (len - 1) * q, then interpolate between floor(pos) and ceil(pos)

    Return nan for an empty list. The tails are the point here: p50 hides the
    behavior that users complain about.
    """
    raise NotImplementedError("stage 16: implement percentile")


class RequestRecord:
    """Required: .rid .arrival .scheduled .token_times .finish
    and the derived properties .ttft .queue_time .itls .num_tokens

    Note: n tokens give n-1 gaps between tokens, not n.
    """

    def __init__(self, rid, arrival):
        raise NotImplementedError("stage 16: implement RequestRecord")


class MetricsCollector:
    """Required methods:

        on_arrival(rid, timestamp)    on_schedule(rid, timestamp)
        on_token(rid, timestamp)      on_finish(rid, timestamp)
        on_preemption(count=1)        on_kv_utilization(used, total)
        snapshot() -> dict
        goodput(ttft_slo, itl_slo) -> int

    snapshot() must contain at least:
        num_requests num_finished total_tokens elapsed_s throughput_tok_s
        ttft_p50 ttft_p99 itl_p50 itl_p99 queue_p50 preemptions
        kv_utilization_mean kv_utilization_max
    """

    def __init__(self):
        raise NotImplementedError("stage 16: implement MetricsCollector")

    def goodput(self, ttft_slo, itl_slo):
        """The requests that met BOTH targets.

        Throughput counts tokens, also tokens that arrived after the user
        stopped waiting. Goodput counts the requests that you served
        acceptably, and it is the metric to optimize.
        """
        raise NotImplementedError


def poisson_arrivals(rate_per_s, num_arrivals, seed=0, start=0.0):
    """The arrival times of a Poisson process.

    The gaps have an exponential distribution (random.expovariate(rate)),
    NOT a uniform one. That is important: real traffic arrives in bursts. A
    uniform generator makes a benchmark with no queue in it. That benchmark
    tells you that your scheduler is perfect, until production shows that it
    is not.
    """
    raise NotImplementedError("stage 16: implement poisson_arrivals")
