"""Stage 16 - the metrics that matter.

`./vc lore 16` for the insight. `./vc test 16` to check yourself.

Six numbers. If you cannot see them you cannot tune anything, and you will
misdiagnose every performance problem you hit:

    TTFT            arrival -> first token   (queue wait + prefill)
    ITL / TPOT      gap between output tokens (streaming smoothness)
    throughput      tokens/sec across everything
    queue time      how long requests waited before being scheduled
    KV utilization  how full the block pool is
    preemptions     how often you had to kick someone out

The last two are the ones people forget, and they are the ones that explain the
other four. Rising preemptions with high KV utilization means you set
max_num_seqs too high and the system is thrashing -- which looks identical to
"the model got slow" from the outside.
"""

import math
import random


def percentile(values, q):
    """Linear-interpolated percentile, matching numpy's default.

        pos = (len - 1) * q, then interpolate between floor(pos) and ceil(pos)

    Return nan for an empty list. Tails are the whole point here -- p50 hides
    exactly the behaviour users complain about.
    """
    raise NotImplementedError("stage 16: implement percentile")


class RequestRecord:
    """Required: .rid .arrival .scheduled .token_times .finish
    and the derived properties .ttft .queue_time .itls .num_tokens

    Note n tokens produce n-1 inter-token gaps, not n.
    """

    def __init__(self, rid, arrival):
        raise NotImplementedError("stage 16: implement RequestRecord")


class MetricsCollector:
    """Required methods:

        on_arrival(rid, t)   on_schedule(rid, t)   on_token(rid, t)
        on_finish(rid, t)    on_preemption(n=1)    on_kv_utilization(used, total)
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
        """Requests that met BOTH targets.

        Throughput counts tokens, including tokens delivered so late that the
        user had already given up. Goodput counts requests you actually served
        acceptably, and it is the metric worth optimizing.
        """
        raise NotImplementedError


def poisson_arrivals(rate_per_s, n, seed=0, start=0.0):
    """Arrival timestamps for a Poisson process.

    Gaps are exponentially distributed (random.expovariate(rate)), NOT uniform.
    That distinction matters: real traffic arrives in bursts, and a uniform
    generator produces a benchmark with no queueing in it, which will tell you
    your scheduler is perfect right up until production disagrees.
    """
    raise NotImplementedError("stage 16: implement poisson_arrivals")
