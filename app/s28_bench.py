"""Stage 28 - what your engine does, against what it could.

`./vc lore 28` for the insight. `./vc test 28` to check yourself.

A throughput number alone tells you nothing. 2,000 tok/s is excellent on one
card and poor on another. So this stage measures your engine against the
floor that physics sets for the SAME steps, on the SAME card, in the SAME
minute. The result is a ratio, and a ratio travels between cards.

WHAT YOU ARE BUILDING

    Hardware(read_bandwidth, flops)                         given
    ModelCost(weight_bytes, kv_bytes_per_token, num_params) given
    model_cost(model) -> ModelCost
    step_floor(cost, hardware, tokens, context_tokens) -> seconds
        The shortest possible time for one step:

          weights = max(weight_bytes / read_bandwidth,
                        2 * num_params * tokens / flops)
          floor   = weights + context_tokens * kv_bytes_per_token
                            / read_bandwidth

        A step reads the weights one time, whatever the batch. The arithmetic
        on them is 2 FLOP for each parameter for each token. The step waits
        for the larger of the two. Then attention reads each context position
        of each sequence one time.

    run_benchmark(engine, requests, hardware) -> dict
        requests: list of (rid, prompt_ids, max_tokens). All arrive at t=0.
        Use ignore_eos, so that every run makes the same number of tokens.
        After each step, add the step_floor of engine.last_step.
        Return at least: output_tokens, seconds, tok_s, steps,
        floor_seconds, efficiency (= floor_seconds / seconds), preemptions.

    report(result) -> str      one line that a person can read

Build the Hardware from cudalib.read_bandwidth() and cudalib.matmul_flops(),
measured just before the run. A laptop GPU throttles, and a floor measured
cold against an engine measured hot compares two different machines.

WHAT THE RATIO TELLS YOU

  - Near 100%: the engine moves bytes as fast as the card can. A faster
    engine must move fewer bytes: quantize the weights (stage 18) or the KV
    cache.
  - Well below: some part of the step does not move bytes. Find it. The
    usual causes are prefill steps that run eager, host time between steps,
    and many small kernels. LORE.md section 11 lists a stretch goal for each.

WHAT THIS DOES NOT MEASURE

  This benchmark sends all requests at t=0 and measures raw tok/s. A
  production metric is GOODPUT: output tokens per second at a fixed p99
  TTFT and ITL, measured at batch 32 and 128 with Poisson arrivals. The
  server in stage 27 has the machinery for that measurement (run_load,
  MetricsCollector), but this stage does not gate on it. The reason:
  goodput depends on the clock and the thermal state of the card, and a
  gate that fails from a hot GPU teaches nothing.
"""

import time
from dataclasses import dataclass

import torch


@dataclass
class Hardware:
    read_bandwidth: float       # bytes/s
    flops: float                # FLOP/s


@dataclass
class ModelCost:
    weight_bytes: float         # read one time in each step
    kv_bytes_per_token: float   # read for each context position
    num_params: float           # 2 FLOP for each parameter for each token


def model_cost(model):
    raise NotImplementedError("stage 28: implement model_cost")


def step_floor(cost, hardware, tokens, context_tokens):
    """The shortest possible time for one step, in seconds. The step waits
    for the larger of the weight read and the arithmetic. Then attention
    reads each context position of each sequence one time."""
    raise NotImplementedError("stage 28: implement step_floor")


def run_benchmark(engine, requests, hardware):
    """requests: [(rid, prompt_ids, max_tokens)], all at t=0, EOS ignored so
    that every run makes the same number of tokens."""
    raise NotImplementedError("stage 28: implement run_benchmark")


def run_goodput_benchmark(engine, requests, hardware, ttft_slo=2.0, itl_slo=0.2):
    """Measure goodput: output tokens per second meeting TTFT and ITL SLOs."""
    raise NotImplementedError("stage 28: implement run_goodput_benchmark")


def report(result):
    raise NotImplementedError("stage 28: implement report")
