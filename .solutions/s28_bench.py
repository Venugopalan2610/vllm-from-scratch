"""Reference solution, stage 28 - what the engine does, against what it could."""

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
    weight_bytes = model.weight_bytes()
    bytes_per_param = torch.empty((), dtype=model.config.dtype).element_size()
    return ModelCost(weight_bytes, model.kv_bytes_per_token(),
                     weight_bytes / bytes_per_param)


def step_floor(cost, hardware, tokens, context_tokens):
    """The shortest possible time for one step, in seconds. The step waits
    for the larger of the weight read and the arithmetic. Then attention
    reads each context position of each sequence one time."""
    weight_read = cost.weight_bytes / hardware.read_bandwidth
    arithmetic = 2 * cost.num_params * tokens / hardware.flops
    kv_read = context_tokens * cost.kv_bytes_per_token / hardware.read_bandwidth
    return max(weight_read, arithmetic) + kv_read


def run_benchmark(engine, requests, hardware):
    """requests: [(rid, prompt_ids, max_tokens)], all at t=0, EOS ignored so
    that every run makes the same number of tokens."""
    cost = model_cost(engine.model)
    for rid, prompt_ids, max_tokens in requests:
        engine.add_request(rid, prompt_ids, max_tokens, ignore_eos=True)
    floor_seconds = 0.0
    torch.cuda.synchronize()
    start = time.perf_counter()
    while engine.has_work():
        engine.step()
        work = engine.last_step
        if work.tokens:
            floor_seconds += step_floor(cost, hardware, work.tokens,
                                        work.context_tokens)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    output_tokens = sum(len(engine.output(rid)) for rid, _, _ in requests)
    return {"output_tokens": output_tokens, "seconds": seconds,
            "tok_s": output_tokens / seconds, "steps": engine.num_steps,
            "floor_seconds": floor_seconds, "efficiency": floor_seconds / seconds,
            "preemptions": engine.num_preemptions}


def report(result):
    return (f"{result['output_tokens']} tokens in {result['seconds']:.2f} s = "
            f"{result['tok_s']:.0f} tok/s over {result['steps']} steps. The "
            f"roofline floor for the same steps is {result['floor_seconds']:.2f} "
            f"s, so the engine runs at {100 * result['efficiency']:.0f}% of the "
            f"roof.")
