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


def run_goodput_benchmark(engine, requests, hardware, ttft_slo=2.0, itl_slo=0.2):
    """Measure goodput: output tokens per second meeting TTFT and ITL SLOs."""
    cost = model_cost(engine.model)
    arrivals = {}
    token_times = {}
    for rid, prompt_ids, max_tokens in requests:
        now = time.perf_counter()
        arrivals[rid] = now
        token_times[rid] = []
        engine.add_request(rid, prompt_ids, max_tokens, ignore_eos=True, arrival=now)
    floor_seconds = 0.0
    torch.cuda.synchronize()
    start = time.perf_counter()
    while engine.has_work():
        events = engine.step()
        now = time.perf_counter()
        for rid, text, _ in events:
            if text:
                token_times[rid].append(now)
        work = engine.last_step
        if work.tokens:
            floor_seconds += step_floor(cost, hardware, work.tokens,
                                        work.context_tokens)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    output_tokens = sum(len(engine.output(rid)) for rid, _, _ in requests)

    met_tokens = 0
    good_reqs = 0
    for rid, _, _ in requests:
        times = token_times.get(rid, [])
        num_tok = len(engine.output(rid))
        if not times:
            continue
        ttft = times[0] - arrivals[rid]
        itls = [t2 - t1 for t1, t2 in zip(times, times[1:])]
        max_itl = max(itls) if itls else 0.0
        if ttft <= ttft_slo and max_itl <= itl_slo:
            good_reqs += 1
            met_tokens += num_tok

    goodput_tok_s = met_tokens / seconds if seconds > 0 else 0.0
    return {"output_tokens": output_tokens, "seconds": seconds,
            "tok_s": output_tokens / seconds if seconds > 0 else 0.0,
            "goodput_tok_s": goodput_tok_s,
            "goodput_ratio": met_tokens / output_tokens if output_tokens > 0 else 0.0,
            "good_requests": good_reqs,
            "total_requests": len(requests),
            "steps": engine.num_steps,
            "floor_seconds": floor_seconds,
            "efficiency": floor_seconds / seconds if seconds > 0 else 0.0,
            "preemptions": engine.num_preemptions}


def report(result):
    text = (f"{result['output_tokens']} tokens in {result['seconds']:.2f} s = "
            f"{result['tok_s']:.0f} tok/s over {result['steps']} steps. The "
            f"roofline floor for the same steps is {result['floor_seconds']:.2f} "
            f"s, so the engine runs at {100 * result['efficiency']:.0f}% of the "
            f"roof.")
    if "goodput_tok_s" in result:
        text += (f" Goodput: {result['goodput_tok_s']:.0f} tok/s "
                 f"({result['good_requests']}/{result['total_requests']} requests met SLO, "
                 f"{100 * result['goodput_ratio']:.0f}%).")
    if result.get('preemptions'):
        text += f" ({result['preemptions']} preemptions.)"
    return text
