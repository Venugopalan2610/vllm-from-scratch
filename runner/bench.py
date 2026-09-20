"""./vc bench [--model NAME] [--seqs N]

Run YOUR capstone engine (stages 21 to 28) on a fixed load. Print what it did
against the roofline floor of the same steps, on this card, now. Then print
the ceiling of this card, from the same formula.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.style import paint  # noqa: E402

MEMORY_RESERVE = 1.5e9          # bytes left free for activations and graphs


def parse_args(argv):
    options = {"model": None, "max_num_seqs": 32}
    for index, arg in enumerate(argv[:-1]):
        if arg == "--model":
            options["model"] = argv[index + 1]
        if arg == "--seqs":
            options["max_num_seqs"] = int(argv[index + 1])
    return options


def pool_blocks(model):
    """Profile the GPU and derive the block count from what is free.

    Real engines do this at startup: run a dummy forward, measure the peak
    memory, take a fraction of what remains, and derive the block count.
    Students hit this on day one of a deployment. The tests use a fixed
    num_blocks to make the checks reproducible, but production code should
    always call a function like this one.
    """
    import torch

    free_bytes, _ = torch.cuda.mem_get_info()
    return int((free_bytes - MEMORY_RESERVE) / (model.kv_bytes_per_token() * 16))


def build_engine(model, num_blocks, max_num_seqs):
    from app.s22_engine import LLMEngine
    from app.s23_graphs import GraphedModelRunner

    runner = GraphedModelRunner(model, num_blocks, max_model_len=2048)
    runner.capture()
    return LLMEngine(model, num_blocks, runner=runner,
                     max_num_seqs=max_num_seqs, token_budget=1024)


def fixed_load(num_requests, tokenizer=None):
    """A workload closer to a real serving trace.

    Random token ids with lognormal prompt and output lengths, and a shared
    system prompt so that the prefix cache contributes to the measurement.
    Real traces are heavy-tailed, and a flat uniform distribution hides
    the prefix cache and makes every request look the same.
    """
    rng = random.Random(0)
    # A shared system prompt. When tokenizer is available, use real tokens;
    # otherwise, use random ids. The prefix cache saves real money here.
    system_ids = [rng.randrange(1000, 20000) for _ in range(64)]
    requests = []
    for rid in range(num_requests):
        # Lognormal: median ~100, heavy tail up to ~1000
        prompt_len = max(8, min(1024, int(rng.lognormvariate(4.5, 0.8))))
        output_len = max(4, min(512, int(rng.lognormvariate(3.5, 0.7))))
        prompt = system_ids + [rng.randrange(1000, 20000)
                               for _ in range(prompt_len)]
        requests.append((rid, prompt, output_len))
    return requests


def print_ceiling(model, hardware, num_blocks):
    from app.s28_bench import model_cost, step_floor

    cost = model_cost(model)
    print("\n" + paint("The ceiling on this card", "bold"))
    print(f"  batch 1:    {hardware.read_bandwidth / cost.weight_bytes:6.0f} "
          f"tok/s   = read bandwidth / weight bytes")
    for context_len in (1024, 16384):
        num_seqs = max(1, min(256, num_blocks * 16 // context_len))
        floor = step_floor(cost, hardware, num_seqs, num_seqs * context_len)
        print(f"  ctx {context_len:>5}:  {num_seqs / floor:6.0f} tok/s   at "
              f"{num_seqs} sequences, which is what fits")
    print("\n" + paint("With a full KV pool, tok/s is close to bandwidth / "
                       "(context x KV bytes for each token). That ratio is "
                       "true on every card.", "dim"))


def main(argv):
    import cudalib
    from tvllm import load_model

    options = parse_args(argv)
    model = load_model(options["model"]) if options["model"] else load_model()
    try:
        from app.s28_bench import Hardware, report, run_benchmark
        num_blocks = pool_blocks(model)
        engine = build_engine(model, num_blocks, options["max_num_seqs"])
    except NotImplementedError as error:
        print(paint(f"The capstone is not finished: {error}.", "yellow"))
        print(paint("./vc bench runs YOUR engine. Finish stages 21 to 28 "
                    "first.", "dim"))
        return 1

    print(paint(model.config.name, "bold") + "  weights "
          f"{model.weight_bytes() / 1e9:.2f} GB, KV "
          f"{model.kv_bytes_per_token() / 1024:.0f} KiB for each token, pool "
          f"{num_blocks * 16:,} tokens")
    hardware = Hardware(cudalib.read_bandwidth(), cudalib.matmul_flops())
    result = run_benchmark(engine, fixed_load(3 * options["max_num_seqs"],
                                              model.tokenizer),
                           hardware)
    print(f"\n  this card: {hardware.read_bandwidth / 1e9:.0f} GB/s read, "
          f"{hardware.flops / 1e12:.1f} TFLOP/s, ridge "
          f"{hardware.flops / hardware.read_bandwidth:.0f} FLOP/byte")
    print(f"  {report(result)}")
    # Show prefix cache contribution (Criticism #18)
    if hasattr(engine, 'prefix_cache') and engine.prefix_cache:
        cache = engine.prefix_cache
        total = cache.hits + cache.misses
        hit_rate = cache.hits / total if total > 0 else 0.0
        print(f"  prefix cache: {cache.hits} hits, {cache.misses} misses, "
              f"{100 * hit_rate:.0f}% hit rate")
    print_ceiling(model, hardware, num_blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
