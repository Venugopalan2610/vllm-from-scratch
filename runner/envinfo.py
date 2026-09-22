"""./vc info

Measure the roofline of THIS GPU. These numbers explain every design
decision in the repo.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import cudalib  # noqa: E402
from runner.style import paint  # noqa: E402

BYTES_PER_BF16 = 2
HEAT_SECONDS = 3.0
# (name, GB of weights in bf16) for the size comparisons
EXAMPLE_MODELS = [("1.7B", 3.4), ("1B", 2.0), ("7B", 14.0), ("8B (fp8)", 8.0)]


def print_device(properties):
    print("\n" + paint(properties.name, "bold"))
    print(f"  VRAM         {properties.total_memory / 1e9:.1f} GB")
    print(f"  SMs          {properties.multi_processor_count}")
    print(f"  Compute cap  {properties.major}.{properties.minor}")
    print(f"  torch        {torch.__version__}")


def print_bandwidth(gbs):
    print("\n" + paint("Memory bandwidth", "bold") + " (measured, streaming copy)")
    print(f"  {gbs:.0f} GB/s")


def print_compute(burst_tflops, sustained_tflops):
    print("\n" + paint("Compute", "bold") + " (bf16 4096^3 matmul)")
    print(f"  {burst_tflops:.0f} TFLOP/s burst  ->  {sustained_tflops:.0f} "
          "TFLOP/s sustained")
    print(paint("  A GPU can boost, then throttle when it is hot. Serving is a\n"
                "  sustained load, so plan with the second number. It also\n"
                "  means that a benchmark run cold gives a wrong answer.",
                "dim"))


def print_hardware_ratio(ridge, tflops, gbs):
    print("\n" + paint("HARDWARE ratio", "bold") + "  "
          + paint("(a property of this chip)", "dim"))
    print(f"     {tflops * 1000:,.0f} GFLOP/s")
    print(f"  {'-' * 20}  =  " + paint(f"{ridge:.0f} FLOP per byte", "bold"))
    print(f"      {gbs:.0f} GB/s")
    print(paint("  The /s is on the top and on the bottom, so this is a plain\n"
                "  ratio of counts, not a rate. That makes it comparable to the\n"
                "  number below.", "dim"))


def print_workload_ratio():
    print("\n" + paint("WORKLOAD ratio", "bold") + "  "
          + paint("(a property of how you batch)", "dim"))
    print("  A weight is 2 bytes (bf16). One multiply-add uses it: 2 FLOP.")
    print("  With N weights and batch B, for each forward pass:")
    print("     bytes read = 2N   "
          + paint("(the weights are read ONE time, for every B)", "dim"))
    print("     operations = 2NB")
    print("\n     2NB")
    print(f"  {'-' * 8}  =  " + paint("B FLOP per byte", "bold") + "   "
          + paint("(the 2 and the N cancel)", "dim"))
    print("      2N")


def print_comparison(ridge):
    print("\n" + paint("Compare them", "bold"))
    print(f"  B  <  {ridge:.0f}   ->  memory-bound, the GPU waits for bytes")
    print(f"  B  >  {ridge:.0f}   ->  compute-bound, the GPU waits for math")
    print(f"\n  At batch 1 the workload is at 1, against {ridge:.0f}. "
          + paint(f"({100 / ridge:.1f}% of peak compute.)", "dim"))
    print("\n    " + paint(f"batch ~{ridge:.0f} is where decode stops being "
                           "memory-bound.", "bold"))
    print(paint("\n  That is the argument for continuous batching, and the\n"
                "  reason that real servers set max_num_seqs in the hundreds.\n"
                "  You do not quite get there. The KV cache traffic also grows\n"
                "  with B, and PagedAttention exists to relax that limit.\n"
                "\n  A profiler shows neither ratio. It shows achieved RATES\n"
                "  (GB/s, GFLOP/s). You compute these two numbers before, to\n"
                "  predict which of those rates gets to its limit.", "dim"))


def print_cache_argument(l2_bytes, gbs):
    print("\n" + paint("Why the weights cannot stay in the cache", "bold"))
    print(f"  L2 cache on this GPU   {l2_bytes / 1e6:>8,.0f} MB")
    for name, weight_gb in [("Qwen3-1.7B bf16", 3.4), ("7B bf16", 14.0)]:
        print(f"  {name:<22} {weight_gb * 1000:>8,.0f} MB   "
              + paint(f"-> {weight_gb * 1e9 / l2_bytes:.0f}x too big", "dim"))
    print(paint("  So the weights are in VRAM, and 'read them from VRAM' IS\n"
                f"  the {14.0 / gbs * 1000:.0f} ms that a 7B forward pass costs "
                "on this card.\n  Read-only does not mean free to read.",
                "dim"))


def print_ceilings(gbs):
    print("\n" + paint("Implied ceilings", "bold")
          + " (bf16, batch=1, weights-only traffic)")
    for name, weight_gb in EXAMPLE_MODELS:
        tokens_per_second = gbs / weight_gb
        print(f"  {name:<10} {weight_gb:>5.1f} GB/step -> "
              f"{tokens_per_second:>7.0f} tok/s ceiling  "
              f"({1000 / tokens_per_second:.2f} ms/token)")
    print(paint("\nNo kernel is faster than these at batch=1. Batching is how "
                "you beat them.", "dim") + "\n")


def main():
    properties = torch.cuda.get_device_properties(0)
    print_device(properties)
    gbs = cudalib.peak_bandwidth(fresh=True)
    print_bandwidth(gbs)
    burst_tflops = cudalib.matmul_flops() / 1e12
    sustained_tflops = cudalib.matmul_flops(heat_seconds=HEAT_SECONDS) / 1e12
    print_compute(burst_tflops, sustained_tflops)
    ridge = sustained_tflops * 1e12 / (gbs * 1e9)
    print_hardware_ratio(ridge, sustained_tflops, gbs)
    print_workload_ratio()
    print_comparison(ridge)
    l2_bytes = getattr(properties, "L2_cache_size", 0)
    if l2_bytes:
        print_cache_argument(l2_bytes, gbs)
    print_ceilings(gbs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
