"""./vc math [params_in_billions] [batch]

Print the two timings, read against compute, and show every division. No
number appears without its arithmetic.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cudalib  # noqa: E402
from runner.style import paint  # noqa: E402

BYTES_PER_BF16 = 2
FLOP_PER_WEIGHT = 2          # one multiply and one add
HEAT_SECONDS = 2.0


def parse_args(argv):
    params_billion = float(argv[0]) if argv else 7.0
    batch_size = int(argv[1]) if len(argv) > 1 else 1
    return params_billion, batch_size


def print_card(gbs, gflops):
    print("\n" + paint("Your card, measured now:", "dim"))
    print("  memory bandwidth   " + paint(f"{gbs:,.0f} GB/s", "cyan")
          + "      " + paint("(how fast bytes arrive)", "dim"))
    print("  compute rate       " + paint(f"{gflops:,.0f} GFLOP/s", "cyan")
          + "  " + paint("(how fast it does arithmetic)", "dim"))


def print_division(numerator, denominator, result):
    print(f"     {numerator}")
    print(f"  {'-' * 22}  =  " + paint(result, "cyan"))
    print(f"   {denominator}")


def main(argv):
    params_billion, batch_size = parse_args(argv)
    gbs = cudalib.peak_bandwidth(fresh=True)
    gflops = cudalib.matmul_flops(heat_seconds=HEAT_SECONDS) / 1e9
    print_card(gbs, gflops)

    weight_gb = params_billion * BYTES_PER_BF16
    print("\n" + paint(f"A {params_billion}B model in bf16", "bold"))
    print(f"  {params_billion}e9 weights x 2 bytes each = "
          + paint(f"{weight_gb:.1f} GB", "cyan"))

    print("\n" + paint("Step 1: read the weights", "bold") + "  "
          + paint("(ONE time, for every batch size)", "dim"))
    read_ms = weight_gb / gbs * 1000
    print_division(f"{weight_gb:.1f} GB", f"{gbs:,.0f} GB/s", f"{read_ms:.2f} ms")

    print("\n" + paint("Step 2: do the math", "bold") + "  "
          + paint("(2 ops for each weight, x batch size)", "dim"))
    gflop = params_billion * FLOP_PER_WEIGHT * batch_size
    compute_ms = gflop / gflops * 1000
    print(f"   {params_billion}e9 weights x 2 ops x {batch_size} request(s) "
          f"= {gflop:,.1f} GFLOP\n")
    print_division(f"{gflop:,.1f} GFLOP", f"{gflops:,.0f} GFLOP/s",
                   f"{compute_ms:.2f} ms")

    step_ms = max(read_ms, compute_ms)
    memory_bound = read_ms > compute_ms
    print("\n" + paint(f"Verdict at batch {batch_size}", "bold"))
    print(f"  reading   {read_ms:6.2f} ms")
    print(f"  computing {compute_ms:6.2f} ms")
    print("  -> " + paint("MEMORY-bound" if memory_bound else "COMPUTE-bound",
                          "yellow" if memory_bound else "cyan")
          + f", ratio {step_ms / min(read_ms, compute_ms):.0f}x")
    tokens_per_second = batch_size / (step_ms / 1000)
    print(f"\n  {batch_size} token(s) for each ~{step_ms:.2f} ms  =  "
          + paint(f"{tokens_per_second:,.0f} tokens/sec", "bold"))

    compute_ms_per_request = compute_ms / batch_size
    crossover = read_ms / compute_ms_per_request
    print("\n" + paint("Where batching stops helping", "bold"))
    print(f"     {read_ms:.2f} ms of reading")
    print(f"  {'-' * 26}  =  " + paint(f"batch {crossover:.0f}", "bold"))
    print(f"   {compute_ms_per_request:.4f} ms for each request")
    print(paint(f"\n  Below batch {crossover:.0f}, one more request costs almost "
                "nothing.\n  It fills idle time. Above it, you pay real "
                "arithmetic for each one.\n  Set max_num_seqs from this number.",
                "dim") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
