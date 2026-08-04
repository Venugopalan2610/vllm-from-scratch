"""./vc math [params_in_billions] [batch]

Prints the two timings -- read vs compute -- with every division shown.
No number appears without its arithmetic.
"""

import sys

import torch

B = "\033[1m"
D = "\033[2m"
C = "\033[36m"
Y = "\033[33m"
X = "\033[0m"


def measure():
    """Your card's two rates. Same benchmarks ./vc info uses."""
    def bench(fn, iters, warmup):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    n = 256 * 1024 * 1024 // 2
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a), 30, 10)
    gbs = (2 * a.numel() * 2) / (ms * 1e-3) / 1e9

    m = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    import time as t
    end = t.perf_counter() + 2.0
    while t.perf_counter() < end:
        m @ m
    torch.cuda.synchronize()
    ms = bench(lambda: m @ m, 50, 0)
    gflops = (2 * 4096**3) / (ms * 1e-3) / 1e9
    return gbs, gflops


def main():
    params_b = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    gbs, gflops = measure()

    print(f"\n{D}Your card, measured just now:{X}")
    print(f"  memory bandwidth   {C}{gbs:,.0f} GB/s{X}      {D}(how fast bytes arrive){X}")
    print(f"  compute rate       {C}{gflops:,.0f} GFLOP/s{X}  {D}(how fast it does arithmetic){X}")

    gb = params_b * 2  # bf16 = 2 bytes per weight
    print(f"\n{B}A {params_b}B model in bf16{X}")
    print(f"  {params_b}e9 weights x 2 bytes each = {C}{gb:.1f} GB{X}")

    print(f"\n{B}Step 1: read the weights{X}  {D}(happens ONCE, no matter the batch){X}")
    read_ms = gb / gbs * 1000
    print(f"     {gb:.1f} GB")
    print(f"  {'-' * 18}  =  {C}{read_ms:.2f} ms{X}")
    print(f"   {gbs:,.0f} GB/s")

    print(f"\n{B}Step 2: do the math{X}  {D}(2 ops per weight, x batch size){X}")
    gflop = params_b * 2 * batch
    comp_ms = gflop / gflops * 1000
    print(f"   {params_b}e9 weights x 2 ops x {batch} request(s) = {gflop:,.1f} GFLOP")
    print(f"\n     {gflop:,.1f} GFLOP")
    print(f"  {'-' * 22}  =  {C}{comp_ms:.2f} ms{X}")
    print(f"   {gflops:,.0f} GFLOP/s")

    total = max(read_ms, comp_ms)
    bound = "MEMORY" if read_ms > comp_ms else "COMPUTE"
    color = Y if read_ms > comp_ms else C
    print(f"\n{B}Verdict at batch {batch}{X}")
    print(f"  reading  {read_ms:6.2f} ms")
    print(f"  computing{comp_ms:6.2f} ms")
    print(f"  -> {color}{bound}-bound{X}, ratio {max(read_ms, comp_ms) / min(read_ms, comp_ms):.0f}x")

    tok_s = batch / (total / 1000)
    print(f"\n  {batch} token(s) per ~{total:.2f} ms  =  {B}{tok_s:,.0f} tokens/sec{X}")

    crossover = read_ms / (comp_ms / batch)
    print(f"\n{B}Where batching stops helping{X}")
    print(f"     {read_ms:.2f} ms of reading")
    print(f"  {'-' * 26}  =  {B}batch {crossover:.0f}{X}")
    print(f"   {comp_ms / batch:.4f} ms per request")
    print(f"\n  {D}Below batch {crossover:.0f}, extra requests are nearly free -- you are")
    print(f"  filling idle time. Above it, you pay real arithmetic for each one.")
    print(f"  This is what max_num_seqs is set from.{X}\n")


if __name__ == "__main__":
    main()
