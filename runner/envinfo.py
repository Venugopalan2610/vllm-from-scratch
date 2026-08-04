"""Measure the roofline on THIS GPU. The numbers here are the ones that
explain every design decision in the repo."""

import torch

D = "\033[2m"
X = "\033[0m"

p = torch.cuda.get_device_properties(0)
print(f"\n\033[1m{p.name}\033[0m")
print(f"  VRAM         {p.total_memory / 1e9:.1f} GB")
print(f"  SMs          {p.multi_processor_count}")
print(f"  Compute cap  {p.major}.{p.minor}")
print(f"  torch        {torch.__version__}")


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


# --- achievable HBM bandwidth: a big streaming copy ---
n = 256 * 1024 * 1024 // 2  # 256MB of bf16
a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
b = torch.empty_like(a)
ms = bench(lambda: b.copy_(a))
bw = (2 * a.numel() * 2) / (ms * 1e-3) / 1e9  # read + write
print(f"\n\033[1mMemory bandwidth\033[0m (measured, streaming copy)")
print(f"  {bw:.0f} GB/s")

# --- bf16 matmul throughput ---
# This is a 150W laptop GPU: it bursts to a high clock, then thermally
# throttles under sustained load. There is no single "the" number, so measure
# both. Serving is a sustained workload, so the second one is the honest one.
import time as _t

m = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
FLOP = 2 * 4096**3

burst = FLOP / (bench(lambda: m @ m, iters=30, warmup=20) * 1e-3) / 1e12

_end = _t.perf_counter() + 3.0  # heat it up
while _t.perf_counter() < _end:
    m @ m
torch.cuda.synchronize()
sustained = FLOP / (bench(lambda: m @ m, iters=100, warmup=0) * 1e-3) / 1e12

tflops = sustained
print(f"\n\033[1mCompute\033[0m (bf16 4096^3 matmul)")
print(f"  {burst:.0f} TFLOP/s burst  ->  {sustained:.0f} TFLOP/s sustained")
print(f"  \033[2mA 150W laptop part boosts, then throttles. Serving is a")
print(f"  sustained workload, so the second number is the one to plan with.")
print(f"  It also means benchmarks you run cold will lie to you.\033[0m")

ridge = tflops * 1e12 / (bw * 1e9)
print(f"\n\033[1mHARDWARE ratio\033[0m  {D}-- a property of this chip{X}")
print(f"     {tflops * 1000:,.0f} GFLOP/s")
print(f"  {'-' * 20}  =  \033[1m{ridge:.0f} FLOP per byte\033[0m")
print(f"      {bw:.0f} GB/s")
print(f"  {D}The /s cancels top and bottom, so this is a plain count ratio,")
print(f"  not a rate. That is what makes it comparable to the number below.{X}")

print(f"\n\033[1mWORKLOAD ratio\033[0m  {D}-- a property of how you batch{X}")
print(f"  A weight is 2 bytes (bf16) and is used in one multiply-add = 2 FLOP.")
print(f"  With N weights and batch B, per forward pass:")
print(f"     bytes read = 2N   {D}(weights fetched ONCE, whatever B is){X}")
print(f"     operations = 2NB")
print(f"\n     2NB")
print(f"  {'-' * 8}  =  \033[1mB FLOP per byte\033[0m   {D}(the 2 and the N cancel){X}")
print(f"      2N")

print(f"\n\033[1mCompare them\033[0m")
print(f"  B  <  {ridge:.0f}   ->  memory-bound, GPU sits waiting on bytes")
print(f"  B  >  {ridge:.0f}   ->  compute-bound, GPU sits waiting on math")
print(f"\n  At batch 1 the workload sits at 1, against {ridge:.0f}. "
      f"{D}({100 / ridge:.1f}% of peak compute.){X}")
print(f"\n    \033[1mbatch ~{ridge:.0f} is where decode stops being memory-bound.\033[0m")
print(f"\n  \033[2mThat is the whole argument for continuous batching, and why")
print(f"  real servers set max_num_seqs in the hundreds. (You will not quite")
print(f"  reach it -- KV cache traffic grows with B too, and that is the")
print(f"  constraint PagedAttention exists to relax.)")
print(f"\n  Neither ratio is something a profiler shows you. A profiler shows")
print(f"  achieved RATES (GB/s, GFLOP/s). These two numbers are what you work")
print(f"  out beforehand to predict which of those rates will be pegged.{X}")

# --- what this implies for a 7B model ---
print(f"\n\033[1mImplied ceilings\033[0m (bf16, batch=1, weights-only traffic)")
for name, gb in [("0.6B", 1.2), ("1B", 2.0), ("7B", 14.0), ("8B (fp8)", 8.0)]:
    tok_s = bw / gb
    print(f"  {name:<10} {gb:>5.1f} GB/step -> {tok_s:>7.0f} tok/s ceiling  ({1000 / tok_s:.2f} ms/token)")

print(f"\n\033[2mNo kernel beats these at batch=1. Batching is how you beat them.\033[0m\n")
