"""Measure the roofline on THIS GPU. The numbers here are the ones that
explain every design decision in the repo."""

import torch

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
m = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
ms = bench(lambda: m @ m)
tflops = (2 * 4096**3) / (ms * 1e-3) / 1e12
print(f"\n\033[1mCompute\033[0m (bf16 4096^3 matmul)")
print(f"  {tflops:.0f} TFLOP/s")

print(f"\n\033[1mThe ratio that matters\033[0m")
print(f"  {tflops * 1e12 / (bw * 1e9):.0f} FLOP per byte of HBM traffic.")
print(f"  \033[2mA decode step does ~2 FLOP per weight byte. You are ~{tflops * 1e12 / (bw * 1e9) / 2:.0f}x")
print(f"  away from saturating compute -- which is exactly the headroom")
print(f"  that batching, and later speculative decoding, cash in.\033[0m")

# --- what this implies for a 7B model ---
print(f"\n\033[1mImplied ceilings\033[0m (bf16, batch=1, weights-only traffic)")
for name, gb in [("0.6B", 1.2), ("1B", 2.0), ("7B", 14.0), ("8B (fp8)", 8.0)]:
    tok_s = bw / gb
    print(f"  {name:<10} {gb:>5.1f} GB/step -> {tok_s:>7.0f} tok/s ceiling  ({1000 / tok_s:.2f} ms/token)")

print(f"\n\033[2mNo kernel beats these at batch=1. Batching is how you beat them.\033[0m\n")
