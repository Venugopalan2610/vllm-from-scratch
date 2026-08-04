"""./vc cliff

Measure the L2-vs-VRAM bandwidth cliff on this GPU, and what it would mean if a
model were small enough to live in cache. Backs Appendix A of LORE.md.
"""

import torch

B = "\033[1m"
D = "\033[2m"
Y = "\033[33m"
X = "\033[0m"

p = torch.cuda.get_device_properties(0)
l2_mb = getattr(p, "L2_cache_size", 0) / 1e6


def bw(mb, iters=200):
    # half the working set is source, half is destination
    n = mb * 1024 * 1024 // 2 // 2
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    for _ in range(50):
        b.copy_(a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        b.copy_(a)
    e.record()
    torch.cuda.synchronize()
    return (2 * a.numel() * 2) / (s.elapsed_time(e) / iters * 1e-3) / 1e9


print(f"\n{B}L2 cache on this GPU: {l2_mb:.0f} MB{X}\n")
print(f"  {'working set':>12}   {'bandwidth':>12}")

results = {}
for mb in (8, 16, 32, 64, 256, 1024):
    g = bw(mb)
    results[mb] = g
    fits = mb < l2_mb
    tag = f"{D}fits in L2{X}" if fits else f"{Y}spills to VRAM{X}"
    print(f"  {mb:>9} MB   {g:>8,.0f} GB/s   {tag}")

cached = max(g for mb, g in results.items() if mb < l2_mb)
vram = min(results.values())
print(f"\n{B}The cliff: {cached / vram:.1f}x{X}  "
      f"{D}({cached:,.0f} GB/s cached vs {vram:,.0f} GB/s from VRAM){X}")

print(f"\n{B}So what would fit in {l2_mb:.0f} MB?{X}")
for name, bytes_per in [("bf16", 2), ("int8", 1), ("int4", 0.5), ("1.58-bit", 0.21)]:
    params_m = l2_mb * 1e6 / bytes_per / 1e6
    print(f"  {name:<9} {params_m:>6,.0f}M parameters")
print(f"  {D}GPT-2 small was 124M. BERT-base was 110M.")
print(f"  'Fits in cache' means roughly 2019-era capability.{X}")

print(f"\n{B}And the catch{X}")
print(f"  {D}The KV cache does not shrink -- its size comes from context length")
print(f"  and layer count, not parameter count. Shrink the weights into L2 and")
print(f"  KV becomes ~all of your memory traffic. The bottleneck relocates")
print(f"  rather than disappearing, and stages 06-09 matter MORE, not less.")
print(f"\n  See Appendix A of LORE.md for where this bet is already shipping")
print(f"  (Groq, Cerebras) and why it is awkward on a GPU.{X}\n")
