"""./vc cliff

Measure the bandwidth cliff between L2 and VRAM on this GPU. Then show what
it means if a model is small enough to stay in the cache. Appendix A of
LORE.md uses it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import cudalib  # noqa: E402
from runner.style import paint  # noqa: E402

WORKING_SETS_MB = (8, 16, 32, 64, 256, 1024)
# (format, bytes for each parameter)
FORMATS = [("bf16", 2), ("int8", 1), ("int4", 0.5), ("1.58-bit", 0.21)]


def measure_working_sets(l2_mb):
    """-> {working set MB: GB/s}, printed as it goes."""
    print(f"  {'working set':>12}   {'bandwidth':>12}")
    bandwidths = {}
    for working_set_mb in WORKING_SETS_MB:
        gbs = cudalib.copy_bandwidth(working_set_mb, iters=200, warmup=50)
        bandwidths[working_set_mb] = gbs
        where = (paint("fits in L2", "dim") if working_set_mb < l2_mb
                 else paint("spills to VRAM", "yellow"))
        print(f"  {working_set_mb:>9} MB   {gbs:>8,.0f} GB/s   {where}")
    return bandwidths


def print_cliff(bandwidths, l2_mb):
    cached = max(gbs for working_set_mb, gbs in bandwidths.items()
                 if working_set_mb < l2_mb)
    from_vram = min(bandwidths.values())
    print("\n" + paint(f"The cliff: {cached / from_vram:.1f}x", "bold") + "  "
          + paint(f"({cached:,.0f} GB/s cached against {from_vram:,.0f} GB/s "
                  "from VRAM)", "dim"))


def print_what_fits(l2_mb):
    print("\n" + paint(f"So what fits in {l2_mb:.0f} MB?", "bold"))
    for name, bytes_per_parameter in FORMATS:
        print(f"  {name:<9} {l2_mb / bytes_per_parameter:>6,.0f}M parameters")
    print(paint("  GPT-2 small was 124M. BERT-base was 110M.\n"
                "  'Fits in cache' means about the capability of 2019.", "dim"))


def print_the_catch():
    print("\n" + paint("And the catch", "bold"))
    print(paint("  The KV cache does not become smaller. The context length and\n"
                "  the layer count set its size, not the parameter count. Put\n"
                "  the weights in L2, and KV becomes almost all of your memory\n"
                "  traffic. The bottleneck moves. It does not go away. Stages 06\n"
                "  to 09 are MORE important, not less.\n"
                "\n  Appendix A of LORE.md tells where this bet already ships\n"
                "  (Groq, Cerebras), and why it is awkward on a GPU.", "dim")
          + "\n")


def main():
    properties = torch.cuda.get_device_properties(0)
    l2_mb = getattr(properties, "L2_cache_size", 0) / 1e6
    print("\n" + paint(f"L2 cache on this GPU: {l2_mb:.0f} MB", "bold") + "\n")
    bandwidths = measure_working_sets(l2_mb)
    print_cliff(bandwidths, l2_mb)
    print_what_fits(l2_mb)
    print_the_catch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
