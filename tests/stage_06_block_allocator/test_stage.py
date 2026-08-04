"""Stage 06 - Blocks, block tables, free list.

Spec in app/s06_blocks.py. No GPU, no model -- this is pure data structure, and
it is the foundation every later stage stands on. Test it hard.
"""

import math
import random

import pytest

from app.s06_blocks import (
    BlockAllocator,
    BlockTable,
    OutOfBlocks,
    capacity_contiguous,
    capacity_paged,
)


# ---- allocator ------------------------------------------------------

def test_allocate_and_free_roundtrip():
    a = BlockAllocator(num_blocks=10, block_size=16)
    assert a.num_free == 10
    blocks = a.allocate(4)
    assert len(blocks) == 4
    assert len(set(blocks)) == 4, "allocate() handed out the same block twice"
    assert a.num_free == 6
    a.free(blocks)
    assert a.num_free == 10


def test_blocks_are_never_double_allocated():
    """The bug that corrupts one sequence with another's tokens."""
    a = BlockAllocator(num_blocks=32, block_size=16)
    held = set()
    for _ in range(8):
        got = a.allocate(4)
        assert not (held & set(got)), (
            f"block(s) {held & set(got)} handed out while still allocated"
        )
        held |= set(got)
    assert a.num_free == 0


def test_out_of_blocks_raises_and_does_not_leak():
    a = BlockAllocator(num_blocks=4, block_size=16)
    a.allocate(3)
    with pytest.raises(OutOfBlocks):
        a.allocate(2)          # only 1 free
    assert a.num_free == 1, (
        "a failed allocation must be all-or-nothing; it leaked a partial grab"
    )
    assert len(a.allocate(1)) == 1


def test_double_free_is_caught():
    a = BlockAllocator(num_blocks=4, block_size=16)
    b = a.allocate(2)
    a.free(b)
    with pytest.raises(Exception):
        a.free(b)


def test_allocator_survives_random_churn():
    """Fuzz: the free count must never drift."""
    rng = random.Random(0)
    a = BlockAllocator(num_blocks=64, block_size=16)
    held = []
    for _ in range(2000):
        if held and rng.random() < 0.5:
            grp = held.pop(rng.randrange(len(held)))
            a.free(grp)
        else:
            n = rng.randint(1, 8)
            if n <= a.num_free:
                held.append(a.allocate(n))
    assert a.num_free + sum(len(g) for g in held) == 64


# ---- block table ----------------------------------------------------

def test_block_table_grows_one_block_at_a_time():
    a = BlockAllocator(num_blocks=10, block_size=4)
    t = BlockTable(a)
    for i in range(1, 13):
        t.append_token()
        assert len(t.blocks) == math.ceil(i / 4), (
            f"after {i} tokens with block_size=4 expected "
            f"{math.ceil(i / 4)} blocks, got {len(t.blocks)}"
        )
    assert a.num_free == 10 - 3


def test_slot_mapping_is_correct():
    a = BlockAllocator(num_blocks=8, block_size=4)
    t = BlockTable(a)
    t.reserve(10)
    for pos in range(10):
        blk, off = t.slot(pos)
        assert blk == t.blocks[pos // 4]
        assert off == pos % 4
        assert t.slot_index(pos) == blk * 4 + off


def test_slot_rejects_unallocated_positions():
    a = BlockAllocator(num_blocks=8, block_size=4)
    t = BlockTable(a)
    t.reserve(4)
    with pytest.raises(IndexError):
        t.slot(99)


def test_logical_positions_are_contiguous_physical_ones_are_not():
    """The whole point of paging: logical order is preserved, physical is free.

    Two sequences interleaving their allocations end up with scrambled
    physical blocks, and neither one cares.
    """
    a = BlockAllocator(num_blocks=8, block_size=4)
    t1, t2 = BlockTable(a), BlockTable(a)
    for _ in range(6):
        t1.append_token()
        t2.append_token()

    assert len(set(t1.blocks) & set(t2.blocks)) == 0, "sequences share a block!"
    # logical order intact for each
    for t in (t1, t2):
        for pos in range(6):
            assert t.slot(pos)[0] == t.blocks[pos // 4]
    print(f"\n  seq1 physical blocks: {t1.blocks}")
    print(f"  seq2 physical blocks: {t2.blocks}")
    print("  \033[2mInterleaved in physical memory, contiguous in logical space.")
    print("  Exactly what a page table buys you.\033[0m")


def test_free_returns_everything():
    a = BlockAllocator(num_blocks=8, block_size=4)
    t = BlockTable(a)
    t.reserve(16)
    assert a.num_free == 4
    t.free()
    assert a.num_free == 8
    assert t.blocks == []


# ---- the payoff -----------------------------------------------------

def test_capacity_math():
    # 1000 token-slots of budget
    vram, kv = 1000, 1
    assert capacity_contiguous(vram, kv, max_len=100) == 10
    # ten 10-token sequences at block_size 16 -> 16 slots each -> 62 fit
    assert capacity_paged(vram, kv, 16, [10] * 100) == 62


def test_paging_beats_reservation_on_real_traffic(hf):
    """The number from the vLLM paper, on your GPU, with your model.

    Real requests have wildly different lengths, but a contiguous allocator has
    to reserve max_len for every one of them because it cannot know the future.
    """
    model, _ = hf
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2

    vram = 6 * 1024**3        # 6 GB of KV budget
    max_len = 4096
    rng = random.Random(0)
    lens = [rng.choice([64, 128, 256, 300, 512, 900]) for _ in range(4000)]

    contig = capacity_contiguous(vram, kv_per_token, max_len)
    paged = capacity_paged(vram, kv_per_token, 16, lens)

    print(f"\n  KV per token:     {kv_per_token / 1024:.1f} KB")
    print(f"  budget:           6 GB")
    print(f"  max_len:          {max_len}")
    print(f"  mean actual len:  {sum(lens) / len(lens):.0f}")
    print(f"\n  contiguous (reserve max_len): {contig:>5} sequences")
    print(f"  paged (pay for what you use): {paged:>5} sequences")
    mean_len = sum(lens) / len(lens)
    print(f"\n  \033[1m{paged / contig:.1f}x more concurrent sequences\033[0m")
    print("  \033[2mAnd batch size is throughput. That is the whole paper.\033[0m")
    print(f"\n  \033[2mThe win is not magic, and it is not a constant: it tracks")
    print(f"  max_len / mean_len = {max_len} / {mean_len:.0f} = {max_len / mean_len:.1f}x,")
    print(f"  because that ratio IS the over-reservation you stopped paying for.")
    print(f"  The vLLM paper reports 3-4x on traffic whose lengths sit closer")
    print(f"  to max_len. Set max_len=1024 here and watch this number fall.\033[0m")

    assert paged > contig * 3


def test_internal_waste_is_under_one_block(hf):
    """Paged waste is bounded: at most block_size-1 tokens per sequence."""
    lens = [17, 33, 100, 5, 64]
    bs = 16
    used = sum(math.ceil(L / bs) * bs for L in lens)
    waste = used - sum(lens)
    assert waste < bs * len(lens)
    print(f"\n  {len(lens)} sequences, {sum(lens)} real tokens, {used} slots used")
    print(f"  waste: {waste} tokens = {waste / len(lens):.1f} per sequence "
          f"(bounded by block_size-1 = {bs - 1})")
