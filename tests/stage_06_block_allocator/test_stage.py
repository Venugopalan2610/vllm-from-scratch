"""Stage 06 - blocks, block tables, the free list.

The spec is in app/s06_blocks.py. There is no GPU and no model. This is a
pure data structure, and every later stage depends on it. Test it hard.
"""

import math
import random

import pytest

from app.s06_blocks import (
    BlockAllocator,
    BlockTable,
    OutOfBlocks,
    VirtualMemoryBlockManager,
    capacity_contiguous,
    capacity_paged,
)


def _table_with(num_blocks, block_size, num_tokens):
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    table = BlockTable(allocator)
    table.reserve(num_tokens)
    return allocator, table


# ---- allocator ------------------------------------------------------

def test_allocate_and_free_roundtrip():
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    assert allocator.num_free == 10
    block_ids = allocator.allocate(4)
    assert len(block_ids) == 4
    assert len(set(block_ids)) == 4, "allocate() gave the same block two times"
    assert allocator.num_free == 6
    allocator.free(block_ids)
    assert allocator.num_free == 10


def test_blocks_are_never_double_allocated():
    """The bug that puts the tokens of one sequence into another."""
    allocator = BlockAllocator(num_blocks=32, block_size=16)
    held = set()
    for _ in range(8):
        block_ids = set(allocator.allocate(4))
        assert not held & block_ids, (
            f"block(s) {held & block_ids} given out while still allocated")
        held |= block_ids
    assert allocator.num_free == 0


def test_out_of_blocks_raises_and_does_not_leak():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    allocator.allocate(3)
    with pytest.raises(OutOfBlocks):
        allocator.allocate(2)          # only 1 is free
    assert allocator.num_free == 1, (
        "an allocation that fails must be all or nothing. It kept some blocks.")
    assert len(allocator.allocate(1)) == 1


def test_double_free_is_caught():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    block_ids = allocator.allocate(2)
    allocator.free(block_ids)
    with pytest.raises(Exception):
        allocator.free(block_ids)


def test_allocator_survives_random_churn():
    """A fuzz check: the free count must never drift."""
    rng = random.Random(0)
    allocator = BlockAllocator(num_blocks=64, block_size=16)
    held_groups = []
    for _ in range(2000):
        if held_groups and rng.random() < 0.5:
            allocator.free(held_groups.pop(rng.randrange(len(held_groups))))
            continue
        count = rng.randint(1, 8)
        if count <= allocator.num_free:
            held_groups.append(allocator.allocate(count))
    held = sum(len(group) for group in held_groups)
    assert allocator.num_free + held == 64


# ---- block table ----------------------------------------------------

def test_block_table_grows_one_block_at_a_time():
    allocator = BlockAllocator(num_blocks=10, block_size=4)
    table = BlockTable(allocator)
    for num_tokens in range(1, 13):
        table.append_token()
        assert len(table.blocks) == math.ceil(num_tokens / 4), (
            f"after {num_tokens} tokens with block_size=4, expected "
            f"{math.ceil(num_tokens / 4)} blocks, got {len(table.blocks)}")
    assert allocator.num_free == 10 - 3


def test_slot_mapping_is_correct():
    _, table = _table_with(num_blocks=8, block_size=4, num_tokens=10)
    for position in range(10):
        block_id, offset = table.slot(position)
        assert block_id == table.blocks[position // 4]
        assert offset == position % 4
        assert table.slot_index(position) == block_id * 4 + offset


def test_slot_rejects_unallocated_positions():
    _, table = _table_with(num_blocks=8, block_size=4, num_tokens=4)
    with pytest.raises(IndexError):
        table.slot(99)


def test_logical_positions_are_contiguous_physical_ones_are_not():
    """This is the point of paging. The logical order stays. The physical
    order can be anything.

    Two sequences that allocate in turns have their physical blocks mixed,
    and that changes nothing for either sequence.
    """
    allocator = BlockAllocator(num_blocks=8, block_size=4)
    first, second = BlockTable(allocator), BlockTable(allocator)
    for _ in range(6):
        first.append_token()
        second.append_token()

    assert not set(first.blocks) & set(second.blocks), (
        "two sequences share a block!")
    for table in (first, second):
        for position in range(6):
            assert table.slot(position)[0] == table.blocks[position // 4]
    print(f"\n  physical blocks of sequence 1: {first.blocks}")
    print(f"  physical blocks of sequence 2: {second.blocks}")
    print("  \033[2mMixed in physical memory, contiguous in logical space.")
    print("  That is what a page table gives you.\033[0m")


def test_free_returns_everything():
    allocator, table = _table_with(num_blocks=8, block_size=4, num_tokens=16)
    assert allocator.num_free == 4
    table.free()
    assert allocator.num_free == 8
    assert table.blocks == []


# ---- the result -----------------------------------------------------

def test_capacity_math():
    budget_slots, bytes_per_slot = 1000, 1
    assert capacity_contiguous(budget_slots, bytes_per_slot, max_len=100) == 10
    # Sequences of 10 tokens at block_size 16 use 16 slots each -> 62 fit.
    assert capacity_paged(budget_slots, bytes_per_slot, 16, [10] * 100) == 62


def test_paging_beats_reservation_on_real_traffic(hf):
    """The number of the vLLM paper, with your model.

    Real requests have very different lengths. A contiguous allocator must
    reserve max_len for each one, because it cannot know the future.
    """
    model, _ = hf
    config = model.config
    head_dim = (getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads)
    kv_per_token = (2 * config.num_hidden_layers * config.num_key_value_heads
                    * head_dim * 2)

    budget_bytes = 6 * 1024**3        # an example KV budget of 6 GB
    max_len = 4096
    rng = random.Random(0)
    seq_lens = [rng.choice([64, 128, 256, 300, 512, 900]) for _ in range(4000)]
    mean_len = sum(seq_lens) / len(seq_lens)

    contiguous = capacity_contiguous(budget_bytes, kv_per_token, max_len)
    paged = capacity_paged(budget_bytes, kv_per_token, 16, seq_lens)

    print(f"\n  KV per token:     {kv_per_token / 1024:.1f} KB")
    print("  budget:           6 GB")
    print(f"  max_len:          {max_len}")
    print(f"  mean length:      {mean_len:.0f}")
    print(f"\n  contiguous (reserve max_len):  {contiguous:>5} sequences")
    print(f"  paged (pay for what you use):  {paged:>5} sequences")
    print(f"\n  \033[1m{paged / contiguous:.1f}x more sequences at one "
          "time\033[0m")
    print("  \033[2mAnd the batch size is the throughput. That is the paper.")
    print("\n  The gain is not magic, and it is not a constant. It follows")
    print(f"  max_len / mean_len = {max_len} / {mean_len:.0f} = "
          f"{max_len / mean_len:.1f}x,")
    print("  because that ratio IS the over-reservation that you stopped")
    print("  paying for. The vLLM paper reports 3-4x on traffic whose lengths")
    print("  are nearer to max_len. Set max_len=1024 here, and this number")
    print("  decreases.\033[0m")
    assert paged > contiguous * 3


def test_internal_waste_is_under_one_block(hf):
    """Paging has a limit on its waste: block_size-1 tokens for each
    sequence."""
    seq_lens = [17, 33, 100, 5, 64]
    block_size = 16
    used_slots = sum(math.ceil(seq_len / block_size) * block_size
                     for seq_len in seq_lens)
    waste = used_slots - sum(seq_lens)
    assert waste < block_size * len(seq_lens)
    print(f"\n  {len(seq_lens)} sequences, {sum(seq_lens)} real tokens, "
          f"{used_slots} slots used")
    print(f"  waste: {waste} tokens = {waste / len(seq_lens):.1f} for each "
          f"sequence (the limit is block_size-1 = {block_size - 1})")


def test_virtual_memory_cu_mem_map_lifecycle():
    """vLLM V1 architecture: dynamic physical page mapping (cuMemMap).
    Virtual address space is reserved up-front, while physical GPU pages are mapped/unmapped
    dynamically on demand without tensor reallocation or VRAM fragmentation."""
    # Reserve a 64-block virtual space
    vmm = VirtualMemoryBlockManager(virtual_capacity_blocks=64, page_size_blocks=16)
    assert vmm.virtual_address_reserved is True
    assert vmm.physical_pages_in_use == 0

    # Dynamically allocate and map physical pages to virtual slots 0 and 5
    h0 = vmm.map_page(0)
    h5 = vmm.map_page(5)
    assert h0 != h5
    assert vmm.physical_pages_in_use == 2

    # Mapping same virtual page returns the existing physical allocation handle
    assert vmm.map_page(0) == h0

    # Unmap virtual page 0 (simulates cuMemUnmap and freeing physical page)
    vmm.unmap_page(0)
    assert vmm.physical_pages_in_use == 1

    # Virtual slot 5 remains valid and mapped
    assert 5 in vmm.mapped_physical_pages
