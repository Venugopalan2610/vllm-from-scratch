"""Stage 09 - copy-on-write and automatic prefix caching.

The spec is in app/s09_prefix.py. This is a pure data structure again, and
it needs no GPU. So test it as hard as you can.

A reference-count bug here shows as an out-of-memory error under load, ten
stages later. That is the worst place to debug it.
"""

import random

from app.s09_prefix import (
    PrefixCache,
    RefCountedAllocator,
    SharedBlockTable,
    block_hashes,
    hash_block,
)


def _parent_and_child(num_blocks, block_size, num_tokens):
    """A table of num_tokens and a fork of it."""
    allocator = RefCountedAllocator(num_blocks=num_blocks,
                                    block_size=block_size)
    parent = SharedBlockTable(allocator)
    parent.reserve(num_tokens)
    return allocator, parent, parent.fork()


# ---- reference counts -----------------------------------------------

def test_block_frees_only_at_zero():
    allocator = RefCountedAllocator(num_blocks=4, block_size=16)
    block_id = allocator.allocate(1)[0]
    assert allocator.ref_count(block_id) == 1
    assert allocator.num_free == 3

    allocator.incref(block_id)
    assert allocator.ref_count(block_id) == 2
    allocator.decref(block_id)
    assert allocator.num_free == 3, "freed a block that still had a reference"
    allocator.decref(block_id)
    assert allocator.num_free == 4
    assert allocator.ref_count(block_id) == 0


def test_refcount_fuzz_never_leaks():
    rng = random.Random(7)
    allocator = RefCountedAllocator(num_blocks=32, block_size=16)
    expected_counts = {}
    for _ in range(3000):
        choice = rng.random()
        if choice < 0.4 and allocator.num_free:
            block_id = allocator.allocate(1)[0]
            expected_counts[block_id] = 1
        elif choice < 0.7 and expected_counts:
            block_id = rng.choice(list(expected_counts))
            allocator.incref(block_id)
            expected_counts[block_id] += 1
        elif expected_counts:
            block_id = rng.choice(list(expected_counts))
            allocator.decref(block_id)
            expected_counts[block_id] -= 1
            if expected_counts[block_id] == 0:
                del expected_counts[block_id]
    assert allocator.num_free == 32 - len(expected_counts)


# ---- fork and copy-on-write -----------------------------------------

def test_fork_shares_blocks_without_copying():
    """n>1 sampling: one prefill of the prompt serves every sample."""
    allocator = RefCountedAllocator(num_blocks=16, block_size=4)
    parent = SharedBlockTable(allocator)
    parent.reserve(12)                     # 3 blocks
    free_before_fork = allocator.num_free

    child = parent.fork()
    assert child.blocks == parent.blocks, "fork must share the SAME block ids"
    assert allocator.num_free == free_before_fork, "fork allocated new blocks"
    for block_id in parent.blocks:
        assert allocator.ref_count(block_id) == 2


def test_copy_on_write_only_copies_shared_blocks():
    allocator, parent, child = _parent_and_child(16, 4, 8)   # 2 blocks

    # The child diverges in block 0.
    private_block, copied_from = child.prepare_write(pos=1)
    assert copied_from is not None, "a write to a shared block must copy it"
    assert private_block != copied_from
    assert child.blocks[0] == private_block
    assert parent.blocks[0] == copied_from, "the parent must not change"
    assert allocator.ref_count(copied_from) == 1, (
        "only the parent must hold the old block now")

    # A second write to the private block copies nothing.
    same_block, nothing_copied = child.prepare_write(pos=1)
    assert same_block == private_block
    assert nothing_copied is None


def test_cow_does_not_leak_blocks():
    """Every cycle of fork, diverge and free must make the pool full again."""
    allocator = RefCountedAllocator(num_blocks=16, block_size=4)
    for _ in range(50):
        parent = SharedBlockTable(allocator)
        parent.reserve(8)
        child = parent.fork()
        child.prepare_write(0)
        child.prepare_write(4)
        parent.free()
        child.free()
    assert allocator.num_free == 16, (
        f"lost {16 - allocator.num_free} blocks in 50 cycles")


def test_freeing_parent_leaves_child_intact():
    allocator, parent, child = _parent_and_child(8, 4, 8)
    shared_blocks = list(parent.blocks)
    parent.free()
    for block_id in shared_blocks:
        assert allocator.ref_count(block_id) == 1, (
            "the blocks of the child were freed with the parent")
    assert child.blocks == shared_blocks


# ---- hashing --------------------------------------------------------

def test_hash_is_content_addressed():
    assert hash_block([1, 2, 3]) == hash_block([1, 2, 3])
    assert hash_block([1, 2, 3]) != hash_block([1, 2, 4])


def test_hash_is_chained_to_the_prefix():
    """The same tokens after a DIFFERENT prefix are a different block.

    With no chain, you serve one user the cached KV of the conversation of
    another user. That is a correctness bug and a data leak.
    """
    after_first = hash_block([9, 9], parent_hash=hash_block([1, 1]))
    after_second = hash_block([9, 9], parent_hash=hash_block([2, 2]))
    assert after_first != after_second, (
        "the block hash must depend on the prefix before it")


def test_block_hashes_skips_the_partial_tail():
    """A block that is half full still grows, so it is not a stable key."""
    hashes = block_hashes(list(range(20)), block_size=8)
    assert len(hashes) == 2, (
        f"expected 2 full blocks from 20 tokens, got {len(hashes)}")
    assert block_hashes(list(range(16)), 8) == hashes[:2]


def test_shared_prefix_produces_shared_hashes():
    system_prompt = list(range(100))
    first = block_hashes(system_prompt + [1000, 1001], 16)
    second = block_hashes(system_prompt + [2000, 2001], 16)
    num_shared = sum(1 for one, other in zip(first, second) if one == other)
    assert num_shared == 6, f"expected 6 shared full blocks, got {num_shared}"


# ---- the cache ------------------------------------------------------

def test_lookup_returns_longest_prefix_and_stops_at_first_miss():
    allocator = RefCountedAllocator(num_blocks=16, block_size=4)
    cache = PrefixCache(allocator)
    block_ids = allocator.allocate(3)
    cache.insert("h0", block_ids[0])
    cache.insert("h2", block_ids[2])        # h1 is not inserted, on purpose

    assert cache.lookup(["h0", "h1", "h2"]) == [block_ids[0]], (
        "lookup must stop at the first miss. A later block is valid only if "
        "every block before it matched.")


def test_lookup_increfs_so_the_blocks_cannot_be_freed_underneath():
    allocator = RefCountedAllocator(num_blocks=8, block_size=4)
    cache = PrefixCache(allocator)
    block_id = allocator.allocate(1)[0]
    cache.insert("h", block_id)
    count_before = allocator.ref_count(block_id)
    assert cache.lookup(["h"]) == [block_id]
    assert allocator.ref_count(block_id) == count_before + 1, (
        "a cache hit must take a reference")


def test_lru_eviction_respects_capacity():
    allocator = RefCountedAllocator(num_blocks=32, block_size=4)
    cache = PrefixCache(allocator, capacity=2)
    block_ids = allocator.allocate(3)
    cache.insert("a", block_ids[0])
    cache.insert("b", block_ids[1])
    cache.lookup(["a"])                     # use a, so b is the oldest
    cache.insert("c", block_ids[2])
    assert cache.num_cached == 2
    assert cache.lookup(["b"]) == [], (
        "expected the eviction of b, the least recently used block")
    assert cache.lookup(["a"]) == [block_ids[0]]


def test_cache_saves_prefill_on_a_shared_system_prompt():
    """The main feature. The same system prompt, many users."""
    block_size = 16
    allocator = RefCountedAllocator(num_blocks=4096, block_size=block_size)
    cache = PrefixCache(allocator)
    system_prompt = list(range(2000))
    hashes = block_hashes(system_prompt, block_size)

    # User 1 is cold. Every block misses, so the engine computes all of them.
    assert cache.lookup(hashes) == []
    for block_hash in hashes:
        cache.insert(block_hash, allocator.allocate(1)[0])
    computed = len(hashes)

    # Users 2 to 10 are warm.
    reused = sum(len(cache.lookup(block_hashes(system_prompt + [999, 998, 997],
                                               block_size)))
                 for _ in range(9))

    print(f"\n  system prompt: {len(system_prompt)} tokens = {len(hashes)} "
          "full blocks")
    print(f"  user 1 (cold): computed {computed} blocks")
    print(f"  users 2-10:    used again {reused} blocks, computed 0")
    print(f"\n  \033[1mPrefill work saved: "
          f"{reused / (computed * 10) * 100:.0f}%\033[0m")
    print("  \033[2mThat is automatic prefix caching. It is a page cache, and")
    print("  it is why the TTFT of the second request of a conversation is")
    print("  so small.\033[0m")
    assert reused == len(hashes) * 9
    assert cache.hits > cache.misses
