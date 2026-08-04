"""Stage 09 - Copy-on-write and automatic prefix caching.

Spec in app/s09_prefix.py. Pure data structure again -- no GPU needed, so test
it as hard as you can. Refcount bugs here surface as OOM-under-load ten stages
later, which is the worst possible place to debug them.
"""

import random

import pytest

from app.s06_blocks import OutOfBlocks
from app.s09_prefix import (
    PrefixCache,
    RefCountedAllocator,
    SharedBlockTable,
    block_hashes,
    hash_block,
)


# ---- refcounting ----------------------------------------------------

def test_block_frees_only_at_zero():
    a = RefCountedAllocator(num_blocks=4, block_size=16)
    b = a.allocate(1)[0]
    assert a.ref_count(b) == 1
    assert a.num_free == 3

    a.incref(b)
    assert a.ref_count(b) == 2
    a.decref(b)
    assert a.num_free == 3, "freed a block that still had a reference"
    a.decref(b)
    assert a.num_free == 4
    assert a.ref_count(b) == 0


def test_refcount_fuzz_never_leaks():
    rng = random.Random(7)
    a = RefCountedAllocator(num_blocks=32, block_size=16)
    live = {}
    for _ in range(3000):
        r = rng.random()
        if r < 0.4 and a.num_free:
            b = a.allocate(1)[0]
            live[b] = live.get(b, 0) + 1
        elif r < 0.7 and live:
            b = rng.choice(list(live))
            a.incref(b)
            live[b] += 1
        elif live:
            b = rng.choice(list(live))
            a.decref(b)
            live[b] -= 1
            if live[b] == 0:
                del live[b]
    assert a.num_free == 32 - len(live)


# ---- fork / copy-on-write -------------------------------------------

def test_fork_shares_blocks_without_copying():
    """n>1 sampling: the prompt is prefilled once and shared."""
    a = RefCountedAllocator(num_blocks=16, block_size=4)
    parent = SharedBlockTable(a)
    parent.reserve(12)                    # 3 blocks
    free_after_parent = a.num_free

    child = parent.fork()
    assert child.blocks == parent.blocks, "fork must share the SAME block ids"
    assert a.num_free == free_after_parent, "fork allocated new blocks"
    for b in parent.blocks:
        assert a.ref_count(b) == 2


def test_copy_on_write_only_copies_shared_blocks():
    a = RefCountedAllocator(num_blocks=16, block_size=4)
    parent = SharedBlockTable(a)
    parent.reserve(8)                     # blocks 0,1
    child = parent.fork()

    # child diverges in block 0
    new, copied_from = child.prepare_write(pos=1)
    assert copied_from is not None, "writing a shared block must copy it"
    assert new != copied_from
    assert child.blocks[0] == new
    assert parent.blocks[0] == copied_from, "the parent must not be disturbed"
    assert a.ref_count(copied_from) == 1, "old block should be down to the parent"

    # second write to the now-private block copies nothing
    again, again_from = child.prepare_write(pos=1)
    assert again == new
    assert again_from is None


def test_cow_does_not_leak_blocks():
    """Every fork+diverge+free cycle must return the pool to full."""
    a = RefCountedAllocator(num_blocks=16, block_size=4)
    for _ in range(50):
        p = SharedBlockTable(a)
        p.reserve(8)
        c = p.fork()
        c.prepare_write(0)
        c.prepare_write(4)
        p.free()
        c.free()
    assert a.num_free == 16, f"leaked {16 - a.num_free} blocks over 50 cycles"


def test_freeing_parent_leaves_child_intact():
    a = RefCountedAllocator(num_blocks=8, block_size=4)
    p = SharedBlockTable(a)
    p.reserve(8)
    shared = list(p.blocks)
    c = p.fork()
    p.free()
    for b in shared:
        assert a.ref_count(b) == 1, "child's blocks were freed with the parent"
    assert c.blocks == shared


# ---- hashing --------------------------------------------------------

def test_hash_is_content_addressed():
    assert hash_block([1, 2, 3]) == hash_block([1, 2, 3])
    assert hash_block([1, 2, 3]) != hash_block([1, 2, 4])


def test_hash_is_chained_to_the_prefix():
    """The same 16 tokens after a DIFFERENT prefix is a different block.

    Without chaining you will serve one user cached KV computed for another
    user's conversation. That is both a correctness bug and a data leak.
    """
    a = hash_block([9, 9], parent_hash=hash_block([1, 1]))
    b = hash_block([9, 9], parent_hash=hash_block([2, 2]))
    assert a != b, "block hash must depend on the prefix that precedes it"


def test_block_hashes_skips_the_partial_tail():
    """A half-full block is still growing, so it is not a stable key."""
    hs = block_hashes(list(range(20)), block_size=8)
    assert len(hs) == 2, f"expected 2 full blocks from 20 tokens, got {len(hs)}"
    assert block_hashes(list(range(16)), 8) == hs[:2]


def test_shared_prefix_produces_shared_hashes():
    sys_prompt = list(range(100))
    a = block_hashes(sys_prompt + [1000, 1001], 16)
    b = block_hashes(sys_prompt + [2000, 2001], 16)
    common = sum(1 for x, y in zip(a, b) if x == y)
    assert common == 6, f"expected 6 shared full blocks, got {common}"


# ---- the cache ------------------------------------------------------

def test_lookup_returns_longest_prefix_and_stops_at_first_miss():
    a = RefCountedAllocator(num_blocks=16, block_size=4)
    cache = PrefixCache(a)
    blocks = a.allocate(3)
    hs = ["h0", "h1", "h2"]
    cache.insert(hs[0], blocks[0])
    cache.insert(hs[2], blocks[2])        # deliberately skip h1

    got = cache.lookup(["h0", "h1", "h2"])
    assert got == [blocks[0]], (
        "lookup must stop at the first miss -- a later block is only valid if "
        "every block before it matched"
    )


def test_lookup_increfs_so_the_blocks_cannot_be_freed_underneath():
    a = RefCountedAllocator(num_blocks=8, block_size=4)
    cache = PrefixCache(a)
    b = a.allocate(1)[0]
    cache.insert("h", b)
    before = a.ref_count(b)
    got = cache.lookup(["h"])
    assert got == [b]
    assert a.ref_count(b) == before + 1, "a cache hit must take a reference"


def test_lru_eviction_respects_capacity():
    a = RefCountedAllocator(num_blocks=32, block_size=4)
    cache = PrefixCache(a, capacity=2)
    bs = a.allocate(3)
    cache.insert("a", bs[0])
    cache.insert("b", bs[1])
    cache.lookup(["a"])                   # touch a, making b the LRU victim
    cache.insert("c", bs[2])
    assert cache.num_cached == 2
    assert cache.lookup(["b"]) == [], "expected b to be evicted as least-recent"
    assert cache.lookup(["a"]) == [bs[0]]


def test_cache_saves_prefill_on_a_shared_system_prompt():
    """The headline feature. Same system prompt, many users."""
    block_size = 16
    a = RefCountedAllocator(num_blocks=4096, block_size=block_size)
    cache = PrefixCache(a)

    system = list(range(2000))
    hs = block_hashes(system, block_size)

    # user 1: cold. every block is a miss and must be computed.
    hit = cache.lookup(hs)
    assert hit == []
    computed = 0
    for h in hs:
        blk = a.allocate(1)[0]
        cache.insert(h, blk)
        computed += 1

    # users 2..10: warm
    reused = 0
    for _ in range(9):
        conv = system + [999, 998, 997]
        got = cache.lookup(block_hashes(conv, block_size))
        reused += len(got)

    print(f"\n  system prompt: {len(system)} tokens = {len(hs)} full blocks")
    print(f"  user 1 (cold): computed {computed} blocks")
    print(f"  users 2-10:    reused   {reused} blocks, computed 0")
    print(f"\n  \033[1mPrefill work saved: {reused / (computed * 10) * 100:.0f}%\033[0m")
    print("  \033[2mThat is automatic prefix caching. It is a page cache, and it")
    print("  is why TTFT collapses on the second request of a conversation.\033[0m")

    assert reused == len(hs) * 9
    assert cache.hits > cache.misses
