"""Tests for Stage 29 - Prefix Cache LRU Eviction & Cache-Aware Admission."""

import pytest
from app.s06_blocks import BlockAllocator
from app.s29_lru_cache import CacheAwareScheduler, LRUPrefixCache


def test_lru_cache_retains_blocks_with_zero_refcount():
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    cache = LRUPrefixCache(alloc)

    b0 = alloc.allocate(1)[0]
    cache.insert((1, 2, 3), b0)
    assert cache.ref_counts[b0] == 1

    # Release reduces refcount to 0, but does NOT immediately free from allocator
    cache.release(b0)
    assert cache.ref_counts[b0] == 0
    assert alloc.num_free == 7  # Not freed yet!

    # Re-acquire works
    b_hit = cache.acquire((1, 2, 3))
    assert b_hit == b0
    assert cache.ref_counts[b0] == 1


def test_lru_eviction_under_memory_pressure():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    cache = LRUPrefixCache(alloc)

    # Allocate 4 blocks
    blocks = [alloc.allocate(1)[0] for _ in range(4)]
    for i, b in enumerate(blocks):
        cache.insert((i,), b)
        cache.release(b)  # All refcounts = 0

    assert alloc.num_free == 0

    # Evict 2 oldest blocks (b0, b1)
    evicted = cache.evict(2)
    assert evicted == 2
    assert alloc.num_free == 2

    # b0 and b1 should no longer be in cache
    assert cache.acquire((0,)) is None
    assert cache.acquire((1,)) is None

    # b2 and b3 should still be cached
    assert cache.acquire((2,)) == blocks[2]
    assert cache.acquire((3,)) == blocks[3]


def test_touch_updates_lru_recency():
    alloc = BlockAllocator(num_blocks=3, block_size=16)
    cache = LRUPrefixCache(alloc)

    b0 = alloc.allocate(1)[0]
    cache.insert((10,), b0)
    cache.release(b0)

    b1 = alloc.allocate(1)[0]
    cache.insert((20,), b1)
    cache.release(b1)

    # Touch b0 (making b0 more recent than b1)
    cache.acquire((10,))
    cache.release(b0)

    # Evict 1 block -> b1 should be evicted first because b0 was touched
    evicted = cache.evict(1)
    assert evicted == 1
    assert cache.acquire((20,)) is None  # b1 evicted
    assert cache.acquire((10,)) == b0    # b0 retained


def test_active_blocks_are_never_evicted():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    cache = LRUPrefixCache(alloc)

    b0 = alloc.allocate(1)[0]
    cache.insert((1,), b0)  # refcount = 1 (active)

    b1 = alloc.allocate(1)[0]
    cache.insert((2,), b1)
    cache.release(b1)       # refcount = 0 (inactive)

    # Evict 2 blocks -> can only evict b1
    evicted = cache.evict(2)
    assert evicted == 1
    assert cache.acquire((2,)) is None
    assert cache.acquire((1,)) == b0  # Still active


def test_cache_aware_scheduler_prioritizes_prefix_hits():
    alloc = BlockAllocator(num_blocks=10, block_size=4)
    cache = LRUPrefixCache(alloc)
    scheduler = CacheAwareScheduler(cache, block_size=4)

    # Populate cache with a system prompt: [1, 2, 3, 4]
    b0 = alloc.allocate(1)[0]
    cache.insert((1, 2, 3, 4), b0)

    req_cold = {"id": "cold", "prompt_tokens": [99, 98, 97, 96, 95]}
    req_warm = {"id": "warm", "prompt_tokens": [1, 2, 3, 4, 10, 20]}

    queue = [req_cold, req_warm]
    prioritized = scheduler.prioritize_queue(queue)

    # req_warm hits 4 tokens, req_cold hits 0 tokens -> warm prioritized first
    assert prioritized[0]["id"] == "warm"
    assert prioritized[1]["id"] == "cold"
