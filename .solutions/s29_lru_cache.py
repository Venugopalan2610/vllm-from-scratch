"""Stage 29 - Prefix Cache LRU Eviction & Cache-Aware Admission.

Reference solution.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple


class LRUPrefixCache:
    """A prefix cache keyed by chained block hashes, with LRU eviction under memory pressure."""

    def __init__(self, block_allocator):
        self.allocator = block_allocator
        # Maps block_id -> prefix
        self.lru_queue = OrderedDict()  # Least recently used at beginning
        self.cached_blocks: Dict[Tuple[int, ...], int] = {}  # token_prefix -> block_id
        self.ref_counts: Dict[int, int] = {}

    def insert(self, prefix: Tuple[int, ...], block_id: int):
        """Insert a block mapped to a prefix into the cache."""
        self.cached_blocks[prefix] = block_id
        self.lru_queue[block_id] = prefix
        self.ref_counts[block_id] = self.ref_counts.get(block_id, 0) + 1

    def touch(self, block_id: int):
        """Mark a block as recently used (moves to end of LRU queue)."""
        if block_id in self.lru_queue:
            prefix = self.lru_queue.pop(block_id)
            self.lru_queue[block_id] = prefix

    def acquire(self, prefix: Tuple[int, ...]) -> Optional[int]:
        """Look up a prefix. If found, increment refcount and touch."""
        block_id = self.cached_blocks.get(prefix)
        if block_id is not None:
            self.ref_counts[block_id] = self.ref_counts.get(block_id, 0) + 1
            self.touch(block_id)
            return block_id
        return None

    def release(self, block_id: int):
        """Decrement refcount. When 0, block becomes eligible for eviction."""
        if block_id in self.ref_counts:
            self.ref_counts[block_id] = max(0, self.ref_counts[block_id] - 1)

    def evict(self, num_blocks_needed: int) -> int:
        """Evict the least recently used blocks that have refcount == 0.
        Returns the number of blocks successfully freed."""
        evicted = 0
        to_remove = []
        for block_id, prefix in self.lru_queue.items():
            if self.ref_counts.get(block_id, 0) == 0:
                to_remove.append((block_id, prefix))
                evicted += 1
                if evicted >= num_blocks_needed:
                    break

        for block_id, prefix in to_remove:
            del self.lru_queue[block_id]
            if prefix in self.cached_blocks:
                del self.cached_blocks[prefix]
            if block_id in self.ref_counts:
                del self.ref_counts[block_id]
            self.allocator.free([block_id])

        return evicted


class CacheAwareScheduler:
    """Prioritizes waiting requests based on their prefix cache hit count."""

    def __init__(self, prefix_cache: LRUPrefixCache, block_size: int = 16):
        self.prefix_cache = prefix_cache
        self.block_size = block_size

    def compute_prefix_hits(self, prompt_tokens: List[int]) -> int:
        """Count how many prompt tokens can be fulfilled by the prefix cache."""
        hits = 0
        n_blocks = len(prompt_tokens) // self.block_size
        for i in range(n_blocks):
            prefix = tuple(prompt_tokens[: (i + 1) * self.block_size])
            if prefix in self.prefix_cache.cached_blocks:
                hits += self.block_size
            else:
                break
        return hits

    def prioritize_queue(self, waiting_requests: List[dict]) -> List[dict]:
        """Sort waiting requests descending by prefix hit token count.
        Requests with equal hits maintain arrival order (stable sort)."""
        # Python's sort is stable (Timsort)
        return sorted(
            waiting_requests,
            key=lambda req: self.compute_prefix_hits(req.get("prompt_tokens", [])),
            reverse=True,
        )
