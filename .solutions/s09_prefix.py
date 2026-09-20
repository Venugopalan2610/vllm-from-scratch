"""Reference solution, stage 09 - reference counts, copy-on-write, prefix
caching."""

import math
from collections import OrderedDict

from app.s06_blocks import OutOfBlocks


class RefCountedAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.ref_counts = {}

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self.free_blocks):
            raise OutOfBlocks(f"need {n}, {len(self.free_blocks)} free")
        block_ids = [self.free_blocks.pop() for _ in range(n)]
        for block_id in block_ids:
            self.ref_counts[block_id] = 1
        return block_ids

    def _check_allocated(self, block_id):
        if block_id not in self.ref_counts:
            raise ValueError(f"block {block_id} is not allocated")

    def incref(self, block_id: int) -> int:
        self._check_allocated(block_id)
        self.ref_counts[block_id] += 1
        return self.ref_counts[block_id]

    def decref(self, block_id: int) -> int:
        self._check_allocated(block_id)
        self.ref_counts[block_id] -= 1
        remaining = self.ref_counts[block_id]
        if remaining == 0:
            del self.ref_counts[block_id]
            self.free_blocks.append(block_id)
        return remaining

    def ref_count(self, block_id: int) -> int:
        return self.ref_counts.get(block_id, 0)


class SharedBlockTable:
    def __init__(self, allocator: RefCountedAllocator):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.blocks = []
        self.num_tokens = 0

    def append_token(self) -> None:
        if self.num_tokens % self.block_size == 0:
            self.blocks.extend(self.allocator.allocate(1))
        self.num_tokens += 1

    def reserve(self, num_tokens: int) -> None:
        missing = math.ceil(num_tokens / self.block_size) - len(self.blocks)
        if missing > 0:
            self.blocks.extend(self.allocator.allocate(missing))
        self.num_tokens = max(self.num_tokens, num_tokens)

    def fork(self) -> "SharedBlockTable":
        child = SharedBlockTable(self.allocator)
        child.blocks = list(self.blocks)
        child.num_tokens = self.num_tokens
        for block_id in child.blocks:
            self.allocator.incref(block_id)
        return child

    def prepare_write(self, pos: int):
        """Make the block that holds `pos` private to this table.

        -> (block_id, copied_from). copied_from is None if no copy is
        necessary. If it is not None, the caller copies the contents.
        """
        index = pos // self.block_size
        shared_block = self.blocks[index]
        if self.allocator.ref_count(shared_block) == 1:
            return shared_block, None
        private_block = self.allocator.allocate(1)[0]
        self.blocks[index] = private_block
        self.allocator.decref(shared_block)
        return private_block, shared_block

    def free(self) -> None:
        for block_id in self.blocks:
            self.allocator.decref(block_id)
        self.blocks = []
        self.num_tokens = 0


def hash_block(token_ids, parent_hash=None):
    """The content hash of one block, CHAINED to its prefix."""
    return hash((parent_hash, tuple(token_ids)))


def block_hashes(token_ids, block_size):
    """The chained hashes of every FULL block of token_ids."""
    hashes = []
    parent_hash = None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent_hash = hash_block(token_ids[start:start + block_size],
                                 parent_hash)
        hashes.append(parent_hash)
    return hashes


class PrefixCache:
    def __init__(self, allocator: RefCountedAllocator, capacity=None):
        self.allocator = allocator
        self.capacity = capacity
        self.block_of_hash = OrderedDict()   # least recently used first
        self.hits = 0
        self.misses = 0

    @property
    def num_cached(self):
        return len(self.block_of_hash)

    def lookup(self, hashes):
        """The longest cached PREFIX of `hashes`. Each block that it returns
        gets one more reference."""
        found = []
        for block_hash in hashes:
            if block_hash not in self.block_of_hash:
                break
            block_id = self.block_of_hash[block_hash]
            self.allocator.incref(block_id)
            self.block_of_hash.move_to_end(block_hash)
            found.append(block_id)
        self.hits += len(found)
        self.misses += len(hashes) - len(found)
        return found

    def insert(self, block_hash, block_id):
        if block_hash in self.block_of_hash:
            return
        self.allocator.incref(block_id)   # the cache holds a reference
        self.block_of_hash[block_hash] = block_id
        while self.capacity is not None and self.num_cached > self.capacity:
            self._evict_oldest()

    def _evict_oldest(self):
        _, block_id = self.block_of_hash.popitem(last=False)
        self.allocator.decref(block_id)

    def evict_all(self):
        while self.block_of_hash:
            self._evict_oldest()

    def match_prefix_len(self, hashes):
        """Count consecutive prefix block matches without taking references."""
        count = 0
        for block_hash in hashes:
            if block_hash not in self.block_of_hash:
                break
            count += 1
        return count

    def evict_lru(self, count=1):
        """Evict the oldest LRU blocks under memory pressure."""
        evicted = 0
        while self.block_of_hash and evicted < count:
            self._evict_oldest()
            evicted += 1
        return evicted
