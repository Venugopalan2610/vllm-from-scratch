"""Reference solution, stage 09 - refcounting, CoW, prefix caching."""

import math
from collections import OrderedDict

from app.s06_blocks import OutOfBlocks


class RefCountedAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))
        self._refs = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(f"need {n}, {len(self._free)} free")
        out = [self._free.pop() for _ in range(n)]
        for b in out:
            self._refs[b] = 1
        return out

    def incref(self, block_id: int) -> int:
        if block_id not in self._refs:
            raise ValueError(f"block {block_id} is not allocated")
        self._refs[block_id] += 1
        return self._refs[block_id]

    def decref(self, block_id: int) -> int:
        if block_id not in self._refs:
            raise ValueError(f"block {block_id} is not allocated")
        self._refs[block_id] -= 1
        n = self._refs[block_id]
        if n == 0:
            del self._refs[block_id]
            self._free.append(block_id)
        return n

    def ref_count(self, block_id: int) -> int:
        return self._refs.get(block_id, 0)


class SharedBlockTable:
    def __init__(self, allocator: RefCountedAllocator):
        self.alloc = allocator
        self.block_size = allocator.block_size
        self.blocks = []
        self.num_tokens = 0

    def append_token(self) -> None:
        if self.num_tokens % self.block_size == 0:
            self.blocks.extend(self.alloc.allocate(1))
        self.num_tokens += 1

    def reserve(self, n_tokens: int) -> None:
        need = math.ceil(n_tokens / self.block_size)
        if need > len(self.blocks):
            self.blocks.extend(self.alloc.allocate(need - len(self.blocks)))
        self.num_tokens = max(self.num_tokens, n_tokens)

    def fork(self) -> "SharedBlockTable":
        child = SharedBlockTable(self.alloc)
        child.blocks = list(self.blocks)
        child.num_tokens = self.num_tokens
        for b in child.blocks:
            self.alloc.incref(b)
        return child

    def prepare_write(self, pos: int):
        """Make block containing `pos` privately writable.

        Returns (block_id, copied_from) -- copied_from is None if no copy was
        needed. The caller is responsible for physically copying the block's
        contents when copied_from is not None.
        """
        idx = pos // self.block_size
        old = self.blocks[idx]
        if self.alloc.ref_count(old) == 1:
            return old, None
        new = self.alloc.allocate(1)[0]
        self.blocks[idx] = new
        self.alloc.decref(old)
        return new, old

    def free(self) -> None:
        for b in self.blocks:
            self.alloc.decref(b)
        self.blocks = []
        self.num_tokens = 0


def hash_block(token_ids, parent_hash=None):
    """Content hash of one block, CHAINED to its prefix."""
    return hash((parent_hash, tuple(token_ids)))


def block_hashes(token_ids, block_size):
    """Chained hashes for every FULL block of token_ids."""
    out = []
    parent = None
    for i in range(0, len(token_ids) - block_size + 1, block_size):
        parent = hash_block(token_ids[i:i + block_size], parent)
        out.append(parent)
    return out


class PrefixCache:
    def __init__(self, allocator: RefCountedAllocator, capacity=None):
        self.alloc = allocator
        self.capacity = capacity
        self._map = OrderedDict()   # hash -> block_id
        self.hits = 0
        self.misses = 0

    @property
    def num_cached(self):
        return len(self._map)

    def lookup(self, hashes):
        """Longest cached PREFIX of `hashes`. Increfs every block returned."""
        out = []
        for h in hashes:
            if h in self._map:
                b = self._map[h]
                self.alloc.incref(b)
                self._map.move_to_end(h)
                out.append(b)
            else:
                break
        if out:
            self.hits += len(out)
        self.misses += len(hashes) - len(out)
        return out

    def insert(self, h, block_id):
        if h in self._map:
            return
        self.alloc.incref(block_id)   # the cache itself holds a reference
        self._map[h] = block_id
        if self.capacity is not None:
            while len(self._map) > self.capacity:
                _, victim = self._map.popitem(last=False)
                self.alloc.decref(victim)

    def evict_all(self):
        while self._map:
            _, b = self._map.popitem(last=False)
            self.alloc.decref(b)
