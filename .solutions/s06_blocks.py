"""Reference solution, stage 06."""

import math


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))
        self._allocated = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(
                f"need {n} blocks, only {len(self._free)} free "
                f"({self.num_blocks} total)"
            )
        out = [self._free.pop() for _ in range(n)]
        self._allocated.update(out)
        return out

    def free(self, block_ids) -> None:
        for b in block_ids:
            if b not in self._allocated:
                raise ValueError(f"block {b} is not allocated (double free?)")
            self._allocated.discard(b)
            self._free.append(b)


class BlockTable:
    def __init__(self, allocator: BlockAllocator):
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

    def slot(self, pos: int) -> tuple[int, int]:
        if pos < 0 or pos >= len(self.blocks) * self.block_size:
            raise IndexError(f"position {pos} not allocated")
        return self.blocks[pos // self.block_size], pos % self.block_size

    def slot_index(self, pos: int) -> int:
        b, off = self.slot(pos)
        return b * self.block_size + off

    def free(self) -> None:
        self.alloc.free(self.blocks)
        self.blocks = []
        self.num_tokens = 0


def capacity_contiguous(vram_bytes: int, kv_bytes_per_token: int,
                        max_len: int) -> int:
    return vram_bytes // (max_len * kv_bytes_per_token)


def capacity_paged(vram_bytes: int, kv_bytes_per_token: int,
                   block_size: int, seq_lens: list[int]) -> int:
    budget = vram_bytes // kv_bytes_per_token  # in token-slots
    used = 0
    n = 0
    for L in seq_lens:
        need = math.ceil(L / block_size) * block_size
        if used + need > budget:
            break
        used += need
        n += 1
    return n
