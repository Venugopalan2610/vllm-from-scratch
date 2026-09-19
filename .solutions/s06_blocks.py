"""Reference solution, stage 06."""

import math


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = set()

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self.free_blocks):
            raise OutOfBlocks(f"need {n} blocks, only {len(self.free_blocks)} "
                              f"free ({self.num_blocks} total)")
        block_ids = [self.free_blocks.pop() for _ in range(n)]
        self.allocated_blocks.update(block_ids)
        return block_ids

    def free(self, block_ids) -> None:
        for block_id in block_ids:
            if block_id not in self.allocated_blocks:
                raise ValueError(f"block {block_id} is not allocated "
                                 f"(a double free?)")
            self.allocated_blocks.discard(block_id)
            self.free_blocks.append(block_id)


def blocks_for(num_tokens, block_size):
    return math.ceil(num_tokens / block_size)


class BlockTable:
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.blocks = []
        self.num_tokens = 0

    def append_token(self) -> None:
        if self.num_tokens % self.block_size == 0:
            self.blocks.extend(self.allocator.allocate(1))
        self.num_tokens += 1

    def reserve(self, num_tokens: int) -> None:
        missing = blocks_for(num_tokens, self.block_size) - len(self.blocks)
        if missing > 0:
            self.blocks.extend(self.allocator.allocate(missing))
        self.num_tokens = max(self.num_tokens, num_tokens)

    def slot(self, pos: int) -> tuple[int, int]:
        if pos < 0 or pos >= len(self.blocks) * self.block_size:
            raise IndexError(f"position {pos} not allocated")
        return self.blocks[pos // self.block_size], pos % self.block_size

    def slot_index(self, pos: int) -> int:
        block_id, offset = self.slot(pos)
        return block_id * self.block_size + offset

    def free(self) -> None:
        self.allocator.free(self.blocks)
        self.blocks = []
        self.num_tokens = 0


def capacity_contiguous(vram_bytes: int, kv_bytes_per_token: int,
                        max_len: int) -> int:
    return vram_bytes // (max_len * kv_bytes_per_token)


def capacity_paged(vram_bytes: int, kv_bytes_per_token: int,
                   block_size: int, seq_lens: list[int]) -> int:
    """The number of sequences, in order, that fit in whole blocks."""
    free_slots = vram_bytes // kv_bytes_per_token
    num_fit = 0
    for seq_len in seq_lens:
        slots_needed = blocks_for(seq_len, block_size) * block_size
        if slots_needed > free_slots:
            break
        free_slots -= slots_needed
        num_fit += 1
    return num_fit
