"""Stage 06 - blocks, block tables, free list.

`./vc lore 6` for the insight. `./vc test 6` to check yourself.

This is virtual memory for the KV cache. No attention changes yet -- just the
memory manager. Get this right and stage 07 is mostly bookkeeping.

The mapping, worth holding in your head:

    OS                          here
    ---------------------------------------------------
    process                     sequence / request
    page (4 KB)                 block (block_size tokens of KV)
    page table                  block table
    physical frame              physical block in the pool
    virtual address             logical token position 0..n-1
    page fault / OOM            allocate() raising OutOfBlocks
"""


class OutOfBlocks(Exception):
    """Raised when the pool cannot satisfy an allocation."""


class BlockAllocator:
    """A free list over `num_blocks` physical blocks, each holding
    `block_size` tokens of KV.

    Required attributes:
        .num_blocks   total blocks in the pool
        .block_size   tokens per block
        .num_free     how many are currently free

    Required methods:
        allocate(n=1) -> list[int]   physical block ids; raises OutOfBlocks
        free(block_ids)              return blocks to the pool
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        raise NotImplementedError("stage 06: implement BlockAllocator")

    @property
    def num_free(self) -> int:
        raise NotImplementedError

    def allocate(self, n: int = 1) -> list[int]:
        """Hand out n free blocks. Raise OutOfBlocks if there aren't n free.

        All-or-nothing: a failed allocation must not leak partial blocks.
        """
        raise NotImplementedError

    def free(self, block_ids) -> None:
        """Return blocks to the pool. Freeing an already-free block should
        raise -- it means two sequences believe they own it, and the resulting
        corruption is very hard to trace back.
        """
        raise NotImplementedError


class BlockTable:
    """One sequence's page table: logical token position -> physical slot.

    Required attributes:
        .blocks       list[int] of physical block ids, in logical order
        .num_tokens   how many tokens are currently stored

    Required methods:
        append_token()          grow by one token, allocating a block only when
                                the current tail block is full
        reserve(n_tokens)       ensure capacity for n_tokens total
        slot(pos) -> (block_id, offset)
        slot_index(pos) -> int  flat index: block_id * block_size + offset
        free()                  return every block to the allocator
    """

    def __init__(self, allocator: BlockAllocator):
        raise NotImplementedError("stage 06: implement BlockTable")

    def append_token(self) -> None:
        raise NotImplementedError

    def reserve(self, n_tokens: int) -> None:
        raise NotImplementedError

    def slot(self, pos: int) -> tuple[int, int]:
        """Logical position -> (physical block id, offset within block).

            block_id = self.blocks[pos // block_size]
            offset   = pos % block_size

        Raise IndexError for a position that has not been allocated yet.
        """
        raise NotImplementedError

    def slot_index(self, pos: int) -> int:
        """Flat slot index, the form a kernel actually wants:

            block_id * block_size + offset
        """
        raise NotImplementedError

    def free(self) -> None:
        raise NotImplementedError


def capacity_contiguous(vram_bytes: int, kv_bytes_per_token: int,
                        max_len: int) -> int:
    """How many sequences fit if each RESERVES max_len tokens up front.

    This is the pre-vLLM world: you cannot know a sequence's final length, so
    you allocate for the worst case and eat the waste.

        vram_bytes // (max_len * kv_bytes_per_token)
    """
    raise NotImplementedError("stage 06: implement capacity_contiguous")


def capacity_paged(vram_bytes: int, kv_bytes_per_token: int,
                   block_size: int, seq_lens: list[int]) -> int:
    """How many of `seq_lens` fit when each pays only for the blocks it uses.

    A sequence of length L occupies ceil(L / block_size) blocks, i.e.
    ceil(L / block_size) * block_size token-slots. Greedily admit sequences in
    the given order until the next one does not fit; return how many fit.
    """
    raise NotImplementedError("stage 06: implement capacity_paged")
