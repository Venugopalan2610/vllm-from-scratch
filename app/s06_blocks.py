"""Stage 06 - blocks, block tables, free list.

`./vc lore 6` for the insight. `./vc test 6` to check yourself.

This is virtual memory for the KV cache. Attention does not change yet. This
stage is only the memory manager. Make it correct, and stage 07 is mostly
bookkeeping.

Keep this mapping in mind:

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
    """The pool does not have the blocks for an allocation."""


class BlockAllocator:
    """A free list over `num_blocks` physical blocks. Each block holds
    `block_size` tokens of KV.

    Required attributes:
        .num_blocks   all the blocks of the pool
        .block_size   the tokens in one block
        .num_free     the number of free blocks now

    Required methods:
        allocate(n=1) -> list[int]   physical block ids, or OutOfBlocks
        free(block_ids)              give blocks back to the pool
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        raise NotImplementedError("stage 06: implement BlockAllocator")

    @property
    def num_free(self) -> int:
        raise NotImplementedError

    def allocate(self, n: int = 1) -> list[int]:
        """Give n free blocks. Raise OutOfBlocks if fewer than n are free.

        All or nothing: an allocation that fails must not keep some blocks.
        """
        raise NotImplementedError

    def free(self, block_ids) -> None:
        """Give blocks back to the pool. A call on a block that is already
        free must raise an error. It means that two sequences both think that
        they own the block. The corruption that follows is very difficult to
        find.
        """
        raise NotImplementedError


class BlockTable:
    """The page table of one sequence: logical token position -> physical
    slot.

    Required attributes:
        .blocks       list[int] of physical block ids, in logical order
        .num_tokens   the number of tokens stored now

    Required methods:
        append_token()          add one token. Allocate a block only when
                                the last block is full.
        reserve(num_tokens)     make room for num_tokens in total
        slot(pos) -> (block_id, offset)
        slot_index(pos) -> int  flat index: block_id * block_size + offset
        free()                  give every block back to the allocator
    """

    def __init__(self, allocator: BlockAllocator):
        raise NotImplementedError("stage 06: implement BlockTable")

    def append_token(self) -> None:
        raise NotImplementedError

    def reserve(self, num_tokens: int) -> None:
        raise NotImplementedError

    def slot(self, pos: int) -> tuple[int, int]:
        """Logical position -> (physical block id, offset in the block).

            block_id = self.blocks[pos // block_size]
            offset   = pos % block_size

        Raise IndexError for a position that has no block yet.
        """
        raise NotImplementedError

    def slot_index(self, pos: int) -> int:
        """The flat slot index, the form that a kernel uses:

            block_id * block_size + offset
        """
        raise NotImplementedError

    def free(self) -> None:
        raise NotImplementedError


def capacity_contiguous(vram_bytes: int, kv_bytes_per_token: int,
                        max_len: int) -> int:
    """The number of sequences that fit if each one RESERVES max_len tokens
    at the start.

    This is the world before vLLM. You cannot know the final length of a
    sequence, so you allocate for the worst case and accept the waste.

        vram_bytes // (max_len * kv_bytes_per_token)
    """
    raise NotImplementedError("stage 06: implement capacity_contiguous")


def capacity_paged(vram_bytes: int, kv_bytes_per_token: int,
                   block_size: int, seq_lens: list[int]) -> int:
    """The number of `seq_lens` that fit when each one pays only for the
    blocks that it uses.

    A sequence of length L uses ceil(L / block_size) blocks, that is
    ceil(L / block_size) * block_size token slots. Admit the sequences in
    order until the next one does not fit. Return the number that fit.
    """
    raise NotImplementedError("stage 06: implement capacity_paged")


class VirtualMemoryBlockManager:
    """A Python model of the CUDA driver's virtual memory API (cuMemMap).

    It calls no driver function. It counts pages, so that you can reason about a
    design where the address of the KV cache stays fixed and the memory behind it
    grows and shrinks. The engine of this course, like vLLM, preallocates its KV
    cache and pages it in software, through the block table. LORE.md section 18
    says which systems use the real API. The real API has these calls:
      1. cuMemAddressReserve: Reserves a large contiguous VIRTUAL address space
         (e.g., 128 GB) without committing physical GPU memory.
      2. cuMemCreate: Allocates physical memory chunks in fixed 2MB OS pages.
      3. cuMemMap: Maps physical pages to arbitrary virtual address ranges.
      4. cuMemSetAccess: Sets read/write permissions for the device.
      5. cuMemUnmap / cuMemRelease: Decouples and frees physical pages without
         moving data or re-allocating tensors.
    """

    def __init__(self, virtual_capacity_blocks: int, page_size_blocks: int = 16):
        self.virtual_capacity_blocks = virtual_capacity_blocks
        self.page_size_blocks = page_size_blocks
        self.virtual_address_reserved = True
        self.mapped_physical_pages = {}
        self.next_handle_id = 1

    def map_page(self, virtual_page_idx: int) -> int:
        raise NotImplementedError("stage 06: implement map_page")

    def unmap_page(self, virtual_page_idx: int) -> None:
        raise NotImplementedError("stage 06: implement unmap_page")

    @property
    def physical_pages_in_use(self) -> int:
        raise NotImplementedError("stage 06: implement physical_pages_in_use")
