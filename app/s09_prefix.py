"""Stage 09 - copy-on-write and automatic prefix caching.

`./vc lore 9` for the insight. `./vc test 9` to check yourself.

Once a sequence's KV lives behind a block table, SHARING becomes a pointer
operation. Two sequences can point at the same physical block. That single fact
buys you three separate features:

  - n>1 sampling: fork the prompt's blocks instead of re-prefilling
  - beam search: all beams share the common prefix
  - automatic prefix caching: a shared system prompt is prefilled ONCE, for
    every user, forever

The price is that you now need refcounts, and copy-on-write when a shared block
is about to be modified. This is `fork()` and page tables, exactly.
"""


class RefCountedAllocator:
    """BlockAllocator from stage 06, plus reference counts.

    Required:
        .num_blocks  .block_size  .num_free
        allocate(n=1) -> list[int]      blocks start with refcount 1
        incref(block_id) -> int         new count
        decref(block_id) -> int         new count; frees the block at 0
        ref_count(block_id) -> int      0 if not allocated
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        raise NotImplementedError("stage 09: implement RefCountedAllocator")


class SharedBlockTable:
    """BlockTable from stage 06, plus fork() and copy-on-write.

    Required:
        .blocks  .num_tokens
        append_token()  reserve(n_tokens)  free()
        fork() -> SharedBlockTable
        prepare_write(pos) -> (block_id, copied_from)
    """

    def __init__(self, allocator: RefCountedAllocator):
        raise NotImplementedError("stage 09: implement SharedBlockTable")

    def fork(self):
        """A second table over the SAME physical blocks.

        Increment the refcount of every shared block. No KV is copied, and no
        prefill is repeated -- that is the entire point.
        """
        raise NotImplementedError

    def prepare_write(self, pos: int):
        """Make the block containing `pos` privately writable. The CoW step.

            if ref_count(block) == 1:  it is already private -> (block, None)
            else:                      allocate a new block, point this table's
                                       entry at it, decref the old one, and
                                       return (new, old) so the caller can
                                       physically copy the contents across

        Forget the decref and you leak a block per divergence, which shows up
        much later as a mysterious OOM under load.
        """
        raise NotImplementedError

    def free(self):
        """decref every block. Only those reaching 0 actually return to the pool."""
        raise NotImplementedError


def hash_block(token_ids, parent_hash=None):
    """Content hash of one block, CHAINED to its prefix hash.

    The chaining is essential. A block's identity is not just its own tokens --
    it is "these tokens, arriving after exactly this prefix". Hash only the
    block's own contents and you will happily serve a cached block from a
    completely different conversation that merely happens to share 16 tokens.
    """
    raise NotImplementedError("stage 09: implement hash_block")


def block_hashes(token_ids, block_size):
    """Chained hashes for each FULL block of token_ids.

    Partial trailing blocks are deliberately excluded: their contents are still
    growing, so they are not yet a stable cache key.
    """
    raise NotImplementedError("stage 09: implement block_hashes")


class PrefixCache:
    """Content-addressed store of already-computed KV blocks, with LRU eviction.

    Required:
        .num_cached  .hits  .misses
        lookup(hashes) -> list[int]   longest cached PREFIX; increfs each hit
        insert(hash, block_id)        cache holds its own reference
        evict_all()

    lookup must stop at the first miss. A cached block is only valid if every
    block before it also matched -- that is what the hash chain encodes.
    """

    def __init__(self, allocator: RefCountedAllocator, capacity=None):
        raise NotImplementedError("stage 09: implement PrefixCache")
