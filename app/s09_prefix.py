"""Stage 09 - copy-on-write and automatic prefix caching.

`./vc lore 9` for the insight. `./vc test 9` to check yourself.

When the KV of a sequence is behind a block table, SHARING is a pointer
operation. Two sequences can point at the same physical block. That one fact
gives three features:

  - n>1 sampling: fork the blocks of the prompt, and do not prefill again
  - beam search: all the beams share the common prefix
  - automatic prefix caching: one prefill of a shared system prompt serves
    every user, for ever

The cost: you now need reference counts, and a copy-on-write before a write
into a shared block. This is exactly `fork()` and a page table.

**What this stage does NOT build.** LRU eviction under memory pressure
(the cache evicts, but it does not choose which blocks to evict based on
recency), and cache-aware scheduling (admitting the request whose prefix
is already cached). In production, these two features decide your
capacity and your hit rate under load.
"""


class RefCountedAllocator:
    """The BlockAllocator of stage 06, with reference counts.

    Required:
        .num_blocks  .block_size  .num_free
        allocate(n=1) -> list[int]      each block starts with a count of 1
        incref(block_id) -> int         the new count
        decref(block_id) -> int         the new count. At 0 the block is free.
        ref_count(block_id) -> int      0 if the block is not allocated
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        raise NotImplementedError("stage 09: implement RefCountedAllocator")


class SharedBlockTable:
    """The BlockTable of stage 06, with fork() and copy-on-write.

    Required:
        .blocks  .num_tokens
        append_token()  reserve(num_tokens)  free()
        fork() -> SharedBlockTable
        prepare_write(pos) -> (block_id, copied_from)
    """

    def __init__(self, allocator: RefCountedAllocator):
        raise NotImplementedError("stage 09: implement SharedBlockTable")

    def fork(self):
        """A second table over the SAME physical blocks.

        Increment the count of every shared block. Copy no KV, and do not
        prefill again. That is the point.
        """
        raise NotImplementedError

    def prepare_write(self, pos: int):
        """Make the block that holds `pos` private to this table. This is
        the copy-on-write step.

            if ref_count(block) == 1:  it is already private -> (block, None)
            else:                      allocate a new block, point the entry
                                       of this table at it, decref the old
                                       block, and return (new, old). The
                                       caller then copies the contents.

        If you forget the decref, you lose one block at each divergence. Much
        later, under load, that shows as an out-of-memory error with no clear
        cause.
        """
        raise NotImplementedError

    def free(self):
        """decref every block. Only a block that gets to 0 goes back to the
        pool."""
        raise NotImplementedError


def hash_block(token_ids, parent_hash=None):
    """The content hash of one block, CHAINED to the hash of its prefix.

    The chain is necessary. The identity of a block is not only its own
    tokens. It is "these tokens, after exactly this prefix". If you hash only
    the contents of the block, you can serve a cached block from a different
    conversation. It only has the same 16 tokens by chance.
    """
    raise NotImplementedError("stage 09: implement hash_block")


def block_hashes(token_ids, block_size):
    """The chained hashes of each FULL block of token_ids.

    A partial block at the end is not included, on purpose: its contents
    still grow, so it is not a stable cache key yet.
    """
    raise NotImplementedError("stage 09: implement block_hashes")


class PrefixCache:
    """A store of computed KV blocks with the content as the key, and LRU
    eviction.

    Required:
        .num_cached  .hits  .misses
        lookup(hashes) -> list[int]   the longest cached PREFIX. incref each hit.
        insert(hash, block_id)        the cache holds its own reference
        evict_all()
        match_prefix_len(hashes) -> int  count matching prefix blocks without incref
        evict_lru(count=1) -> int     evict oldest LRU blocks under pressure

    lookup must stop at the first miss. A cached block is valid only if every
    block before it also matched. The hash chain records that.
    """

    def __init__(self, allocator: RefCountedAllocator, capacity=None):
        raise NotImplementedError("stage 09: implement PrefixCache")
