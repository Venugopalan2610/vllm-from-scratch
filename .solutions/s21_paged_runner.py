"""Reference solution, stage 21 - the model runs on your paged cache."""

from dataclasses import dataclass, field

import torch

from app.s08_paged_cuda import write_kv_cuda
from app.s08c_cuda_warps import paged_attention_split
from tvllm import attend_to_context


@dataclass
class SeqChunk:
    """The work for one sequence in one step."""
    token_ids: list
    start_pos: int
    block_ids: list

    @property
    def is_decode(self):
        return len(self.token_ids) == 1

    @property
    def context_len(self):
        """The positions that attention reads, this chunk included."""
        return self.start_pos + len(self.token_ids)


@dataclass
class PrefillChunk:
    token_start: int          # the first row of this chunk in the flat batch
    num_tokens: int
    context_len: int
    block_ids: torch.Tensor   # int64, the blocks that cover context_len


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor              # (num_tokens,) int64
    num_decodes: int                        # the first rows are the decodes
    block_tables: torch.Tensor | None       # (num_decodes, width) int32
    context_lens: torch.Tensor | None       # (num_decodes,) int32
    prefills: list = field(default_factory=list)


def slot_of(block_ids, position, block_size):
    return block_ids[position // block_size] * block_size + position % block_size


def blocks_for(num_tokens, block_size):
    return (num_tokens + block_size - 1) // block_size


# ---------------------------------------------------------------- backend


class PagedAttention:
    """The attention backend of tvllm, over your paged cache."""

    def __init__(self, kv_caches, block_size):
        self.kv_caches = kv_caches
        self.block_size = block_size
        self.metadata = None

    # Stage 24b overrides write, attend_decodes and load_context for an FP8
    # cache. The other steps stay the same.

    def write(self, layer, key, value):
        key_cache, value_cache = self.kv_caches[layer]
        write_kv_cuda(key_cache, value_cache, key, value,
                      self.metadata.slot_mapping)

    def attend_decodes(self, layer, query):
        key_cache, value_cache = self.kv_caches[layer]
        return paged_attention_split(query, key_cache, value_cache,
                                     self.metadata.block_tables,
                                     self.metadata.context_lens)

    def gather_context(self, cache, block_ids, context_len):
        """(num_blocks, kv_heads, block_size, head_dim) blocks of one sequence
        -> (kv_heads, context_len, head_dim)."""
        blocks = cache.index_select(0, block_ids)
        num_kv_heads, head_dim = cache.shape[1], cache.shape[3]
        heads_first = blocks.transpose(0, 1).reshape(num_kv_heads, -1, head_dim)
        return heads_first[:, :context_len]

    def load_context(self, layer, chunk, dtype):
        """The keys and values of the whole context of a prefill chunk."""
        key_cache, value_cache = self.kv_caches[layer]
        keys = self.gather_context(key_cache, chunk.block_ids, chunk.context_len)
        values = self.gather_context(value_cache, chunk.block_ids,
                                     chunk.context_len)
        return keys.to(dtype), values.to(dtype)

    def attend_prefill(self, layer, query, chunk):
        keys, values = self.load_context(layer, chunk, query.dtype)
        return attend_to_context(query, keys, values)

    def __call__(self, layer, query, key, value):
        num_tokens, num_heads, head_dim = query.shape
        output = torch.empty(num_tokens, num_heads * head_dim, dtype=query.dtype,
                             device=query.device)
        self.write(layer, key, value)
        num_decodes = self.metadata.num_decodes
        if num_decodes:
            decoded = self.attend_decodes(layer, query[:num_decodes])
            output[:num_decodes] = decoded.reshape(num_decodes, -1)
        for chunk in self.metadata.prefills:
            rows = slice(chunk.token_start, chunk.token_start + chunk.num_tokens)
            output[rows] = self.attend_prefill(layer, query[rows], chunk)
        return output


# ---------------------------------------------------------------- runner


@dataclass
class StepInputs:
    token_ids: torch.Tensor
    positions: torch.Tensor
    metadata: AttentionMetadata
    logits_rows: torch.Tensor
    order: list               # order[row] = the index of its chunk


def decodes_first(chunks):
    """The decodes form one slice, so one kernel launch serves all of them."""
    decodes = [i for i, c in enumerate(chunks) if c.is_decode]
    prefills = [i for i, c in enumerate(chunks) if not c.is_decode]
    return decodes + prefills


def pad_rows(rows, width, fill=0):
    return [row + [fill] * (width - len(row)) for row in rows]


class ModelRunner:
    def __init__(self, model, num_blocks, block_size=16):
        self.model = model
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = model.device
        self.kv_caches = self.allocate_cache()
        self.attention = self.make_backend()

    # Two hooks, so that stage 24b can store the cache in FP8 with no first
    # allocation in bf16.

    def allocate_cache(self):
        return self.model.allocate_kv_cache(self.num_blocks, self.block_size)

    def make_backend(self):
        return PagedAttention(self.kv_caches, self.block_size)

    # ------------------------------------------------------------ one step

    def _tensor(self, values, dtype=torch.long):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def _decode_tables(self, decodes):
        if not decodes:
            return None, None
        width = max(len(c.block_ids) for c in decodes)
        block_tables = self._tensor(pad_rows([c.block_ids for c in decodes],
                                             width), torch.int32)
        context_lens = self._tensor([c.context_len for c in decodes],
                                    torch.int32)
        return block_tables, context_lens

    def _prefill_chunk(self, chunk, token_start):
        needed = blocks_for(chunk.context_len, self.block_size)
        return PrefillChunk(token_start, len(chunk.token_ids), chunk.context_len,
                            self._tensor(chunk.block_ids[:needed]))

    def prepare(self, chunks):
        order = decodes_first(chunks)
        token_ids, positions, slots, logits_rows, prefills = [], [], [], [], []
        for index in order:
            chunk = chunks[index]
            token_start = len(token_ids)
            for offset, token in enumerate(chunk.token_ids):
                position = chunk.start_pos + offset
                token_ids.append(token)
                positions.append(position)
                slots.append(slot_of(chunk.block_ids, position, self.block_size))
            logits_rows.append(len(token_ids) - 1)
            if not chunk.is_decode:
                prefills.append(self._prefill_chunk(chunk, token_start))
        decodes = [chunks[i] for i in order if chunks[i].is_decode]
        block_tables, context_lens = self._decode_tables(decodes)
        metadata = AttentionMetadata(self._tensor(slots), len(decodes),
                                     block_tables, context_lens, prefills)
        return StepInputs(self._tensor(token_ids), self._tensor(positions),
                          metadata, self._tensor(logits_rows), order)

    def _restore_order(self, logits, order):
        rows_of_chunk = torch.empty(len(order), dtype=torch.long,
                                    device=logits.device)
        rows_of_chunk[self._tensor(order)] = torch.arange(len(order),
                                                          device=logits.device)
        return logits.index_select(0, rows_of_chunk)

    def execute(self, chunks):
        """Logits for the LAST token of each chunk, in the order of chunks."""
        if not chunks:
            return None
        inputs = self.prepare(chunks)
        self.attention.metadata = inputs.metadata
        logits = self.model.forward(inputs.token_ids, inputs.positions,
                                    self.attention, inputs.logits_rows)
        return self._restore_order(logits, inputs.order)

    def copy_block(self, source, destination):
        """Copy one physical block in every layer. Copy-on-write needs it."""
        for key_cache, value_cache in self.kv_caches:
            key_cache[destination].copy_(key_cache[source])
            value_cache[destination].copy_(value_cache[source])
