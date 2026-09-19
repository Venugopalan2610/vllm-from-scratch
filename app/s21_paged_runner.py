"""Stage 21 - the model runs on your paged cache.

`./vc lore 21` for the insight. `./vc test 21` to check yourself.

Stages 06 to 08c built a block allocator, a KV write kernel and a paged
attention kernel. Each one passed its checks on random tensors. Not one of
them ran inside a model. This stage puts them there.

Read tvllm/model.py first. The model takes a FLAT batch: every token of every
sequence in one (num_tokens,) tensor. For each layer it calls your backend:

    output = attention(layer, query, key, value)

WHAT YOU ARE BUILDING

    SeqChunk(token_ids, start_pos, block_ids)       given
        the work for one sequence in one step. One token is a DECODE. More
        tokens are a PREFILL. .context_len = start_pos + len(token_ids)
    PrefillChunk, AttentionMetadata, StepInputs     given: what one step needs
    slot_of, blocks_for, decodes_first, pad_rows    four small helpers
    PagedAttention(kv_caches, block_size)           the backend, in steps:
        write(layer, key, value)          your stage 08 write kernel
        attend_decodes(layer, query)      your stage 08c kernel, one launch
        gather_context(cache, block_ids, context_len)
        load_context(layer, chunk, dtype) the keys and values of a prefill
        attend_prefill(layer, query, chunk)
        __call__(layer, query, key, value)
    ModelRunner(model, num_blocks, block_size=16)
        .prepare(chunks) -> StepInputs
        .execute(chunks) -> logits, one row for each chunk, IN CHUNK ORDER
        .copy_block(source, destination)

Stage 24b stores the cache in FP8. It overrides write, attend_decodes and
load_context, and nothing else. So keep each step in its own method.

THE LAYOUT OF ONE STEP

Put the decodes first and the prefills after them. The decodes then form one
slice query[:num_decodes], and your stage 08c kernel takes the whole slice in
one launch. The logits must still come back in the order of the chunks, so
remember the order, and undo it at the end.

    slot of position p = block_ids[p // block_size] * block_size
                         + p % block_size

A PREFILL CHUNK SEES THE WHOLE CONTEXT

A prefill chunk attends to its own tokens AND to every earlier token of its
sequence, which earlier chunks wrote. So gather its blocks, trim them to its
context, and call tvllm.attend_to_context. Read how tvllm.causal_mask builds
the mask: the queries are the LAST positions of the context. SDPA with
is_causal=True aligns the mask to the top left instead. The first chunk is
then correct, and every later chunk is wrong.

TRAPS

  - The rows of a block table have different lengths. Pad each row with any
    valid block id. The kernel reads only up to context_len.
  - The stage 08 and 08c kernels take an int64 slot_mapping, and int32
    block_tables and context_lens.
  - Build each tensor from one Python list, with one torch.tensor() call.
    A call for each row costs more host time than the step costs on the GPU.
"""

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
    raise NotImplementedError("stage 21: implement slot_of")


def blocks_for(num_tokens, block_size):
    raise NotImplementedError("stage 21: implement blocks_for")


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
        raise NotImplementedError("stage 21: implement PagedAttention.write")

    def attend_decodes(self, layer, query):
        raise NotImplementedError("stage 21: implement PagedAttention.attend_decodes")

    def gather_context(self, cache, block_ids, context_len):
        """(num_blocks, kv_heads, block_size, head_dim) blocks of one sequence
        -> (kv_heads, context_len, head_dim)."""
        raise NotImplementedError("stage 21: implement PagedAttention.gather_context")

    def load_context(self, layer, chunk, dtype):
        """The keys and values of the whole context of a prefill chunk."""
        raise NotImplementedError("stage 21: implement PagedAttention.load_context")

    def attend_prefill(self, layer, query, chunk):
        raise NotImplementedError("stage 21: implement PagedAttention.attend_prefill")

    def __call__(self, layer, query, key, value):
        raise NotImplementedError("stage 21: implement PagedAttention.__call__")


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
    raise NotImplementedError("stage 21: implement decodes_first")


def pad_rows(rows, width, fill=0):
    raise NotImplementedError("stage 21: implement pad_rows")


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
        raise NotImplementedError("stage 21: implement ModelRunner._decode_tables")

    def _prefill_chunk(self, chunk, token_start):
        raise NotImplementedError("stage 21: implement ModelRunner._prefill_chunk")

    def prepare(self, chunks):
        raise NotImplementedError("stage 21: implement ModelRunner.prepare")

    def _restore_order(self, logits, order):
        raise NotImplementedError("stage 21: implement ModelRunner._restore_order")

    def execute(self, chunks):
        """Logits for the LAST token of each chunk, in the order of chunks."""
        raise NotImplementedError("stage 21: implement ModelRunner.execute")

    def copy_block(self, source, destination):
        """Copy one physical block in every layer. Copy-on-write needs it."""
        raise NotImplementedError("stage 21: implement ModelRunner.copy_block")
