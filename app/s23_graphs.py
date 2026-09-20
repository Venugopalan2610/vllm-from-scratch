"""Stage 23 - the real decode step, captured.

`./vc lore 23` for the insight. `./vc test 23` to check yourself.

Stage 12 captured a stack of Linear layers. This stage captures the real
decode step of the real model, with your paged attention inside it.

At batch 1 the eager step takes about two times the weight-read time. The
GPU is not slow. The step is about a thousand small kernels, and the CPU
issues them one at a time. A graph issues them all with one call. How much
that saves depends on your CPU.

WHAT YOU ARE BUILDING

    StaticInputs                    given: the tensors that one graph reads
    warm_up(run)                    run on a side stream before a capture
    GraphedModelRunner(model, num_blocks, block_size=16, max_model_len=4096,
                       buckets=(1, 2, 4, 8, 16, 32, 64))
        .bucket_for(num_rows)       the smallest bucket, or None
        .capture()                  one graph for each bucket
        .execute(chunks)            replay when every chunk is a decode and
                                    a bucket fits; otherwise run eager
        .replays, .eager_steps      counters that the checks read

CAUTION: your stage 12 runner fills the padding rows with zeros. Here a zero
slot writes K and V into block 0, and block 0 belongs to a real sequence.
Its output then changes, quietly, a few tokens later. A padding row must use
slot -1, which write_kv skips, and context 0, which the attention kernel
treats as empty.

TRAPS

  - Make the block table as wide as max_model_len needs. Its width then never
    changes, and neither does the split count that stage 08c picks from it.
  - Capture the largest bucket first, and share one memory pool across all
    buckets. The smaller graphs then reuse the pool of the largest.
  - Return logits[:num_rows].clone(). The next replay overwrites them.
"""

from dataclasses import dataclass

import torch

from app.s21_paged_runner import (AttentionMetadata, ModelRunner, blocks_for,
                                  pad_rows, slot_of)

PADDING_SLOT = -1          # write_kv skips it
PADDING_CONTEXT = 0        # the attention kernel treats it as empty


@dataclass
class StaticInputs:
    """The tensors that one captured graph reads. A replay copies into them."""
    token_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    logits_rows: torch.Tensor

    @classmethod
    def zeros(cls, batch_size, max_blocks, device):
        def longs(value=0):
            return torch.full((batch_size,), value, dtype=torch.long,
                              device=device)
        return cls(token_ids=longs(), positions=longs(),
                   slot_mapping=longs(PADDING_SLOT),
                   block_tables=torch.zeros(batch_size, max_blocks,
                                            dtype=torch.int32, device=device),
                   context_lens=torch.zeros(batch_size, dtype=torch.int32,
                                            device=device),
                   logits_rows=torch.arange(batch_size, device=device))

    def metadata(self):
        return AttentionMetadata(self.slot_mapping, len(self.token_ids),
                                 self.block_tables, self.context_lens, [])


@dataclass
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    inputs: StaticInputs
    logits: torch.Tensor


@dataclass
class StagingBuffers:
    """Pre-allocated pinned host memory buffers for asynchronous zero-copy transfers.
    Allocating host tensors on every step thrashes the CPU memory allocator and
    forces synchronous pageable transfers."""
    token_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor

    @classmethod
    def create(cls, batch_size, max_blocks):
        return cls(
            token_ids=torch.empty(batch_size, dtype=torch.long, pin_memory=True),
            positions=torch.empty(batch_size, dtype=torch.long, pin_memory=True),
            slot_mapping=torch.empty(batch_size, dtype=torch.long, pin_memory=True),
            context_lens=torch.empty(batch_size, dtype=torch.int32, pin_memory=True),
            block_tables=torch.empty((batch_size, max_blocks), dtype=torch.int32, pin_memory=True),
        )


def warm_up(run, times=3):
    """Run once on a side stream, so that the first-call allocations of
    cuBLAS happen before the capture and not inside it."""
    raise NotImplementedError("stage 23: implement warm_up")


class GraphedModelRunner(ModelRunner):
    def __init__(self, model, num_blocks, block_size=16, max_model_len=4096,
                 buckets=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__(model, num_blocks, block_size)
        self.max_blocks = blocks_for(max_model_len, block_size)
        self.buckets = sorted(buckets)
        self.graphs = {}
        self.replays = 0
        self.eager_steps = 0

    def bucket_for(self, num_rows):
        """The smallest bucket that holds num_rows, or None."""
        raise NotImplementedError("stage 23: implement GraphedModelRunner.bucket_for")

    # ------------------------------------------------------------ capture

    def _capture_bucket(self, batch_size, memory_pool):
        raise NotImplementedError("stage 23: implement GraphedModelRunner._capture_bucket")

    def capture(self):
        """Largest bucket first, so that the smaller graphs fit inside its
        memory pool."""
        raise NotImplementedError("stage 23: implement GraphedModelRunner.capture")

    # ------------------------------------------------------------ replay

    def _can_replay(self, chunks):
        raise NotImplementedError("stage 23: implement GraphedModelRunner._can_replay")

    def _fill(self, inputs, chunks):
        """Copy one decode step into the static tensors. A padding row writes
        to slot -1 and reads an empty context. A zero slot writes into block
        0, which belongs to a real sequence."""
        raise NotImplementedError("stage 23: implement GraphedModelRunner._fill")

    def _replay(self, chunks):
        raise NotImplementedError("stage 23: implement GraphedModelRunner._replay")

    def execute(self, chunks):
        raise NotImplementedError("stage 23: implement GraphedModelRunner.execute")
