"""Reference solution, stage 23 - the real decode step, captured."""

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


def warm_up(run, times=3):
    """Run once on a side stream, so that the first-call allocations of
    cuBLAS happen before the capture and not inside it."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(times):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()


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
        return next((b for b in self.buckets if b >= num_rows), None)

    # ------------------------------------------------------------ capture

    def _capture_bucket(self, batch_size, memory_pool):
        inputs = StaticInputs.zeros(batch_size, self.max_blocks, self.device)
        self.attention.metadata = inputs.metadata()

        def run():
            return self.model.forward(inputs.token_ids, inputs.positions,
                                      self.attention, inputs.logits_rows)

        warm_up(run)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):
            logits = run()
        return CapturedGraph(graph, inputs, logits)

    def capture(self):
        """Largest bucket first, so that the smaller graphs fit inside its
        memory pool."""
        memory_pool = torch.cuda.graph_pool_handle()
        for batch_size in sorted(self.buckets, reverse=True):
            self.graphs[batch_size] = self._capture_bucket(batch_size,
                                                           memory_pool)
        torch.cuda.synchronize()

    # ------------------------------------------------------------ replay

    def _can_replay(self, chunks):
        return (bool(self.graphs) and bool(chunks)
                and all(c.is_decode for c in chunks)
                and self.bucket_for(len(chunks)) is not None)

    def _fill(self, inputs, chunks):
        """Copy one decode step into the static tensors. A padding row writes
        to slot -1 and reads an empty context. A zero slot writes into block
        0, which belongs to a real sequence."""
        padding = len(inputs.token_ids) - len(chunks)
        token_ids = [c.token_ids[0] for c in chunks] + [0] * padding
        positions = [c.start_pos for c in chunks] + [0] * padding
        slots = ([slot_of(c.block_ids, c.start_pos, self.block_size)
                  for c in chunks] + [PADDING_SLOT] * padding)
        context_lens = ([c.context_len for c in chunks]
                        + [PADDING_CONTEXT] * padding)
        block_tables = pad_rows([c.block_ids for c in chunks] + [[]] * padding,
                                self.max_blocks)
        host = torch.tensor([token_ids, positions, slots, context_lens])
        on_device = host.to(self.device, non_blocking=True)
        inputs.token_ids.copy_(on_device[0])
        inputs.positions.copy_(on_device[1])
        inputs.slot_mapping.copy_(on_device[2])
        inputs.context_lens.copy_(on_device[3])
        inputs.block_tables.copy_(torch.tensor(block_tables, dtype=torch.int32),
                                  non_blocking=True)

    def _replay(self, chunks):
        captured = self.graphs[self.bucket_for(len(chunks))]
        self._fill(captured.inputs, chunks)
        captured.graph.replay()
        self.replays += 1
        return captured.logits[:len(chunks)].clone()   # the next replay overwrites

    def execute(self, chunks):
        if self._can_replay(chunks):
            return self._replay(chunks)
        self.eager_steps += 1
        return super().execute(chunks)
