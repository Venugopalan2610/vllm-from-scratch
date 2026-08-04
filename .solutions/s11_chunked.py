"""Reference solution, stage 11 - chunked prefill and mixed batches."""

import math
from collections import deque

from app.s06_blocks import OutOfBlocks


class ChunkSeq:
    def __init__(self, rid, prompt_len, max_tokens):
        self.id = rid
        self.prompt_len = prompt_len
        self.max_tokens = max_tokens
        self.num_computed = 0        # prompt tokens prefilled so far
        self.num_generated = 0
        self.blocks = []

    @property
    def is_prefilling(self):
        return self.num_computed < self.prompt_len

    @property
    def num_tokens(self):
        return self.num_computed + self.num_generated

    @property
    def done(self):
        return self.num_generated >= self.max_tokens

    def __repr__(self):
        return f"ChunkSeq({self.id}, {self.num_computed}/{self.prompt_len}p, " \
               f"{self.num_generated}/{self.max_tokens}d)"


class ChunkedScheduler:
    def __init__(self, allocator, max_num_seqs=8, token_budget=512,
                 prefill_first=False):
        self.alloc = allocator
        self.block_size = allocator.block_size
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.prefill_first = prefill_first
        self.waiting = deque()
        self.running = []
        self.finished = []
        self.steps = 0

    def _grow(self, seq, n_tokens):
        need = math.ceil(n_tokens / self.block_size) - len(seq.blocks)
        if need <= 0:
            return True
        if self.alloc.num_free < need:
            return False
        seq.blocks.extend(self.alloc.allocate(need))
        return True

    def add_request(self, rid, prompt_len, max_tokens):
        self.waiting.append(ChunkSeq(rid, prompt_len, max_tokens))

    def has_work(self):
        return bool(self.waiting or self.running)

    def step(self):
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self._grow(seq, seq.prompt_len):
                break
            self.waiting.popleft()
            self.running.append(seq)

        budget = self.token_budget
        prefill, decoded, finished = [], [], []

        decoders = [s for s in self.running if not s.is_prefilling]
        prefillers = [s for s in self.running if s.is_prefilling]

        def do_decodes():
            nonlocal budget
            for seq in decoders:
                if budget < 1:
                    break
                if not self._grow(seq, seq.num_tokens + 1):
                    continue
                seq.num_generated += 1
                budget -= 1
                decoded.append(seq)

        def do_prefills():
            nonlocal budget
            for seq in prefillers:
                if budget <= 0:
                    break
                want = min(budget, seq.prompt_len - seq.num_computed)
                if not self._grow(seq, seq.num_computed + want):
                    continue
                seq.num_computed += want
                budget -= want
                prefill.append((seq.id, want))
                if seq.num_computed >= seq.prompt_len:
                    # the last prefill chunk also emits the first token
                    if self._grow(seq, seq.num_tokens + 1):
                        seq.num_generated += 1

        if self.prefill_first:
            do_prefills()
            do_decodes()
        else:
            do_decodes()
            do_prefills()

        used = self.token_budget - budget
        for seq in list(self.running):
            if seq.done:
                self.alloc.free(seq.blocks)
                seq.blocks = []
                self.running.remove(seq)
                self.finished.append(seq)
                finished.append(seq)

        if used:
            self.steps += 1
        return {
            "prefill": prefill,
            "decoded": decoded,
            "finished": finished,
            "tokens_used": used,
        }

    def run_to_completion(self, max_steps=100000):
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        return self.finished


class UnchunkedScheduler(ChunkedScheduler):
    """A prefill must complete in a single step, however long it is.

    This is the stage-10 behaviour, kept around so the tests can measure what
    chunking actually bought.
    """

    def __init__(self, allocator, max_num_seqs=8, token_budget=512):
        super().__init__(allocator, max_num_seqs, token_budget)

    def step(self):
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self._grow(seq, seq.prompt_len):
                break
            self.waiting.popleft()
            self.running.append(seq)

        prefill, decoded, finished = [], [], []
        used = 0
        decoders_before = [s for s in self.running if not s.is_prefilling]

        for seq in [s for s in self.running if s.is_prefilling]:
            n = seq.prompt_len - seq.num_computed
            seq.num_computed = seq.prompt_len
            used += n                       # the WHOLE prompt, in one step
            prefill.append((seq.id, n))
            if self._grow(seq, seq.num_tokens + 1):
                seq.num_generated += 1

        for seq in decoders_before:
            if not self._grow(seq, seq.num_tokens + 1):
                continue
            seq.num_generated += 1
            used += 1
            decoded.append(seq)

        for seq in list(self.running):
            if seq.done:
                self.alloc.free(seq.blocks)
                seq.blocks = []
                self.running.remove(seq)
                self.finished.append(seq)
                finished.append(seq)

        if used:
            self.steps += 1
        return {
            "prefill": prefill,
            "decoded": decoded,
            "finished": finished,
            "tokens_used": used,
        }
