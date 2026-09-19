"""Reference solution, stage 11 - chunked prefill and mixed batches."""

import math
from collections import deque


class ChunkSeq:
    def __init__(self, rid, prompt_len, max_tokens):
        self.id = rid
        self.prompt_len = prompt_len
        self.max_tokens = max_tokens
        self.num_computed = 0        # the prompt tokens prefilled so far
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
        return (f"ChunkSeq({self.id}, {self.num_computed}/{self.prompt_len}p, "
                f"{self.num_generated}/{self.max_tokens}d)")


class ChunkedScheduler:
    def __init__(self, allocator, max_num_seqs=8, token_budget=512,
                 prefill_first=False):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.prefill_first = prefill_first
        self.waiting = deque()
        self.running = []
        self.finished = []
        self.steps = 0

    def add_request(self, rid, prompt_len, max_tokens):
        self.waiting.append(ChunkSeq(rid, prompt_len, max_tokens))

    def has_work(self):
        return bool(self.waiting or self.running)

    def _grow(self, seq, num_tokens):
        missing = math.ceil(num_tokens / self.block_size) - len(seq.blocks)
        if missing <= 0:
            return True
        if self.allocator.num_free < missing:
            return False
        seq.blocks.extend(self.allocator.allocate(missing))
        return True

    def _admit(self):
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self._grow(seq, seq.prompt_len):
                break
            self.waiting.popleft()
            self.running.append(seq)

    def _decode_one(self, seq):
        """-> True if seq got one more token."""
        if not self._grow(seq, seq.num_tokens + 1):
            return False
        seq.num_generated += 1
        return True

    def _decodes(self, decoders, budget):
        decoded = []
        for seq in decoders:
            if len(decoded) >= budget:
                break
            if self._decode_one(seq):
                decoded.append(seq)
        return decoded

    def _prefill_chunk(self, seq, num_tokens):
        seq.num_computed += num_tokens
        if not seq.is_prefilling:
            # The last prefill chunk also makes the first token.
            self._decode_one(seq)

    def _prefills(self, prefillers, budget):
        """-> [(rid, chunk length)]."""
        chunks = []
        for seq in prefillers:
            if budget <= 0:
                break
            chunk_len = min(budget, seq.prompt_len - seq.num_computed)
            if not self._grow(seq, seq.num_computed + chunk_len):
                continue
            self._prefill_chunk(seq, chunk_len)
            budget -= chunk_len
            chunks.append((seq.id, chunk_len))
        return chunks

    def _plan(self):
        """-> (prefill chunks, decoded seqs), inside the token budget."""
        decoders = [seq for seq in self.running if not seq.is_prefilling]
        prefillers = [seq for seq in self.running if seq.is_prefilling]
        if self.prefill_first:
            prefill = self._prefills(prefillers, self.token_budget)
            used = sum(length for _, length in prefill)
            return prefill, self._decodes(decoders, self.token_budget - used)
        decoded = self._decodes(decoders, self.token_budget)
        return self._prefills(prefillers, self.token_budget - len(decoded)), \
            decoded

    def _retire_done(self):
        finished = [seq for seq in self.running if seq.done]
        for seq in finished:
            self.allocator.free(seq.blocks)
            seq.blocks = []
            self.running.remove(seq)
        self.finished.extend(finished)
        return finished

    def step(self):
        self._admit()
        prefill, decoded = self._plan()
        tokens_used = sum(length for _, length in prefill) + len(decoded)
        finished = self._retire_done()
        if tokens_used:
            self.steps += 1
        return {"prefill": prefill, "decoded": decoded, "finished": finished,
                "tokens_used": tokens_used}

    def run_to_completion(self, max_steps=100000):
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        return self.finished


class UnchunkedScheduler(ChunkedScheduler):
    """A prefill completes in one step, at every length.

    This is the stage 10 behavior. The tests use it to measure what the
    chunks give.
    """

    def __init__(self, allocator, max_num_seqs=8, token_budget=512):
        super().__init__(allocator, max_num_seqs, token_budget)

    def _plan(self):
        decoders = [seq for seq in self.running if not seq.is_prefilling]
        prefill = []
        for seq in self.running:
            if seq.is_prefilling:
                prompt_left = seq.prompt_len - seq.num_computed
                self._prefill_chunk(seq, prompt_left)   # the WHOLE prompt
                prefill.append((seq.id, prompt_left))
        decoded = [seq for seq in decoders if self._decode_one(seq)]
        return prefill, decoded
