"""Reference solution, stage 10 - admission and preemption."""

import math
from collections import deque

from app.s06_blocks import OutOfBlocks


class SeqState:
    def __init__(self, rid, prompt_len, max_tokens):
        self.id = rid
        self.prompt_len = prompt_len
        self.max_tokens = max_tokens
        self.num_generated = 0
        self.prefilled = False
        self.blocks = []
        self.preempted_count = 0

    @property
    def num_tokens(self):
        return self.prompt_len + self.num_generated

    @property
    def done(self):
        return self.num_generated >= self.max_tokens

    def __repr__(self):
        return f"Seq({self.id}, {self.num_generated}/{self.max_tokens})"


class Scheduler:
    def __init__(self, allocator, max_num_seqs=8):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.max_num_seqs = max_num_seqs
        self.waiting = deque()
        self.running = []
        self.finished = []
        self.preemptions = 0
        self.steps = 0

    def add_request(self, rid, prompt_len, max_tokens):
        self.waiting.append(SeqState(rid, prompt_len, max_tokens))

    def has_work(self):
        return bool(self.waiting or self.running)

    def _grow(self, seq, num_tokens):
        """Give seq the blocks for num_tokens. -> False if they are not free."""
        missing = math.ceil(num_tokens / self.block_size) - len(seq.blocks)
        if missing <= 0:
            return True
        if self.allocator.num_free < missing:
            return False
        seq.blocks.extend(self.allocator.allocate(missing))
        return True

    def _release(self, seq):
        if seq.blocks:
            self.allocator.free(seq.blocks)
            seq.blocks = []

    def _admit(self):
        admitted = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self._grow(seq, seq.prompt_len):
                break                      # no KV for it: it stays queued
            self.waiting.popleft()
            seq.prefilled = True
            self.running.append(seq)
            admitted.append(seq)
        return admitted

    def _preempt_newest(self):
        """Preempt by RECOMPUTE: free the blocks of the newest sequence, and
        put it at the front of the queue. A prefill is cheap. PCIe is not."""
        victim = self.running.pop()
        self._release(victim)
        victim.num_generated = 0
        victim.prefilled = False
        victim.preempted_count += 1
        self.waiting.appendleft(victim)
        self.preemptions += 1
        return victim

    def _decode_running(self, just_prefilled):
        """Give each running sequence one token. Preempt the newest sequence
        until the blocks fit."""
        decoded, preempted = [], []
        index = 0
        while index < len(self.running):
            seq = self.running[index]
            if seq in just_prefilled:
                index += 1
            elif self._grow(seq, seq.num_tokens + 1):
                seq.num_generated += 1
                decoded.append(seq)
                index += 1
            elif len(self.running) == 1:
                raise OutOfBlocks(f"cannot fit even one sequence (seq "
                                  f"{seq.id}, {seq.num_tokens} tokens)")
            else:
                # Try the same index again, with the freed blocks.
                preempted.append(self._preempt_newest())
        return decoded, preempted

    def _retire_done(self):
        finished = [seq for seq in self.running if seq.done]
        for seq in finished:
            self._release(seq)
            self.running.remove(seq)
        self.finished.extend(finished)
        return finished

    def step(self):
        """One engine iteration. -> a record of what happened."""
        prefilled = self._admit()
        decoded, preempted = self._decode_running(prefilled)
        finished = self._retire_done()
        if prefilled or decoded:
            self.steps += 1
        return {"prefilled": prefilled, "decoded": decoded,
                "preempted": preempted, "finished": finished}

    def run_to_completion(self, max_steps=100000):
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        return self.finished
