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
        self.alloc = allocator
        self.block_size = allocator.block_size
        self.max_num_seqs = max_num_seqs
        self.waiting = deque()
        self.running = []
        self.finished = []
        self.preemptions = 0
        self.steps = 0

    # ---- helpers ----
    def _blocks_for(self, n_tokens):
        return math.ceil(n_tokens / self.block_size)

    def _grow(self, seq, n_tokens):
        need = self._blocks_for(n_tokens) - len(seq.blocks)
        if need <= 0:
            return True
        if self.alloc.num_free < need:
            return False
        seq.blocks.extend(self.alloc.allocate(need))
        return True

    def _release(self, seq):
        if seq.blocks:
            self.alloc.free(seq.blocks)
            seq.blocks = []

    # ---- public ----
    def add_request(self, rid, prompt_len, max_tokens):
        self.waiting.append(SeqState(rid, prompt_len, max_tokens))

    def has_work(self):
        return bool(self.waiting or self.running)

    def _admit(self):
        admitted = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self._grow(seq, seq.prompt_len):
                break                      # not enough KV; leave it queued
            self.waiting.popleft()
            seq.prefilled = True
            self.running.append(seq)
            admitted.append(seq)
        return admitted

    def _preempt_one(self):
        """Preempt by RECOMPUTE: drop the newest running sequence's blocks and
        send it back to the front of the queue. Prefill is cheap; PCIe is not."""
        victim = self.running.pop()
        self._release(victim)
        victim.num_generated = 0
        victim.prefilled = False
        victim.preempted_count += 1
        self.waiting.appendleft(victim)
        self.preemptions += 1
        return victim

    def step(self):
        """One engine iteration. Returns a record of what happened."""
        prefilled = self._admit()
        decoded, finished, preempted = [], [], []

        i = 0
        while i < len(self.running):
            seq = self.running[i]
            if seq in prefilled:
                i += 1
                continue
            if not self._grow(seq, seq.num_tokens + 1):
                # out of KV -- free memory by preempting the newest sequence
                if self.running[-1] is seq and len(self.running) == 1:
                    raise OutOfBlocks(
                        f"cannot fit even one sequence (seq {seq.id}, "
                        f"{seq.num_tokens} tokens)"
                    )
                v = self._preempt_one()
                preempted.append(v)
                if v is seq:
                    continue           # this row is gone; re-check index i
                continue               # retry the same seq with freed memory
            seq.num_generated += 1
            decoded.append(seq)
            i += 1

        for seq in list(self.running):
            if seq.done:
                self._release(seq)
                self.running.remove(seq)
                self.finished.append(seq)
                finished.append(seq)

        if prefilled or decoded:
            self.steps += 1
        return {
            "prefilled": prefilled,
            "decoded": decoded,
            "preempted": preempted,
            "finished": finished,
        }

    def run_to_completion(self, max_steps=100000):
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        return self.finished
