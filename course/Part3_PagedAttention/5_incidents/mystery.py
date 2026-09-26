"""Three block allocators. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part3_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 9.

Each pool has the methods of lab.Allocator: allocate, append, fork, free and
num_free.
"""

from lab import BLOCK, Allocator


class PoolA(Allocator):
    # free() gives every block back at once
    def free(self, seq):
        for block in self.tables.pop(seq):
            self.refs[block] = 0
            if block not in self.free_list:
                self.free_list.append(block)
        del self.lengths[seq]


class PoolB(Allocator):
    # a table never grows beyond 32 blocks, to bound the kernel's table width
    MAX_BLOCKS = 32

    def append(self, seq):
        position = self.lengths[seq]
        if position // BLOCK == len(self.tables[seq]) and len(self.tables[seq]) < self.MAX_BLOCKS:
            self.tables[seq].append(self._take())
        self.lengths[seq] += 1


class PoolC(Allocator):
    # append() takes the next free block, and removes it from the list later
    def __init__(self, num_blocks):
        super().__init__(num_blocks)
        self.pending = []

    def append(self, seq):
        position = self.lengths[seq]
        if position // BLOCK == len(self.tables[seq]):
            block = self.free_list[-1]
            self.refs[block] = 1
            self.tables[seq].append(block)
            self.pending.append(block)
        self.lengths[seq] += 1

    def allocate(self, seq, num_tokens):
        for block in self.pending:
            if block in self.free_list:
                self.free_list.remove(block)
        self.pending = []
        super().allocate(seq, num_tokens)
