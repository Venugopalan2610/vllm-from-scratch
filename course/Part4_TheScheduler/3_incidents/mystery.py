"""Three schedulers. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part4_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 8.

Each scheduler takes the arguments of lab.Scheduler.
"""

import math
from collections import deque

from lab import BLOCK, Scheduler


class SchedulerA(Scheduler):
    # admission: the shortest prompt first, which lowers the mean wait
    def schedule(self):
        self.waiting = deque(sorted(self.waiting, key=lambda r: (r.computed == 0, r.pending())))
        return super().schedule()


class SchedulerB(Scheduler):
    # the budget is for the prefill: a decode is one token, so it is free
    def schedule(self):
        plan, used = [], 0
        for request in sorted(self.running, key=lambda r: r.pending()):
            if request not in self.running:
                continue
            tokens = request.pending() if request.pending() == 1 else min(request.pending(), self.budget - used)
            if tokens <= 0:
                continue
            grown = self._grow(request, tokens)
            while not grown and self._victim(request) is not None:
                victim = self._victim(request)
                self.preempt(victim)
                used -= sum(t for r, t in plan if r is victim and t > 1)
                plan = [(r, t) for r, t in plan if r is not victim]
                grown = self._grow(request, tokens)
            if not grown:
                self.preempt(request)
                continue
            plan.append((request, tokens))
            if tokens > 1:
                used += tokens
        while self.waiting and used < self.budget:
            request = self.waiting[0]
            chunk = min(request.pending(), self.budget - used)
            if not self._can_admit(request, chunk) or not self._grow(request, chunk):
                break
            self.waiting.popleft()
            self.running.append(request)
            plan.append((request, chunk))
            used += chunk
        return plan


class SchedulerC(Scheduler):
    # preemption gives the blocks back
    def preempt(self, victim):
        self._release(victim)
        victim.preemptions += 1
        self.preemptions += 1
        self.running.remove(victim)
        self.waiting.appendleft(victim)
