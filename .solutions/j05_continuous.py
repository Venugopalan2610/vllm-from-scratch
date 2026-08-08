"""Reference solution, stage 05 (jax) - continuous batching over slots."""

from collections import deque

import jax.numpy as jnp
import numpy as np


class Request:
    def __init__(self, rid, prompt_ids, max_tokens):
        self.id = rid
        self.prompt_ids = list(prompt_ids)
        self.max_tokens = max_tokens
        self.output_ids = []
        self.finished = False

    def __repr__(self):
        return f"Request({self.id}, {len(self.output_ids)}/{self.max_tokens})"


class ContinuousEngine:
    def __init__(self, model, max_batch_size=8, max_len=512):
        self.model = model
        self.B = max_batch_size
        self.max_len = max_len

        # Allocated once. Nothing here ever changes shape again.
        self.cache = model.init_cache(max_batch_size, max_len)
        self.slot_req = [None] * max_batch_size
        self.cache_len = np.zeros(max_batch_size, dtype=np.int32)
        self.next_tok = np.zeros(max_batch_size, dtype=np.int32)

        self.waiting = deque()
        self.steps = 0            # FORWARD PASSES, not loop iterations
        self.slot_steps = 0       # occupied-slot-steps, for the occupancy metric
        self.tokens = 0

    # ---- public API -------------------------------------------------
    def add_request(self, rid, prompt, max_tokens):
        ids = [int(t) for t in self.model.encode(prompt)]
        self.waiting.append(Request(rid, ids, max_tokens))

    def has_work(self):
        return bool(self.waiting) or any(r is not None for r in self.slot_req)

    def occupied(self):
        return [i for i, r in enumerate(self.slot_req) if r is not None]

    def step(self):
        self._admit()
        busy = self.occupied()
        if not busy:
            return []

        # harvest the pending token from every occupied slot
        finished = []
        for i in busy:
            req = self.slot_req[i]
            t = int(self.next_tok[i])
            if t in self.model.eos_ids or len(req.output_ids) >= req.max_tokens:
                req.finished = True
                finished.append(req)
            else:
                req.output_ids.append(t)
                self.tokens += 1
                if len(req.output_ids) >= req.max_tokens:
                    req.finished = True
                    finished.append(req)

        # free the slots BEFORE the forward, so an admission next iteration
        # can reuse them immediately
        for i in busy:
            if self.slot_req[i] is not None and self.slot_req[i].finished:
                self.slot_req[i] = None
                self.cache_len[i] = 0

        busy = self.occupied()
        if not busy:
            return finished

        self.slot_steps += len(busy)
        self.steps += 1
        pos = jnp.asarray(self.cache_len, jnp.int32)[:, None]
        logits, self.cache = self.model.forward(
            jnp.asarray(self.next_tok, jnp.int32)[:, None],
            positions=pos, cache=self.cache,
            cache_len=jnp.asarray(self.cache_len, jnp.int32),
        )
        # ONE host sync for the whole batch. np.array, not np.asarray: the
        # latter hands back a read-only view of the device buffer, and the
        # next per-slot write to it raises.
        self.next_tok = np.array(jnp.argmax(logits, axis=-1), dtype=np.int32)
        for i in busy:
            self.cache_len[i] += 1
        return finished

    def run_to_completion(self):
        done = {}
        while self.has_work():
            for r in self.step():
                done[r.id] = r.output_ids
        return done

    # ---- internals --------------------------------------------------
    def _admit(self):
        while self.waiting:
            free = [i for i, r in enumerate(self.slot_req) if r is None]
            if not free:
                return
            req = self.waiting.popleft()
            if len(req.prompt_ids) + req.max_tokens + 1 > self.max_len:
                raise ValueError(
                    f"request {req.id} needs "
                    f"{len(req.prompt_ids) + req.max_tokens + 1} slots but "
                    f"max_len is {self.max_len}"
                )
            self._prefill_into(free[0], req)

    def _prefill_into(self, slot, req):
        ids = jnp.asarray([req.prompt_ids], jnp.int32)
        scratch = self.model.init_cache(1, self.max_len)
        logits, (k1, v1) = self.model.forward(
            ids, cache=scratch, cache_len=jnp.zeros(1, jnp.int32)
        )
        kc, vc = self.cache
        self.cache = (kc.at[:, slot].set(k1[:, 0]),
                      vc.at[:, slot].set(v1[:, 0]))
        self.cache_len[slot] = len(req.prompt_ids)
        self.next_tok[slot] = int(jnp.argmax(logits[0]))
        self.slot_req[slot] = req


def run_all(model, jobs, max_batch_size=8, max_len=512):
    eng = ContinuousEngine(model, max_batch_size, max_len)
    for rid, prompt, mt in jobs:
        eng.add_request(rid, prompt, mt)
    return eng.run_to_completion()
