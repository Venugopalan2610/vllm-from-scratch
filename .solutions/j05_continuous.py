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

    def record(self, token, stop_ids):
        """Keep the token, or finish at a stop token or at max_tokens."""
        if token in stop_ids or len(self.output_ids) >= self.max_tokens:
            self.finished = True
            return
        self.output_ids.append(token)
        if len(self.output_ids) >= self.max_tokens:
            self.finished = True


class ContinuousEngine:
    def __init__(self, model, max_batch_size=8, max_len=512):
        self.model = model
        self.max_len = max_len

        # Allocated one time. No shape here changes again.
        self.cache = model.init_cache(max_batch_size, max_len)
        self.slot_requests = [None] * max_batch_size
        self.cache_len = np.zeros(max_batch_size, dtype=np.int32)
        self.pending_tokens = np.zeros(max_batch_size, dtype=np.int32)

        self.waiting = deque()
        self.steps = 0            # forward passes, not loop iterations
        self.slot_steps = 0       # the sum of busy slots over the steps
        self.tokens = 0

    def add_request(self, rid, prompt, max_tokens):
        prompt_ids = [int(token) for token in self.model.encode(prompt)]
        self.waiting.append(Request(rid, prompt_ids, max_tokens))

    def has_work(self):
        return bool(self.waiting or self.occupied())

    def occupied(self):
        return [slot for slot, request in enumerate(self.slot_requests)
                if request is not None]

    def free_slots(self):
        return [slot for slot, request in enumerate(self.slot_requests)
                if request is None]

    def step(self):
        self._admit()
        finished = self._record_pending()
        # Free the slots BEFORE the forward pass, so that the next admission
        # can use them.
        self._free_finished()
        if self.occupied():
            self._decode()
        return finished

    def run_to_completion(self):
        outputs = {}
        while self.has_work():
            for request in self.step():
                outputs[request.id] = request.output_ids
        return outputs

    def _record_pending(self):
        finished = []
        for slot in self.occupied():
            request = self.slot_requests[slot]
            num_before = len(request.output_ids)
            request.record(int(self.pending_tokens[slot]), self.model.eos_ids)
            self.tokens += len(request.output_ids) - num_before
            if request.finished:
                finished.append(request)
        return finished

    def _free_finished(self):
        for slot in self.occupied():
            if self.slot_requests[slot].finished:
                self.slot_requests[slot] = None
                self.cache_len[slot] = 0

    def _decode(self):
        busy_slots = self.occupied()
        self.slot_steps += len(busy_slots)
        self.steps += 1
        cache_len = jnp.asarray(self.cache_len, jnp.int32)
        logits, self.cache = self.model.forward(
            jnp.asarray(self.pending_tokens, jnp.int32)[:, None],
            positions=cache_len[:, None], cache=self.cache,
            cache_len=cache_len)
        # One host sync for the whole batch. Use np.array, not np.asarray:
        # np.asarray gives a read-only view, and the next write to a slot
        # then fails.
        self.pending_tokens = np.array(jnp.argmax(logits, axis=-1),
                                       dtype=np.int32)
        for slot in busy_slots:
            self.cache_len[slot] += 1

    def _check_fits(self, request):
        slots_needed = len(request.prompt_ids) + request.max_tokens + 1
        if slots_needed > self.max_len:
            raise ValueError(f"request {request.id} needs {slots_needed} "
                             f"slots but max_len is {self.max_len}")

    def _admit(self):
        while self.waiting and self.free_slots():
            request = self.waiting.popleft()
            self._check_fits(request)
            self._prefill_into(self.free_slots()[0], request)

    def _prefill_into(self, slot, request):
        scratch_cache = self.model.init_cache(1, self.max_len)
        logits, (new_keys, new_values) = self.model.forward(
            jnp.asarray([request.prompt_ids], jnp.int32), cache=scratch_cache,
            cache_len=jnp.zeros(1, jnp.int32))
        key_cache, value_cache = self.cache
        self.cache = (key_cache.at[:, slot].set(new_keys[:, 0]),
                      value_cache.at[:, slot].set(new_values[:, 0]))
        self.cache_len[slot] = len(request.prompt_ids)
        self.pending_tokens[slot] = int(jnp.argmax(logits[0]))
        self.slot_requests[slot] = request


def run_all(model, jobs, max_batch_size=8, max_len=512):
    engine = ContinuousEngine(model, max_batch_size, max_len)
    for rid, prompt, max_tokens in jobs:
        engine.add_request(rid, prompt, max_tokens)
    return engine.run_to_completion()
