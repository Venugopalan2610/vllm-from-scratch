"""Reference solution, stage 05 - continuous batching."""

from collections import deque

import torch
from transformers import DynamicCache


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


def left_pad(tensor, length, dim):
    """Put zeros in front of `tensor` along `dim`, up to `length`."""
    missing = length - tensor.shape[dim]
    if missing <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = missing
    zeros = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([zeros, tensor], dim=dim)


def merge_caches(running_cache, new_cache, length):
    """Left-pad both caches to one length, and stack the batch rows."""
    layers = []
    for running_layer, new_layer in zip(running_cache.layers, new_cache.layers):
        keys = torch.cat([left_pad(running_layer.keys, length, dim=2),
                          left_pad(new_layer.keys, length, dim=2)], dim=0)
        values = torch.cat([left_pad(running_layer.values, length, dim=2),
                            left_pad(new_layer.values, length, dim=2)], dim=0)
        layers.append((keys, values))
    return DynamicCache(layers)


class ContinuousEngine:
    def __init__(self, model, tokenizer, max_batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.device = next(model.parameters()).device
        stop = model.generation_config.eos_token_id
        self.stop_ids = set(stop) if isinstance(stop, list) else {stop}

        self.waiting = deque()
        self.running = []
        self.cache = None
        self.attention_mask = None     # (rows, cache length)
        self.next_positions = None     # (rows,)
        self.pending_tokens = None     # (rows,) not yet recorded
        self.steps = 0                 # forward passes, not loop iterations

    def add_request(self, rid, prompt, max_tokens):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        self.waiting.append(Request(rid, prompt_ids.tolist(), max_tokens))

    def has_work(self):
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self):
        """Admit, record, evict, then run one forward pass.

        At the start of a step, each running row has one pending token that
        is not yet recorded. A prefill and a forward pass each make one.
        """
        self._admit()
        if not self.running:
            return []
        finished = self._record_pending()
        # Evict BEFORE the forward pass. A finished row must not cost
        # compute. That is the difference from stage 04.
        if finished:
            self._evict()
        if self.running:
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
        for row, request in enumerate(self.running):
            request.record(int(self.pending_tokens[row]), self.stop_ids)
            if request.finished:
                finished.append(request)
        return finished

    def _decode(self):
        # The model writes one new KV entry. The mask must cover it first.
        one_column = torch.ones(len(self.running), 1,
                                dtype=self.attention_mask.dtype,
                                device=self.device)
        self.attention_mask = torch.cat([self.attention_mask, one_column], dim=1)
        self.steps += 1
        result = self.model(self.pending_tokens.unsqueeze(1),
                            attention_mask=self.attention_mask,
                            position_ids=self.next_positions.unsqueeze(1),
                            past_key_values=self.cache, use_cache=True)
        self.cache = result.past_key_values
        self.pending_tokens = result.logits[:, -1].argmax(-1)
        self.next_positions = self.next_positions + 1

    def _admit(self):
        while self.waiting and len(self.running) < self.max_batch_size:
            self._prefill_and_join(self.waiting.popleft())

    def _prefill(self, request):
        prompt_len = len(request.prompt_ids)
        token_ids = torch.tensor([request.prompt_ids], device=self.device)
        attention_mask = torch.ones(1, prompt_len, dtype=torch.long,
                                    device=self.device)
        positions = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        result = self.model(token_ids, attention_mask=attention_mask,
                            position_ids=positions, use_cache=True)
        first_token = result.logits[0, -1].argmax().view(1)
        return result.past_key_values, attention_mask, first_token

    @torch.inference_mode()
    def _prefill_and_join(self, request):
        cache, attention_mask, first_token = self._prefill(request)
        next_position = torch.tensor([len(request.prompt_ids)],
                                     device=self.device)
        if not self.running:
            self.cache = cache
            self.attention_mask = attention_mask
            self.next_positions = next_position
            self.pending_tokens = first_token
        else:
            length = max(self.attention_mask.shape[1], attention_mask.shape[1])
            self.cache = merge_caches(self.cache, cache, length)
            self.attention_mask = torch.cat(
                [left_pad(self.attention_mask, length, dim=1),
                 left_pad(attention_mask, length, dim=1)], dim=0)
            self.next_positions = torch.cat([self.next_positions,
                                             next_position])
            self.pending_tokens = torch.cat([self.pending_tokens, first_token])
        self.running.append(request)

    def _evict(self):
        kept_rows = [row for row, request in enumerate(self.running)
                     if not request.finished]
        self.running = [self.running[row] for row in kept_rows]
        if not self.running:
            self.cache = self.attention_mask = None
            self.next_positions = self.pending_tokens = None
            return
        rows = torch.tensor(kept_rows, device=self.device)
        self.cache.batch_select_indices(rows)
        self.attention_mask = self.attention_mask[rows]
        self.next_positions = self.next_positions[rows]
        self.pending_tokens = self.pending_tokens[rows]


def run_all(model, tokenizer, jobs, max_batch_size=8):
    """jobs: a list of (rid, prompt, max_tokens). -> {rid: [token ids]}."""
    engine = ContinuousEngine(model, tokenizer, max_batch_size)
    for rid, prompt, max_tokens in jobs:
        engine.add_request(rid, prompt, max_tokens)
    return engine.run_to_completion()
