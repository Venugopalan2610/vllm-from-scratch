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


def _lpad(t, to):
    """Left-pad a (B, H, L, D) cache tensor to length `to`."""
    if t.shape[2] >= to:
        return t
    z = torch.zeros(t.shape[0], t.shape[1], to - t.shape[2], t.shape[3],
                    dtype=t.dtype, device=t.device)
    return torch.cat([z, t], dim=2)


class ContinuousEngine:
    def __init__(self, model, tokenizer, max_batch_size=8):
        self.model = model
        self.tok = tokenizer
        self.max_batch_size = max_batch_size
        self.dev = next(model.parameters()).device
        e = model.generation_config.eos_token_id
        self.eos = set(e) if isinstance(e, list) else {e}

        self.waiting = deque()
        self.running = []
        self.cache = None
        self.mask = None      # (B, L)
        self.pos = None       # (B,) next absolute position
        self.next_tok = None  # (B,)
        self.steps = 0

    # ---- public API -------------------------------------------------
    def add_request(self, rid, prompt, max_tokens):
        ids = self.tok(prompt, return_tensors="pt").input_ids[0].tolist()
        self.waiting.append(Request(rid, ids, max_tokens))

    def has_work(self):
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self):
        """One iteration: admit -> harvest -> evict -> forward.

        Invariant: at the top of step(), every running row's next_tok holds a
        token that has NOT yet been recorded. Both prefill and the forward pass
        produce exactly one such pending token per row.
        """
        self._admit()
        if not self.running:
            return []

        # harvest the pending token from each row
        finished = []
        for i, req in enumerate(self.running):
            t = int(self.next_tok[i])
            if t in self.eos or len(req.output_ids) >= req.max_tokens:
                req.finished = True
                finished.append(req)
            else:
                req.output_ids.append(t)
                if len(req.output_ids) >= req.max_tokens:
                    req.finished = True
                    finished.append(req)

        # evict BEFORE the forward -- finished rows must not cost us compute.
        # That is the entire difference from stage 04.
        if finished:
            self._evict()
        if not self.running:
            return finished

        # extend the mask BEFORE the forward: the model is about to write one
        # new KV entry, so the mask must already cover cache_len + 1
        self.mask = torch.cat(
            [self.mask, torch.ones(len(self.running), 1,
                                   dtype=self.mask.dtype, device=self.dev)],
            dim=1,
        )
        self.steps += 1   # counts FORWARD PASSES, not loop iterations
        out = self.model(
            self.next_tok.unsqueeze(1),
            attention_mask=self.mask,
            position_ids=self.pos.unsqueeze(1),
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = out.past_key_values
        self.next_tok = out.logits[:, -1].argmax(-1)
        self.pos = self.pos + 1
        return finished

    def run_to_completion(self):
        done = {}
        while self.has_work():
            for r in self.step():
                done[r.id] = r.output_ids
        return done

    # ---- internals --------------------------------------------------
    def _admit(self):
        while self.waiting and len(self.running) < self.max_batch_size:
            self._prefill_and_splice(self.waiting.popleft())

    @torch.inference_mode()
    def _prefill_and_splice(self, req):
        ids = torch.tensor([req.prompt_ids], device=self.dev)
        L = ids.shape[1]
        m = torch.ones(1, L, dtype=torch.long, device=self.dev)
        p = torch.arange(L, device=self.dev).unsqueeze(0)
        out = self.model(ids, attention_mask=m, position_ids=p, use_cache=True)
        new_cache = out.past_key_values
        first = out.logits[0, -1].argmax().view(1)

        if not self.running:
            self.cache = new_cache
            self.mask = m
            self.pos = torch.tensor([L], device=self.dev)
            self.next_tok = first
        else:
            cur_L = self.mask.shape[1]
            tgt = max(cur_L, L)
            layers = []
            for i in range(len(self.cache.layers)):
                ok = _lpad(self.cache.layers[i].keys, tgt)
                ov = _lpad(self.cache.layers[i].values, tgt)
                nk = _lpad(new_cache.layers[i].keys, tgt)
                nv = _lpad(new_cache.layers[i].values, tgt)
                layers.append((torch.cat([ok, nk], 0), torch.cat([ov, nv], 0)))
            self.cache = DynamicCache(layers)

            def pad_mask(mm, to):
                if mm.shape[1] >= to:
                    return mm
                z = torch.zeros(mm.shape[0], to - mm.shape[1],
                                dtype=mm.dtype, device=self.dev)
                return torch.cat([z, mm], 1)

            self.mask = torch.cat([pad_mask(self.mask, tgt), pad_mask(m, tgt)], 0)
            self.pos = torch.cat([self.pos, torch.tensor([L], device=self.dev)])
            self.next_tok = torch.cat([self.next_tok, first])

        self.running.append(req)

    def _evict(self):
        keep = [i for i, r in enumerate(self.running) if not r.finished]
        self.running = [self.running[i] for i in keep]
        if not self.running:
            self.cache = self.mask = self.pos = self.next_tok = None
            return
        idx = torch.tensor(keep, device=self.dev)
        self.cache.batch_select_indices(idx)
        self.mask = self.mask[idx]
        self.pos = self.pos[idx]
        self.next_tok = self.next_tok[idx]


def run_all(model, tokenizer, jobs, max_batch_size=8):
    """jobs: list of (rid, prompt, max_tokens). Returns {rid: [token ids]}."""
    eng = ContinuousEngine(model, tokenizer, max_batch_size)
    for rid, prompt, mt in jobs:
        eng.add_request(rid, prompt, mt)
    return eng.run_to_completion()
