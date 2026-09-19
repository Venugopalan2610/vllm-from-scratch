"""Reference solution, stage 22 - one engine from the parts."""

import math
import time
from collections import deque
from dataclasses import dataclass

import torch

from app.s06_blocks import OutOfBlocks
from app.s09_prefix import PrefixCache, RefCountedAllocator, block_hashes
from app.s13_sampler import SamplingParams, sample
from app.s14_detokenizer import IncrementalDetokenizer
from app.s21_paged_runner import ModelRunner, SeqChunk

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"


class Sequence:
    """The state of one request."""

    def __init__(self, rid, prompt_ids, max_tokens, params, detokenizer,
                 ignore_eos, arrival):
        self.rid = rid
        self.prompt_ids = list(prompt_ids)
        self.output_ids = []
        self.max_tokens = max_tokens
        self.params = params
        self.detokenizer = detokenizer
        self.ignore_eos = ignore_eos
        self.arrival = arrival
        self.blocks = []
        self.num_computed = 0        # tokens whose K and V are in the cache
        self.num_cached_blocks = 0   # full blocks already in the prefix cache
        self.draft = []              # stage 25: draft tokens for this step
        self.status = WAITING
        self.finish_reason = None

    @property
    def all_ids(self):
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self):
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_pending(self):
        return self.num_tokens - self.num_computed

    @property
    def tokens_left(self):
        return self.max_tokens - len(self.output_ids)


# ---------------------------------------------------------------- blocks


class KVBlockManager:
    """The blocks of every sequence, and the prefix cache on top of them."""

    def __init__(self, num_blocks, block_size, prefix_cache_blocks):
        self.block_size = block_size
        self.allocator = RefCountedAllocator(num_blocks, block_size)
        self.prefix_cache = (PrefixCache(self.allocator,
                                         capacity=prefix_cache_blocks)
                             if prefix_cache_blocks else None)
        self.cached_tokens = 0

    @property
    def num_blocks(self):
        return self.allocator.num_blocks

    @property
    def num_used(self):
        return self.allocator.num_blocks - self.allocator.num_free

    def blocks_needed(self, num_tokens):
        return math.ceil(num_tokens / self.block_size)

    def _allocate(self, count):
        """Drop the prefix cache before you drop work."""
        if count > self.allocator.num_free and self.prefix_cache is not None:
            self.prefix_cache.evict_all()
        return self.allocator.allocate(count)

    def grow(self, seq, num_tokens):
        """Give seq the blocks for num_tokens. False if the pool is empty."""
        missing = self.blocks_needed(num_tokens) - len(seq.blocks)
        if missing <= 0:
            return True
        try:
            seq.blocks.extend(self._allocate(missing))
            return True
        except OutOfBlocks:
            return False

    def release(self, seq):
        for block in seq.blocks:
            self.allocator.decref(block)
        seq.blocks = []
        seq.num_computed = 0
        seq.num_cached_blocks = 0

    def take_cached_prefix(self, seq):
        """Start seq from the longest cached prefix. Leave at least one token
        to compute, because the last token must produce logits."""
        if self.prefix_cache is None:
            return
        usable = (seq.num_tokens - 1) // self.block_size * self.block_size
        hits = self.prefix_cache.lookup(block_hashes(seq.all_ids[:usable],
                                                     self.block_size))
        seq.blocks = list(hits)
        seq.num_computed = len(hits) * self.block_size
        seq.num_cached_blocks = len(hits)
        self.cached_tokens += seq.num_computed

    def cache_full_blocks(self, seq):
        """Give each newly full block to the prefix cache. A verify (stage 25)
        counts accepted drafts before they reach the output, so count only
        the ids that exist."""
        if self.prefix_cache is None:
            return
        full = min(seq.num_computed, seq.num_tokens) // self.block_size
        if full <= seq.num_cached_blocks:
            return
        hashes = block_hashes(seq.all_ids[:full * self.block_size],
                              self.block_size)
        for index in range(seq.num_cached_blocks, full):
            self.prefix_cache.insert(hashes[index], seq.blocks[index])
        seq.num_cached_blocks = full


# ---------------------------------------------------------------- scheduler


class Scheduler:
    """One rule: give pending tokens to sequences until the budget runs out.
    Running sequences first, oldest first. Then admit new ones."""

    def __init__(self, block_manager, max_num_seqs, token_budget, propose):
        self.blocks = block_manager
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.propose = propose        # the draft hook of the engine
        self.waiting = deque()
        self.running = []
        self.num_preemptions = 0
        self.on_preempt = None
        self.on_admit = None

    def has_work(self):
        return bool(self.waiting or self.running)

    def _preempt_newest(self):
        """The newest sequence loses the least work."""
        victim = self.running.pop()
        self.blocks.release(victim)
        victim.status = WAITING
        self.waiting.appendleft(victim)
        self.num_preemptions += 1
        if self.on_preempt:
            self.on_preempt()
        return victim

    def _make_room(self, seq, num_tokens):
        """Preempt until seq has its blocks. False if seq preempted itself."""
        while not self.blocks.grow(seq, num_tokens):
            if self._preempt_newest() is seq:
                if not self.running:
                    raise OutOfBlocks(f"request {seq.rid} needs more blocks "
                                      f"than the whole pool holds")
                return False
        return True

    def _plan_running(self, budget):
        plan, index = [], 0
        while index < len(self.running) and budget > 0:
            seq = self.running[index]
            num_tokens = min(seq.num_pending, budget)
            seq.draft = (self.propose(seq, budget - num_tokens)
                         if seq.num_pending == 1 else [])
            num_tokens += len(seq.draft)
            if not self._make_room(seq, seq.num_computed + num_tokens):
                continue                     # seq was preempted: index holds
            plan.append((seq, num_tokens))
            budget -= num_tokens
            index += 1
        return plan, budget

    def _admit_one(self, budget):
        seq = self.waiting[0]
        if not seq.blocks:
            self.blocks.take_cached_prefix(seq)
        num_tokens = min(seq.num_pending, budget)
        if not self.blocks.grow(seq, seq.num_computed + num_tokens):
            self.blocks.release(seq)          # give back the cache hits
            return None
        self.waiting.popleft()
        seq.status = RUNNING
        self.running.append(seq)
        if self.on_admit:
            self.on_admit(seq)
        return seq, num_tokens

    def _plan_admissions(self, budget):
        plan = []
        while (self.waiting and budget > 0
               and len(self.running) < self.max_num_seqs):
            admitted = self._admit_one(budget)
            if admitted is None:
                if not self.running and not plan:
                    raise OutOfBlocks(f"request {self.waiting[0].rid} cannot "
                                      f"start in an empty engine")
                break
            plan.append(admitted)
            budget -= admitted[1]
        return plan

    def schedule(self):
        """-> [(seq, num_tokens)]. No admission after a preemption: the new
        request takes the blocks of the victim, and the engine thrashes."""
        preemptions_before = self.num_preemptions
        plan, budget = self._plan_running(self.token_budget)
        if self.num_preemptions == preemptions_before:
            plan += self._plan_admissions(budget)
        return plan

    def remove(self, seq):
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.running:
            self.running.remove(seq)


# ---------------------------------------------------------------- engine


@dataclass
class StepWork:
    """What one step moved. Stage 28 turns it into a roofline floor."""
    tokens: int = 0
    context_tokens: int = 0


def all_greedy(seqs):
    return all(s.params.greedy and s.params.repetition_penalty == 1.0
               for s in seqs)


class LLMEngine:
    def __init__(self, model, num_blocks, block_size=16, max_num_seqs=64,
                 token_budget=512, prefix_cache_blocks=None, runner=None,
                 metrics=None, clock=time.perf_counter):
        self.model = model
        self.tokenizer = model.tokenizer
        self.runner = runner or ModelRunner(model, num_blocks, block_size)
        if prefix_cache_blocks is None:
            prefix_cache_blocks = num_blocks // 4
        self.blocks = KVBlockManager(num_blocks, block_size, prefix_cache_blocks)
        self.scheduler = Scheduler(self.blocks, max_num_seqs, token_budget,
                                   self._propose)
        self.metrics = metrics
        self.clock = clock
        if metrics:
            self.scheduler.on_preempt = metrics.on_preemption
            self.scheduler.on_admit = lambda s: metrics.on_schedule(s.rid,
                                                                    clock())
        self.seqs = {}
        self.num_steps = 0
        self.prefill_tokens = 0
        self.last_step = StepWork()

    # ------------------------------------------------------------ counters

    @property
    def allocator(self):
        return self.blocks.allocator

    @property
    def prefix_cache(self):
        return self.blocks.prefix_cache

    @property
    def cached_tokens(self):
        return self.blocks.cached_tokens

    @property
    def num_preemptions(self):
        return self.scheduler.num_preemptions

    # ------------------------------------------------------------ requests

    def _check_fits(self, rid, num_tokens):
        needed = self.blocks.blocks_needed(num_tokens)
        if needed > self.blocks.num_blocks:
            raise ValueError(f"request {rid} needs {needed} blocks and the "
                             f"pool has {self.blocks.num_blocks}. It can "
                             f"never run.")

    def add_request(self, rid, prompt, max_tokens=16, params=None, stop=(),
                    ignore_eos=False, arrival=None):
        prompt_ids = (self.tokenizer(prompt).input_ids
                      if isinstance(prompt, str) else list(prompt))
        self._check_fits(rid, len(prompt_ids) + max_tokens)
        seq = Sequence(rid, prompt_ids, max_tokens,
                       params or SamplingParams(temperature=0.0),
                       IncrementalDetokenizer(self.tokenizer, stop), ignore_eos,
                       self.clock() if arrival is None else arrival)
        self.seqs[rid] = seq
        self.scheduler.waiting.append(seq)
        if self.metrics:
            self.metrics.on_arrival(rid, seq.arrival)

    def has_work(self):
        return self.scheduler.has_work()

    def abort(self, rid):
        seq = self.seqs.get(rid)
        if seq is None or seq.status == FINISHED:
            return
        self.scheduler.remove(seq)
        self.blocks.release(seq)
        seq.status = FINISHED
        seq.finish_reason = "abort"

    def output(self, rid):
        return self.seqs[rid].output_ids

    # ------------------------------------------------------------ hooks
    # Stage 25 (speculation) and stage 26 (JSON mode) override these.

    def _propose(self, seq, room):
        """Draft tokens to verify in this step, at most `room`. None here."""
        return []

    def _chunks_for(self, seq, num_tokens):
        """The runner chunks for num_tokens of seq. One chunk here."""
        start = seq.num_computed
        return [SeqChunk(seq.all_ids[start:start + num_tokens], start,
                         seq.blocks)]

    def _process(self, seq, num_tokens, logits_rows):
        """Count the computed tokens. Return the new tokens, or None to let
        the batched sampler draw one when nothing is pending."""
        seq.num_computed += num_tokens
        return None

    def _mask(self, seq, logits):
        """Restrict one row of logits before sampling. Nothing here."""
        return logits

    def _on_token(self, seq, token):
        """Called for every token that the engine emits. Nothing here."""

    # ------------------------------------------------------------ one step

    def _run_model(self, plan):
        """-> (logits, spans). spans[i] = (first row, rows) of plan[i]."""
        chunks, spans = [], []
        for seq, num_tokens in plan:
            seq_chunks = self._chunks_for(seq, num_tokens)
            spans.append((len(chunks), len(seq_chunks)))
            chunks += seq_chunks
        logits = self.runner.execute(chunks)
        self.last_step = StepWork(sum(len(c.token_ids) for c in chunks),
                                  sum(c.context_len for c in chunks))
        return logits, spans

    def _count_prefill(self, seq, num_tokens):
        prompt_left = len(seq.prompt_ids) - seq.num_computed
        if prompt_left > 0:
            self.prefill_tokens += min(num_tokens, prompt_left)

    def _sample(self, seqs, logits):
        logits = torch.stack([self._mask(s, row) for s, row in zip(seqs, logits)])
        if all_greedy(seqs):
            return logits.argmax(-1).tolist()   # no sampler to run
        previous = None
        if any(s.params.repetition_penalty != 1.0 for s in seqs):
            previous = [s.all_ids for s in seqs]
        return sample(logits, [s.params for s in seqs], previous).tolist()

    def _new_tokens(self, plan, logits, spans):
        """-> [(seq, tokens)] for every sequence that makes tokens now."""
        new, ready, ready_rows = [], [], []
        for (seq, num_tokens), (first_row, num_rows) in zip(plan, spans):
            self._count_prefill(seq, num_tokens)
            tokens = self._process(seq, num_tokens,
                                   logits[first_row:first_row + num_rows])
            self.blocks.cache_full_blocks(seq)
            if tokens is not None:
                new.append((seq, tokens))
            elif seq.num_pending == 0:
                ready.append(seq)
                ready_rows.append(first_row + num_rows - 1)
        if ready:
            rows = torch.tensor(ready_rows, device=logits.device)
            sampled = self._sample(ready, logits.index_select(0, rows))
            new += [(seq, [token]) for seq, token in zip(ready, sampled)]
        return new

    def _finish_reason(self, seq, token):
        if not seq.ignore_eos and token in self.model.eos_ids:
            return "stop"
        if seq.detokenizer.stopped:
            return "stop"
        if seq.tokens_left <= 0:
            return "length"
        return None

    def _finish(self, seq, now):
        self.scheduler.remove(seq)
        self.blocks.release(seq)
        seq.status = FINISHED
        if self.metrics:
            self.metrics.on_finish(seq.rid, now)
        return seq.detokenizer.finalize()

    def _emit(self, seq, tokens, now):
        """Append tokens until one finishes the sequence. -> the new text."""
        text = ""
        for token in tokens:
            seq.output_ids.append(token)
            self._on_token(seq, token)
            if self.metrics:
                self.metrics.on_token(seq.rid, now)
            is_stop_token = not seq.ignore_eos and token in self.model.eos_ids
            if not is_stop_token:               # a stop token is not text
                text += seq.detokenizer.add_token(token)
            seq.finish_reason = self._finish_reason(seq, token)
            if seq.finish_reason:
                return text + self._finish(seq, now)
        return text

    def step(self):
        """-> [(rid, new_text, finished)], the stage 15 contract."""
        plan = self.scheduler.schedule()
        if not plan:
            return []
        logits, spans = self._run_model(plan)
        self.num_steps += 1
        now = self.clock()
        events = [(seq.rid, self._emit(seq, tokens, now),
                   seq.finish_reason is not None)
                  for seq, tokens in self._new_tokens(plan, logits, spans)]
        if self.metrics:
            self.metrics.on_kv_utilization(self.blocks.num_used,
                                           self.blocks.num_blocks)
        return events

    def run_to_completion(self, max_steps=1_000_000):
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        return {rid: seq.output_ids for rid, seq in self.seqs.items()}
