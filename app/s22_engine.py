"""Stage 22 - one engine from the parts.

`./vc lore 22` for the insight. `./vc test 22` to check yourself.

Stages 10 and 11 were two separate schedulers in a simulation. Neither one
ran a model, and neither one could stop at EOS. Stage 09 was a prefix cache
with nothing to cache. Stages 13 and 14 were a sampler and a detokenizer
with no engine to call them. This stage makes one engine from all of them,
on top of your stage 21 runner.

WHAT YOU ARE BUILDING

Four classes, and each one has one job:

    Sequence          given: the state of one request
    KVBlockManager    the blocks and the prefix cache: grow, release, take a
                      cached prefix, cache the full blocks
    Scheduler         the one rule below: plan a step, admit, preempt
    LLMEngine         requests, one step on the runner, sampling, emission

    LLMEngine(model, num_blocks, block_size=16, max_num_seqs=64,
              token_budget=512, prefix_cache_blocks=None, runner=None,
              metrics=None, clock=time.perf_counter)
      .add_request(rid, prompt, max_tokens=16, params=None, stop=(),
                   ignore_eos=False, arrival=None)
      .step() -> [(rid, new_text, finished)]     the stage 15 contract
      .has_work(), .abort(rid), .output(rid), .run_to_completion()
      .seqs, .scheduler, .blocks, .allocator, .prefix_cache
      .num_preemptions, .num_steps, .prefill_tokens, .cached_tokens
      .last_step = StepWork(tokens, context_tokens) of the last step

ONE IDEA HOLDS THE WHOLE SCHEDULER

A sequence has token ids (prompt + output), and num_computed of them have K
and V in the cache. The difference is its PENDING work. A decode has one
pending token. A prefill has many. A preempted sequence has all of them
again.

So there is one rule: each step, give pending tokens to sequences until the
token budget runs out. Running sequences first, oldest first. Then admit new
ones, but not in a step that preempted. vLLM V1 schedules like this.

PREEMPTION

When the blocks run out, drop the prefix cache first. Then preempt the
NEWEST running sequence: release its blocks, and put it at the FRONT of the
waiting queue. It keeps its output. It computes everything again later.

FIVE HOOKS, FOR STAGES 25 AND 26

Stage 25 adds speculation and stage 26 adds a JSON mode, as subclasses. They
must not copy step(). So step() calls five methods. LLMEngine holds their
plain versions:

    _propose(seq, room)            the scheduler calls it for a decode
    _chunks_for(seq, num_tokens)   the runner chunks of one plan entry
    _process(seq, num_tokens, logits_rows)   count the computed tokens
    _mask(seq, logits)             before sampling
    _on_token(seq, token)          for each emitted token

TRAPS

  - A stop token (EOS) finishes the request. It is not text: do not feed it
    to the detokenizer.
  - A block in the prefix cache has a reference from the cache itself. After
    a run, num_free is not num_blocks until you evict the cache.
  - A request that needs more blocks than the pool holds can never run.
    Reject it in add_request with a ValueError, or the engine spins forever.
  - When every sequence is greedy, do not call the sampler. argmax is one
    kernel.
"""

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
        raise NotImplementedError("stage 22: implement KVBlockManager.blocks_needed")

    def _allocate(self, count):
        """Drop the prefix cache before you drop work."""
        raise NotImplementedError("stage 22: implement KVBlockManager._allocate")

    def grow(self, seq, num_tokens):
        """Give seq the blocks for num_tokens. False if the pool is empty."""
        raise NotImplementedError("stage 22: implement KVBlockManager.grow")

    def release(self, seq):
        raise NotImplementedError("stage 22: implement KVBlockManager.release")

    def take_cached_prefix(self, seq):
        """Start seq from the longest cached prefix. Leave at least one token
        to compute, because the last token must produce logits."""
        raise NotImplementedError("stage 22: implement KVBlockManager.take_cached_prefix")

    def cache_full_blocks(self, seq):
        """Give each newly full block to the prefix cache. A verify (stage 25)
        counts accepted drafts before they reach the output, so count only
        the ids that exist."""
        raise NotImplementedError("stage 22: implement KVBlockManager.cache_full_blocks")


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
        raise NotImplementedError("stage 22: implement Scheduler._preempt_newest")

    def _make_room(self, seq, num_tokens):
        """Preempt until seq has its blocks. False if seq preempted itself."""
        raise NotImplementedError("stage 22: implement Scheduler._make_room")

    def _plan_running(self, budget):
        raise NotImplementedError("stage 22: implement Scheduler._plan_running")

    def _admit_one(self, budget):
        raise NotImplementedError("stage 22: implement Scheduler._admit_one")

    def _plan_admissions(self, budget):
        raise NotImplementedError("stage 22: implement Scheduler._plan_admissions")

    def schedule(self):
        """-> [(seq, num_tokens)]. No admission after a preemption: the new
        request takes the blocks of the victim, and the engine thrashes."""
        raise NotImplementedError("stage 22: implement Scheduler.schedule")

    def remove(self, seq):
        raise NotImplementedError("stage 22: implement Scheduler.remove")


# ---------------------------------------------------------------- engine


@dataclass
class StepWork:
    """What one step moved. Stage 28 turns it into a roofline floor."""
    tokens: int = 0
    context_tokens: int = 0


def all_greedy(seqs):
    raise NotImplementedError("stage 22: implement all_greedy")


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
        raise NotImplementedError("stage 22: implement LLMEngine._check_fits")

    def add_request(self, rid, prompt, max_tokens=16, params=None, stop=(),
                    ignore_eos=False, arrival=None):
        raise NotImplementedError("stage 22: implement LLMEngine.add_request")

    def has_work(self):
        return self.scheduler.has_work()

    def abort(self, rid):
        raise NotImplementedError("stage 22: implement LLMEngine.abort")

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
        raise NotImplementedError("stage 22: implement LLMEngine._run_model")

    def _count_prefill(self, seq, num_tokens):
        raise NotImplementedError("stage 22: implement LLMEngine._count_prefill")

    def _sample(self, seqs, logits):
        raise NotImplementedError("stage 22: implement LLMEngine._sample")

    def _new_tokens(self, plan, logits, spans):
        """-> [(seq, tokens)] for every sequence that makes tokens now."""
        raise NotImplementedError("stage 22: implement LLMEngine._new_tokens")

    def _finish_reason(self, seq, token):
        raise NotImplementedError("stage 22: implement LLMEngine._finish_reason")

    def _finish(self, seq, now):
        raise NotImplementedError("stage 22: implement LLMEngine._finish")

    def _emit(self, seq, tokens, now):
        """Append tokens until one finishes the sequence. -> the new text."""
        raise NotImplementedError("stage 22: implement LLMEngine._emit")

    def step(self):
        """-> [(rid, new_text, finished)], the stage 15 contract."""
        raise NotImplementedError("stage 22: implement LLMEngine.step")

    def run_to_completion(self, max_steps=1_000_000):
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        return {rid: seq.output_ids for rid, seq in self.seqs.items()}
