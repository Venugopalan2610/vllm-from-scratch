"""The live engine: your parts, inside one running engine, from stage 06.

The repo gives you this file. Read it. Do not edit it.

WHY THIS FILE EXISTS

Stage 05 was an engine. Stages 06 to 20 build parts, and each part passes its
checks alone: on random tensors, or in a simulation. Without this file you
would not see an engine again until stage 21. This file keeps one engine
running the whole time. `./vc run` runs a fixed workload on it.

Each box of docs/MAP.md is a slot here. A slot holds YOUR part when you have
finished its stage, and a GIVEN part before that. A given part is always the
naive version that your stage replaces. It is never the answer:

    slot        given (before your stage)                  yours (from stage)
    memory      reserve max_model_len for each sequence     06 BlockTable, 09 prefix cache
    attention   DenseReference: one growing K and V tensor  07 paged, 08 / 08b / 08c kernels
    scheduler   first come, first served, no preemption     10 Scheduler, 11 ChunkedScheduler
    launch      eager                                       12 CUDAGraphRunner
    sampler     argmax only                                 13 sample
    text        decode each token alone                     14 IncrementalDetokenizer
    metrics     none                                        16 MetricsCollector

Stages 15, 17, 18, 18b and 19 plug in through `./vc run` too. Stage 20 does
not: it needs two GPUs to be real.

THIS ENGINE IS SIMPLE AND SLOW ON PURPOSE

It is glue: it moves tokens between your parts and the model. It does not
try to be fast. The capstone (stages 21 to 28) asks you to write the real
engine, and to make it fast. Where a stage contract has a gap, this file
works around it, and `./vc run` names the gap. Each gap is a reason for a
later stage.
"""

import importlib
import math
import time
from collections import deque

import torch

from tvllm.model import DenseReference

LIVE_STAGES = ("06-block-allocator", "07-paged-attention", "08-paged-attention-cuda",
               "08b-cuda-memory", "08c-cuda-warps", "09-prefix-sharing",
               "10-admission-preemption", "11-chunked-prefill", "12-cuda-graphs",
               "13-sampling", "14-detokenization", "15-async-engine", "16-observability",
               "17-speculative-decoding", "18-quantization", "18b-int8-gemv-cuda",
               "19-guided-decoding")


def _module(name):
    return importlib.import_module(f"app.{name}")


class LiveStall(RuntimeError):
    """The engine made no progress. The message says where."""


# ---------------------------------------------------------------- the parts


class Parts:
    """Which slot holds whose part. `yours` is the set of stage ids to use
    from app/. Everything else is given."""

    def __init__(self, yours=()):
        self.yours = set(yours)

    def has(self, stage_id):
        return stage_id in self.yours

    def attention(self):
        """-> (name, stage, write_kv, attend) or None for the given dense one.
        The fastest kernel that you finished wins."""
        if not self.has("07-paged-attention"):
            return None
        write_kv = _module("s07_paged_attn").write_kv
        if self.has("08-paged-attention-cuda"):
            write_kv = _module("s08_paged_cuda").write_kv_cuda
        if self.has("08c-cuda-warps"):
            module = _module("s08c_cuda_warps")

            def attend(query, key_cache, value_cache, block_tables, context_lens, max_len):
                splits = module.splits_for(query.shape[0], query.shape[1], max_len)
                return module.paged_attention_split(query, key_cache, value_cache,
                                                    block_tables, context_lens, splits=splits)
            return "your split-K kernel", "08c", write_kv, attend
        for stage, file, function, name in [
                ("08b-cuda-memory", "s08b_cuda_memory", "paged_attention_vec", "your coalesced kernel"),
                ("08-paged-attention-cuda", "s08_paged_cuda", "paged_attention_cuda", "your CUDA kernel"),
                ("07-paged-attention", "s07_paged_attn", "paged_attention", "your PyTorch paged attention")]:
            if self.has(stage):
                inner = getattr(_module(file), function)
                return name, stage[:3].rstrip("-"), write_kv, (
                    lambda q, kc, vc, bt, cl, max_len, inner=inner: inner(q, kc, vc, bt, cl))

    def table(self):
        """-> [(slot, whose, what)] for the report."""
        attention = self.attention()
        rows = [
            ("memory", *(("yours", "stage 09: refcounts and the prefix cache") if self.has("09-prefix-sharing")
                         else ("yours", "stage 06: blocks on demand") if self.has("06-block-allocator")
                         else ("given", "reserve max_model_len for each sequence"))),
            ("attention", *(("yours", f"stage {attention[1]}: {attention[0]}") if attention
                            else ("given", "DenseReference: one growing tensor for each sequence"))),
            ("scheduler", *(("yours", "stage 11: token budget, chunked prefill") if self.has("11-chunked-prefill")
                            else ("yours", "stage 10: admission and preemption") if self.has("10-admission-preemption")
                            else ("given", "first come, first served, no preemption"))),
            ("launch", *(("yours", "stage 12: CUDA graphs for decode steps") if self.has("12-cuda-graphs")
                         else ("given", "eager"))),
            ("sampler", *(("yours", "stage 13: one vectorized pass") if self.has("13-sampling")
                          else ("given", "argmax only"))),
            ("text", *(("yours", "stage 14: incremental detokenizer") if self.has("14-detokenization")
                       else ("given", "decode each token alone"))),
            ("metrics", *(("yours", "stage 16: TTFT, ITL, KV use") if self.has("16-observability")
                          else ("given", "none"))),
        ]
        return rows


# ---------------------------------------------------------------- a request


class LiveRequest:
    def __init__(self, rid, prompt_ids, max_tokens, params=None, stop=(), guided=None):
        self.id = rid
        self.prompt_ids = list(prompt_ids)
        self.max_tokens = max_tokens
        self.params = params
        self.stop = tuple(stop)
        self.guided = guided
        self.reset()
        self.arrival = time.perf_counter()
        self.first_scheduled = None
        self.finished = False
        self.aborted = False
        self.emitted_chars = 0         # text already sent: a restart does not send it again
        self.recomputed_tokens = 0

    def reset(self):
        """A fresh start: no KV, no output. Used again after a preemption."""
        self.computed = 0              # tokens whose K and V are in the cache
        self.outputs = []
        self.blocks = []               # physical blocks, when this file owns them
        self.table = None              # your BlockTable, from stage 06
        self.dense = None              # the given DenseReference, before stage 07
        self.detokenizer = None
        self.text = ""
        self.cached_tokens = 0
        self.promised_blocks = 0

    @property
    def prompt_len(self):
        return len(self.prompt_ids)

    def next_input(self):
        """The tokens that the next forward pass must feed for this request."""
        if self.computed < self.prompt_len:
            return self.prompt_ids[self.computed:]
        return [self.outputs[-1]]


# ---------------------------------------------------------------- memory


class ReservedMemory:
    """GIVEN, before stage 06. The world before vLLM: each sequence reserves
    max_model_len tokens of KV when it starts, because nobody knows how long
    it will be. The dense attention holds the real tensors. This only counts."""

    whose = "given"

    def __init__(self, num_blocks, block_size, max_model_len):
        self.capacity = num_blocks * block_size
        self.reservation = max_model_len
        self.reserved = 0
        self.total_blocks, self.block_size = num_blocks, block_size

    def can_admit(self, request, running):
        return self.capacity - self.reserved >= self.reservation

    def admit(self, request):
        self.reserved += self.reservation

    def ensure(self, request, num_tokens):
        pass

    def release(self, request):
        self.reserved -= self.reservation

    def after_prefill(self, request):
        pass

    def used_blocks(self):
        return self.reserved // self.block_size


class YourBlocks:
    """YOURS from stage 06: your BlockAllocator and one BlockTable for each
    sequence. Blocks come on demand. Admission is careful, because the given
    scheduler cannot preempt: it admits a request only if the blocks for its
    whole prompt + max_tokens are still free after the promises to the
    running requests. Stage 10 removes that caution."""

    whose = "yours"

    def __init__(self, num_blocks, block_size):
        blocks = _module("s06_blocks")
        self.allocator = blocks.BlockAllocator(num_blocks, block_size)
        self.table_class = blocks.BlockTable
        self.total_blocks, self.block_size = num_blocks, block_size

    def _need(self, request):
        return math.ceil((request.prompt_len + request.max_tokens) / self.block_size)

    def can_admit(self, request, running):
        promised = sum(r.promised_blocks - len(r.table.blocks) for r in running)
        return self.allocator.num_free - promised >= self._need(request)

    def admit(self, request):
        request.table = self.table_class(self.allocator)
        request.promised_blocks = self._need(request)

    def ensure(self, request, num_tokens):
        request.table.reserve(num_tokens)
        request.blocks = list(request.table.blocks)

    def release(self, request):
        request.table.free()

    def after_prefill(self, request):
        pass

    def used_blocks(self):
        return self.total_blocks - self.allocator.num_free


class YourPrefixCache:
    """YOURS from stage 09: your RefCountedAllocator, your PrefixCache and
    your block hashes. A new request starts from the cached blocks of its
    longest cached prefix, and computes only the rest."""

    whose = "yours"

    def __init__(self, num_blocks, block_size):
        self.prefix = _module("s09_prefix")
        self.allocator = self.prefix.RefCountedAllocator(num_blocks, block_size)
        self.cache = self.prefix.PrefixCache(self.allocator)
        self.total_blocks, self.block_size = num_blocks, block_size
        self.hashes = {}

    def _need(self, request):
        return math.ceil((request.prompt_len + request.max_tokens) / self.block_size)

    def can_admit(self, request, running):
        hashes = self.prefix.block_hashes(request.prompt_ids, self.block_size)
        cached = min(self.cache.match_prefix_len(hashes), (request.prompt_len - 1) // self.block_size)
        need = self._need(request) - cached
        promised = sum(r.promised_blocks - len(r.blocks) for r in running)
        free = self.allocator.num_free - promised
        if free < need and self.cache.num_cached:
            self.cache.evict_lru(need - free)
            free = self.allocator.num_free - promised
        return free >= need

    def admit(self, request):
        hashes = self.prefix.block_hashes(request.prompt_ids, self.block_size)
        self.hashes[request.id] = hashes
        hits = self.cache.lookup(hashes)                # each hit has a new reference
        keep = min(len(hits), (request.prompt_len - 1) // self.block_size)
        for block in hits[keep:]:                       # at least one token must be computed
            self.allocator.decref(block)
        request.blocks = list(hits[:keep])
        request.computed = request.cached_tokens = keep * self.block_size
        request.promised_blocks = self._need(request)

    def ensure(self, request, num_tokens):
        missing = math.ceil(num_tokens / self.block_size) - len(request.blocks)
        if missing > 0:
            request.blocks += self.allocator.allocate(missing)

    def release(self, request):
        for block in request.blocks:
            self.allocator.decref(block)
        self.hashes.pop(request.id, None)

    def after_prefill(self, request):
        """The full blocks of the prompt are now computed: cache them."""
        for index, block_hash in enumerate(self.hashes.get(request.id, [])):
            if index * self.block_size >= request.cached_tokens:
                self.cache.insert(block_hash, request.blocks[index])

    def used_blocks(self):
        return self.total_blocks - self.allocator.num_free


# ---------------------------------------------------------------- attention


class DenseBackend:
    """GIVEN, before stage 07: the DenseReference of tvllm/model.py, one for
    each sequence. The K and V grow by concatenation. No pages."""

    def __init__(self, segments):
        self.segments = segments                # [(DenseReference, start, end)]

    def __call__(self, layer, query, key, value):
        outputs = [reference(layer, query[start:end], key[start:end], value[start:end])
                   for reference, start, end in self.segments]
        return torch.cat(outputs)


class PagedBackend:
    """YOURS from stage 07: your write_kv, then your attention, on the paged
    pool. Every token of the step is one row. A prefill token at position p
    is a row with context p + 1, so one decode kernel serves prefill too."""

    def __init__(self, write_kv, attend, kv_caches, slots, block_tables, context_lens, max_len):
        self.write_kv, self.attend, self.kv_caches = write_kv, attend, kv_caches
        self.slots, self.block_tables, self.context_lens = slots, block_tables, context_lens
        self.max_len = max_len

    def __call__(self, layer, query, key, value):
        key_cache, value_cache = self.kv_caches[layer]
        self.write_kv(key_cache, value_cache, key, value, self.slots)
        output = self.attend(query, key_cache, value_cache, self.block_tables,
                             self.context_lens, self.max_len)
        return output.reshape(query.shape[0], -1)


# ---------------------------------------------------------------- the engine


class LiveEngine:
    """The contract of stage 15, so that your server can serve it:

        add_request(rid, prompt, max_tokens)     prompt: str, or token ids
        step() -> [(rid, new_text, finished)]
        has_work()
        abort(rid)
    """

    def __init__(self, model, parts, num_blocks=192, block_size=16, max_num_seqs=32,
                 max_model_len=1024, token_budget=256, ignore_eos=False):
        self.model, self.parts = model, parts
        self.tokenizer = model.tokenizer
        self.block_size, self.max_model_len = block_size, max_model_len
        self.max_blocks = math.ceil(max_model_len / block_size)
        self.max_num_seqs, self.ignore_eos = max_num_seqs, ignore_eos
        self.device = model.device
        self.requests, self.waiting, self.running = {}, deque(), []
        self.notes = []                          # the gaps that ./vc run reports

        self.attention = parts.attention()
        self.scheduler = None
        if parts.has("11-chunked-prefill"):
            allocator = _module("s06_blocks").BlockAllocator(num_blocks, block_size)
            self.scheduler = _module("s11_chunked").ChunkedScheduler(
                allocator, max_num_seqs=max_num_seqs, token_budget=token_budget)
        elif parts.has("10-admission-preemption"):
            allocator = _module("s06_blocks").BlockAllocator(num_blocks, block_size)
            self.scheduler = _module("s10_scheduler").Scheduler(allocator, max_num_seqs=max_num_seqs)
        if self.scheduler is not None:
            self.allocator, self.memory = allocator, None
            self.ignore_eos = True
            self.notes.append("Your scheduler finishes a sequence by its count of tokens, so the "
                              "engine ignores EOS, and an abort frees the blocks only when the "
                              "count ends. The stage 10 contract has no abort. Stage 22 adds one.")
            if parts.has("09-prefix-sharing"):
                self.notes.append("The prefix cache is off: your stage 10 scheduler does not know "
                                  "about shared blocks. Stage 22 joins the two.")
        elif parts.has("09-prefix-sharing") and self.attention:
            self.memory = YourPrefixCache(num_blocks, block_size)
        elif parts.has("06-block-allocator"):
            self.memory = YourBlocks(num_blocks, block_size)
        else:
            self.memory = ReservedMemory(num_blocks, block_size, max_model_len)
        self.total_blocks = num_blocks
        # One extra block that no allocator knows: the padding rows of a
        # captured graph write there, not into a block of a real sequence.
        self.kv_caches = (model.allocate_kv_cache(num_blocks + 1, block_size)
                          if self.attention else None)
        self.scratch_slot = num_blocks * block_size

        self.sampler = _module("s13_sampler") if parts.has("13-sampling") else None
        self.detokenizer_class = (_module("s14_detokenizer").IncrementalDetokenizer
                                  if parts.has("14-detokenization") else None)
        self.metrics = (_module("s16_metrics").MetricsCollector()
                        if parts.has("16-observability") else None)
        self.graphs, self.graphs_off = None, None
        if parts.has("12-cuda-graphs"):
            if not self.attention:
                self.graphs_off = "a graph needs the paged attention of stage 07"
            else:
                self.graph_runner_class = _module("s12_cudagraph").CUDAGraphRunner

        self.stats = {"steps": 0, "tokens": 0, "prefill_computed": 0, "prefill_cached": 0,
                      "recomputed": 0, "preemptions": 0, "peak_running": 0, "peak_blocks": 0,
                      "step_ms": [], "step_tokens": [], "graph_steps": 0}

    # ------------------------------------------------------------ the contract

    def add_request(self, rid, prompt, max_tokens, params=None, stop=(), guided=None):
        prompt_ids = self.tokenizer(prompt).input_ids if isinstance(prompt, str) else list(prompt)
        request = LiveRequest(rid, prompt_ids, max_tokens, params, stop, guided)
        self.requests[rid] = request
        if self.scheduler is not None:
            # Your scheduler counts the tokens. One more than asked: the last
            # decode of a finished sequence never runs here (its blocks are
            # already free), so its token must be one that nobody needs.
            self.scheduler.add_request(rid, request.prompt_len, max_tokens + 1)
        else:
            self.waiting.append(request)
        if self.metrics:
            self.metrics.on_arrival(rid, request.arrival)

    def has_work(self):
        if self.scheduler is not None:
            return self.scheduler.has_work()
        return bool(self.waiting or self.running)

    def abort(self, rid):
        request = self.requests.get(rid)
        if request is None or request.finished:
            return
        request.aborted = True
        if self.scheduler is None:
            if request in self.waiting:
                self.waiting.remove(request)
            if request in self.running:
                self.running.remove(request)
                self.memory.release(request)
            request.finished = True
            request.dense = None

    def run_to_completion(self, max_idle_steps=200):
        idle = 0
        while self.has_work():
            self.step()
            idle = 0 if self.stats["step_tokens"][-1] else idle + 1
            if idle >= max_idle_steps:
                raise LiveStall(self._stall_report(idle))
        return {rid: request.outputs for rid, request in self.requests.items()}

    def _stall_report(self, idle):
        if self.scheduler is not None:
            where = (f"your scheduler planned no work for {idle} steps, with "
                     f"{len(self.scheduler.waiting)} waiting and {len(self.scheduler.running)} running")
        else:
            where = (f"the given scheduler admitted nothing for {idle} steps, with "
                     f"{len(self.waiting)} waiting and {len(self.running)} running")
        return (f"The engine stopped making progress: {where}. {self.used_blocks()} of "
                f"{self.total_blocks} blocks are in use. A sequence that needs a block when "
                "none is free waits for ever unless something frees one: a finish, or a "
                "preemption. Make the pool larger (--blocks), or look at how your scheduler "
                "handles a full pool.")

    def used_blocks(self):
        if self.scheduler is not None:
            return self.total_blocks - self.allocator.num_free
        return self.memory.used_blocks()

    # ------------------------------------------------------------ one step

    @torch.inference_mode()
    def step(self):
        start = time.perf_counter()
        work = self._plan_with_your_scheduler() if self.scheduler else self._plan_given()
        events = self._run(work) if work else []
        torch.cuda.synchronize()
        self.stats["steps"] += bool(work)
        capture = self.stats.pop("capture_ms", 0.0)     # one time, not a step
        self.stats["step_ms"].append((time.perf_counter() - start) * 1e3 - capture)
        self.stats["step_tokens"].append(sum(len(tokens) for _, tokens in work))
        self.stats["peak_blocks"] = max(self.stats["peak_blocks"], self.used_blocks())
        if self.metrics:
            self.metrics.on_kv_utilization(self.used_blocks(), self.total_blocks)
        return events

    def _plan_given(self):
        """GIVEN, before stage 10: admit in order while memory says yes, then
        every running request feeds all of its next tokens. No token budget,
        no preemption."""
        while (self.waiting and len(self.running) < self.max_num_seqs
               and self.memory.can_admit(self.waiting[0], self.running)):
            request = self.waiting.popleft()
            self.memory.admit(request)
            self.running.append(request)
        self.stats["peak_running"] = max(self.stats["peak_running"], len(self.running))
        work = []
        for request in self.running:
            tokens = request.next_input()
            self.memory.ensure(request, request.computed + len(tokens))
            work.append((request, tokens))
        return work

    def _plan_with_your_scheduler(self):
        """YOURS from stage 10: your scheduler decides; this only executes."""
        plan = self.scheduler.step()
        seqs = {seq.id: seq for seq in list(self.scheduler.running) + list(self.scheduler.waiting)}
        self.stats["peak_running"] = max(self.stats["peak_running"], len(self.scheduler.running))
        for seq in plan.get("preempted", []):
            request = self.requests[seq.id]
            self.stats["preemptions"] += 1
            self.stats["recomputed"] += request.computed
            request.recomputed_tokens += request.computed
            request.reset()
            if self.metrics:
                self.metrics.on_preemption()
        finished = {seq.id for seq in plan.get("finished", [])}
        preempted = {seq.id for seq in plan.get("preempted", [])}
        work = []
        if "prefill" in plan:                                    # stage 11
            chunks = [(rid, length) for rid, length in plan["prefill"]]
        else:                                                    # stage 10
            chunks = [(seq.id, None) for seq in plan.get("prefilled", [])]
        for rid, length in chunks:
            request = self.requests[rid]
            if rid in finished or rid in preempted:
                continue
            tokens = request.prompt_ids[request.computed:]
            work.append((request, tokens if length is None else tokens[:length]))
        for seq in plan.get("decoded", []):
            request = self.requests[seq.id]
            if (seq.id in finished or seq.id in preempted or not request.outputs
                    or request.computed < request.prompt_len or request.finished):
                continue
            work.append((request, [request.outputs[-1]]))
        for request, _ in work:
            request.blocks = list(seqs[request.id].blocks)
        for rid in finished:
            self._finish(self.requests[rid], quiet=True)
        return work

    # ------------------------------------------------------------ the model

    def _run(self, work):
        """One forward pass for every token of the step, then one sample for
        each request whose prompt is complete."""
        token_ids, positions, rows, sample_rows = [], [], [], []
        for request, tokens in work:
            if request.first_scheduled is None:
                request.first_scheduled = time.perf_counter()
                if self.metrics:
                    self.metrics.on_schedule(request.id, request.first_scheduled)
            start = request.computed
            token_ids += tokens
            positions += range(start, start + len(tokens))
            rows.append((request, len(token_ids) - len(tokens), len(token_ids)))
            if start + len(tokens) >= request.prompt_len:
                sample_rows.append((request, len(token_ids) - 1))
        tokens_tensor = torch.tensor(token_ids, device=self.device)
        positions_tensor = torch.tensor(positions, device=self.device)
        logits_rows = torch.tensor([row for _, row in sample_rows], device=self.device)

        if self.attention and self._decode_only(work):
            logits = self._run_graph_or_eager(work, tokens_tensor, positions_tensor)
        else:
            backend = self._backend(rows, positions)
            logits = self.model.forward(tokens_tensor, positions_tensor, backend, logits_rows)

        for request, tokens in work:
            before = request.computed
            request.computed += len(tokens)
            if before < request.prompt_len:
                self.stats["prefill_computed"] += min(len(tokens), request.prompt_len - before)
                if request.computed >= request.prompt_len:
                    self.stats["prefill_cached"] += request.cached_tokens
                    if self.memory is not None:
                        self.memory.after_prefill(request)
        return self._sample_and_emit([request for request, _ in sample_rows], logits)

    def _decode_only(self, work):
        return all(request.computed >= request.prompt_len and len(tokens) == 1
                   for request, tokens in work)

    def _backend(self, rows, positions):
        if not self.attention:
            segments = []
            for request, start, end in rows:
                if request.dense is None:
                    request.dense = DenseReference(self.model.config.num_layers)
                segments.append((request.dense, start, end))
            return DenseBackend(segments)
        slots, tables, lengths = self._paged_inputs(rows, positions)
        _, _, write_kv, attend = self.attention
        return PagedBackend(write_kv, attend, self.kv_caches, slots, tables, lengths,
                            self.max_model_len)

    def _paged_inputs(self, rows, positions):
        """For each token: its slot, the block table of its sequence, and its
        context length (its position + 1)."""
        size = self.block_size
        slots, tables = [], []
        for request, start, end in rows:
            padded = request.blocks + [0] * (self.max_blocks - len(request.blocks))
            for position in positions[start:end]:
                slots.append(request.blocks[position // size] * size + position % size)
                tables.append(padded)
        return (torch.tensor(slots, device=self.device),
                torch.tensor(tables, dtype=torch.int32, device=self.device),
                torch.tensor([p + 1 for p in positions], dtype=torch.int32, device=self.device))

    def _run_graph_or_eager(self, work, tokens, positions):
        rows = [(request, index, index + 1) for index, (request, _) in enumerate(work)]
        slots, tables, lengths = self._paged_inputs(rows, positions.tolist())
        if self.graph_runner_class_ready(len(work)):
            self.stats["graph_steps"] += 1
            return self.graphs.run(tokens, positions, slots, tables, lengths)
        return self._decode_step(tokens, positions, slots, tables, lengths)

    def _decode_step(self, tokens, positions, slots, tables, lengths):
        """The decode step, as one function of tensors, so that stage 12 can
        capture it. A padding row has a context of 0: it writes into the
        scratch block, never into a block of a real sequence."""
        slots = torch.where(lengths > 0, slots, torch.full_like(slots, self.scratch_slot))
        _, _, write_kv, attend = self.attention
        backend = PagedBackend(write_kv, attend, self.kv_caches, slots, tables, lengths,
                               self.max_model_len)
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return self.model.forward(tokens, positions, backend, rows)

    def graph_runner_class_ready(self, batch_size):
        if not self.parts.has("12-cuda-graphs") or self.graphs_off:
            return False
        buckets = tuple(b for b in (1, 2, 4, 8, 16, 32, 64) if b <= max(self.max_num_seqs, 1))
        if batch_size > buckets[-1]:
            return False
        if self.graphs is None:
            try:
                runner = self.graph_runner_class(self._decode_step, buckets=buckets)

                def make_inputs(batch):
                    zeros = lambda dtype: torch.zeros(batch, dtype=dtype, device=self.device)
                    return (zeros(torch.long), zeros(torch.long), zeros(torch.long),
                            torch.zeros(batch, self.max_blocks, dtype=torch.int32, device=self.device),
                            zeros(torch.int32))
                began = time.perf_counter()
                runner.capture(make_inputs)
                self.stats["capture_ms"] = (time.perf_counter() - began) * 1e3
                self.graphs = runner
            except Exception as error:                  # noqa: BLE001 - reported, not hidden
                torch.cuda.synchronize()
                self.graphs_off = (f"the capture failed ({type(error).__name__}). An attention "
                                   "that reads a GPU value on the CPU cannot be captured. "
                                   "Stage 21 gives you a backend that can be.")
                return False
        return True

    # ------------------------------------------------------------ the output

    def _sample_and_emit(self, requests, logits):
        if not requests:
            return []
        logits = logits.clone()
        for row, request in enumerate(requests):
            if request.guided is not None:
                text = self.tokenizer.decode(request.outputs)
                mask = request.guided.mask_for(text).to(logits.device)
                logits[row, :mask.shape[0]][~mask] = float("-inf")
                logits[row, mask.shape[0]:] = float("-inf")
        if self.sampler is not None:
            params = [request.params or self.sampler.SamplingParams(temperature=0.0)
                      for request in requests]
            previous = [request.prompt_ids + request.outputs for request in requests]
            tokens = self.sampler.sample(logits, params, previous).tolist()
        else:
            tokens = logits.argmax(-1).tolist()
        events, now = [], time.perf_counter()
        for request, token in zip(requests, tokens):
            if request.finished:
                continue
            request.outputs.append(token)
            self.stats["tokens"] += 1
            if self.metrics:
                self.metrics.on_token(request.id, now)
            delta = self._text(request, token)
            done = (len(request.outputs) >= request.max_tokens
                    or (not self.ignore_eos and token in self.model.eos_ids)
                    or (request.guided is not None and request.guided.is_complete(request.text))
                    or getattr(request.detokenizer, "stopped", False))
            if done and self.scheduler is None:
                delta += self._finish(request)
            elif done:
                request.finished = True
                if self.metrics:
                    self.metrics.on_finish(request.id, time.perf_counter())
                delta += self._final_text(request)
            if request.aborted:
                continue
            events.append((request.id, delta, done))
        return events

    def _text(self, request, token):
        """-> the new text to stream. A restart after a preemption does not
        send again the text that the client already has."""
        if self.detokenizer_class is not None:
            if request.detokenizer is None:
                request.detokenizer = self.detokenizer_class(self.tokenizer, request.stop)
            request.detokenizer.add_token(token)
            request.text = request.detokenizer.text
        else:
            request.text += self.tokenizer.decode([token])
        new = request.text[request.emitted_chars:]
        request.emitted_chars = max(request.emitted_chars, len(request.text))
        return new

    def _final_text(self, request):
        if request.detokenizer is not None:
            request.detokenizer.finalize()
            request.text = request.detokenizer.text
        new = request.text[request.emitted_chars:]
        request.emitted_chars = max(request.emitted_chars, len(request.text))
        return new

    def _finish(self, request, quiet=False):
        if request.finished and quiet:
            return ""
        request.finished = True
        request.dense = None                       # free the given dense K and V
        if self.scheduler is None and request in self.running:
            self.running.remove(request)
            self.memory.release(request)
        if self.metrics:
            self.metrics.on_finish(request.id, time.perf_counter())
        return "" if quiet else self._final_text(request)


# ---------------------------------------------------------------- the oracle


@torch.inference_mode()
def oracle_greedy(model, prompt_ids, num_tokens):
    """The trusted version: one sequence alone, the given dense attention,
    argmax, EOS ignored. -> (tokens, the gap between the two best scores at
    each position). A small gap is a near tie that rounding can flip."""
    reference = DenseReference(model.config.num_layers)
    tokens, gaps = [], []
    inputs, start = list(prompt_ids), 0
    for _ in range(num_tokens):
        ids = torch.tensor(inputs, device=model.device)
        positions = torch.arange(start, start + len(inputs), device=model.device)
        rows = torch.tensor([len(inputs) - 1], device=model.device)
        logits = model.forward(ids, positions, reference, rows)[0]
        top = logits.topk(2).values
        gaps.append(float(top[0] - top[1]))
        tokens.append(int(logits.argmax()))
        start += len(inputs)
        inputs = [tokens[-1]]
    return tokens, gaps


# ---------------------------------------------------------------- stage 17


@torch.inference_mode()
def speculative_greedy(model, parts, prompt_ids, num_tokens, num_draft=4, block_size=16):
    """Greedy decode of one sequence, with your n-gram drafts (stage 17) and
    your paged attention. A verify is one forward pass over the last token and
    the drafts: each draft is one more row of the same sequence. num_draft=0
    is the plain decode, to compare with.

    -> (tokens, target_calls, seconds, gaps): gaps[i] is the distance between
    the two best scores when token i was chosen."""
    name, stage, write_kv, attend = parts.attention()
    propose = _module("s17_speculative").ngram_propose
    max_len = len(prompt_ids) + num_tokens + num_draft + 1
    max_blocks = math.ceil(max_len / block_size)
    allocator = _module("s06_blocks").BlockAllocator(max_blocks + 1, block_size)
    blocks = allocator.allocate(max_blocks)
    kv_caches = model.allocate_kv_cache(max_blocks + 1, block_size)
    device = model.device

    def forward(tokens, start):
        positions = list(range(start, start + len(tokens)))
        slots = torch.tensor([blocks[p // block_size] * block_size + p % block_size
                              for p in positions], device=device)
        tables = torch.tensor([blocks] * len(tokens), dtype=torch.int32, device=device)
        lengths = torch.tensor([p + 1 for p in positions], dtype=torch.int32, device=device)
        backend = PagedBackend(write_kv, attend, kv_caches, slots, tables, lengths, max_len)
        return model.forward(torch.tensor(tokens, device=device),
                             torch.tensor(positions, device=device), backend,
                             torch.arange(len(tokens), device=device))

    torch.cuda.synchronize()
    begin = time.perf_counter()
    logits = forward(list(prompt_ids), 0)
    output, calls = [int(logits[-1].argmax())], 1
    top = logits[-1].topk(2).values
    gaps = [float(top[0] - top[1])]
    while len(output) < num_tokens:
        history = list(prompt_ids) + output
        draft = list(propose(history, num_draft))[:num_draft] if num_draft else []
        logits = forward([output[-1]] + draft, len(history) - 1)
        targets = logits.argmax(-1).tolist()
        top2 = logits.topk(2, dim=-1).values
        calls += 1
        accepted = 0
        while accepted < len(draft) and draft[accepted] == targets[accepted]:
            accepted += 1
        output += draft[:accepted] + [targets[accepted]]
        gaps += (top2[:accepted + 1, 0] - top2[:accepted + 1, 1]).tolist()
    torch.cuda.synchronize()
    return output[:num_tokens], calls, time.perf_counter() - begin, gaps[:num_tokens]


# ---------------------------------------------------------------- stage 18


def quantize_weights_(model, parts):
    """Replace every matmul weight of the model (not lm_head) with your
    quantized Linear: stage 18b if you finished it, else stage 18. The model
    takes a callable weight (see `linear` in tvllm/model.py).
    -> the name of the Linear that it used."""
    if parts.has("18b-int8-gemv-cuda"):
        quantized, name = _module("s18b_gemv_cuda").QuantizedLinearCUDA, "your stage 18b GEMV"
    else:
        quantized, name = _module("s18_quantization").QuantizedLinear, "your stage 18 int8 Linear"
    for layer in model.layers:
        for attribute in layer.MATMULS:
            weight = getattr(layer, attribute)
            dense = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                                    device=weight.device, dtype=weight.dtype)
            dense.weight.data = weight
            setattr(layer, attribute, quantized.from_linear(dense))
    torch.cuda.empty_cache()
    return name
