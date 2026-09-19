"""Stage 25 - speculative decoding inside the engine.

`./vc lore 25` for the insight. `./vc test 25` to check yourself.

Stage 17 proved the rejection sampler on synthetic distributions. Nothing ran
it on a model. Here it runs inside your engine, on every decode step that has
a draft.

WHAT YOU ARE BUILDING

    target_probs(logits_rows, params) -> (rows, vocab)
        the distribution that plain sampling draws from: temperature, top-k,
        top-p, as in stage 13
    verify_greedy(logits_rows, draft) -> tokens to emit
    verify_sampled(logits_rows, draft, params, generator=None) -> tokens
    verify(logits_rows, draft, params, generator=None) -> tokens
        logits_rows (len(draft) + 1, vocab): the logits after the last real
        token and after each draft token
    SpeculativeEngine(*args, num_speculative=4, ngram=3, **kwargs)
        your stage 22 LLMEngine, with three hooks overridden:
        _propose, _chunks_for, _process. .stats is a stage 17
        SpeculativeStats.

THE VERIFY IS A DECODE BATCH

A draft of K tokens needs logits at K+1 positions. Do not send them to the
runner as one K+1-token prefill chunk: that takes the slow SDPA path, and it
never replays a graph.

Send K+1 single-token chunks instead, at positions p, p+1, ..., p+K, with the
SAME block list. Your stage 21 backend writes all K+1 tokens first, then runs
attention. Row i has context p+i+1, so it sees exactly the tokens before it.
The verify runs on your stage 08c kernel, and it replays a captured graph.

HOW TO ACCEPT

  - Greedy: accept each draft token while it equals the argmax of its row.
    Then emit the argmax of the first row that disagrees, or of the last row.
  - Sampled: an n-gram draft is a point mass, q = one-hot. Stage 17's rule
    with that q: accept draft t with probability p(t). On a rejection, draw
    from p with t removed and renormalised. The output distribution is then
    exactly p.

After a verify with a accepted drafts, num_computed grows by 1 + a. The slots
of the rejected drafts hold garbage, and the next step overwrites them.

TRAPS

  - Limit the draft to max_tokens - len(output) - 1. A draft that runs past the
    end emits tokens that the request did not ask for.
  - A request with a repetition penalty does not speculate. Its distribution
    depends on every earlier token, and the verify would need it for each row.
  - Seeded sampling: keep one torch.Generator for each seeded request, so that
    the same seed gives the same text.
  - The greedy output with speculation must equal the output without it,
    token for token. A check proves it in fp32.
"""

import torch

from app.s13_sampler import apply_top_k, apply_top_p
from app.s17_speculative import SpeculativeStats, ngram_propose
from app.s21_paged_runner import SeqChunk
from app.s22_engine import LLMEngine


def target_probs(logits_rows, params):
    """The distribution that plain sampling draws from, for each row:
    temperature, then top-k, then top-p, as in stage 13."""
    raise NotImplementedError("stage 25: implement target_probs")


def verify_greedy(logits_rows, draft):
    """Accept each draft token while it equals the argmax of its row. Then
    emit the argmax of the first row that disagrees, or of the last row."""
    raise NotImplementedError("stage 25: implement verify_greedy")


def draw(probs, generator):
    return int(torch.multinomial(probs, 1, generator=generator))


def verify_sampled(logits_rows, draft, params, generator=None):
    """Modified rejection sampling (stage 17) with an n-gram draft. The draft
    is a point mass, so accept token t with probability p(t). On a rejection,
    draw from p with t removed. The output is then distributed exactly as p."""
    raise NotImplementedError("stage 25: implement verify_sampled")


def verify(logits_rows, draft, params, generator=None):
    """logits_rows (len(draft) + 1, vocab): the logits after the last real
    token and after each draft token. -> the tokens to emit."""
    raise NotImplementedError("stage 25: implement verify")


class SpeculativeEngine(LLMEngine):
    def __init__(self, *args, num_speculative=4, ngram=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_speculative = num_speculative
        self.ngram = ngram
        self.stats = SpeculativeStats()
        self._generators = {}

    def _draft_length(self, seq, room):
        """Never draft past max_tokens: the last emitted token is not a draft."""
        raise NotImplementedError("stage 25: implement SpeculativeEngine._draft_length")

    def _propose(self, seq, room):
        # A repetition penalty makes each row depend on the rows before it.
        raise NotImplementedError("stage 25: implement SpeculativeEngine._propose")

    def _chunks_for(self, seq, num_tokens):
        """A verify is K+1 decode rows that share one block list. Row i has
        its own context, so the verify is causal, runs on your stage 08c
        kernel, and replays a captured graph."""
        raise NotImplementedError("stage 25: implement SpeculativeEngine._chunks_for")

    def _generator(self, seq):
        """One generator for each seeded request: the same seed, the same
        text."""
        raise NotImplementedError("stage 25: implement SpeculativeEngine._generator")

    def _process(self, seq, num_tokens, logits_rows):
        raise NotImplementedError("stage 25: implement SpeculativeEngine._process")
