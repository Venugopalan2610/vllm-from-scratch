"""Reference solution, stage 25 - speculative decoding inside the engine."""

import torch

from app.s13_sampler import apply_top_k, apply_top_p
from app.s17_speculative import SpeculativeStats, ngram_propose
from app.s21_paged_runner import SeqChunk
from app.s22_engine import LLMEngine


def target_probs(logits_rows, params):
    """The distribution that plain sampling draws from, for each row:
    temperature, then top-k, then top-p, as in stage 13."""
    num_rows = logits_rows.shape[0]
    device = logits_rows.device
    scaled = logits_rows.float() / max(params.temperature, 1e-6)
    if params.top_k:
        scaled = apply_top_k(scaled, torch.full((num_rows,), params.top_k,
                                                dtype=torch.long, device=device))
    if params.top_p < 1.0:
        scaled = apply_top_p(scaled, torch.full((num_rows,), params.top_p,
                                                device=device))
    return torch.softmax(scaled, dim=-1)


def verify_greedy(logits_rows, draft):
    """Accept each draft token while it equals the argmax of its row. Then
    emit the argmax of the first row that disagrees, or of the last row."""
    best = logits_rows.argmax(-1).tolist()
    accepted = []
    for row, token in enumerate(draft):
        if best[row] != token:
            return accepted + [best[row]]
        accepted.append(token)
    return accepted + [best[len(draft)]]


def draw(probs, generator):
    return int(torch.multinomial(probs, 1, generator=generator))


def verify_sampled(logits_rows, draft, params, generator=None):
    """Modified rejection sampling (stage 17) with an n-gram draft. The draft
    is a point mass, so accept token t with probability p(t). On a rejection,
    draw from p with t removed. The output is then distributed exactly as p."""
    probs = target_probs(logits_rows, params)
    coins = torch.rand(len(draft), generator=generator,
                       device=logits_rows.device).tolist()
    accepted = []
    for row, token in enumerate(draft):
        if coins[row] < float(probs[row, token]):
            accepted.append(token)
            continue
        without_token = probs[row].clone()
        without_token[token] = 0.0
        return accepted + [draw(without_token / without_token.sum(), generator)]
    return accepted + [draw(probs[len(draft)], generator)]


def verify(logits_rows, draft, params, generator=None):
    """logits_rows (len(draft) + 1, vocab): the logits after the last real
    token and after each draft token. -> the tokens to emit."""
    if params.greedy:
        return verify_greedy(logits_rows, draft)
    return verify_sampled(logits_rows, draft, params, generator)


class SpeculativeEngine(LLMEngine):
    def __init__(self, *args, num_speculative=4, ngram=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_speculative = num_speculative
        self.ngram = ngram
        self.stats = SpeculativeStats()
        self._generators = {}

    def _draft_length(self, seq, room):
        """Never draft past max_tokens: the last emitted token is not a draft."""
        return min(self.num_speculative, room, seq.tokens_left - 1)

    def _propose(self, seq, room):
        # A repetition penalty makes each row depend on the rows before it.
        if seq.params.repetition_penalty != 1.0:
            return []
        length = self._draft_length(seq, room)
        if length <= 0:
            return []
        return ngram_propose(seq.all_ids, length, self.ngram)

    def _chunks_for(self, seq, num_tokens):
        """A verify is K+1 decode rows that share one block list. Row i has
        its own context, so the verify is causal, runs on your stage 08c
        kernel, and replays a captured graph."""
        if not seq.draft:
            return super()._chunks_for(seq, num_tokens)
        start = seq.num_computed
        tokens = seq.all_ids[start:start + 1] + seq.draft
        return [SeqChunk([token], start + offset, seq.blocks)
                for offset, token in enumerate(tokens)]

    def _generator(self, seq):
        """One generator for each seeded request: the same seed, the same
        text."""
        if seq.params.seed is None:
            return None
        if seq.rid not in self._generators:
            generator = torch.Generator(device=self.runner.device)
            self._generators[seq.rid] = generator.manual_seed(seq.params.seed)
        return self._generators[seq.rid]

    def _process(self, seq, num_tokens, logits_rows):
        if not seq.draft:
            return super()._process(seq, num_tokens, logits_rows)
        tokens = verify(logits_rows, seq.draft, seq.params, self._generator(seq))
        num_accepted = len(tokens) - 1
        # The cache holds the last real token and the accepted drafts. The
        # slots of the rejected drafts hold garbage. The next step overwrites.
        seq.num_computed += 1 + num_accepted
        self.stats.record(len(seq.draft), num_accepted)
        seq.draft = []
        return tokens
