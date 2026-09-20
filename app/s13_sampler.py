"""Stage 13 - a real batched sampler.

`./vc lore 13` for the insight. `./vc test 13` to check yourself.

Each request of the batch has its OWN temperature, top_k, top_p, penalties
and seed. ONE vectorized pass must apply all of them. When the kernels are
fast, after stages 08 and 12, a Python loop over 256 requests here becomes
your bottleneck, with no warning.
"""

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0            # 0 = off
    top_p: float = 1.0        # 1.0 = off
    min_p: float = 0.0        # 0.0 = off
    repetition_penalty: float = 1.0
    seed: int | None = None
    logprobs: int | None = None  # None or number of top logprobs to return

    @property
    def greedy(self):
        return self.temperature == 0.0


def check_logits(logits):
    """Assert that no logit is NaN. A bad quantization scale, a bad FP8
    calibration, or an overflow in attention produces NaN logits. The sampler
    then samples garbage, and the user sees fluent nonsense with no error.
    NaN is silent; only a check makes it loud."""
    if torch.isnan(logits).any():
        raise ValueError(
            "NaN detected in logits. Common causes: bad quantization scale, "
            "FP8 calibration overflow, or attention overflow. Check your "
            "weights and KV cache.")


def apply_repetition_penalty(logits, prev_tokens, penalties):
    """logits (rows, vocab), prev_tokens list[list[int]], penalties (rows,).

    The convention (from CTRL, and all others copied it):

        logit > 0  ->  logit / penalty
        logit <= 0 ->  logit * penalty

    The two cases are on purpose: a division makes a negative logit LARGER,
    and that makes the token more probable, not less.
    """
    raise NotImplementedError("stage 13: implement apply_repetition_penalty")


def apply_top_k(logits, k):
    """Keep only the k highest logits of each row. k is (rows,). A 0 turns
    top-k off for that row.

    Vectorize it: sort one time, take the k-th largest value as a threshold
    for each row, then mask everything below it. Do not loop over the batch.
    """
    raise NotImplementedError("stage 13: implement apply_top_k")


def apply_top_p(logits, p):
    """Nucleus sampling. p is (rows,), and 1.0 = off.

    Keep the SMALLEST set of tokens whose cumulative probability gets to p.

    Sort in descending order, cumsum, and remove the tokens whose mass
    BEFORE them already got to p:

        remove = (cumsum - probs) >= p

    Two important details:
      - always keep the top token. If not, a peaked distribution with a small
        p leaves no token to sample.
      - compare with a small epsilon. Exact limits such as 0.5+0.25+0.15 are
        not exactly 0.9 in float32.
    """
    raise NotImplementedError("stage 13: implement apply_top_p")


def apply_min_p(logits, min_p):
    """Filter out tokens whose probability is less than min_p * max_prob.
    min_p is (rows,), and 0.0 = off. Always keep the top token."""
    raise NotImplementedError("stage 13: implement apply_min_p")


def sample(logits, params, prev_tokens=None):
    """logits (rows, vocab) -> (rows,) token ids.

    The order is important: repetition penalty, then temperature, then
    top_k, then top_p, then the draw. A temperature after the truncations
    changes the tokens that remain.

    temperature == 0 means greedy (argmax) for that row.

    For the draw, the Gumbel-max method vectorizes, and torch.multinomial
    does not:

        gumbel = -log(-log(uniform))
        token  = argmax(logits + gumbel)

    That is an exact categorical sample, and each row can use its own seeded
    noise. Look at the parentheses. `-torch.log(x).clamp_min(e)` is
    `-(torch.log(x).clamp_min(e))`. That gives NaNs, and a sampler that
    returns token 0 for ever.

    The same seed and the same logits must give the same token, every time.
    """
    raise NotImplementedError("stage 13: implement sample")
