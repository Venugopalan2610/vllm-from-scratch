"""Stage 21 - the model runs on your paged cache.

The spec is in app/s21_paged_runner.py. The oracle is HuggingFace in fp32.
Every check that compares tokens uses fp32, for the reason that the
`hf_exact` fixture gives.
"""

import random
from dataclasses import dataclass, field

import torch

import app.s21_paged_runner as s21
from app.s21_paged_runner import ModelRunner, SeqChunk
from tests.helpers import CAPSTONE_PROMPTS, hf_greedy

LOGIT_TOLERANCE = dict(rtol=1e-4, atol=1e-4)


@dataclass
class _Sequence:
    """One prompt, driven by hand through the runner."""
    prompt_ids: list
    block_ids: list
    output_ids: list = field(default_factory=list)
    num_computed: int = 0

    @property
    def all_ids(self):
        return self.prompt_ids + self.output_ids

    def next_chunk(self, max_prefill_chunk):
        pending = self.all_ids[self.num_computed:]
        if max_prefill_chunk and self.num_computed < len(self.prompt_ids):
            pending = pending[:max_prefill_chunk]
        return SeqChunk(pending, self.num_computed, self.block_ids)

    def advance(self, chunk_len, logits_row):
        self.num_computed += chunk_len
        if self.num_computed == len(self.all_ids):
            self.output_ids.append(logits_row.argmax().item())


def _greedy(runner, tokenizer, prompts, num_tokens, blocks_per_seq=8,
            max_prefill_chunk=None, seed=0):
    """All prompts in one batch, num_tokens each, on shuffled blocks."""
    free_blocks = list(range(runner.num_blocks))
    random.Random(seed).shuffle(free_blocks)
    sequences = [_Sequence(tokenizer(prompt).input_ids,
                           [free_blocks.pop() for _ in range(blocks_per_seq)])
                 for prompt in prompts]
    while True:
        active = [seq for seq in sequences if len(seq.output_ids) < num_tokens]
        if not active:
            return [seq.output_ids for seq in sequences]
        chunks = [seq.next_chunk(max_prefill_chunk) for seq in active]
        for seq, chunk, logits_row in zip(active, chunks,
                                          runner.execute(chunks)):
            seq.advance(len(chunk.token_ids), logits_row)


def _assert_matches_hf(hf_exact, generated, prompts, num_tokens):
    model, tokenizer = hf_exact
    for prompt, tokens in zip(prompts, generated):
        assert tokens == hf_greedy(model, tokenizer, prompt, num_tokens), prompt


def _prompt_ids(model, index):
    return model.tokenizer(CAPSTONE_PROMPTS[index]).input_ids


def test_prefill_logits_match_hf(nvcc, tmodel_exact, hf_exact):
    """One prompt, one chunk. The last row must equal the last row of HF."""
    model, _ = hf_exact
    prompt_ids = _prompt_ids(tmodel_exact, 2)
    last_row = ModelRunner(tmodel_exact, 16).execute(
        [SeqChunk(prompt_ids, 0, list(range(16)))])[0]
    with torch.no_grad():
        expected = model(torch.tensor([prompt_ids],
                                      device=model.device)).logits[0, -1]
    torch.testing.assert_close(last_row, expected.float(), rtol=1e-4,
                               atol=1e-3)


def test_batched_decode_matches_hf(nvcc, tmodel_exact, hf_exact):
    """Six ragged prompts in one batch, 24 tokens each, on shuffled
    blocks."""
    generated = _greedy(ModelRunner(tmodel_exact, 64), tmodel_exact.tokenizer,
                        CAPSTONE_PROMPTS, 24)
    _assert_matches_hf(hf_exact, generated, CAPSTONE_PROMPTS, 24)


def test_chunked_prefill_matches_hf(nvcc, tmodel_exact, hf_exact):
    """Prefill in chunks of 3 tokens, next to the decodes of other
    sequences.

    A causal mask on the wrong corner passes the first chunk and breaks
    every later one. is_causal=True does exactly that."""
    generated = _greedy(ModelRunner(tmodel_exact, 64), tmodel_exact.tokenizer,
                        CAPSTONE_PROMPTS, 12, max_prefill_chunk=3)
    _assert_matches_hf(hf_exact, generated, CAPSTONE_PROMPTS, 12)


def test_logits_come_back_in_the_order_of_the_chunks(nvcc, tmodel_exact):
    """Decodes go first inside the step. The caller must not see that."""
    runner = ModelRunner(tmodel_exact, 32)
    first_ids = _prompt_ids(tmodel_exact, 0)
    second_ids = _prompt_ids(tmodel_exact, 1)
    runner.execute([SeqChunk(second_ids, 0, [4, 5])])   # second is cached now
    prefill = SeqChunk(first_ids, 0, [0, 1])
    decode = SeqChunk([7], len(second_ids), [4, 5])
    prefill_alone = runner.execute([prefill])[0]
    decode_alone = runner.execute([decode])[0]
    mixed = runner.execute([prefill, decode])
    torch.testing.assert_close(mixed[0], prefill_alone, **LOGIT_TOLERANCE)
    torch.testing.assert_close(mixed[1], decode_alone, **LOGIT_TOLERANCE)


def test_other_blocks_are_untouched(nvcc, tmodel_exact):
    """A step writes only the slots of its own tokens."""
    runner = ModelRunner(tmodel_exact, 16)
    runner.execute([SeqChunk(_prompt_ids(tmodel_exact, 0), 0, [3])])
    before = [(keys[3].clone(), values[3].clone())
              for keys, values in runner.kv_caches]
    runner.execute([SeqChunk(_prompt_ids(tmodel_exact, 1), 0, [9, 2])])
    for (keys, values), (old_keys, old_values) in zip(runner.kv_caches, before):
        assert torch.equal(keys[3], old_keys)
        assert torch.equal(values[3], old_values)


def test_copy_block_copies_every_layer(nvcc, tmodel_exact):
    runner = ModelRunner(tmodel_exact, 8)
    runner.execute([SeqChunk(_prompt_ids(tmodel_exact, 0), 0, [1])])
    runner.copy_block(1, 6)
    for keys, values in runner.kv_caches:
        assert torch.equal(keys[6], keys[1])
        assert torch.equal(values[6], values[1])


def test_it_runs_your_kernels(nvcc, tmodel_exact, monkeypatch):
    """The decodes go through your stage 08c kernel, one launch for each
    layer, and every token goes through your stage 08 write."""
    calls = {"write": 0, "attention": 0}

    def counted(name, function):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(s21, "write_kv_cuda",
                        counted("write", s21.write_kv_cuda))
    monkeypatch.setattr(s21, "paged_attention_split",
                        counted("attention", s21.paged_attention_split))
    runner = ModelRunner(tmodel_exact, 16)
    prompt_ids = _prompt_ids(tmodel_exact, 0)
    runner.execute([SeqChunk(prompt_ids, 0, [0, 1])])
    calls.update(write=0, attention=0)
    runner.execute([SeqChunk([5], len(prompt_ids), [0, 1]),
                    SeqChunk([6], len(prompt_ids), [0, 1])])
    num_layers = tmodel_exact.config.num_layers
    assert calls == {"write": num_layers, "attention": num_layers}, calls


def test_bf16_decode_is_sane(nvcc, tmodel, hf):
    """bf16 does not match token for token. It must still make sense: the
    first token agrees with HF, and no logit is NaN."""
    logits = ModelRunner(tmodel, 64).execute(
        [SeqChunk(_prompt_ids(tmodel, 0), 0, [0, 1])])
    assert torch.isfinite(logits).all()
    prompts = CAPSTONE_PROMPTS[:3]
    generated = _greedy(ModelRunner(tmodel, 64), tmodel.tokenizer, prompts, 1)
    _assert_matches_hf(hf, generated, prompts, 1)
