"""Stage 24b - the KV cache in FP8.

The spec is in app/s24b_kv_fp8.py and app/cuda/s24b_kv_fp8.cu. Dense
attention on the same FP8 values is the oracle of the kernel. The
perplexity through the decode kernel and the speed at long context check the
engine.
"""

import math

import torch
import torch.nn.functional as F

import cudalib
from app.s21_paged_runner import SeqChunk
from app.s23_graphs import GraphedModelRunner
from app.s24b_kv_fp8 import (
    FP8_MAX,
    Fp8GraphedModelRunner,
    KVScale,
    calibrate_kv_scales,
    kv_bytes_per_token_fp8,
    paged_attention_fp8,
    write_kv_fp8,
)
from tests.helpers import PROSE_SAMPLE

BLOCK_SIZE = 16
NEXT_TOKEN = 11


def _empty_fp8_cache(num_blocks, num_kv_heads, head_dim, device):
    shape = (num_blocks, num_kv_heads, BLOCK_SIZE, head_dim)
    return (torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device),
            torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device))


def _prose_ids(model, num_tokens):
    return model.tokenizer(PROSE_SAMPLE).input_ids[:num_tokens]


def test_write_then_read_back(nvcc, device):
    """A value comes back as value / scale, rounded to e4m3."""
    num_kv_heads, head_dim = 2, 64
    key_cache, value_cache = _empty_fp8_cache(4, num_kv_heads, head_dim, device)
    key = torch.randn(5, num_kv_heads, head_dim, device=device,
                      dtype=torch.bfloat16)
    slots = torch.tensor([0, 17, 18, 40, -1], device=device)
    scale = KVScale(key=0.01, value=0.02)
    write_kv_fp8(key_cache, value_cache, key, torch.randn_like(key), slots,
                 scale)

    slot_17 = key_cache[1, :, 1].float() * scale.key
    torch.testing.assert_close(slot_17, key[1].float(), rtol=0.07, atol=0.02)
    assert key_cache.float().abs().max() <= FP8_MAX
    assert torch.count_nonzero(value_cache[0, :, 1].float()) == 0, (
        "slot 1 was never a target. A slot of -1 must be skipped.")


def _dense_context(cache, block_table, context_len, scale):
    """The dequantized (kv_heads, context_len, head_dim) of one sequence."""
    num_kv_heads, head_dim = cache.shape[1], cache.shape[3]
    rows = cache[block_table.long()].transpose(0, 1)
    return rows.reshape(num_kv_heads, -1, head_dim)[:, :context_len].float() * scale


def test_attention_matches_dense_on_the_same_values(nvcc, device):
    """Dense attention on the dequantized cache is the oracle."""
    num_seqs, num_heads, num_kv_heads, head_dim = 3, 8, 4, 128
    context_len = 200
    blocks_per_seq = math.ceil(context_len / BLOCK_SIZE)
    key_cache, value_cache = _empty_fp8_cache(num_seqs * blocks_per_seq,
                                              num_kv_heads, head_dim, device)
    scale = KVScale(key=0.02, value=0.03)
    for seq in range(num_seqs):
        first_slot = seq * blocks_per_seq * BLOCK_SIZE
        key = torch.randn(context_len, num_kv_heads, head_dim, device=device,
                          dtype=torch.bfloat16)
        write_kv_fp8(key_cache, value_cache, key, torch.randn_like(key),
                     torch.arange(context_len, device=device) + first_slot,
                     scale)
    block_tables = torch.arange(num_seqs * blocks_per_seq, dtype=torch.int32,
                                device=device).view(num_seqs, blocks_per_seq)
    context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32,
                              device=device)
    query = torch.randn(num_seqs, num_heads, head_dim, device=device,
                        dtype=torch.bfloat16)
    output = paged_attention_fp8(query, key_cache, value_cache, block_tables,
                                 context_lens, scale)

    for seq in range(num_seqs):
        expected = F.scaled_dot_product_attention(
            query[seq].float()[:, None],
            _dense_context(key_cache, block_tables[seq], context_len,
                           scale.key),
            _dense_context(value_cache, block_tables[seq], context_len,
                           scale.value),
            enable_gqa=True)[:, 0]
        torch.testing.assert_close(output[seq].float(), expected, rtol=2e-2,
                                   atol=2e-2)


def test_scales_come_from_calibration(nvcc, tmodel):
    scales = calibrate_kv_scales(tmodel, _prose_ids(tmodel, 256))
    assert len(scales) == tmodel.config.num_layers
    assert all(scale.key > 0 and scale.value > 0 for scale in scales)


def test_the_cache_is_one_byte(nvcc, tmodel):
    runner = Fp8GraphedModelRunner(
        tmodel, 8, calibrate_kv_scales(tmodel, _prose_ids(tmodel, 128)),
        max_model_len=128, buckets=(1,))
    key_cache, _ = runner.kv_caches[0]
    assert key_cache.dtype == torch.float8_e4m3fn
    assert kv_bytes_per_token_fp8(tmodel) * 2 == tmodel.kv_bytes_per_token()


def _decode_perplexity(runner, token_ids, prefill_len=16):
    """Prefill a few tokens, then give one token at a time, so that every
    logit comes through the decode kernel."""
    block_ids = list(range(math.ceil(len(token_ids) / BLOCK_SIZE) + 1))
    logits = runner.execute([SeqChunk(token_ids[:prefill_len], 0, block_ids)])
    losses = []
    for position in range(prefill_len, len(token_ids)):
        target = torch.tensor([token_ids[position]], device=logits.device)
        losses.append(F.cross_entropy(logits, target).item())
        logits = runner.execute([SeqChunk([token_ids[position]], position,
                                          block_ids)])
    return math.exp(sum(losses) / len(losses))


def _bf16_runner(model, num_blocks, max_model_len, bucket):
    runner = GraphedModelRunner(model, num_blocks, max_model_len=max_model_len,
                                buckets=(bucket,))
    runner.capture()
    return runner


def _fp8_runner(model, num_blocks, max_model_len, bucket, scales):
    runner = Fp8GraphedModelRunner(model, num_blocks, scales,
                                   max_model_len=max_model_len,
                                   buckets=(bucket,))
    runner.capture()
    return runner


def test_perplexity_barely_moves(nvcc, tmodel):
    token_ids = _prose_ids(tmodel, 300)
    scales = calibrate_kv_scales(tmodel, token_ids[:128])
    bf16_perplexity = _decode_perplexity(_bf16_runner(tmodel, 32, 512, 1),
                                         token_ids)
    fp8_perplexity = _decode_perplexity(
        _fp8_runner(tmodel, 32, 512, 1, scales), token_ids)
    print(f"\n  perplexity bf16 KV {bf16_perplexity:.3f}, FP8 KV "
          f"{fp8_perplexity:.3f} "
          f"({100 * (fp8_perplexity / bf16_perplexity - 1):+.2f}%)")
    assert fp8_perplexity / bf16_perplexity < 1.03


def _long_context_step_ms(runner, num_seqs, context_len):
    blocks_per_seq = context_len // BLOCK_SIZE
    decodes = [SeqChunk([NEXT_TOKEN], context_len - 1,
                        list(range(seq * blocks_per_seq,
                                   (seq + 1) * blocks_per_seq)))
               for seq in range(num_seqs)]
    step_ms = cudalib.bench_ms(lambda: runner.execute(decodes), iters=20,
                               best_of=3)
    runner.kv_caches.clear()          # free the pool before the next runner
    torch.cuda.empty_cache()
    return step_ms


def test_long_context_is_faster(nvcc, tmodel):
    """16 sequences of 2048 tokens: the KV read is three times the weight
    read, so half the KV bytes must show."""
    num_seqs, context_len = 16, 2048
    num_blocks = num_seqs * context_len // BLOCK_SIZE + 8
    max_model_len = context_len + BLOCK_SIZE
    scales = calibrate_kv_scales(tmodel, _prose_ids(tmodel, 128))

    bf16_ms = _long_context_step_ms(
        _bf16_runner(tmodel, num_blocks, max_model_len, num_seqs), num_seqs,
        context_len)
    fp8_ms = _long_context_step_ms(
        _fp8_runner(tmodel, num_blocks, max_model_len, num_seqs, scales),
        num_seqs, context_len)
    print(f"\n  {num_seqs} x {context_len} tokens: bf16 KV {bf16_ms:.1f} ms, "
          f"FP8 KV {fp8_ms:.1f} ms = {bf16_ms / fp8_ms:.2f}x")
    assert bf16_ms / fp8_ms >= 1.2
