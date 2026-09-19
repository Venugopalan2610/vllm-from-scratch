"""Stage 24 - quantized weights in the engine.

The spec is in app/s24_quantized.py. The accuracy is the fidelity of stage 18
on an ISL/OSL workload: the int8 engine against the bf16 engine, at each
output position. The speed is the graphed decode step, int8 against bf16, on
your card, one after the other.
"""

from contextlib import contextmanager

import pytest
import torch

import cudalib
from app.s21_paged_runner import SeqChunk
from app.s23_graphs import GraphedModelRunner
from app.s18_quantization import fidelity
from app.s24_quantized import MAX_ROWS, Int8Linear, continuation_logits, quantize_model
from tests.helpers import CAPSTONE_PROMPTS, ISL, MODEL, OSL, hf_workload
from tvllm import LayerWeights, Model



@pytest.fixture(scope="module")
def qmodel(hf):
    """A second tvllm model on the same HF weights, then quantized."""
    hf_model, tokenizer = hf
    model = Model(hf_model, tokenizer, MODEL)
    quantize_model(model)
    return model


def test_int8_linear_matches_the_bf16_matmul(nvcc, device):
    torch.manual_seed(0)
    weight = torch.randn(512, 256, device=device, dtype=torch.bfloat16) * 0.05
    layer = Int8Linear(weight, max_rows=4)
    for num_rows in (1, 3, 4, 9):
        inputs = torch.randn(num_rows, 256, device=device, dtype=torch.bfloat16)
        expected = inputs.float() @ weight.float().t()
        error = (layer(inputs).float() - expected).norm() / expected.norm()
        assert error < 0.02, (num_rows, float(error))


def test_both_paths_are_taken(nvcc, device, monkeypatch):
    """At or below max_rows your GEMV. Above it the bf16 weight."""
    import app.s24_quantized as s24
    gemv_calls = []
    real_gemv = s24.gemv_int8

    def counting_gemv(*args):
        gemv_calls.append(args)
        return real_gemv(*args)

    monkeypatch.setattr(s24, "gemv_int8", counting_gemv)
    weight = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
    layer = Int8Linear(weight, max_rows=2)
    layer(torch.randn(2, 64, device=device, dtype=torch.bfloat16))
    layer(torch.randn(3, 64, device=device, dtype=torch.bfloat16))
    assert len(gemv_calls) == 1


def test_every_matmul_is_replaced(nvcc, qmodel):
    assert all(not isinstance(getattr(layer, name), torch.Tensor)
               for layer in qmodel.layers for name in LayerWeights.MATMULS)
    assert not isinstance(qmodel.lm_head, torch.Tensor)


def test_weight_bytes_are_half(nvcc, tmodel, qmodel):
    ratio = qmodel.weight_bytes() / tmodel.weight_bytes()
    print(f"\n  int8 reads {100 * ratio:.0f}% of the bf16 bytes at a small batch")
    assert 0.48 < ratio < 0.55


@contextmanager
def dequantized_weights(model):
    """A prefill has more rows than MAX_ROWS, so it takes the bf16 path and
    measures nothing. Put the dequantized int8 weights in place for it."""
    saved = [(layer, name, getattr(layer, name))
             for layer in model.layers for name in LayerWeights.MATMULS]
    saved.append((model, "lm_head", model.lm_head))
    for owner, name, weight in saved:
        setattr(owner, name, weight.dequantized_weight())
    try:
        yield model
    finally:
        for owner, name, weight in saved:
            setattr(owner, name, weight)


# Stage 18 measured the gate. The lm_head is int8 here too, so the engine
# moves a little more than the model of stage 18.
TOP1_FLOOR = 0.90
KL_CEILING = 0.015


def test_int8_keeps_the_choices_of_the_engine(nvcc, hf, tmodel, qmodel):
    sequences = hf_workload(*hf)
    before = torch.cat([continuation_logits(tmodel, ids, ISL) for ids in sequences])
    with dequantized_weights(qmodel):
        after = torch.cat([continuation_logits(qmodel, ids, ISL) for ids in sequences])
    result = fidelity(before, after)
    print(f"\n  ISL {ISL}, OSL {OSL}: the same top token at "
          f"{100 * result.top1_agreement:.1f}% of the output positions, "
          f"mean KL {result.mean_kl:.4f} nats")
    assert result.top1_agreement >= TOP1_FLOOR and result.mean_kl <= KL_CEILING


def _step_ms(model, batch_size):
    """One graphed decode step for batch_size sequences, in ms."""
    runner = GraphedModelRunner(model, 16 * batch_size + 16, max_model_len=256,
                                buckets=(batch_size,))
    runner.capture()
    prompt_ids = model.tokenizer(CAPSTONE_PROMPTS[0]).input_ids
    block_lists = [list(range(i * 16, i * 16 + 16)) for i in range(batch_size)]
    runner.execute([SeqChunk(prompt_ids, 0, blocks) for blocks in block_lists])
    decodes = [SeqChunk([11], len(prompt_ids), blocks) for blocks in block_lists]
    return cudalib.bench_ms(lambda: runner.execute(decodes), iters=30, best_of=3)


def _set_max_rows(model, max_rows):
    for layer in model.layers:
        for name in LayerWeights.MATMULS:
            getattr(layer, name).max_rows = max_rows


def test_the_crossover(nvcc, tmodel, qmodel):
    """Informational: the table that sets MAX_ROWS. Measured in the model."""
    print(f"\n  {'batch':>5} {'bf16':>9} {'int8 GEMV':>10}")
    _set_max_rows(qmodel, 8)
    for batch_size in (1, 2, 4, 8):
        int8_ms = _step_ms(qmodel, batch_size)
        print(f"  {batch_size:>5} {_step_ms(tmodel, batch_size):>7.2f}ms "
              f"{int8_ms:>8.2f}ms")
    _set_max_rows(qmodel, MAX_ROWS)
    print(f"  MAX_ROWS is {MAX_ROWS}.")


def test_batch_one_is_faster(nvcc, tmodel, qmodel):
    bf16_ms, int8_ms = _step_ms(tmodel, 1), _step_ms(qmodel, 1)
    print(f"\n  batch-1 step: bf16 {bf16_ms:.2f} ms, int8 {int8_ms:.2f} ms = "
          f"{bf16_ms / int8_ms:.2f}x")
    assert bf16_ms / int8_ms >= 1.3


def test_a_big_batch_is_not_slower(nvcc, tmodel, qmodel):
    """Above MAX_ROWS the dispatch takes the bf16 weight, so the step loses
    nothing."""
    bf16_ms, int8_ms = _step_ms(tmodel, 32), _step_ms(qmodel, 32)
    assert bf16_ms / int8_ms >= 0.93
