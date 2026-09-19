"""Stage 23 - the real decode step, captured.

The spec is in app/s23_graphs.py. The gates are ratios on your own card: the
graphed step against the eager step, and against the weight-read floor.
"""

import torch

import cudalib
from app.s21_paged_runner import ModelRunner, SeqChunk
from app.s22_engine import LLMEngine
from app.s23_graphs import GraphedModelRunner
from tests.helpers import CAPSTONE_PROMPTS, hf_greedy

NEXT_TOKEN = 11


def _sequences(tokenizer, num_seqs, blocks_per_seq=4):
    """num_seqs sequences, each with a real prefix: [(prompt_ids, block_ids)].
    Block 0 stays free."""
    sequences = []
    for index in range(num_seqs):
        prompt = CAPSTONE_PROMPTS[index % len(CAPSTONE_PROMPTS)]
        first_block = 1 + index * blocks_per_seq
        sequences.append((tokenizer(prompt).input_ids,
                          list(range(first_block, first_block + blocks_per_seq))))
    return sequences


def _prefill(runner, sequences):
    runner.execute([SeqChunk(prompt_ids, 0, block_ids)
                    for prompt_ids, block_ids in sequences])


def _decode_chunks(sequences):
    return [SeqChunk([NEXT_TOKEN], len(prompt_ids), block_ids)
            for prompt_ids, block_ids in sequences]


def _captured(model, num_blocks, buckets):
    runner = GraphedModelRunner(model, num_blocks, max_model_len=256,
                                buckets=buckets)
    runner.capture()
    return runner


def test_bucket_for(nvcc, tmodel_exact):
    runner = GraphedModelRunner(tmodel_exact, 8, buckets=(1, 2, 4, 8))
    assert [runner.bucket_for(size) for size in (1, 2, 3, 5, 8, 9)] == [
        1, 2, 4, 8, 8, None]


def test_replay_equals_eager(nvcc, tmodel_exact):
    """Not bit for bit. The graph runs a padded batch of 4 and a block table
    as wide as max_model_len. So cuBLAS selects a different algorithm for
    the matmuls, and stage 08c selects a different number of splits. In fp32
    the two still agree far inside the margin that greedy decode needs."""
    sequences = _sequences(tmodel_exact.tokenizer, 3)
    graphed = _captured(tmodel_exact, 32, buckets=(4,))
    eager = ModelRunner(tmodel_exact, 32)
    _prefill(graphed, sequences)
    _prefill(eager, sequences)
    chunks = _decode_chunks(sequences)
    replayed = graphed.execute(chunks)
    assert graphed.replays == 1
    torch.testing.assert_close(replayed, eager.execute(chunks), rtol=1e-4,
                               atol=1e-4)


def test_padding_rows_do_not_touch_block_zero(nvcc, tmodel_exact):
    """Three sequences in a bucket of four. One of them owns block 0. The
    padding row must not write into it."""
    tokenizer = tmodel_exact.tokenizer
    runner = _captured(tmodel_exact, 32, buckets=(4,))
    owner_ids = tokenizer(CAPSTONE_PROMPTS[0]).input_ids
    sequences = [(owner_ids, [0, 20])] + _sequences(tokenizer, 2)
    _prefill(runner, sequences)
    before = [(keys[0].clone(), values[0].clone())
              for keys, values in runner.kv_caches]
    runner.execute(_decode_chunks(sequences))
    written_slot = len(owner_ids)          # the one slot that the step can write
    for (keys, values), (old_keys, old_values) in zip(runner.kv_caches, before):
        old_keys[:, written_slot] = keys[0, :, written_slot]
        old_values[:, written_slot] = values[0, :, written_slot]
        assert torch.equal(keys[0], old_keys)
        assert torch.equal(values[0], old_values)


def test_prefills_and_big_batches_run_eager(nvcc, tmodel_exact):
    runner = _captured(tmodel_exact, 64, buckets=(1, 2))
    sequences = _sequences(tmodel_exact.tokenizer, 3)
    _prefill(runner, sequences)
    assert runner.eager_steps == 1 and runner.replays == 0
    runner.execute(_decode_chunks(sequences))     # 3 rows, the largest bucket is 2
    assert runner.eager_steps == 2 and runner.replays == 0


def test_engine_with_graphs_matches_hf(nvcc, tmodel_exact, hf_exact):
    model, tokenizer = hf_exact
    output_lens = [30, 8, 45, 12]
    runner = _captured(tmodel_exact, 64, buckets=(1, 2, 4))
    engine = LLMEngine(tmodel_exact, 64, runner=runner, max_num_seqs=4)
    for rid, num_tokens in enumerate(output_lens):
        engine.add_request(rid, CAPSTONE_PROMPTS[rid], num_tokens,
                           ignore_eos=True)
    outputs = engine.run_to_completion()
    assert runner.replays > sum(output_lens) // 4
    for rid, num_tokens in enumerate(output_lens):
        assert outputs[rid] == hf_greedy(model, tokenizer,
                                         CAPSTONE_PROMPTS[rid], num_tokens)


def _batch_one_step(model, runner):
    """Prefill one sequence on the runner. -> the chunks of a decode step."""
    sequences = _sequences(model.tokenizer, 1)
    _prefill(runner, sequences)
    return _decode_chunks(sequences)


def test_batch_one_is_much_faster_than_eager(nvcc, tmodel):
    """Eager and graphed, with the rounds interleaved. The gain depends on
    how fast your CPU issues kernels: a slow host CPU gains much more than
    1.2x."""
    graphed = _captured(tmodel, 16, buckets=(1,))
    eager = ModelRunner(tmodel, 16)
    chunks = _batch_one_step(tmodel, graphed)
    _batch_one_step(tmodel, eager)
    eager_ms, graphed_ms = cudalib.compare_ms(
        lambda: eager.execute(chunks), lambda: graphed.execute(chunks),
        rounds=3, iters=30, warmup=5)
    print(f"\n  batch-1 step: eager {eager_ms:.2f} ms, graphed "
          f"{graphed_ms:.2f} ms, {eager_ms / graphed_ms:.1f}x")
    assert eager_ms / graphed_ms >= 1.2


def test_batch_one_reaches_the_weight_floor(nvcc, tmodel):
    """A decode step at batch 1 must read every weight one time. That sets a
    floor: weight bytes / read bandwidth. Get within 2x of it."""
    runner = _captured(tmodel, 16, buckets=(1,))
    chunks = _batch_one_step(tmodel, runner)
    step_ms = cudalib.bench_ms(lambda: runner.execute(chunks), iters=30,
                               warmup=5, best_of=3)
    floor_ms = tmodel.weight_bytes() / cudalib.read_bandwidth() * 1e3
    print(f"\n  batch-1 step {step_ms:.2f} ms against a floor of "
          f"{floor_ms:.2f} ms = {100 * floor_ms / step_ms:.0f}% of the roof")
    assert floor_ms / step_ms >= 0.5
