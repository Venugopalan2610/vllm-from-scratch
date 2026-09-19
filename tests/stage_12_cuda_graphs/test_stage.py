"""Stage 12 - CUDA graphs for the decode step.

The spec is in app/s12_cudagraph.py.
"""

import pytest
import torch

from app.s12_cudagraph import CUDAGraphRunner
from tests.helpers import bench_ms

WIDTH = 512


def _random_batch(batch_size, device):
    return torch.randn(batch_size, WIDTH, device=device, dtype=torch.float16)


@pytest.fixture(scope="module")
def stack(device):
    """A launch-bound load: many small kernels, little arithmetic.

    It has the shape of decode on purpose. If each kernel did much work, the
    launch overhead would not be important, and graphs would give nothing.
    """
    torch.manual_seed(0)
    layers = [torch.nn.Linear(WIDTH, WIDTH, device=device, dtype=torch.float16)
              for _ in range(32)]

    def forward(inputs):
        for layer in layers:
            inputs = torch.relu(layer(inputs))
        return inputs

    return forward


@pytest.fixture(scope="module")
def runner(stack, device):
    graph_runner = CUDAGraphRunner(stack, buckets=(1, 2, 4, 8))
    graph_runner.capture(lambda batch_size: (
        torch.zeros(batch_size, WIDTH, device=device, dtype=torch.float16),))
    return graph_runner


def test_captures_one_graph_per_bucket(runner):
    assert runner.num_captured == 4


def test_bucket_selection():
    graph_runner = CUDAGraphRunner(lambda inputs: inputs, buckets=(1, 2, 4, 8))
    assert graph_runner.bucket_for(1) == 1
    assert graph_runner.bucket_for(3) == 4
    assert graph_runner.bucket_for(5) == 8
    assert graph_runner.bucket_for(8) == 8
    with pytest.raises(ValueError):
        graph_runner.bucket_for(9)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_replay_matches_eager_exactly(runner, stack, device, batch_size):
    """The same kernels run in the same order. The result must be the same,
    bit for bit."""
    inputs = _random_batch(batch_size, device)
    torch.testing.assert_close(runner.run(inputs), stack(inputs), rtol=0,
                               atol=0)


@pytest.mark.parametrize("batch_size", [1, 3, 5, 7])
def test_padded_batches_are_correct(runner, stack, device, batch_size):
    """Batch 3 runs on the bucket-4 graph, and it must still be exact."""
    inputs = _random_batch(batch_size, device)
    output = runner.run(inputs)
    assert output.shape[0] == batch_size, (
        "cut the output back to the real batch size")
    torch.testing.assert_close(output, stack(inputs), rtol=0, atol=0)


def test_output_is_not_a_view_of_the_static_buffer(runner, device):
    """The usual graph bug: return a view, and the next replay changes it.

    In a server, the tokens of one request then show in the response of
    another. Few bugs are worse.
    """
    first_output = runner.run(_random_batch(2, device))
    snapshot = first_output.clone()
    runner.run(_random_batch(2, device))     # overwrites the static output
    torch.testing.assert_close(first_output, snapshot, rtol=0, atol=0)


def test_stale_padding_cannot_leak_nans(runner, stack, device):
    """Poison the padding rows, then run a smaller batch through the
    bucket."""
    runner.run(torch.full((8, WIDTH), float("nan"), device=device,
                          dtype=torch.float16))       # NaN in the static buffers
    inputs = _random_batch(3, device)
    output = runner.run(inputs)
    assert torch.isfinite(output).all(), (
        "NaNs from the padding rows of the replay before got into a real "
        "result. Put zeros in the padding in run().")
    torch.testing.assert_close(output, stack(inputs), rtol=0, atol=0)


def test_graphs_are_faster_than_eager(runner, stack, device):
    """This is the point. You removed the Python and the launches, and the
    clock must show it."""
    rows = []
    for batch_size in (1, 2, 4, 8):
        inputs = _random_batch(batch_size, device)
        eager_ms = bench_ms(lambda: stack(inputs), iters=200, warmup=20)
        graph_ms = bench_ms(lambda: runner.run(inputs), iters=200, warmup=20)
        rows.append((batch_size, eager_ms, graph_ms))

    print(f"\n  {'batch':>6} {'eager':>10} {'graph':>10} {'speedup':>9}")
    for batch_size, eager_ms, graph_ms in rows:
        print(f"  {batch_size:>6} {eager_ms:>8.3f}ms {graph_ms:>8.3f}ms "
              f"{eager_ms / graph_ms:>8.2f}x")

    worst = min(eager_ms / graph_ms for _, eager_ms, graph_ms in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x in the worst case. The launch overhead must be "
        "most of the time of 32 small layers. Make sure that capture() "
        "captured.")
    print(f"\n  \033[1mWorst case {worst:.2f}x faster.\033[0m")
    print("  \033[2mNo arithmetic changed. You removed CPU work that the GPU")
    print("  waited on. That is the gap that stage 03 measured between the")
    print("  decode time and its roofline floor.\033[0m")
