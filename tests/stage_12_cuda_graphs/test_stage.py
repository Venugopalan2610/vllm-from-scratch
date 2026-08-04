"""Stage 12 - CUDA graphs for the decode step.

Spec in app/s12_cudagraph.py.
"""

import pytest
import torch

from app.s12_cudagraph import CUDAGraphRunner
from tests.helpers import bench_ms


@pytest.fixture(scope="module")
def stack(dev):
    """A launch-bound workload: many tiny kernels, little arithmetic.

    This is deliberately shaped like decode -- if the work per kernel were
    large, launch overhead would not matter and graphs would buy nothing.
    """
    torch.manual_seed(0)
    layers = [torch.nn.Linear(512, 512, device=dev, dtype=torch.float16)
              for _ in range(32)]

    def fn(x):
        for l in layers:
            x = torch.relu(l(x))
        return x

    return fn


@pytest.fixture(scope="module")
def runner(stack, dev):
    r = CUDAGraphRunner(stack, buckets=(1, 2, 4, 8))
    r.capture(lambda bs: (torch.zeros(bs, 512, device=dev, dtype=torch.float16),))
    return r


def test_captures_one_graph_per_bucket(runner):
    assert runner.num_captured == 4


def test_bucket_selection():
    r = CUDAGraphRunner(lambda x: x, buckets=(1, 2, 4, 8))
    assert r.bucket_for(1) == 1
    assert r.bucket_for(3) == 4
    assert r.bucket_for(5) == 8
    assert r.bucket_for(8) == 8
    with pytest.raises(ValueError):
        r.bucket_for(9)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
def test_replay_matches_eager_exactly(runner, stack, dev, bs):
    """Same kernels in the same order -- this should be bit-identical."""
    x = torch.randn(bs, 512, device=dev, dtype=torch.float16)
    want = stack(x)
    got = runner.run(x)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [1, 3, 5, 7])
def test_padded_batches_are_correct(runner, stack, dev, bs):
    """Batch 3 runs on the bucket-4 graph and must still be exact."""
    x = torch.randn(bs, 512, device=dev, dtype=torch.float16)
    got = runner.run(x)
    assert got.shape[0] == bs, "output must be sliced back to the real batch size"
    torch.testing.assert_close(got, stack(x), rtol=0, atol=0)


def test_output_is_not_a_view_of_the_static_buffer(runner, dev):
    """The classic graph bug: hand back a view, and the next replay mutates it.

    In a server this shows up as one request's tokens appearing in another's
    response, which is about as bad as bugs get.
    """
    a = torch.randn(2, 512, device=dev, dtype=torch.float16)
    b = torch.randn(2, 512, device=dev, dtype=torch.float16)
    out_a = runner.run(a)
    snapshot = out_a.clone()
    runner.run(b)                      # second replay overwrites static output
    torch.testing.assert_close(out_a, snapshot, rtol=0, atol=0)


def test_stale_padding_cannot_leak_nans(runner, stack, dev):
    """Poison the padding rows, then run a smaller batch through the bucket."""
    big = torch.full((8, 512), float("nan"), device=dev, dtype=torch.float16)
    runner.run(big)                    # fills static buffers with NaN
    x = torch.randn(3, 512, device=dev, dtype=torch.float16)
    got = runner.run(x)
    assert torch.isfinite(got).all(), (
        "NaNs from the previous replay's padding rows leaked into a real "
        "result -- zero the padding in run()"
    )
    torch.testing.assert_close(got, stack(x), rtol=0, atol=0)


def test_graphs_are_faster_than_eager(runner, stack, dev):
    """The point. Deleting the Python and the launches should be visible."""
    rows = []
    for bs in (1, 2, 4, 8):
        x = torch.randn(bs, 512, device=dev, dtype=torch.float16)
        eager = bench_ms(lambda: stack(x), iters=200, warmup=20)
        graph = bench_ms(lambda: runner.run(x), iters=200, warmup=20)
        rows.append((bs, eager, graph))

    print(f"\n  {'batch':>6} {'eager':>10} {'graph':>10} {'speedup':>9}")
    for bs, e, g in rows:
        print(f"  {bs:>6} {e:>8.3f}ms {g:>8.3f}ms {e / g:>8.2f}x")

    worst = min(e / g for _, e, g in rows)
    assert worst > 1.5, (
        f"only {worst:.2f}x at best. 32 tiny layers should be dominated by "
        "launch overhead; check that capture() actually captured."
    )
    print(f"\n  \033[1mWorst case {worst:.2f}x faster.\033[0m")
    print("  \033[2mNo arithmetic changed. You deleted CPU work that the GPU")
    print("  was waiting on -- which is exactly the gap stage 03 measured")
    print("  between decode's 7.8 ms/token and its 3.14 ms roofline floor.\033[0m")
