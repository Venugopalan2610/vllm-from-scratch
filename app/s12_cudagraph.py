"""Stage 12 - CUDA graphs for the decode step.

`./vc lore 12` for the insight. `./vc test 12` to check yourself.

Stage 03 measured the problem: decode ran at ~7.8 ms/token against a 3.14 ms
roofline floor. That 2.5x gap is not arithmetic and it is not memory -- it is
Python, and it is thousands of individual kernel launches per step.

A CUDA graph records an entire sequence of launches once and replays it as a
single unit. The CPU stops being in the loop.

The catch: a graph records EXACT pointers and EXACT shapes. So you must

  - allocate static input/output buffers and reuse them forever
  - copy new data INTO those buffers rather than passing new tensors
  - capture one graph per batch size you intend to run, and pad up to the
    nearest captured size ("bucketing")

Real servers capture at 1, 2, 4, 8, 16, ... and fall back to eager above the
largest bucket. Prefill is never graphed -- its shapes change every request.
"""

import torch


class CUDAGraphRunner:
    """Required attributes:

        .buckets        sorted batch sizes that have captured graphs
        .num_captured   how many graphs exist
        .replays        how many times run() replayed a graph

    Required methods:
        bucket_for(batch_size) -> int      smallest bucket >= batch_size;
                                           raises ValueError if too large
        capture(make_inputs)               make_inputs(bs) -> tuple of tensors
        run(*inputs) -> Tensor             pad, replay, slice back down
        run_eager(*inputs) -> Tensor       the un-graphed path, for comparison
    """

    def __init__(self, fn, buckets=(1, 2, 4, 8)):
        raise NotImplementedError("stage 12: implement CUDAGraphRunner")

    def capture(self, make_inputs):
        """Capture one graph per bucket.

        The warmup is not optional. Run fn() a few times on a SIDE STREAM
        before capturing:

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self.fn(*inputs)
            torch.cuda.current_stream().wait_stream(s)

        Lazy one-time work (cuBLAS handle creation, algorithm autotuning,
        allocator growth) must happen BEFORE capture, or it gets baked into
        the graph or makes capture fail outright.

        Then:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.fn(*inputs)

        Keep `inputs` and `out` -- those exact tensors ARE the graph's I/O.
        """
        raise NotImplementedError

    def run(self, *inputs):
        """Copy into the static buffers, replay, return a slice of the output.

        Two things that will bite you:
          - Zero the padding rows. A bucket-4 graph run with batch 3 still
            computes row 3 from whatever was left there last time. It does not
            affect your answer, but NaNs and infs from stale data will.
          - Return a .clone() of the output slice. The static output buffer is
            overwritten by the very next replay, so handing out a view means
            the caller's data silently changes underneath them.
        """
        raise NotImplementedError

    def run_eager(self, *inputs):
        raise NotImplementedError
