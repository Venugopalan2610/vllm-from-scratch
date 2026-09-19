"""Stage 12 - CUDA graphs for the decode step.

`./vc lore 12` for the insight. `./vc test 12` to check yourself.

Stage 03 measured the problem. At batch 1, decode on the HF model takes
about 2.5 times its roofline floor. That gap is not arithmetic, and it is not
memory. It is Python, and thousands of separate kernel launches in each step.

This stage captures a small stack of Linear layers, so that you learn only
the mechanism. Stage 23 captures the real decode step of the real model.

A CUDA graph records a whole sequence of launches one time, and replays it
as one unit. The CPU is not in the loop now.

The condition: a graph records EXACT pointers and EXACT shapes. So you must

  - allocate static input and output buffers, and use them for ever
  - copy new data INTO those buffers, and not give new tensors
  - capture one graph for each batch size that you run, and pad up to the
    nearest captured size ("bucketing")

A real server captures at 1, 2, 4, 8, 16 and more, and it uses the eager
path above the largest bucket. It never captures a prefill. The shapes of a
prefill change with every request.
"""

import torch


class CUDAGraphRunner:
    """Required attributes:

        .buckets        the sorted batch sizes that have captured graphs
        .num_captured   the number of graphs
        .replays        the number of times that run() replayed a graph

    Required methods:
        bucket_for(batch_size) -> int      the smallest bucket >= batch_size.
                                           Raise ValueError if it is too large.
        capture(make_inputs)               make_inputs(bs) -> tuple of tensors
        run(*inputs) -> Tensor             pad, replay, cut back to the batch
        run_eager(*inputs) -> Tensor       the path with no graph, to compare
    """

    def __init__(self, fn, buckets=(1, 2, 4, 8)):
        raise NotImplementedError("stage 12: implement CUDAGraphRunner")

    def capture(self, make_inputs):
        """Capture one graph for each bucket.

        The warmup is necessary. Run fn() a few times on a SIDE STREAM before
        the capture:

            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(3): self.fn(*inputs)
            torch.cuda.current_stream().wait_stream(side_stream)

        Lazy work that occurs one time (a cuBLAS handle, an algorithm
        autotune, an allocator increase) must occur BEFORE the capture. If
        not, the graph records it, or the capture fails.

        Then:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.fn(*inputs)

        Keep `inputs` and `output`. Those exact tensors ARE the inputs and
        the outputs of the graph.
        """
        raise NotImplementedError

    def run(self, *inputs):
        """Copy into the static buffers, replay, and return a slice of the
        output.

        Two traps:
          - Put zeros in the padding rows. A bucket-4 graph at batch 3 still
            computes row 3 from the data that was there before. It does not
            change your answer, but NaNs and infs from old data do.
          - Return a .clone() of the output slice. The next replay overwrites
            the static output buffer. A view changes the data of the caller
            later, with no error.
        """
        raise NotImplementedError

    def run_eager(self, *inputs):
        raise NotImplementedError
