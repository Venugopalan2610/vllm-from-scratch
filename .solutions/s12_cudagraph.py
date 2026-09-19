"""Reference solution, stage 12 - CUDA graphs for the decode step."""

import torch

WARMUP_CALLS = 3


def warm_up(function, inputs):
    """Run on a side stream before the capture. A lazy initialization (a
    cuBLAS handle, an autotune) inside a capture fails or records garbage."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(WARMUP_CALLS):
            function(*inputs)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()


class CUDAGraphRunner:
    def __init__(self, fn, buckets=(1, 2, 4, 8)):
        self.fn = fn
        self.buckets = sorted(buckets)
        self.graphs = {}     # bucket -> (graph, static inputs, static output)
        self.eager_calls = 0
        self.replays = 0

    @property
    def num_captured(self):
        return len(self.graphs)

    def bucket_for(self, batch_size):
        for bucket in self.buckets:
            if bucket >= batch_size:
                return bucket
        raise ValueError(f"batch {batch_size} exceeds largest captured bucket "
                         f"{self.buckets[-1]}")

    def capture(self, make_inputs):
        for bucket in self.buckets:
            static_inputs = make_inputs(bucket)
            warm_up(self.fn, static_inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self.fn(*static_inputs)
            self.graphs[bucket] = (graph, static_inputs, static_output)

    def run(self, *inputs):
        batch_size = inputs[0].shape[0]
        graph, static_inputs, static_output = self.graphs[
            self.bucket_for(batch_size)]
        for static_input, value in zip(static_inputs, inputs):
            static_input[:batch_size].copy_(value)
            static_input[batch_size:].zero_()   # a padding row is deterministic
        graph.replay()
        self.replays += 1
        return static_output[:batch_size].clone()

    def run_eager(self, *inputs):
        self.eager_calls += 1
        return self.fn(*inputs)
