"""Reference solution, stage 12 - CUDA graphs for the decode step."""

import torch


class CUDAGraphRunner:
    def __init__(self, fn, buckets=(1, 2, 4, 8)):
        self.fn = fn
        self.buckets = sorted(buckets)
        self.graphs = {}
        self.eager_calls = 0
        self.replays = 0

    @property
    def num_captured(self):
        return len(self.graphs)

    def bucket_for(self, batch_size):
        for b in self.buckets:
            if b >= batch_size:
                return b
        raise ValueError(
            f"batch {batch_size} exceeds largest captured bucket "
            f"{self.buckets[-1]}"
        )

    def capture(self, make_inputs):
        for bs in self.buckets:
            inputs = make_inputs(bs)

            # Warm up on a side stream first. Capture will fail or record
            # garbage if lazy init (cuBLAS handles, autotuning) happens inside.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.fn(*inputs)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.fn(*inputs)
            self.graphs[bs] = (g, inputs, out)

    def run(self, *inputs):
        bs = inputs[0].shape[0]
        bucket = self.bucket_for(bs)
        g, static_in, static_out = self.graphs[bucket]

        for dst, src in zip(static_in, inputs):
            dst[:bs].copy_(src)
            if bucket > bs:
                dst[bs:].zero_()      # padding rows must be deterministic

        g.replay()
        self.replays += 1
        return static_out[:bs].clone()

    def run_eager(self, *inputs):
        self.eager_calls += 1
        return self.fn(*inputs)
