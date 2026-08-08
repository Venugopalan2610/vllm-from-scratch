"""Reference solution, stage 12 (jax) - bucketed AOT compilation."""

import jax
import jax.numpy as jnp


class CompiledDecode:
    def __init__(self, fn, buckets=(1, 2, 4, 8), donate=()):
        self.fn = fn
        self.buckets = sorted(buckets)
        self.donate = tuple(donate)
        self.exe = {}
        self.eager_calls = 0
        self.replays = 0

    @property
    def num_compiled(self):
        return len(self.exe)

    def bucket_for(self, batch_size):
        for b in self.buckets:
            if b >= batch_size:
                return b
        raise ValueError(
            f"batch {batch_size} exceeds the largest bucket {self.buckets[-1]}"
        )

    def compile(self, make_inputs):
        """One executable per bucket, built at startup where nobody waits."""
        for bs in self.buckets:
            inputs = make_inputs(bs)
            jitted = jax.jit(self.fn, donate_argnums=self.donate)
            self.exe[bs] = jitted.lower(*inputs).compile()

    def run(self, *inputs):
        bs = inputs[0].shape[0]
        bucket = self.bucket_for(bs)
        padded = tuple(self._pad(x, bs, bucket) for x in inputs)
        out = self.exe[bucket](*padded)
        self.replays += 1
        return jax.tree.map(lambda y: y[:bs], out)

    def run_eager(self, *inputs):
        self.eager_calls += 1
        return self.fn(*inputs)

    @staticmethod
    def _pad(x, bs, bucket):
        # Only the BATCHED arguments get padded. Weights ride along unchanged,
        # and padding them would be nonsense (and, for a (512, 512) weight in a
        # bucket-4 call, a negative pad width).
        if x.shape[0] != bs or bs == bucket:
            return x
        # Zeros, deterministically. Padding rows are real rows to the model.
        pad = jnp.zeros((bucket - bs,) + x.shape[1:], x.dtype)
        return jnp.concatenate([x, pad], axis=0)
