"""Reference solution, stage 12 (jax) - bucketed AOT compilation."""

import jax
import jax.numpy as jnp


def pad_batch(array, batch_size, bucket):
    """Pad only a BATCHED argument. A weight goes through with no change: a
    pad on a (512, 512) weight in a bucket-4 call is a negative pad width."""
    if array.shape[0] != batch_size or batch_size == bucket:
        return array
    # Zeros, so that the result is deterministic. The model sees a padding
    # row as a real row.
    zeros = jnp.zeros((bucket - batch_size,) + array.shape[1:], array.dtype)
    return jnp.concatenate([array, zeros], axis=0)


class CompiledDecode:
    def __init__(self, fn, buckets=(1, 2, 4, 8), donate=()):
        self.fn = fn
        self.buckets = sorted(buckets)
        self.donate = tuple(donate)
        self.executables = {}
        self.eager_calls = 0
        self.replays = 0

    @property
    def num_compiled(self):
        return len(self.executables)

    def bucket_for(self, batch_size):
        for bucket in self.buckets:
            if bucket >= batch_size:
                return bucket
        raise ValueError(f"batch {batch_size} exceeds the largest bucket "
                         f"{self.buckets[-1]}")

    def compile(self, make_inputs):
        """One executable for each bucket, made at startup, where nobody
        waits."""
        jitted = jax.jit(self.fn, donate_argnums=self.donate)
        for bucket in self.buckets:
            self.executables[bucket] = jitted.lower(
                *make_inputs(bucket)).compile()

    def run(self, *inputs):
        batch_size = inputs[0].shape[0]
        bucket = self.bucket_for(batch_size)
        padded = tuple(pad_batch(array, batch_size, bucket) for array in inputs)
        outputs = self.executables[bucket](*padded)
        self.replays += 1
        return jax.tree.map(lambda output: output[:batch_size], outputs)

    def run_eager(self, *inputs):
        self.eager_calls += 1
        return self.fn(*inputs)
