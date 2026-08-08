"""Stage 12 (JAX) - killing recompilation with shape buckets.

`./vc lore 12 --jax` for the insight. `./vc test 12 --jax` to check yourself.

The torch track captures CUDA graphs here, because at batch 1 a decode step can
be as much Python and kernel-launch overhead as it is GPU work. You do not have
that problem: XLA already fused the whole step into one program and there is
one launch, not two hundred.

You have the other one. Every distinct shape XLA sees is a fresh trace and a
fresh compile -- seconds, on the critical path, while the GPU sits idle. A
server whose batch size wanders between 1 and 64 compiles sixty-four programs,
and every one of them was paid for by a user waiting on a request.

The fix is the same fix, for a different reason: pick a handful of BUCKETS,
compile once for each, and pad up to the nearest one.

WHAT YOU'RE BUILDING

    class CompiledDecode(fn, buckets=(1, 2, 4, 8), donate=())
        .compile(make_inputs)     AOT-compile one executable per bucket
        .run(*inputs)             pad to a bucket, run, slice the answer back
        .run_eager(*inputs)       the un-bucketed path, for comparison
        .num_compiled             how many executables exist
        .bucket_for(batch_size)   smallest bucket >= batch_size
        .replays, .eager_calls    counters

    make_inputs(bs) -> a tuple of example arrays with leading dim bs

AHEAD-OF-TIME COMPILATION

`jax.jit(f)` compiles lazily, on first call. You want it to have happened
already, at startup, where nobody is waiting:

    exe = jax.jit(fn).lower(*example_inputs).compile()
    out = exe(*real_inputs)          # no tracing, no compiling, ever

`lower` traces to HLO for those exact shapes; `compile` produces the
executable. Calling it with different shapes is an ERROR rather than a silent
recompile, which is exactly what you want -- it turns "mysteriously slow" into
"loudly wrong".

DONATION

    jax.jit(fn, donate_argnums=(1,))

tells XLA that argument 1 will not be used again, so its buffer can be reused
for the output instead of allocating and copying a new one. For a KV cache that
is the difference between rewriting hundreds of MB per step and updating it in
place.

The cost is that the donated array is INVALIDATED. Touch it after the call and
JAX raises about a deleted buffer. That is a feature: it makes the aliasing
visible instead of letting you accidentally read a buffer someone else now
owns.

PADDING

    run(*inputs) with a batch of 3 and buckets (1, 2, 4, 8)
        -> pad every input's leading dim to 4
        -> execute the bucket-4 program
        -> return out[:3]

Only the BATCHED arguments get padded. A weight matrix passed alongside the
batch keeps its own shape -- pad it and you get a negative pad width and a
confusing error out of broadcast_in_dim. The workable rule is "pad every
argument whose leading dimension equals the batch size", and it is worth
knowing that the rule is a heuristic: an argument that coincidentally has that
leading dimension will be padded too.

Pad with ZEROS, deterministically. Padding rows are real rows to the model: it
will happily attend over whatever garbage is in them, and if that garbage is
uninitialised memory your padded batch is not reproducible run to run. It also
must not affect rows 0..bs-1 -- attention is per row, so it does not, but the
moment you add anything that reduces across the batch it will.

HOW YOU'LL KNOW IT WORKED

    jax.jit(f)._cache_size()    number of shape signatures compiled

Run a hundred calls at wandering batch sizes. The bucketed path's compile count
must not move after startup. The un-bucketed path's climbs forever.
"""


class CompiledDecode:
    """Bucketed, ahead-of-time-compiled decode steps."""

    def __init__(self, fn, buckets=(1, 2, 4, 8), donate=()):
        raise NotImplementedError("stage 12 (jax): implement CompiledDecode")
