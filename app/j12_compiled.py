"""Stage 12 (JAX) - shape buckets, to stop the recompiles.

`./vc lore 12 --jax` for the insight. `./vc test 12 --jax` to check yourself.

The torch track captures CUDA graphs here. At batch 1 a torch decode step can
spend as much time on Python and kernel launches as on GPU work. You do not
have that problem. XLA already fused the whole step into one program, so there
is one launch, and not two hundred.

You have the other problem. Each new shape that XLA sees is a new trace and a
new compile. That costs seconds, on the critical path, while the GPU is idle.
A server whose batch size moves between 1 and 64 compiles sixty-four programs,
and a user who waits for a request pays for each one.

The fix is the same fix, for a different reason. Pick a few BUCKETS. Compile
one time for each bucket, and pad up to the nearest one.

WHAT YOU ARE BUILDING

    class CompiledDecode(fn, buckets=(1, 2, 4, 8), donate=())
        .compile(make_inputs)     AOT-compile one executable for each bucket
        .run(*inputs)             pad to a bucket, run, slice the answer back
        .run_eager(*inputs)       the path with no buckets, for comparison
        .num_compiled             how many executables exist
        .bucket_for(batch_size)   the smallest bucket >= batch_size
        .replays, .eager_calls    counters

    make_inputs(bs) -> a tuple of example arrays with leading dim bs

AHEAD-OF-TIME COMPILATION

`jax.jit(f)` compiles lazily, on the first call. You want the compile to
happen at startup, where nobody waits:

    executable = jax.jit(fn).lower(*example_inputs).compile()
    outputs = executable(*real_inputs)   # no tracing, no compile, never again

`lower` traces to HLO for those exact shapes. `compile` makes the executable.
A call with different shapes is an ERROR, and not a silent recompile. That is
exactly what you want. It turns "mysteriously slow" into "loudly wrong".

DONATION

    jax.jit(fn, donate_argnums=(1,))

This tells XLA that the program does not use argument 1 again. So XLA can use
its buffer for the output, and it does not allocate and copy a new one. For a
KV cache, that is the difference between a rewrite of hundreds of MB at each
step and an update in place.

The cost: the donated array becomes INVALID. If you touch it after the call,
JAX raises an error about a deleted buffer. That is a feature. It makes the
aliasing visible, so you cannot read a buffer that another array now owns.

PADDING

    run(*inputs) with a batch of 3 and buckets (1, 2, 4, 8)
        -> pad the leading dim of every input to 4
        -> execute the bucket-4 program
        -> return out[:3]

Pad only the BATCHED arguments. A weight matrix that travels with the batch
keeps its own shape. If you pad it, you get a negative pad width and a
confusing error from broadcast_in_dim.

The practical rule is "pad every argument whose leading dimension equals the
batch size". Know that this rule is a heuristic. An argument that has that
leading dimension by coincidence also gets padding.

Pad with ZEROS, so that the result is deterministic. The model treats padding
rows as real rows, and it attends over whatever garbage they hold. If that
garbage is memory with no initial value, the padded batch changes from run to
run.

Padding must also not change rows 0..bs-1. Attention works on each row alone,
so it does not change them. But the moment you add an operation that reduces
across the batch, it will.

HOW YOU WILL KNOW IT WORKED

    jax.jit(f)._cache_size()    the number of shape signatures compiled

Run a hundred calls at batch sizes that move around. After startup, the
compile count of the bucketed path must not change. The count of the path
with no buckets climbs forever.
"""


class CompiledDecode:
    """Bucketed, ahead-of-time-compiled decode steps."""

    def __init__(self, fn, buckets=(1, 2, 4, 8), donate=()):
        raise NotImplementedError("stage 12 (jax): implement CompiledDecode")
