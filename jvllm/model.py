"""Qwen3 in pure JAX. Provided -- read it, don't edit it.

The torch track gets its model from HuggingFace. There is no Flax Qwen3, so
this is one. It loads the same safetensors file, and its logits match the torch
model to within bf16 noise.

WHAT IS DIFFERENT FROM THE TORCH MODEL, AND WHY IT MATTERS FOR EVERY STAGE

torch grows the KV cache by concatenation: every decode step produces a cache
one token longer. That is fine when the kernel is dispatched at runtime from a
Python shape. XLA is not that: it compiles for exact shapes, and a cache that
grows by one every step means a full recompile every step. Thirty seconds of
XLA per token.

So the cache here is PREALLOCATED and written into:

    cache = model.init_cache(batch=4, max_len=1024)     # zeros, fixed forever
    logits, cache = model.forward(ids, positions, cache, cache_len)

`cache_len` is a per-row (B,) count of how many slots are already valid. The
model writes this call's K/V at slots cache_len[b] .. cache_len[b]+T-1 and
masks attention to slots <= cache_len[b] + t for query token t. Nothing about
the buffer's SHAPE changes, so one compile serves every step.

That single design difference is the spine of the JAX track:

  - stage 02, the cache is a buffer you own, not a thing the model hands back
  - stage 04, you right-pad prompts and read logits from the right index,
    instead of left-padding to make the last index correct for everyone
  - stage 05, evicting a row is a slot you reuse, not a tensor you re-index
  - stage 12, the enemy is recompilation, not kernel-launch overhead

THE API

    model = load_model()                       # Qwen/Qwen3-0.6B, bf16
    model.config                               # Qwen3Config
    model.params                               # the pytree, for stages 18/20
    model.tokenizer                            # HF tokenizer, framework-free
    model.eos_ids                              # set[int]

    ids  = model.encode("hello")               # (T,) int32 jnp array
    text = model.decode([1, 2, 3])

    cache = model.init_cache(batch, max_len)
    logits, cache = model.forward(ids, positions=None, cache=None,
                                  cache_len=None, logits_index=-1)

      ids           (B, T) int32
      positions     (B, T) int32 absolute RoPE positions. None -> arange(T)
      cache         from init_cache(). None -> a scratch cache, discarded
      cache_len     (B,) int32 valid slots BEFORE this call. None -> zeros
      logits_index  -1        logits at the last position      -> (B, V)
                    (B,) arr  logits at a per-row position     -> (B, V)
                    None      logits at every position         -> (B, T, V)

    Returns (logits, cache). The cache is a NEW pytree -- JAX is functional,
    the old one is still valid and still holds the old contents.

A trap worth stating once: `init_cache(max_len=S)` and then writing past S
silently clamps, because that is what `dynamic_update_slice` does. Size the
cache for the longest sequence you will run, and check `cache_len` yourself.
"""

import functools
import json
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_MODEL = os.environ.get("VC_MODEL", "Qwen/Qwen3-0.6B")


@dataclass(frozen=True)
class Qwen3Config:
    """Frozen so it can ride along as a static jit argument."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int

    @property
    def group(self):
        """Query heads per KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, d):
        head_dim = d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"]
        return cls(
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=head_dim,
            intermediate_size=d["intermediate_size"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=d["rope_theta"],
            vocab_size=d["vocab_size"],
        )


# ---------------------------------------------------------------- pieces


def rms_norm(x, weight, eps):
    """Always in float32. In bf16 the mean-of-squares of a 1024-wide vector
    loses enough precision to move logits by more than the tolerances allow."""
    f = x.astype(jnp.float32)
    f = f * jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps)
    return (f * weight.astype(jnp.float32)).astype(x.dtype)


def rope_tables(positions, head_dim, theta, dtype=jnp.float32):
    """(B, T) positions -> cos, sin of shape (B, T, 1, head_dim).

    The extra axis is the head axis, so the tables broadcast over heads.
    """
    inv = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = positions.astype(jnp.float32)[..., None] * inv        # (B, T, D/2)
    ang = jnp.concatenate([freqs, freqs], axis=-1)                # (B, T, D)
    return jnp.cos(ang)[:, :, None, :].astype(dtype), \
        jnp.sin(ang)[:, :, None, :].astype(dtype)


def apply_rope(x, cos, sin):
    """x is (B, T, heads, head_dim). Half-split convention, as HF uses."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    out = x.astype(jnp.float32) * cos + rotated.astype(jnp.float32) * sin
    return out.astype(x.dtype)


def _write_slots(buf, new, start):
    """Write (T, KVH, D) into a (S, KVH, D) buffer at `start`, per batch row.

    dynamic_update_slice CLAMPS an out-of-range start rather than erroring, so
    overflowing the cache corrupts the tail instead of raising. That is the
    single nastiest failure mode in this file.
    """
    return jax.vmap(
        lambda b, n, s: jax.lax.dynamic_update_slice_in_dim(b, n, s, axis=0)
    )(buf, new, start)


def attend(q, k, v, mask, scale):
    """q (B,T,H,D) against a full cache k/v (B,S,KVH,D), masked by (B,T,S).

    GQA is a reshape: query head h reads KV head h // group, which is exactly
    what splitting H into (KVH, group) gives you.

    The matmuls run in the model's dtype with `preferred_element_type=float32`,
    so they ACCUMULATE in fp32 without ever materialising an fp32 copy of K or
    V. Casting the inputs instead would double the bandwidth of the single
    largest tensor in the whole forward pass. The softmax is fp32 regardless --
    that is the part that actually needs the range.

    This is the dense, score-materialising reference. It costs O(T*S) memory,
    which is fine for decode (T=1) and is the reason prefill stops amortising
    somewhere past 512 tokens. Stage 08 is where you stop materialising it.
    """
    B, T, H, D = q.shape
    KVH = k.shape[2]
    G = H // KVH
    qg = q.reshape(B, T, KVH, G, D)

    scores = jnp.einsum("btkgd,bskd->bkgts", qg, k,
                        preferred_element_type=jnp.float32) * scale
    scores = jnp.where(mask[:, None, None, :, :], scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    out = jnp.einsum("bkgts,bskd->btkgd", probs, v,
                     preferred_element_type=jnp.float32)
    return out.reshape(B, T, H * D).astype(q.dtype)


# ---------------------------------------------------------------- forward


def _layer(cfg, carry, xs):
    """One transformer block. Runs under lax.scan over the stacked layers.

    Scanning instead of unrolling 28 copies is why this model compiles in
    seconds instead of a minute -- XLA sees one block, not twenty-eight.
    """
    x, cos, sin, mask, cache_len = carry
    p, kc, vc = xs
    eps = cfg.rms_norm_eps
    B, T, _ = x.shape
    NH, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    h = rms_norm(x, p["attn_norm"], eps)
    q = (h @ p["q_proj"].T).reshape(B, T, NH, D)
    k = (h @ p["k_proj"].T).reshape(B, T, KVH, D)
    v = (h @ p["v_proj"].T).reshape(B, T, KVH, D)

    # Qwen3 normalises q and k PER HEAD, before RoPE. Skip this and the model
    # produces fluent-looking garbage rather than an obvious error.
    q = rms_norm(q, p["q_norm"], eps)
    k = rms_norm(k, p["k_norm"], eps)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    kc = _write_slots(kc, k, cache_len)
    vc = _write_slots(vc, v, cache_len)

    o = attend(q, kc, vc, mask, 1.0 / np.sqrt(D))
    x = x + o @ p["o_proj"].T

    h = rms_norm(x, p["mlp_norm"], eps)
    x = x + (jax.nn.silu(h @ p["gate"].T) * (h @ p["up"].T)) @ p["down"].T

    return (x, cos, sin, mask, cache_len), (kc, vc)


@functools.partial(jax.jit, static_argnums=(1, 7))
def _forward(params, cfg, ids, positions, kc, vc, cache_len, all_positions,
             logits_index):
    B, T = ids.shape
    S = kc.shape[2]

    x = params["embed"][ids]
    cos, sin = rope_tables(positions, cfg.head_dim, cfg.rope_theta)

    # slot s is visible to query t of row b iff s <= cache_len[b] + t.
    slots = jnp.arange(S)[None, None, :]
    horizon = cache_len[:, None, None] + jnp.arange(T)[None, :, None]
    mask = slots <= horizon

    carry = (x, cos, sin, mask, cache_len)
    (x, *_), (kc, vc) = jax.lax.scan(
        functools.partial(_layer, cfg), carry, (params["layers"], kc, vc)
    )
    x = rms_norm(x, params["final_norm"], cfg.rms_norm_eps)

    # Gather BEFORE the lm_head. The vocab is 152k wide; projecting every
    # position of a 512-token prefill would allocate 300MB of logits to throw
    # away all but one row of.
    if not all_positions:
        x = jnp.take_along_axis(x, logits_index[:, None, None], axis=1)[:, 0]

    return (x @ params["lm_head"].T).astype(jnp.float32), kc, vc


# ---------------------------------------------------------------- model


class Qwen3:
    def __init__(self, params, config, tokenizer, eos_ids):
        self.params = params
        self.config = config
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.dtype = params["embed"].dtype

    # ---- tokenizer passthrough ------------------------------------
    def encode(self, text):
        return jnp.asarray(self.tokenizer(text).input_ids, dtype=jnp.int32)

    def decode(self, ids, **kw):
        return self.tokenizer.decode(list(ids), **kw)

    # ---- cache ----------------------------------------------------
    def init_cache(self, batch, max_len, dtype=None):
        """Zeros, shaped (layers, B, max_len, kv_heads, head_dim), K and V.

        Sized once and never resized. `max_len` is a hard ceiling: writing
        past it silently clamps.

        Defaults to the model's own dtype. A bf16 cache under fp32 weights is
        a legitimate thing to want -- KV is the biggest reader in decode, and
        stage 18 quantizes it on purpose -- but it has to be a choice, not an
        accident, so the default follows the weights.
        """
        c = self.config
        shape = (c.num_hidden_layers, batch, max_len,
                 c.num_key_value_heads, c.head_dim)
        dt = self.dtype if dtype is None else dtype
        return jnp.zeros(shape, dt), jnp.zeros(shape, dt)

    def cache_bytes(self, batch, max_len, dtype_bytes=2):
        c = self.config
        return (2 * c.num_hidden_layers * batch * max_len
                * c.num_key_value_heads * c.head_dim * dtype_bytes)

    # ---- forward --------------------------------------------------
    def forward(self, ids, positions=None, cache=None, cache_len=None,
                logits_index=-1):
        ids = jnp.asarray(ids, dtype=jnp.int32)
        if ids.ndim == 1:
            ids = ids[None]
        B, T = ids.shape

        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
        positions = jnp.asarray(positions, dtype=jnp.int32)
        if positions.ndim == 1:
            positions = jnp.broadcast_to(positions[None], (B, T))

        if cache_len is None:
            cache_len = jnp.zeros((B,), dtype=jnp.int32)
        cache_len = jnp.asarray(cache_len, dtype=jnp.int32)

        if cache is None:
            # Stage 01 runs with no cache at all: a scratch buffer exactly T
            # long, filled, used once, dropped on the floor. That IS the
            # quadratic recompute stage 02 exists to kill.
            cache = self.init_cache(B, T)
        kc, vc = cache

        all_positions = logits_index is None
        if all_positions:
            idx = jnp.zeros((B,), dtype=jnp.int32)
        elif isinstance(logits_index, int):
            idx = jnp.full((B,), (T + logits_index) % T, dtype=jnp.int32)
        else:
            idx = jnp.asarray(logits_index, dtype=jnp.int32)

        logits, kc, vc = _forward(self.params, self.config, ids, positions,
                                  kc, vc, cache_len, all_positions, idx)
        return logits, (kc, vc)

    def nbytes(self):
        return sum(int(x.size) * x.dtype.itemsize
                   for x in jax.tree.leaves(self.params))


# ---------------------------------------------------------------- loading


_LAYER_KEYS = {
    "attn_norm": "input_layernorm.weight",
    "mlp_norm": "post_attention_layernorm.weight",
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


def _snapshot(model_id):
    from huggingface_hub import snapshot_download

    if os.path.isdir(model_id):
        return model_id
    return snapshot_download(
        model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"]
    )


def _read_safetensors(path):
    from safetensors import safe_open

    tensors = {}
    files = [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    for fn in files:
        with safe_open(os.path.join(path, fn), framework="np") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    return tensors


def load_model(model_id=DEFAULT_MODEL, dtype=jnp.bfloat16):
    """Load Qwen3 into a JAX pytree. Cached per (model_id, dtype)."""
    key = (model_id, jnp.dtype(dtype).name)
    if key in _CACHE:
        return _CACHE[key]

    from transformers import AutoTokenizer

    path = _snapshot(model_id)
    cfg = Qwen3Config.from_json(json.load(open(os.path.join(path, "config.json"))))
    raw = _read_safetensors(path)

    def arr(x):
        # bf16 arrives from safetensors as a numpy view ml_dtypes understands.
        return jnp.asarray(np.asarray(x), dtype=dtype)

    layers = {
        name: jnp.stack([arr(raw[f"model.layers.{i}.{suffix}"])
                         for i in range(cfg.num_hidden_layers)])
        for name, suffix in _LAYER_KEYS.items()
    }
    embed = arr(raw["model.embed_tokens.weight"])
    params = {
        "embed": embed,
        "final_norm": arr(raw["model.norm.weight"]),
        "lm_head": arr(raw["lm_head.weight"]) if "lm_head.weight" in raw else embed,
        "layers": layers,
    }

    tok = AutoTokenizer.from_pretrained(path)
    gen_path = os.path.join(path, "generation_config.json")
    eos = json.load(open(gen_path))["eos_token_id"] if os.path.exists(gen_path) \
        else tok.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}

    model = Qwen3(params, cfg, tok, eos_ids)
    _CACHE[key] = model
    return model


_CACHE = {}
