"""Qwen3 in pure JAX. The repo gives you this file. Read it. Do not edit it.

The torch track gets its model from HuggingFace. There is no Flax Qwen3, so
this is one. It loads the same safetensors file, and its logits match the
torch model to within bf16 noise.

THE DIFFERENCE FROM THE TORCH MODEL, AND WHY IT IS IMPORTANT FOR EVERY STAGE

torch grows the KV cache by concatenation. Every decode step makes a cache
one token longer. That works when a Python shape selects the kernel at run
time.

XLA does not work that way. It compiles for exact shapes. A cache that grows
by one token at every step forces a full compile at every step. That is
thirty seconds of XLA for each token.

So here you ALLOCATE the cache BEFORE the run, and write into it:

    cache = model.init_cache(batch=4, max_len=1024)     # zeros, fixed for ever
    logits, cache = model.forward(token_ids, positions, cache, cache_len)

`cache_len` is a (B,) count, for each row, of the slots that are already
valid. The model writes the K and V of this call at slots cache_len[b] ..
cache_len[b]+T-1, and it masks attention to slots <= cache_len[b] + t for
query token t. The SHAPE of the buffer never changes, so one compile serves
every step.

That one design difference is the base of the JAX track:

  - stage 02: the cache is a buffer that you own, not a thing that the model
    gives back
  - stage 04: you right-pad the prompts and read the logits from the correct
    index. You do not left-pad to make the last index correct for all rows.
  - stage 05: to evict a row, you use its slot again. You do not index a
    tensor again.
  - stage 12: the problem is compilation, not kernel-launch overhead

THE API

    model = load_model()                       # Qwen/Qwen3-0.6B, bf16
    model.config                               # Qwen3Config
    model.params                               # the pytree, for stages 18/20
    model.tokenizer                            # HF tokenizer, no framework
    model.eos_ids                              # set[int]

    token_ids = model.encode("hello")          # (T,) int32 jnp array
    text = model.decode([1, 2, 3])

    cache = model.init_cache(batch, max_len)
    logits, cache = model.forward(token_ids, positions=None, cache=None,
                                  cache_len=None, logits_index=-1)

      token_ids     (B, T) int32
      positions     (B, T) int32 absolute RoPE positions. None -> arange(T)
      cache         from init_cache(). None -> a scratch cache, discarded
      cache_len     (B,) int32 valid slots BEFORE this call. None -> zeros
      logits_index  -1        the logits at the last position    -> (B, V)
                    (B,) arr  the logits at one position of each row -> (B, V)
                    None      the logits at every position        -> (B, T, V)

    -> (logits, cache). The cache is a NEW pytree. JAX is functional: the
    old cache is still valid and still holds the old contents.

A trap: after `init_cache(max_len=S)`, `dynamic_update_slice` clamps a
write past S, with no error. Make the cache large enough
for the longest sequence that you run, and check `cache_len` yourself.
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
    """Frozen, so that it can be a static jit argument."""

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
        """The query heads for each KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, fields):
        head_dim = (fields.get("head_dim")
                    or fields["hidden_size"] // fields["num_attention_heads"])
        return cls(
            hidden_size=fields["hidden_size"],
            num_hidden_layers=fields["num_hidden_layers"],
            num_attention_heads=fields["num_attention_heads"],
            num_key_value_heads=fields["num_key_value_heads"],
            head_dim=head_dim,
            intermediate_size=fields["intermediate_size"],
            rms_norm_eps=fields["rms_norm_eps"],
            rope_theta=fields["rope_theta"],
            vocab_size=fields["vocab_size"],
        )


# ---------------------------------------------------------------- pieces


def rms_norm(inputs, weight, eps):
    """Always in float32. In bf16, the mean of squares of a vector of 1024
    loses enough precision to move the logits past the tolerances."""
    values = inputs.astype(jnp.float32)
    values = values * jax.lax.rsqrt(jnp.mean(values * values, axis=-1,
                                             keepdims=True) + eps)
    return (values * weight.astype(jnp.float32)).astype(inputs.dtype)


def rope_tables(positions, head_dim, theta, dtype=jnp.float32):
    """(B, T) positions -> cos, sin of shape (B, T, 1, head_dim).

    The extra axis is the head axis, so the tables broadcast over the heads.
    """
    inverse_frequencies = 1.0 / (theta ** (
        jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    frequencies = positions.astype(jnp.float32)[..., None] * inverse_frequencies
    angles = jnp.concatenate([frequencies, frequencies], axis=-1)   # (B, T, D)
    return (jnp.cos(angles)[:, :, None, :].astype(dtype),
            jnp.sin(angles)[:, :, None, :].astype(dtype))


def apply_rope(values, cos, sin):
    """values is (B, T, heads, head_dim). The half-split convention of HF."""
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    rotated = jnp.concatenate([-second, first], axis=-1)
    output = (values.astype(jnp.float32) * cos
              + rotated.astype(jnp.float32) * sin)
    return output.astype(values.dtype)


def _write_slots(cache, new_values, start):
    """Write (T, kv_heads, D) into a (S, kv_heads, D) buffer at `start`, for
    each batch row.

    dynamic_update_slice CLAMPS a start that is out of range, and gives no
    error. So a cache overflow corrupts the tail and raises nothing. That is
    the worst failure in this file.
    """
    return jax.vmap(
        lambda row_cache, row_values, row_start:
            jax.lax.dynamic_update_slice_in_dim(row_cache, row_values,
                                                row_start, axis=0)
    )(cache, new_values, start)


def attend(query, keys, values, mask, scale):
    """query (B, T, H, D) against a full cache of keys and values
    (B, S, kv_heads, D), with the mask (B, T, S).

    GQA is a reshape: query head h reads KV head h // group, and a split of
    H into (kv_heads, group) gives exactly that.

    The matmuls run in the dtype of the model with
    `preferred_element_type=float32`. So they ACCUMULATE in fp32, and they
    make no fp32 copy of K or V. A cast of the inputs doubles the bandwidth of the
    largest tensor of the whole forward pass. The softmax is always fp32:
    that is the part that needs the range.

    This is the dense reference, and it makes the scores in memory. It costs
    O(T*S) memory. That is acceptable for decode (T=1), and it is the reason
    that prefill stops amortizing after about 512 tokens. In stage 08 you
    stop making the scores in memory.
    """
    batch, seq_len, num_heads, head_dim = query.shape
    num_kv_heads = keys.shape[2]
    grouped = query.reshape(batch, seq_len, num_kv_heads,
                            num_heads // num_kv_heads, head_dim)
    scores = jnp.einsum("btkgd,bskd->bkgts", grouped, keys,
                        preferred_element_type=jnp.float32) * scale
    scores = jnp.where(mask[:, None, None, :, :], scores,
                       jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(values.dtype)
    output = jnp.einsum("bkgts,bskd->btkgd", probs, values,
                        preferred_element_type=jnp.float32)
    return output.reshape(batch, seq_len, num_heads * head_dim).astype(
        query.dtype)


# ---------------------------------------------------------------- forward


def _attention_block(config, weights, hidden, cos, sin, mask, cache_len,
                     key_cache, value_cache):
    """-> (the attention output, the new key cache, the new value cache)."""
    batch, seq_len, _ = hidden.shape
    head_dim = config.head_dim

    def heads(projection, num_heads):
        return (hidden @ weights[projection].T).reshape(batch, seq_len,
                                                        num_heads, head_dim)

    query = heads("q_proj", config.num_attention_heads)
    key = heads("k_proj", config.num_key_value_heads)
    value = heads("v_proj", config.num_key_value_heads)
    # Qwen3 normalizes q and k FOR EACH HEAD, before RoPE. If you skip this,
    # the model makes fluent garbage, not an obvious error.
    query = apply_rope(rms_norm(query, weights["q_norm"], config.rms_norm_eps),
                       cos, sin)
    key = apply_rope(rms_norm(key, weights["k_norm"], config.rms_norm_eps),
                     cos, sin)

    key_cache = _write_slots(key_cache, key, cache_len)
    value_cache = _write_slots(value_cache, value, cache_len)
    attended = attend(query, key_cache, value_cache, mask,
                      1.0 / np.sqrt(head_dim))
    return attended @ weights["o_proj"].T, key_cache, value_cache


def _mlp_block(weights, hidden):
    gate = jax.nn.silu(hidden @ weights["gate"].T)
    return (gate * (hidden @ weights["up"].T)) @ weights["down"].T


def _layer(config, carry, layer_inputs):
    """One transformer block. It runs under lax.scan over the stacked layers.

    A scan, not 28 unrolled copies, is why this model compiles in seconds,
    not in a minute: XLA sees one block, not twenty-eight.
    """
    residual, cos, sin, mask, cache_len = carry
    weights, key_cache, value_cache = layer_inputs
    attention_output, key_cache, value_cache = _attention_block(
        config, weights, rms_norm(residual, weights["attn_norm"],
                                  config.rms_norm_eps),
        cos, sin, mask, cache_len, key_cache, value_cache)
    residual = residual + attention_output
    residual = residual + _mlp_block(
        weights, rms_norm(residual, weights["mlp_norm"], config.rms_norm_eps))
    return (residual, cos, sin, mask, cache_len), (key_cache, value_cache)


def _visibility_mask(cache_len, seq_len, cache_size):
    """(B, T, S): slot s is visible to query t of row b if
    s <= cache_len[b] + t."""
    slots = jnp.arange(cache_size)[None, None, :]
    horizon = cache_len[:, None, None] + jnp.arange(seq_len)[None, :, None]
    return slots <= horizon


@functools.partial(jax.jit, static_argnums=(1, 7))
def _forward(params, config, token_ids, positions, key_cache, value_cache,
             cache_len, all_positions, logits_index):
    seq_len = token_ids.shape[1]
    hidden = params["embed"][token_ids]
    cos, sin = rope_tables(positions, config.head_dim, config.rope_theta)
    mask = _visibility_mask(cache_len, seq_len, key_cache.shape[2])

    carry = (hidden, cos, sin, mask, cache_len)
    (hidden, *_), (key_cache, value_cache) = jax.lax.scan(
        functools.partial(_layer, config), carry,
        (params["layers"], key_cache, value_cache))
    hidden = rms_norm(hidden, params["final_norm"], config.rms_norm_eps)

    # Gather BEFORE the lm_head. The vocabulary is 152k wide. A projection of
    # every position of a 512-token prefill allocates 300 MB of logits, and
    # you then keep one row and discard the rest.
    if not all_positions:
        hidden = jnp.take_along_axis(hidden, logits_index[:, None, None],
                                     axis=1)[:, 0]
    logits = (hidden @ params["lm_head"].T).astype(jnp.float32)
    return logits, key_cache, value_cache


# ---------------------------------------------------------------- model


def _as_batch(token_ids):
    token_ids = jnp.asarray(token_ids, dtype=jnp.int32)
    return token_ids[None] if token_ids.ndim == 1 else token_ids


def _as_positions(positions, batch, seq_len):
    if positions is None:
        return jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32),
                                (batch, seq_len))
    positions = jnp.asarray(positions, dtype=jnp.int32)
    if positions.ndim == 1:
        return jnp.broadcast_to(positions[None], (batch, seq_len))
    return positions


def _as_logits_index(logits_index, batch, seq_len):
    """-> (all_positions, one index for each row)."""
    if logits_index is None:
        return True, jnp.zeros((batch,), dtype=jnp.int32)
    if isinstance(logits_index, int):
        return False, jnp.full((batch,), (seq_len + logits_index) % seq_len,
                               dtype=jnp.int32)
    return False, jnp.asarray(logits_index, dtype=jnp.int32)


class Qwen3:
    def __init__(self, params, config, tokenizer, eos_ids):
        self.params = params
        self.config = config
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.dtype = params["embed"].dtype

    # ---- the tokenizer --------------------------------------------
    def encode(self, text):
        return jnp.asarray(self.tokenizer(text).input_ids, dtype=jnp.int32)

    def decode(self, token_ids, **options):
        return self.tokenizer.decode(list(token_ids), **options)

    # ---- the cache ------------------------------------------------
    def init_cache(self, batch, max_len, dtype=None):
        """Zeros of shape (layers, B, max_len, kv_heads, head_dim), for K and
        for V.

        Allocated one time, and never resized. `max_len` is a hard limit:
        dynamic_update_slice clamps a write past it, with no error.

        The default dtype is the dtype of the model. A bf16 cache under fp32
        weights can be correct. KV is the largest reader in decode, and
        stage 18 quantizes it on purpose. But it must be a
        choice, not an accident, so the default follows the weights.
        """
        config = self.config
        shape = (config.num_hidden_layers, batch, max_len,
                 config.num_key_value_heads, config.head_dim)
        cache_dtype = self.dtype if dtype is None else dtype
        return jnp.zeros(shape, cache_dtype), jnp.zeros(shape, cache_dtype)

    def cache_bytes(self, batch, max_len, dtype_bytes=2):
        config = self.config
        return (2 * config.num_hidden_layers * batch * max_len
                * config.num_key_value_heads * config.head_dim * dtype_bytes)

    # ---- the forward pass -----------------------------------------
    def forward(self, ids, positions=None, cache=None, cache_len=None,
                logits_index=-1):
        token_ids = _as_batch(ids)
        batch, seq_len = token_ids.shape
        positions = _as_positions(positions, batch, seq_len)
        cache_len = (jnp.zeros((batch,), dtype=jnp.int32) if cache_len is None
                     else jnp.asarray(cache_len, dtype=jnp.int32))
        if cache is None:
            # Stage 01 runs with no cache: a scratch buffer exactly T long,
            # filled, used one time, and dropped. That IS the quadratic
            # recompute that stage 02 removes.
            cache = self.init_cache(batch, seq_len)
        all_positions, rows = _as_logits_index(logits_index, batch, seq_len)
        logits, key_cache, value_cache = _forward(
            self.params, self.config, token_ids, positions, *cache, cache_len,
            all_positions, rows)
        return logits, (key_cache, value_cache)

    def nbytes(self):
        return sum(int(leaf.size) * leaf.dtype.itemsize
                   for leaf in jax.tree.leaves(self.params))


# ---------------------------------------------------------------- loading


# jvllm name -> the name of the HF weight inside a layer
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

_LOADED_MODELS = {}


def _snapshot(model_id):
    from huggingface_hub import snapshot_download

    if os.path.isdir(model_id):
        return model_id
    return snapshot_download(
        model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"])


def _read_safetensors(path):
    from safetensors import safe_open

    tensors = {}
    for file_name in sorted(os.listdir(path)):
        if not file_name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(path, file_name), framework="np") as handle:
            for name in handle.keys():
                tensors[name] = handle.get_tensor(name)
    return tensors


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def _stop_token_ids(path, tokenizer):
    generation_config = os.path.join(path, "generation_config.json")
    stop = (_read_json(generation_config)["eos_token_id"]
            if os.path.exists(generation_config) else tokenizer.eos_token_id)
    return set(stop) if isinstance(stop, list) else {stop}


def _params(hf_tensors, config, dtype):
    def as_array(tensor):
        # bf16 comes from safetensors as a numpy view that ml_dtypes knows.
        return jnp.asarray(np.asarray(tensor), dtype=dtype)

    layers = {
        name: jnp.stack([as_array(hf_tensors[f"model.layers.{index}.{suffix}"])
                         for index in range(config.num_hidden_layers)])
        for name, suffix in _LAYER_KEYS.items()
    }
    embedding = as_array(hf_tensors["model.embed_tokens.weight"])
    lm_head = (as_array(hf_tensors["lm_head.weight"])
               if "lm_head.weight" in hf_tensors else embedding)
    return {"embed": embedding,
            "final_norm": as_array(hf_tensors["model.norm.weight"]),
            "lm_head": lm_head, "layers": layers}


def load_model(model_id=DEFAULT_MODEL, dtype=jnp.bfloat16):
    """Load Qwen3 into a JAX pytree. One copy for each (model_id, dtype)."""
    key = (model_id, jnp.dtype(dtype).name)
    if key not in _LOADED_MODELS:
        from transformers import AutoTokenizer

        path = _snapshot(model_id)
        config = Qwen3Config.from_json(_read_json(os.path.join(path,
                                                               "config.json")))
        tokenizer = AutoTokenizer.from_pretrained(path)
        _LOADED_MODELS[key] = Qwen3(_params(_read_safetensors(path), config,
                                            dtype),
                                    config, tokenizer,
                                    _stop_token_ids(path, tokenizer))
    return _LOADED_MODELS[key]
