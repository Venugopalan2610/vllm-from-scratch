"""A decoder-only transformer in plain torch, with a paged KV cache.

The repo gives you this file. Read it. Do not edit it.

WHY THIS FILE EXISTS

Stages 06 to 20 built the parts of an engine. Each part passed its checks
alone. None of them ran inside a model, because the HuggingFace model cannot
host them. Its cache grows by concatenation, and it hides the attention call.

This model is the host. It does three things differently:

  - It takes a FLAT, RAGGED batch. All tokens of all sequences sit in one
    (num_tokens,) tensor. There is no padding and no batch dimension.
  - It does not own a KV cache. You own it. The model gives you its shape.
  - It does not do attention. It calls an attention BACKEND that you write,
    one time for each layer.

THE API

    model = load_model()                    # Qwen/Qwen3-0.6B by default
    model.config                            # a ModelConfig
    model.tokenizer, model.eos_ids
    model.weight_bytes()                    # bytes that decode reads each step
    model.kv_bytes_per_token()

    kv_caches = model.allocate_kv_cache(num_blocks, block_size)
        # one (key_cache, value_cache) pair for each layer. Each tensor is
        # (num_blocks, num_kv_heads, block_size, head_dim).

    logits = model.forward(token_ids, positions, attention, logits_rows)

      token_ids     (num_tokens,) int64, every token of every sequence
      positions     (num_tokens,) int64, the absolute position of each token
      attention     your backend: attention(layer, query, key, value)
                    -> (num_tokens, num_heads * head_dim)
                    query (num_tokens, num_heads, head_dim)
                    key, value (num_tokens, num_kv_heads, head_dim), after RoPE
      logits_rows   (num_rows,) int64, the tokens to project to logits

    Returns (num_rows, vocab_size) float32 logits.

A weight can be a tensor or any callable (see `linear`). Stage 24 puts
quantized layers there.

The model never reads your cache and never allocates during forward(), except
for the activations. So a CUDA graph can capture forward() if your backend is
also free of host syncs. Stage 23 depends on that.

The same file runs Llama-family checkpoints too. It takes the inverse
frequencies of RoPE from the HF rotary module, so the RoPE scaling of each
family is correct.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DEFAULT_MODEL = os.environ.get("VC_MODEL", "Qwen/Qwen3-0.6B")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    has_qk_norm: bool
    dtype: torch.dtype


# ---------------------------------------------------------------- math


def linear(inputs, weight, bias=None):
    """A matmul with a weight that is a tensor, or a callable weight."""
    if isinstance(weight, torch.Tensor):
        return F.linear(inputs, weight, bias)
    outputs = weight(inputs)
    return outputs if bias is None else outputs + bias


def rms_norm(hidden, weight, eps):
    """One fused kernel. It accumulates in float32 internally, so bf16 input
    loses no precision."""
    return F.rms_norm(hidden, (hidden.shape[-1],), weight, eps)


def rope_tables(positions, inverse_frequencies, attention_scale):
    """cos and sin for each position, shaped (num_tokens, 1, head_dim)."""
    angles = positions.float()[:, None] * inverse_frequencies[None, :]
    angles = torch.cat([angles, angles], dim=-1)[:, None, :]
    return angles.cos() * attention_scale, angles.sin() * attention_scale


def apply_rope(heads, cos, sin):
    """heads (num_tokens, num_heads, head_dim). The half-split convention."""
    half = heads.shape[-1] // 2
    rotated = torch.cat([-heads[..., half:], heads[..., :half]], dim=-1)
    return (heads.float() * cos + rotated.float() * sin).to(heads.dtype)


def causal_mask(num_queries, context_len, device):
    """The queries are the LAST num_queries positions of the context. Query i
    sees the keys 0 to (context_len - num_queries + i)."""
    first_query_position = context_len - num_queries
    key_positions = torch.arange(context_len, device=device)
    query_positions = first_query_position + torch.arange(num_queries,
                                                          device=device)
    return key_positions[None, :] <= query_positions[:, None]


def attend_to_context(query, keys, values):
    """Causal attention for a chunk of queries at the end of a context.

    query        (num_queries, num_heads, head_dim)
    keys, values (num_kv_heads, context_len, head_dim)
    Returns (num_queries, num_heads * head_dim)."""
    num_queries, num_heads, head_dim = query.shape
    mask = causal_mask(num_queries, keys.shape[1], query.device)
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1), keys, values, attn_mask=mask, enable_gqa=True)
    return output.transpose(0, 1).reshape(num_queries, num_heads * head_dim)


# ---------------------------------------------------------------- weights


def _fuse(modules, attribute):
    """Concatenate one parameter of several Linear modules. Then point each
    module at its slice of the result, so that the weights exist one time."""
    fused = torch.cat([getattr(m, attribute).data for m in modules]).contiguous()
    start = 0
    for module in modules:
        parameter = getattr(module, attribute)
        rows = parameter.shape[0]
        parameter.data = fused[start:start + rows]
        start += rows
    return fused


class LayerWeights:
    """The weights of one block. Q, K and V are one fused matmul, and so are
    gate and up. Two matmuls in the place of five save launches."""

    def __init__(self, hf_layer, has_qk_norm):
        attention, mlp = hf_layer.self_attn, hf_layer.mlp
        projections = [attention.q_proj, attention.k_proj, attention.v_proj]
        self.qkv_proj = _fuse(projections, "weight")
        self.qkv_bias = (_fuse(projections, "bias")
                         if attention.q_proj.bias is not None else None)
        self.o_proj = attention.o_proj.weight
        self.gate_up_proj = _fuse([mlp.gate_proj, mlp.up_proj], "weight")
        self.down_proj = mlp.down_proj.weight
        self.attention_norm = hf_layer.input_layernorm.weight
        self.mlp_norm = hf_layer.post_attention_layernorm.weight
        self.q_norm = attention.q_norm.weight if has_qk_norm else None
        self.k_norm = attention.k_norm.weight if has_qk_norm else None

    MATMULS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def _read_config(hf_model, name):
    config = hf_model.config
    first_attention = hf_model.model.layers[0].self_attn
    return ModelConfig(
        name=name,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=(getattr(config, "head_dim", None)
                  or config.hidden_size // config.num_attention_heads),
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        has_qk_norm=hasattr(first_attention, "q_norm"),
        dtype=hf_model.dtype,
    )


def _as_set(ids):
    if ids is None:
        return set()
    return set(ids if isinstance(ids, list) else [ids])


def _stop_token_ids(hf_model, tokenizer):
    """A chat model often stops on a second token (<|im_end|> for Qwen3). The
    full list is in the generation config, not only in the model config."""
    generation_config = getattr(hf_model, "generation_config", None)
    return (_as_set(hf_model.config.eos_token_id)
            | _as_set(tokenizer.eos_token_id)
            | _as_set(getattr(generation_config, "eos_token_id", None)))


# ---------------------------------------------------------------- model


class Model:
    def __init__(self, hf_model, tokenizer, name):
        inner = hf_model.model
        self.config = _read_config(hf_model, name)
        self.tokenizer = tokenizer
        self.eos_ids = _stop_token_ids(hf_model, tokenizer)
        self.embedding = inner.embed_tokens.weight
        self.final_norm = inner.norm.weight
        self.lm_head = hf_model.lm_head.weight
        self.layers = [LayerWeights(layer, self.config.has_qk_norm)
                       for layer in inner.layers]
        self.inverse_frequencies = inner.rotary_emb.inv_freq.float()
        self.rope_attention_scale = float(
            getattr(inner.rotary_emb, "attention_scaling", 1.0))
        self.device = self.embedding.device

    # ------------------------------------------------------------ accounting

    def _weights_read_by_decode(self):
        """The embedding table is not read in full: a step reads one row of
        it for each token. A tied lm_head counts one time."""
        weights = [self.lm_head, self.final_norm]
        for layer in self.layers:
            weights += [getattr(layer, name) for name in LayerWeights.MATMULS]
            weights += [layer.attention_norm, layer.mlp_norm]
        return weights

    def weight_bytes(self):
        """The bytes that one decode step reads."""
        seen, total = set(), 0
        for weight in self._weights_read_by_decode():
            if not isinstance(weight, torch.Tensor):
                total += weight.nbytes()        # a quantized layer knows its size
            elif weight.data_ptr() not in seen:
                seen.add(weight.data_ptr())
                total += weight.numel() * weight.element_size()
        return total

    def kv_bytes_per_token(self):
        config = self.config
        bytes_per_value = torch.empty((), dtype=config.dtype).element_size()
        return (2 * config.num_layers * config.num_kv_heads * config.head_dim
                * bytes_per_value)

    def allocate_kv_cache(self, num_blocks, block_size, dtype=None):
        config = self.config
        shape = (num_blocks, config.num_kv_heads, block_size, config.head_dim)
        dtype = dtype or config.dtype

        def empty_cache():
            return torch.zeros(shape, dtype=dtype, device=self.device)

        return [(empty_cache(), empty_cache()) for _ in range(config.num_layers)]

    # ------------------------------------------------------------ forward

    def _split_heads(self, qkv, num_tokens):
        config = self.config
        query_size = config.num_heads * config.head_dim
        kv_size = config.num_kv_heads * config.head_dim
        query, key, value = qkv.split([query_size, kv_size, kv_size], dim=-1)
        return (query.view(num_tokens, config.num_heads, config.head_dim),
                key.view(num_tokens, config.num_kv_heads, config.head_dim),
                value.view(num_tokens, config.num_kv_heads, config.head_dim))

    def _attention_block(self, layer_index, layer, hidden, cos, sin, attention):
        eps = self.config.rms_norm_eps
        normed = rms_norm(hidden, layer.attention_norm, eps)
        qkv = linear(normed, layer.qkv_proj, layer.qkv_bias)
        query, key, value = self._split_heads(qkv, hidden.shape[0])
        # Qwen3 normalises q and k for each head, before RoPE. Without that,
        # the model writes fluent nonsense and gives no error.
        if layer.q_norm is not None:
            query = rms_norm(query, layer.q_norm, eps)
            key = rms_norm(key, layer.k_norm, eps)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        attended = attention(layer_index, query, key, value.contiguous())
        return hidden + linear(attended, layer.o_proj)

    def _mlp_block(self, layer, hidden):
        normed = rms_norm(hidden, layer.mlp_norm, self.config.rms_norm_eps)
        gate, up = linear(normed, layer.gate_up_proj).chunk(2, dim=-1)
        return hidden + linear(F.silu(gate) * up, layer.down_proj)

    @torch.no_grad()
    def forward(self, token_ids, positions, attention, logits_rows):
        hidden = F.embedding(token_ids, self.embedding)
        cos, sin = rope_tables(positions, self.inverse_frequencies,
                               self.rope_attention_scale)
        for layer_index, layer in enumerate(self.layers):
            hidden = self._attention_block(layer_index, layer, hidden, cos, sin,
                                           attention)
            hidden = self._mlp_block(layer, hidden)
        # Select the rows BEFORE the lm_head. A 512-token prefill projected at
        # every position is 300 MB of logits, and you keep one row.
        hidden = rms_norm(hidden.index_select(0, logits_rows), self.final_norm,
                          self.config.rms_norm_eps)
        return linear(hidden, self.lm_head).float()


# ---------------------------------------------------------------- reference


class DenseReference:
    """A slow attention backend with no paging, for one sequence at a time.

    It keeps a plain K and V for each layer, which grow by concatenation. The
    tests check this model against HF with it, so that a failure in your
    paged backend is never a failure of this file."""

    def __init__(self, num_layers):
        self.keys = [None] * num_layers
        self.values = [None] * num_layers

    def _append(self, layer, key, value):
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = key, value
        else:
            self.keys[layer] = torch.cat([self.keys[layer], key])
            self.values[layer] = torch.cat([self.values[layer], value])

    def __call__(self, layer, query, key, value):
        self._append(layer, key, value)
        return attend_to_context(query, self.keys[layer].transpose(0, 1),
                                 self.values[layer].transpose(0, 1))


def load_model(name=DEFAULT_MODEL, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    hf_model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    hf_model = hf_model.to(device).eval()
    for parameter in hf_model.parameters():
        parameter.requires_grad_(False)
    return Model(hf_model, tokenizer, name)
