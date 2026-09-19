"""Reference solution, stage 24b - the KV cache in FP8."""

import math
from dataclasses import dataclass

import torch

from app.s21_paged_runner import PagedAttention
from app.s23_graphs import GraphedModelRunner
from cudalib import build
from tvllm import DenseReference

SOURCE = "app/cuda/s24b_kv_fp8.cu"
FP8_MAX = 448.0            # the largest value that e4m3 holds


def _extension():
    return build("s24b_kv_fp8", SOURCE)


@dataclass
class KVScale:
    """The scales of one layer: stored value = real value / scale."""
    key: float
    value: float


def write_kv_fp8(key_cache, value_cache, key, value, slots, scale):
    _extension().write_kv_fp8(key_cache, value_cache, key.contiguous(),
                              value.contiguous(), slots, scale.key, scale.value)


def paged_attention_fp8(query, key_cache, value_cache, block_tables,
                        context_lens, scale, softmax_scale=None, splits=0):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query.shape[-1])
    return _extension().paged_attn_fp8(query.contiguous(), key_cache,
                                       value_cache, block_tables, context_lens,
                                       float(softmax_scale), scale.key,
                                       scale.value, int(splits))


class _MaxRecorder(DenseReference):
    """Dense attention that also records the largest |key| and |value| of
    each layer."""

    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.max_key = [0.0] * num_layers
        self.max_value = [0.0] * num_layers

    def __call__(self, layer, query, key, value):
        self.max_key[layer] = max(self.max_key[layer], key.abs().max().item())
        self.max_value[layer] = max(self.max_value[layer],
                                    value.abs().max().item())
        return super().__call__(layer, query, key, value)


@torch.no_grad()
def calibrate_kv_scales(model, token_ids, margin=2.0):
    """One KVScale for each layer: max / 448, times a margin, because a later
    prompt can hold a larger value than this one."""
    recorder = _MaxRecorder(model.config.num_layers)
    positions = torch.arange(len(token_ids), device=model.device)
    model.forward(torch.tensor(token_ids, device=model.device), positions,
                  recorder, positions[-1:])

    def to_scale(maximum):
        return max(maximum, 1e-6) * margin / FP8_MAX

    return [KVScale(to_scale(k), to_scale(v))
            for k, v in zip(recorder.max_key, recorder.max_value)]


class Fp8PagedAttention(PagedAttention):
    """The stage 21 backend, with three steps changed for an FP8 cache."""

    def __init__(self, kv_caches, block_size, scales):
        super().__init__(kv_caches, block_size)
        self.scales = scales

    def write(self, layer, key, value):
        key_cache, value_cache = self.kv_caches[layer]
        write_kv_fp8(key_cache, value_cache, key, value,
                     self.metadata.slot_mapping, self.scales[layer])

    def attend_decodes(self, layer, query):
        key_cache, value_cache = self.kv_caches[layer]
        return paged_attention_fp8(query, key_cache, value_cache,
                                   self.metadata.block_tables,
                                   self.metadata.context_lens,
                                   self.scales[layer])

    def load_context(self, layer, chunk, dtype):
        keys, values = super().load_context(layer, chunk, dtype)
        scale = self.scales[layer]
        return keys * scale.key, values * scale.value


class Fp8GraphedModelRunner(GraphedModelRunner):
    """The stage 23 runner with an FP8 cache: half the bytes to store, and
    half to read in each decode step."""

    def __init__(self, model, num_blocks, scales, block_size=16, **kwargs):
        self.scales = scales
        super().__init__(model, num_blocks, block_size, **kwargs)

    def allocate_cache(self):
        return self.model.allocate_kv_cache(self.num_blocks, self.block_size,
                                            dtype=torch.float8_e4m3fn)

    def make_backend(self):
        return Fp8PagedAttention(self.kv_caches, self.block_size, self.scales)


def kv_bytes_per_token_fp8(model):
    config = model.config
    return 2 * config.num_layers * config.num_kv_heads * config.head_dim
