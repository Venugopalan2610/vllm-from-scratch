"""Stage 24b - the KV cache in FP8.

`./vc lore 24b` for the insight. `./vc test 24b` to check yourself.

At long context the KV cache, not the weights, is most of the bytes of a
decode step. FP8 (e4m3) stores K and V in one byte. That halves the bytes to
read, and it doubles the tokens that fit in the pool.

The kernels are in app/cuda/s24b_kv_fp8.cu. Copy your stage 08c kernel into
it and change the load path. The spec is in the header of that file.

WHAT YOU ARE BUILDING

    KVScale(key, value)                given: the scales of one layer
    write_kv_fp8(key_cache, value_cache, key, value, slots, scale)
    paged_attention_fp8(query, key_cache, value_cache, block_tables,
                        context_lens, scale, softmax_scale=None, splits=0)
    calibrate_kv_scales(model, token_ids, margin=2.0) -> [KVScale], one a layer
    Fp8PagedAttention(kv_caches, block_size, scales)
        your stage 21 backend. Override write, attend_decodes and
        load_context, and nothing else.
    Fp8GraphedModelRunner(model, num_blocks, scales, block_size=16, **kwargs)
        your stage 23 runner. Override allocate_cache and make_backend.
    kv_bytes_per_token_fp8(model)

ONE SCALE FOR K AND ONE FOR V, IN EACH LAYER

e4m3 holds values up to 448. Store k / k_scale, with k_scale = amax / 448.
Get amax from a calibration prefill: run the model with a backend that records
the largest |k| and |v| of each layer. A later prompt can hold a larger value
than the calibration prompt, so multiply by a margin, and clamp at write.

The scales cost almost nothing in the kernel:

    q . (k8 * ks)  =  (q * ks) . k8        fold ks into the query scale
    sum p * (v8 * vs)  =  vs * sum p * v8  fold vs into the output

THE RUNNER

Stage 21 builds its cache and its backend with two methods, allocate_cache()
and make_backend(). Override both. model.allocate_kv_cache takes a dtype. Do
not allocate a bf16 cache first and replace it: that needs the memory of both
at the same time.

TRAPS

  - A prefill chunk still attends with SDPA over gathered blocks. So
    load_context converts the FP8 blocks and multiplies them by the scale.
  - Measure the accuracy through the DECODE kernel, one token at a time. A
    prefill through SDPA in bf16 does not test your FP8 kernel at all.
"""

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
    raise NotImplementedError("stage 24b: implement write_kv_fp8")


def paged_attention_fp8(query, key_cache, value_cache, block_tables,
                        context_lens, scale, softmax_scale=None, splits=0):
    raise NotImplementedError("stage 24b: implement paged_attention_fp8")


class _MaxRecorder(DenseReference):
    """Dense attention that also records the largest |key| and |value| of
    each layer."""

    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.max_key = [0.0] * num_layers
        self.max_value = [0.0] * num_layers

    def __call__(self, layer, query, key, value):
        raise NotImplementedError("stage 24b: implement _MaxRecorder.__call__")


@torch.no_grad()
def calibrate_kv_scales(model, token_ids, margin=2.0):
    """One KVScale for each layer: max / 448, times a margin, because a later
    prompt can hold a larger value than this one."""
    raise NotImplementedError("stage 24b: implement calibrate_kv_scales")


class Fp8PagedAttention(PagedAttention):
    """The stage 21 backend, with three steps changed for an FP8 cache."""

    def __init__(self, kv_caches, block_size, scales):
        super().__init__(kv_caches, block_size)
        self.scales = scales

    def write(self, layer, key, value):
        raise NotImplementedError("stage 24b: implement Fp8PagedAttention.write")

    def attend_decodes(self, layer, query):
        raise NotImplementedError("stage 24b: implement Fp8PagedAttention.attend_decodes")

    def load_context(self, layer, chunk, dtype):
        raise NotImplementedError("stage 24b: implement Fp8PagedAttention.load_context")


class Fp8GraphedModelRunner(GraphedModelRunner):
    """The stage 23 runner with an FP8 cache: half the bytes to store, and
    half to read in each decode step."""

    def __init__(self, model, num_blocks, scales, block_size=16, **kwargs):
        self.scales = scales
        super().__init__(model, num_blocks, block_size, **kwargs)

    def allocate_cache(self):
        raise NotImplementedError("stage 24b: implement Fp8GraphedModelRunner.allocate_cache")

    def make_backend(self):
        raise NotImplementedError("stage 24b: implement Fp8GraphedModelRunner.make_backend")


def kv_bytes_per_token_fp8(model):
    raise NotImplementedError("stage 24b: implement kv_bytes_per_token_fp8")
