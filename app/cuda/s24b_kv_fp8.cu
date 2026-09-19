// Stage 24b - the KV cache in FP8.
//
// `./vc lore 24b`. `./vc test 24b`.
//
// Start from your stage 08c kernel. The query and the output stay in bf16.
// Only the cache changes: it holds __nv_fp8_e4m3 values, one byte each.
//
// ---------------------------------------------------------------------------
// THE CHANGES IN THE ATTENTION KERNEL
// ---------------------------------------------------------------------------
//
//   1. A second template type. scalar_t stays the type of the query and the
//      output. A new cache_t is the type of the caches. Your load_vec must
//      work for both.
//
//   2. The load path. Use VEC = 16 for the cache: 16 FP8 values are one
//      16-byte load (a uint4). Then 8 lanes cover one row of head_dim 128,
//      and a block keeps 16 rows in flight. kMaxVec must increase to 16.
//
//      Convert two values at a time: __nv_cvt_fp8x2_to_halfraw2 changes two
//      FP8 bytes into a half2, and __half22float2 changes that into two
//      floats. One conversion for each value costs twice the instructions,
//      and at long context the kernel then waits on arithmetic, not on
//      memory.
//
//   3. The scales, where they cost one multiply:
//
//          q . (k8 * key_scale)        =  (q * key_scale) . k8     into the query scale
//          sum p * (v8 * value_scale)  =  value_scale * sum p * v8 into the output
//
//      With splits, apply value_scale to each partial accumulator before the
//      merge. The merge is linear, so the result is the same.
//
// ---------------------------------------------------------------------------
// THE WRITE KERNEL
// ---------------------------------------------------------------------------
//
// Your stage 08 write kernel, with a conversion. Divide each value by its
// scale, clamp it to [-448, 448], and store __nv_fp8_e4m3(value). Skip a slot
// of -1, as before. A clamp is rare, because the scale comes from a
// calibration run with a margin. It is a small error, not a NaN.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
//
//   - torch keeps the cache as torch.float8_e4m3fn. Get its pointer with
//     data_ptr() and reinterpret_cast it to __nv_fp8_e4m3*. There is no
//     data_ptr<__nv_fp8_e4m3>().
//   - Check the dtype with TORCH_CHECK(key_cache.scalar_type() ==
//     at::kFloat8_e4m3fn). A bf16 cache given here reads garbage, with no
//     error.
//   - #include <cuda_fp16.h> and <cuda_fp8.h>.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {

using fp8 = __nv_fp8_e4m3;

// TODO: your stage 08c kernel, with cache_t, the FP8 load and the scales.
// TODO: write_kv_fp8_kernel

}  // namespace

void write_kv_fp8(torch::Tensor key_cache, torch::Tensor value_cache,
                  torch::Tensor key, torch::Tensor value, torch::Tensor slots,
                  double key_scale, double value_scale) {
  TORCH_CHECK(false, "stage 24b: implement write_kv_fp8");
}

torch::Tensor paged_attn_fp8(torch::Tensor query, torch::Tensor key_cache,
                             torch::Tensor value_cache,
                             torch::Tensor block_tables,
                             torch::Tensor context_lens, double scale,
                             double key_scale, double value_scale,
                             int64_t splits) {
  TORCH_CHECK(false, "stage 24b: implement paged_attn_fp8");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("write_kv_fp8", &write_kv_fp8, "scatter K and V into an FP8 cache");
  m.def("paged_attn_fp8", &paged_attn_fp8,
        "split-K paged decode attention over an FP8 cache");
}
