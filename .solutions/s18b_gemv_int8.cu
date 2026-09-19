// Reference solution, stage 18b - int8 weight-only GEMV, dequantized after
// the dot product.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kThreads / kWarp;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kSharedBytes = 32768;       // the budget for the staged activations
constexpr int kChunk = 16;                // int8 weights in one 16-byte load
constexpr int kRowsPerWarp = 4;

// The largest batch for this kernel. Above it, the batch spreads the weight
// read far enough that a real GEMM is faster. Call one.
constexpr int kMaxBatch = 8;

__device__ __forceinline__ float warp_sum(float value) {
  for (int stride = kWarp / 2; stride > 0; stride >>= 1)
    value += __shfl_xor_sync(kFullMask, value, stride);
  return value;
}

// 16 values with 16-byte vector loads: 2 loads for a 2-byte type, 4 for float.
template <typename scalar_t>
__device__ __forceinline__ void load16(const scalar_t* source, float* target) {
  constexpr int kLoads = kChunk * sizeof(scalar_t) / 16;
  uint4 raw[kLoads];
#pragma unroll
  for (int load = 0; load < kLoads; ++load)
    raw[load] = reinterpret_cast<const uint4*>(source)[load];
  const scalar_t* values = reinterpret_cast<const scalar_t*>(raw);
#pragma unroll
  for (int i = 0; i < kChunk; ++i) target[i] = static_cast<float>(values[i]);
}

// Copy columns [tile_start, tile_start + tile_len) of every batch row into
// shared memory. All threads of the block take part.
template <typename scalar_t>
__device__ void stage_activations(scalar_t* staged, const scalar_t* inputs,
                                  int batch_size, int in_features,
                                  int tile_start, int tile_len) {
  __syncthreads();
  for (int index = threadIdx.x; index < tile_len * batch_size;
       index += kThreads) {
    const int batch_row = index / tile_len, column = index % tile_len;
    staged[batch_row * tile_len + column] =
        inputs[(int64_t)batch_row * in_features + tile_start + column];
  }
  __syncthreads();
}

// One warp owns kRowsPerWarp output rows, and it serves every batch row.
//
// There are two reuse decisions, and they go in opposite directions:
//
//   The kernel reads each WEIGHT one time, and never again. A copy into
//   shared memory has no purpose. The weights go directly from global memory
//   into registers, 16 bytes for each lane.
//
//   Every output row reads the ACTIVATIONS. They go to shared memory one
//   time for each block. So a block must own MANY output rows. If not, it
//   stages the activations again and again. With 4 rows for each block, the
//   activation traffic at a batch of 8 is four times the weight traffic.
//
// So each warp owns kRowsPerWarp rows. A lane loads 16 activations of one
// batch row from shared memory one time. Those 16 values serve all the
// weight rows of the warp. The block stages the activations one time for
// all its rows.
template <typename scalar_t, bool VECTOR_LOADS, int MAX_BATCH>
__global__ void __launch_bounds__(kThreads) gemv_int8(
    scalar_t* __restrict__ outputs,          // (batch, out_features)
    const scalar_t* __restrict__ inputs,     // (batch, in_features)
    const int8_t* __restrict__ weight,       // (out_features, in_features)
    const float* __restrict__ scales,        // (out_features,)
    const int batch_size, const int out_features, const int in_features,
    const int max_tile_len) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int first_row = (blockIdx.x * kWarpsPerBlock + warp) * kRowsPerWarp;
  // No early return: stage_activations has block-wide barriers.
  extern __shared__ char shared_bytes[];
  scalar_t* staged = reinterpret_cast<scalar_t*>(shared_bytes);

  const int8_t* weight_rows[kRowsPerWarp];
  bool row_exists[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    row_exists[r] = first_row + r < out_features;
    weight_rows[r] = weight + (int64_t)(row_exists[r] ? first_row + r : 0) * in_features;
  }
  float sums[kRowsPerWarp][MAX_BATCH];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) sums[r][b] = 0.f;

  for (int tile_start = 0; tile_start < in_features; tile_start += max_tile_len) {
    const int tile_len = min(max_tile_len, in_features - tile_start);
    stage_activations(staged, inputs, batch_size, in_features, tile_start, tile_len);
    if (first_row >= out_features) continue;

    if constexpr (VECTOR_LOADS) {
      for (int chunk = lane; chunk < tile_len / kChunk; chunk += kWarp) {
        int4 packed[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r)
          packed[r] = row_exists[r]
              ? *reinterpret_cast<const int4*>(weight_rows[r] + tile_start + chunk * kChunk)
              : make_int4(0, 0, 0, 0);
#pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
          if (b >= batch_size) break;
          float activations[kChunk];
          load16(staged + b * tile_len + chunk * kChunk, activations);
#pragma unroll
          for (int r = 0; r < kRowsPerWarp; ++r) {
            const int8_t* weights = reinterpret_cast<const int8_t*>(&packed[r]);
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kChunk; ++i)
              dot += activations[i] * static_cast<float>(weights[i]);
            sums[r][b] += dot;
          }
        }
      }
    } else {
      for (int column = lane; column < tile_len; column += kWarp) {
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
          const float weight_value =
              row_exists[r] ? static_cast<float>(weight_rows[r][tile_start + column]) : 0.f;
#pragma unroll
          for (int b = 0; b < MAX_BATCH; ++b) {
            if (b >= batch_size) break;
            sums[r][b] += static_cast<float>(staged[b * tile_len + column]) * weight_value;
          }
        }
      }
    }
  }

  // The dequantize: one multiply, on a value already in a register.
  // A dequantize of the WEIGHTS writes in_features bf16 values to memory and
  // reads them back. This stage exists to avoid that traffic.
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const float scale = row_exists[r] ? scales[first_row + r] : 0.f;
#pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) {
      if (b >= batch_size) break;
      const float total = warp_sum(sums[r][b]);   // every lane must call it
      if (lane == 0 && row_exists[r])
        outputs[(int64_t)b * out_features + first_row + r] =
            static_cast<scalar_t>(total * scale);
    }
  }
}

struct GemvTensors {
  torch::Tensor outputs, inputs, weight, scales;
};

template <typename scalar_t, int MAX_BATCH>
void launch(GemvTensors& tensors) {
  const int batch_size = tensors.inputs.size(0);
  const int in_features = tensors.inputs.size(1);
  const int out_features = tensors.weight.size(0);
  int max_tile_len = kSharedBytes / (MAX_BATCH * (int)sizeof(scalar_t));
  max_tile_len = std::min(max_tile_len / kChunk * kChunk, in_features);
  const size_t shared_bytes = (size_t)max_tile_len * batch_size * sizeof(scalar_t);
  const int rows_per_block = kWarpsPerBlock * kRowsPerWarp;
  const dim3 grid((out_features + rows_per_block - 1) / rows_per_block);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto kernel = in_features % kChunk == 0 ? gemv_int8<scalar_t, true, MAX_BATCH>
                                          : gemv_int8<scalar_t, false, MAX_BATCH>;
  kernel<<<grid, kThreads, shared_bytes, stream>>>(
      tensors.outputs.data_ptr<scalar_t>(), tensors.inputs.data_ptr<scalar_t>(),
      tensors.weight.data_ptr<int8_t>(), tensors.scales.data_ptr<float>(),
      batch_size, out_features, in_features, max_tile_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// MAX_BATCH is a bound at compile time, so the sums can be registers.
template <typename scalar_t>
void launch_for_batch(GemvTensors& tensors) {
  const int batch_size = tensors.inputs.size(0);
  if (batch_size == 1) launch<scalar_t, 1>(tensors);
  else if (batch_size <= 2) launch<scalar_t, 2>(tensors);
  else if (batch_size <= 4) launch<scalar_t, 4>(tensors);
  else launch<scalar_t, 8>(tensors);
}

}  // namespace

torch::Tensor gemv_int8_cuda(torch::Tensor inputs, torch::Tensor weight,
                             torch::Tensor scales) {
  TORCH_CHECK(inputs.is_cuda() && weight.is_cuda(), "everything must be on the GPU");
  TORCH_CHECK(weight.dtype() == torch::kChar, "weights must be int8");
  TORCH_CHECK(weight.is_contiguous(), "weights must be contiguous");

  const auto rows = (inputs.dim() == 1 ? inputs.unsqueeze(0) : inputs).contiguous();
  const int batch_size = rows.size(0), in_features = rows.size(1);
  const int out_features = weight.size(0);
  TORCH_CHECK(weight.size(1) == in_features, "shape mismatch: weight is (",
              out_features, ",", weight.size(1), ") but inputs are (",
              batch_size, ",", in_features, ")");
  TORCH_CHECK(scales.numel() == out_features, "one scale for each output channel");
  TORCH_CHECK(batch_size <= kMaxBatch, "this kernel is a decode path: at ",
              batch_size, " rows the weight read is already amortized, so "
              "call a GEMM");

  GemvTensors tensors{torch::empty({batch_size, out_features}, rows.options()),
                      rows, weight, scales.to(torch::kFloat).contiguous()};
  AT_DISPATCH_SWITCH(
      rows.scalar_type(), "gemv_int8",
      AT_DISPATCH_CASE(at::kFloat, [&] { launch_for_batch<scalar_t>(tensors); })
      AT_DISPATCH_CASE(at::kHalf, [&] { launch_for_batch<scalar_t>(tensors); })
      AT_DISPATCH_CASE(at::kBFloat16, [&] { launch_for_batch<scalar_t>(tensors); }));
  return inputs.dim() == 1 ? tensors.outputs.squeeze(0) : tensors.outputs;
}

int64_t max_batch() { return kMaxBatch; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_int8", &gemv_int8_cuda,
        "int8 weight-only GEMV, dequantized after the dot product");
  m.def("max_batch", &max_batch, "the largest batch that this kernel accepts");
}
