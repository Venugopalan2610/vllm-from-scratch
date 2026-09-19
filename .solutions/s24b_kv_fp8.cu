// Reference solution, stage 24b - the KV cache in FP8.
//
// Your stage 08c kernel, with the cache read as FP8. The query and the output
// stay in bf16. Only the loads of K and V change. Two scales move to where
// they cost one multiply: key_scale into the query, value_scale into the
// output.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 16;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kThreads / kWarp;
constexpr unsigned kFullMask = 0xffffffffu;

// A split shorter than this does not justify a block: the launch and the
// merge cost more than the tokens save.
constexpr int kMinChunk = 128;
constexpr int kMaxSplits = 32;

// 16 FP8 values are one 16-byte load. 8 lanes cover a row of head_dim 128,
// and a block keeps 16 rows in flight.
constexpr int kFp8Vec = 16;
constexpr float kFp8Max = 448.f;          // the largest e4m3 value
constexpr int kWriteThreads = 256;
constexpr int kMaxWriteBlocks = 4096;

using fp8 = __nv_fp8_e4m3;

struct PagedShape {
  int num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq;
  int num_splits;
};

// One partial softmax state for each (seq, head, split).
struct SplitStates {
  float* accumulators;   // (num_seqs, num_heads, num_splits, head_dim)
  float* maxima;         // (num_seqs, num_heads, num_splits)
  float* sums;
};

// Two FP8 values to two floats with one conversion instruction, through
// half2. A conversion of each value alone costs twice the instructions, and
// at long context the kernel then waits on arithmetic, not on memory.
__device__ __forceinline__ float2 fp8x2_to_float2(__nv_fp8x2_storage_t pair) {
  const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&raw));
}

template <typename scalar_t, int VEC>
__device__ __forceinline__ void load_vec(float* target,
                                         const scalar_t* source) {
  if constexpr (std::is_same_v<scalar_t, fp8> && VEC == kFp8Vec) {
    const uint4 raw = *reinterpret_cast<const uint4*>(source);
    const __nv_fp8x2_storage_t* pairs =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
      const float2 two = fp8x2_to_float2(pairs[i]);
      target[2 * i] = two.x;
      target[2 * i + 1] = two.y;
    }
  } else if constexpr (VEC * sizeof(scalar_t) == 16) {
    const float4 raw = *reinterpret_cast<const float4*>(source);
    const scalar_t* values = reinterpret_cast<const scalar_t*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC; ++i) target[i] = static_cast<float>(values[i]);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) target[i] = static_cast<float>(source[i]);
  }
}

// --- warp primitives -------------------------------------------------------
//
// xor is a butterfly: after the loop every lane of the group holds the
// result, so there is no broadcast step. shfl_down leaves it in lane 0 only.

__device__ __forceinline__ float warp_sum(float value, int width) {
  for (int stride = width / 2; stride > 0; stride >>= 1)
    value += __shfl_xor_sync(kFullMask, value, stride);
  return value;
}

__device__ __forceinline__ float warp_max(float value, int width) {
  for (int stride = width / 2; stride > 0; stride >>= 1)
    value = fmaxf(value, __shfl_xor_sync(kFullMask, value, stride));
  return value;
}

// Both block reductions end with TWO barriers, and the second one is
// necessary. Every thread reads scratch[0] to get the answer, and the next
// call writes scratch[0] in its first round. Without a barrier between the
// read and that write, a fast warp overwrites the result before a slow warp
// reads it. The bug is intermittent, it depends on the data, and it looks
// like bad arithmetic.
__device__ __forceinline__ float block_max(float value, float* scratch) {
  const int thread = threadIdx.x;
  value = warp_max(value, kWarp);
  if (thread % kWarp == 0) scratch[thread / kWarp] = value;
  __syncthreads();
  value = thread < kWarpsPerBlock ? scratch[thread] : -INFINITY;
  if (thread < kWarp) value = warp_max(value, kWarpsPerBlock);
  if (thread == 0) scratch[0] = value;
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int thread = threadIdx.x;
  value = warp_sum(value, kWarp);
  if (thread % kWarp == 0) scratch[thread / kWarp] = value;
  __syncthreads();
  value = thread < kWarpsPerBlock ? scratch[thread] : 0.f;
  if (thread < kWarp) value = warp_sum(value, kWarpsPerBlock);
  if (thread == 0) scratch[0] = value;
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

// An empty split still reports: an empty state, or a zero output.
template <typename scalar_t, bool SINGLE>
__device__ void report_empty_split(scalar_t* output, SplitStates states,
                                   int64_t output_offset, int64_t state_index,
                                   int head_dim) {
  for (int dim = threadIdx.x; dim < head_dim; dim += kThreads) {
    if constexpr (SINGLE)
      output[output_offset + dim] = static_cast<scalar_t>(0.f);
    else
      states.accumulators[state_index * head_dim + dim] = 0.f;
  }
  if (!SINGLE && threadIdx.x == 0) {
    states.maxima[state_index] = -INFINITY;
    states.sums[state_index] = 0.f;
  }
}

// --- the split kernel ------------------------------------------------------
//
// The loop below has no __syncthreads, and that is the point.
//
// Every lane of a row group ends the butterfly with the same score. So every
// lane of a group runs the same online softmax, and gets the same max and
// sum with no message. Each group is an INDEPENDENT partial softmax over its
// own stride of the context, as a split is.
//
// So the kernel does not need the block-wide max and sum of stage 08b for
// each tile. One merge at the end replaces them. It merges the partial
// states of the row groups, with the same exp(max_r - M) rescale that merges
// the splits. The same method works at three levels: tile, group and block.

template <typename scalar_t, typename cache_t, int VEC, bool SINGLE>
__global__ void __launch_bounds__(kThreads) paged_attn_split(
    scalar_t* __restrict__ output,              // (num_seqs, num_heads, D), SINGLE only
    SplitStates states,                         // not SINGLE only
    const scalar_t* __restrict__ query,         // (num_seqs, num_heads, D)
    const cache_t* __restrict__ key_cache,      // (num_blocks, kv_heads, block_size, D)
    const cache_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,   // (num_seqs, max_blocks_per_seq)
    const int32_t* __restrict__ context_lens,   // (num_seqs,)
    const float scale,                          // softmax scale * key_scale
    const float value_scale, const PagedShape shape) {
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int split = blockIdx.z;
  const int head_dim = shape.head_dim;
  const int kv_head = head / (shape.num_heads / shape.num_kv_heads);
  const int thread = threadIdx.x;
  const int32_t* block_table = block_tables + seq * shape.max_blocks_per_seq;

  const int context_len = context_lens[seq];
  const int chunk_len = (context_len + shape.num_splits - 1) / shape.num_splits;
  const int chunk_start = split * chunk_len;
  const int chunk_end = min(context_len, chunk_start + chunk_len);

  const int lanes_per_row = head_dim / VEC;
  const int rows_per_tile = kThreads / lanes_per_row;
  const int lane = thread % lanes_per_row;
  const int row = thread / lanes_per_row;

  extern __shared__ float shared[];
  float* scaled_query = shared;                         // head_dim, then partials
  float* row_maxima = shared + kThreads * kMaxVec;      // rows_per_tile
  float* row_sums = row_maxima + kThreads;              // rows_per_tile

  const int64_t output_offset = ((int64_t)seq * shape.num_heads + head) * head_dim;
  const int64_t state_index =
      ((int64_t)seq * shape.num_heads + head) * shape.num_splits + split;

  for (int dim = thread; dim < head_dim; dim += kThreads)
    scaled_query[dim] = static_cast<float>(query[output_offset + dim]) * scale;
  __syncthreads();

  if (chunk_start >= chunk_end) {
    report_empty_split<scalar_t, SINGLE>(output, states, output_offset,
                                         state_index, head_dim);
    return;
  }

  float query_slice[kMaxVec], accumulator[kMaxVec];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    query_slice[i] = scaled_query[lane * VEC + i];
    accumulator[i] = 0.f;
  }
  float running_max = -INFINITY, running_sum = 0.f;

  // Every group walks the same NUMBER of positions. So no lane of a warp
  // leaves the loop early and makes the others wait at a shuffle.
  const int num_tiles = (chunk_end - chunk_start + rows_per_tile - 1) / rows_per_tile;
  __syncthreads();                             // scaled_query is not needed now

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int position = chunk_start + tile * rows_per_tile + row;
    const bool in_chunk = position < chunk_end;

    int64_t row_offset = 0;
    float partial_score = 0.f;
    if (in_chunk) {
      const int64_t block_id = block_table[position / shape.block_size];
      row_offset = ((block_id * shape.num_kv_heads + kv_head) * shape.block_size +
                    position % shape.block_size) * head_dim + lane * VEC;
      float key_slice[kMaxVec];
      load_vec<cache_t, VEC>(key_slice, key_cache + row_offset);
#pragma unroll
      for (int i = 0; i < VEC; ++i) partial_score += query_slice[i] * key_slice[i];
    }
    // Every lane of the warp must reach the shuffle, in the chunk or not. A
    // _sync primitive whose mask names a lane that never arrives does not
    // return. At the warp level, this trap replaces __syncthreads.
    const float row_score = warp_sum(partial_score, lanes_per_row);
    const float score = in_chunk ? row_score : -INFINITY;

    const float new_max = fmaxf(running_max, score);
    // A whole group can be empty: more groups than positions left. Both
    // maxima are then -inf, and -inf minus -inf is NaN, which poisons the
    // merge for all. Stage 08b cannot get this, because its maximum came
    // from the whole block, and the block always had a position.
    const float rescale = new_max == -INFINITY ? 0.f : __expf(running_max - new_max);
    const float prob = in_chunk ? __expf(score - new_max) : 0.f;
    running_sum = running_sum * rescale + prob;
    running_max = new_max;

    float value_slice[kMaxVec];
    if (in_chunk) load_vec<cache_t, VEC>(value_slice, value_cache + row_offset);
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      accumulator[i] = accumulator[i] * rescale + (in_chunk ? prob * value_slice[i] : 0.f);
  }

  // --- merge the partial states of the row groups. One time, not per tile.
  float* partials = shared;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC; ++i) partials[thread * kMaxVec + i] = accumulator[i];
  if (lane == 0) {
    row_maxima[row] = running_max;
    row_sums[row] = running_sum;
  }
  __syncthreads();
  if (row != 0) return;

  float block_max_value = -INFINITY;
  for (int other = 0; other < rows_per_tile; ++other)
    block_max_value = fmaxf(block_max_value, row_maxima[other]);

  float block_sum_value = 0.f;
  float merged[kMaxVec];
#pragma unroll
  for (int i = 0; i < VEC; ++i) merged[i] = 0.f;
  for (int other = 0; other < rows_per_tile; ++other) {
    const float weight = __expf(row_maxima[other] - block_max_value);
    block_sum_value += row_sums[other] * weight;
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      merged[i] += partials[(other * lanes_per_row + lane) * kMaxVec + i] * weight;
  }

  // One split has nothing to merge. Normalize here and write the answer. Do
  // not write a partial state to memory for a second kernel to read back.
  // That round trip is small next to the K and V reads. The launch and its
  // allocations are not small. At a full grid they cost and give nothing.
  if constexpr (SINGLE) {
    // value_scale comes out of the sum: the output is linear in V.
    const float inverse_sum =
        block_sum_value > 0.f ? value_scale / block_sum_value : 0.f;
    scalar_t* output_slice = output + output_offset + lane * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      output_slice[i] = static_cast<scalar_t>(merged[i] * inverse_sum);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      states.accumulators[state_index * head_dim + lane * VEC + i] =
          merged[i] * value_scale;
    if (lane == 0) {
      states.maxima[state_index] = block_max_value;
      states.sums[state_index] = block_sum_value;
    }
  }
}

// --- the merge -------------------------------------------------------------
//
// The online-softmax rescale, one level up: each split is a tile.

template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) combine_splits(
    scalar_t* __restrict__ output,              // (num_seqs, num_heads, D)
    const SplitStates states, const int num_heads, const int head_dim,
    const int num_splits) {
  const int thread = threadIdx.x;
  const int64_t output_offset =
      ((int64_t)blockIdx.x * num_heads + blockIdx.y) * head_dim;
  const int64_t first_state = ((int64_t)blockIdx.x * num_heads + blockIdx.y) * num_splits;

  extern __shared__ float shared[];
  float* scratch = shared;                        // kWarpsPerBlock
  float* split_weights = shared + kWarpsPerBlock; // num_splits

  const bool owns_split = thread < num_splits;
  const float global_max = block_max(
      owns_split ? states.maxima[first_state + thread] : -INFINITY, scratch);
  if (owns_split)
    split_weights[thread] = __expf(states.maxima[first_state + thread] - global_max);
  __syncthreads();

  const float total = block_sum(
      owns_split ? states.sums[first_state + thread] * split_weights[thread] : 0.f,
      scratch);
  const float inverse_total = total > 0.f ? 1.f / total : 0.f;

  for (int dim = thread; dim < head_dim; dim += kThreads) {
    float weighted_sum = 0.f;
    for (int split = 0; split < num_splits; ++split)
      weighted_sum += states.accumulators[(first_state + split) * head_dim + dim] *
                      split_weights[split];
    output[output_offset + dim] = static_cast<scalar_t>(weighted_sum * inverse_total);
  }
}

int choose_splits(int num_seqs, int num_heads, int max_context) {
  // Enough blocks to give every SM work, and never so many that a split is
  // shorter than kMinChunk.
  const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int num_blocks = num_seqs * num_heads;
  const int to_fill_sms = (2 * num_sms + num_blocks - 1) / num_blocks;
  const int by_context = std::max(1, max_context / kMinChunk);
  return std::max(1, std::min({to_fill_sms, by_context, kMaxSplits}));
}


// --- the FP8 write ---------------------------------------------------------
//
// Stage 08 wrote K and V as they came. Here the kernel divides each value by
// the scale of its layer and rounds it to e4m3, which holds values up to 448.
// The kernel clamps a larger value. The scale comes from a calibration run,
// so a clamp is rare, and it is a small error, not a NaN.
__device__ __forceinline__ fp8 to_fp8(float value, float inverse_scale) {
  return fp8(fminf(fmaxf(value * inverse_scale, -kFp8Max), kFp8Max));
}

template <typename scalar_t>
__global__ void write_kv_fp8_kernel(
    fp8* __restrict__ key_cache, fp8* __restrict__ value_cache,
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    const int64_t* __restrict__ slots, const float inverse_key_scale,
    const float inverse_value_scale, const int num_tokens,
    const int num_kv_heads, const int head_dim, const int block_size) {
  const int64_t num_values = (int64_t)num_tokens * num_kv_heads * head_dim;
  for (int64_t index = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       index < num_values; index += (int64_t)gridDim.x * blockDim.x) {
    const int dim = index % head_dim;
    const int kv_head = (index / head_dim) % num_kv_heads;
    const int token = index / (head_dim * num_kv_heads);
    const int64_t slot = slots[token];
    if (slot < 0) continue;               // padding: not a real token
    const int64_t target =
        (((slot / block_size) * num_kv_heads + kv_head) * block_size +
         slot % block_size) * head_dim + dim;
    key_cache[target] = to_fp8(static_cast<float>(key[index]), inverse_key_scale);
    value_cache[target] = to_fp8(static_cast<float>(value[index]), inverse_value_scale);
  }
}

template <typename scalar_t>
void launch_write(torch::Tensor& key_cache, torch::Tensor& value_cache,
                  const torch::Tensor& key, const torch::Tensor& value,
                  const torch::Tensor& slot_indices, double key_scale,
                  double value_scale, int num_blocks) {
  write_kv_fp8_kernel<scalar_t><<<num_blocks, kWriteThreads, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<fp8*>(key_cache.data_ptr()),
      reinterpret_cast<fp8*>(value_cache.data_ptr()),
      key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
      slot_indices.data_ptr<int64_t>(), (float)(1.0 / key_scale),
      (float)(1.0 / value_scale), key.size(0), key.size(1), key.size(2),
      key_cache.size(2));
}

struct AttentionTensors {
  torch::Tensor output, query, key_cache, value_cache, block_tables, context_lens;
};

template <typename scalar_t>
void launch(AttentionTensors& tensors, float scale, float value_scale,
            PagedShape shape) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int num_seqs = tensors.query.size(0);
  const int head_dim = shape.head_dim;
  const size_t split_shared_bytes =
      (std::max<size_t>(head_dim, kThreads * kMaxVec) + 2 * kThreads) * sizeof(float);
  const scalar_t* query = tensors.query.data_ptr<scalar_t>();
  const fp8* key_cache = reinterpret_cast<const fp8*>(tensors.key_cache.data_ptr());
  const fp8* value_cache = reinterpret_cast<const fp8*>(tensors.value_cache.data_ptr());
  const int32_t* block_tables = tensors.block_tables.data_ptr<int32_t>();
  const int32_t* context_lens = tensors.context_lens.data_ptr<int32_t>();
  scalar_t* output = tensors.output.data_ptr<scalar_t>();

  if (shape.num_splits == 1) {
    paged_attn_split<scalar_t, fp8, kFp8Vec, true>
        <<<dim3(num_seqs, shape.num_heads, 1), kThreads, split_shared_bytes, stream>>>(
            output, SplitStates{nullptr, nullptr, nullptr}, query, key_cache,
            value_cache, block_tables, context_lens, scale, value_scale, shape);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  auto float_options = tensors.query.options().dtype(torch::kFloat);
  auto accumulators = torch::empty({num_seqs, shape.num_heads, shape.num_splits, head_dim},
                                   float_options);
  auto maxima = torch::empty({num_seqs, shape.num_heads, shape.num_splits}, float_options);
  auto sums = torch::empty({num_seqs, shape.num_heads, shape.num_splits}, float_options);
  const SplitStates states{accumulators.data_ptr<float>(), maxima.data_ptr<float>(),
                           sums.data_ptr<float>()};

  paged_attn_split<scalar_t, fp8, kFp8Vec, false>
      <<<dim3(num_seqs, shape.num_heads, shape.num_splits), kThreads,
         split_shared_bytes, stream>>>(nullptr, states, query, key_cache,
                                       value_cache, block_tables, context_lens,
                                       scale, value_scale, shape);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t combine_shared_bytes = (kWarpsPerBlock + shape.num_splits) * sizeof(float);
  combine_splits<scalar_t>
      <<<dim3(num_seqs, shape.num_heads), kThreads, combine_shared_bytes, stream>>>(
          output, states, shape.num_heads, head_dim, shape.num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void write_kv_fp8(torch::Tensor key_cache, torch::Tensor value_cache,
                  torch::Tensor key, torch::Tensor value, torch::Tensor slots,
                  double key_scale, double value_scale) {
  TORCH_CHECK(key_cache.scalar_type() == at::kFloat8_e4m3fn,
              "the cache must be float8_e4m3fn");
  TORCH_CHECK(key.is_contiguous() && value.is_contiguous(),
              "key and value must be contiguous");
  if (key.size(0) == 0) return;
  auto slot_indices = slots.to(torch::kLong).contiguous();
  const int64_t num_values = key.numel();
  const int num_blocks = (int)std::min<int64_t>(
      (num_values + kWriteThreads - 1) / kWriteThreads, kMaxWriteBlocks);

  AT_DISPATCH_SWITCH(
      key.scalar_type(), "write_kv_fp8",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        launch_write<scalar_t>(key_cache, value_cache, key, value, slot_indices,
                               key_scale, value_scale, num_blocks);
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        launch_write<scalar_t>(key_cache, value_cache, key, value, slot_indices,
                               key_scale, value_scale, num_blocks);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor paged_attn_fp8(torch::Tensor query, torch::Tensor key_cache,
                             torch::Tensor value_cache,
                             torch::Tensor block_tables,
                             torch::Tensor context_lens, double scale,
                             double key_scale, double value_scale,
                             int64_t splits) {
  TORCH_CHECK(query.is_cuda() && query.is_contiguous(),
              "query must be contiguous and on the GPU");
  TORCH_CHECK(key_cache.scalar_type() == at::kFloat8_e4m3fn,
              "the cache must be float8_e4m3fn");
  const int num_seqs = query.size(0), num_heads = query.size(1);
  const int head_dim = query.size(2);
  const int num_kv_heads = key_cache.size(1), block_size = key_cache.size(2);
  const int max_blocks_per_seq = block_tables.size(1);
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be a multiple of num_kv_heads");
  TORCH_CHECK(head_dim % kFp8Vec == 0 && head_dim / kFp8Vec <= kThreads,
              "head_dim must divide by 16");

  const int num_splits = splits > 0 ? (int)splits
                                    : choose_splits(num_seqs, num_heads,
                                                    max_blocks_per_seq * block_size);
  const PagedShape shape{num_heads, num_kv_heads, head_dim, block_size,
                         max_blocks_per_seq, num_splits};
  AttentionTensors tensors{torch::empty_like(query), query, key_cache, value_cache,
                           block_tables.to(torch::kInt).contiguous(),
                           context_lens.to(torch::kInt).contiguous()};
  // key_scale goes into the query scale: q . (k8 * ks) = (q * ks) . k8.
  const float query_scale = (float)(scale * key_scale);

  AT_DISPATCH_SWITCH(
      query.scalar_type(), "paged_attn_fp8",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        launch<scalar_t>(tensors, query_scale, (float)value_scale, shape);
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        launch<scalar_t>(tensors, query_scale, (float)value_scale, shape);
      }));
  return tensors.output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("write_kv_fp8", &write_kv_fp8, "scatter K and V into an FP8 cache");
  m.def("paged_attn_fp8", &paged_attn_fp8,
        "split-K paged decode attention over an FP8 cache");
}
