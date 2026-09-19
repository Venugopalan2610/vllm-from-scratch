// Reference solution, stage 08b - the same kernel, with good memory access.
//
// Stage 08 gave one thread one position, and the thread walked head_dim
// alone. The arithmetic does not change here. The change is which thread
// touches which byte:
//
//     stage 08   thread t reads key[position_t][0..D)     addresses D apart
//     stage 08b  thread t reads key[position][t*VEC..]    addresses 16 bytes apart
//
// The threads now cooperate ALONG head_dim. lanes_per_row = D/VEC threads
// share one row, and each thread moves 16 bytes. So a warp issues a few full
// 128-byte transactions, not 32 separate ones. The other
// rows_per_tile = 128/lanes_per_row groups work on other positions at the
// same time.
//
// The accumulator stays in registers. Each thread owns VEC elements of the
// output, for the positions that its group walks. The online-softmax rescale
// applies equally to every partial sum. So the sum across groups can wait
// until the end: one shared-memory reduction for each block, not one for
// each tile.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 8;

// 16 bytes is the widest load of one thread, and the width at which the
// requests of a warp join into whole 128-byte transactions. Everything below
// exists to get it.
template <typename scalar_t, int VEC>
__device__ __forceinline__ void load_vec(float* target,
                                         const scalar_t* source) {
  if constexpr (VEC * sizeof(scalar_t) == 16) {
    const float4 raw = *reinterpret_cast<const float4*>(source);
    const scalar_t* values = reinterpret_cast<const scalar_t*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC; ++i) target[i] = static_cast<float>(values[i]);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) target[i] = static_cast<float>(source[i]);
  }
}

// Tree reductions through shared memory. Every thread must call them.
// scratch holds kThreads floats. All threads get the result.
__device__ float block_max(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__device__ float block_sum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float result = scratch[0];
  __syncthreads();
  return result;
}

// The sum over the lanes_per_row threads of one row. Only lane 0 of each row
// gets the result.
__device__ float row_sum(float value, int lane, int lanes_per_row,
                         float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
    if (lane < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float result = scratch[threadIdx.x];
  __syncthreads();
  return result;
}

template <typename scalar_t, int VEC>
__global__ void __launch_bounds__(kThreads) paged_attn_vec(
    scalar_t* __restrict__ output,              // (num_seqs, num_heads, D)
    const scalar_t* __restrict__ query,         // (num_seqs, num_heads, D)
    const scalar_t* __restrict__ key_cache,     // (num_blocks, kv_heads, block_size, D)
    const scalar_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,   // (num_seqs, max_blocks_per_seq)
    const int32_t* __restrict__ context_lens,   // (num_seqs,)
    const float scale, const int num_heads, const int num_kv_heads,
    const int head_dim, const int block_size, const int max_blocks_per_seq) {
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int thread = threadIdx.x;
  const int context_len = context_lens[seq];
  const int32_t* block_table = block_tables + seq * max_blocks_per_seq;

  const int lanes_per_row = head_dim / VEC;
  const int rows_per_tile = kThreads / lanes_per_row;   // positions in flight
  const int lane = thread % lanes_per_row;              // the part of head_dim
  const int row = thread / lanes_per_row;               // the position in the tile

  extern __shared__ float shared[];
  float* scaled_query = shared;                         // head_dim
  float* row_scores = scaled_query + head_dim;          // rows_per_tile
  float* scratch = row_scores + kThreads;               // kThreads

  const int64_t query_offset = ((int64_t)seq * num_heads + head) * head_dim;
  for (int dim = thread; dim < head_dim; dim += kThreads)
    scaled_query[dim] = static_cast<float>(query[query_offset + dim]) * scale;
  __syncthreads();

  float query_slice[kMaxVec];                           // this thread's part of q
#pragma unroll
  for (int i = 0; i < VEC; ++i) query_slice[i] = scaled_query[lane * VEC + i];

  float accumulator[kMaxVec];
#pragma unroll
  for (int i = 0; i < VEC; ++i) accumulator[i] = 0.f;

  float running_max = -INFINITY;
  float running_sum = 0.f;
  for (int tile_start = 0; tile_start < context_len;
       tile_start += rows_per_tile) {
    const int position = tile_start + row;
    const bool in_context = position < context_len;

    // The score. The lanes of a row read the row between them, 16 bytes each.
    float partial_score = 0.f;
    int64_t row_offset = 0;
    if (in_context) {
      const int64_t block_id = block_table[position / block_size];
      row_offset = ((block_id * num_kv_heads + kv_head) * block_size +
                    position % block_size) * head_dim + lane * VEC;
      float key_slice[kMaxVec];
      load_vec<scalar_t, VEC>(key_slice, key_cache + row_offset);
#pragma unroll
      for (int i = 0; i < VEC; ++i) partial_score += query_slice[i] * key_slice[i];
    }
    const float score = row_sum(partial_score, lane, lanes_per_row, scratch);
    if (lane == 0) row_scores[row] = in_context ? score : -INFINITY;
    __syncthreads();

    // The running maximum and denominator over the scores of the tile.
    const float tile_max =
        block_max(thread < rows_per_tile ? row_scores[thread] : -INFINITY,
                  scratch);
    const float new_max = fmaxf(running_max, tile_max);
    const float rescale = __expf(running_max - new_max);
    const float prob =
        in_context ? __expf(row_scores[row] - new_max) : 0.f;
    running_sum = running_sum * rescale +
                  block_sum(lane == 0 ? prob : 0.f, scratch);
    running_max = new_max;

    // The numerator, in registers, from the same coalesced part of V.
    float value_slice[kMaxVec];
    if (in_context) load_vec<scalar_t, VEC>(value_slice, value_cache + row_offset);
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      accumulator[i] = accumulator[i] * rescale +
                       (in_context ? prob * value_slice[i] : 0.f);
  }

  // One reduction across the row groups, one time, at the end.
  float* partials = shared;               // scaled_query is not needed now
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC; ++i) partials[thread * kMaxVec + i] = accumulator[i];
  __syncthreads();
  if (row != 0) return;

  for (int other_row = 1; other_row < rows_per_tile; ++other_row)
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      accumulator[i] += partials[(other_row * lanes_per_row + lane) * kMaxVec + i];

  const float inverse_sum = context_len > 0 ? 1.f / running_sum : 0.f;
  scalar_t* output_slice = output + query_offset + lane * VEC;
#pragma unroll
  for (int i = 0; i < VEC; ++i)
    output_slice[i] = static_cast<scalar_t>(accumulator[i] * inverse_sum);
}

struct LaunchShape {
  dim3 grid;
  size_t shared_bytes;
  int num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq;
};

template <typename scalar_t, int VEC>
void launch_with_width(torch::Tensor& output, const torch::Tensor& query,
                       const torch::Tensor& key_cache,
                       const torch::Tensor& value_cache,
                       const torch::Tensor& block_tables,
                       const torch::Tensor& context_lens, float scale,
                       const LaunchShape& shape) {
  paged_attn_vec<scalar_t, VEC>
      <<<shape.grid, kThreads, shape.shared_bytes,
         at::cuda::getCurrentCUDAStream()>>>(
          output.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(),
          key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
          block_tables.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
          scale, shape.num_heads, shape.num_kv_heads, shape.head_dim,
          shape.block_size, shape.max_blocks_per_seq);
}

// The launcher selects the vector width. The target is 16 bytes for each
// thread. A head_dim that does not divide by it uses scalar loads, and never
// reads past the end of a row. The alignment is a precondition, not a hope.
template <typename scalar_t, int WIDE>
void launch(torch::Tensor& output, const torch::Tensor& query,
            const torch::Tensor& key_cache, const torch::Tensor& value_cache,
            const torch::Tensor& block_tables,
            const torch::Tensor& context_lens, float scale,
            const LaunchShape& shape) {
  const int head_dim = shape.head_dim;
  if (head_dim % WIDE == 0 && head_dim / WIDE <= kThreads) {
    launch_with_width<scalar_t, WIDE>(output, query, key_cache, value_cache,
                                      block_tables, context_lens, scale, shape);
    return;
  }
  TORCH_CHECK(head_dim <= kThreads, "head_dim ", head_dim,
              " needs more than ", kThreads,
              " threads for each row; make kThreads larger");
  launch_with_width<scalar_t, 1>(output, query, key_cache, value_cache,
                                 block_tables, context_lens, scale, shape);
}

}  // namespace

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale) {
  TORCH_CHECK(query.is_cuda() && query.is_contiguous(),
              "query must be contiguous and on the GPU");
  TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
              "the KV cache must be contiguous");

  const int num_seqs = query.size(0), num_heads = query.size(1);
  const int head_dim = query.size(2);
  const int num_kv_heads = key_cache.size(1);
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be a multiple of num_kv_heads");

  auto output = torch::empty_like(query);
  auto tables = block_tables.to(torch::kInt).contiguous();
  auto lens = context_lens.to(torch::kInt).contiguous();
  const LaunchShape shape{
      dim3(num_seqs, num_heads),
      std::max<size_t>(head_dim + 2 * kThreads, kThreads * kMaxVec) *
          sizeof(float),
      num_heads, num_kv_heads, head_dim, (int)key_cache.size(2),
      (int)block_tables.size(1)};

  AT_DISPATCH_SWITCH(
      query.scalar_type(), "paged_attn",
      AT_DISPATCH_CASE(at::kFloat, [&] {
        launch<scalar_t, 4>(output, query, key_cache, value_cache, tables,
                            lens, (float)scale, shape);
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        launch<scalar_t, 8>(output, query, key_cache, value_cache, tables,
                            lens, (float)scale, shape);
      })
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        launch<scalar_t, 8>(output, query, key_cache, value_cache, tables,
                            lens, (float)scale, shape);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "coalesced paged decode attention (CUDA)");
}
