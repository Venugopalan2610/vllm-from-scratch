// Reference solution, stage 08 - paged decode attention in CUDA.
//
// The mapping is the lesson:
//
//     grid  = (num_seqs, num_heads)     one block for each (sequence, query head)
//     block = 128 threads               they share one output vector
//
// One thread for each context position, and each thread walks head_dim
// alone. It is correct and it is slow. The reason: thread t reads
// key_row[0..head_dim), and thread t+1 reads an address head_dim elements
// further on. The warp touches 32 cache lines to start 32 dot products.
// Stage 08b fixes that.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kWriteThreads = 256;
constexpr int kMaxWriteBlocks = 4096;

// The first element of one cached token row, for one KV head.
// cache: (num_blocks, num_kv_heads, block_size, head_dim).
template <typename scalar_t>
__device__ const scalar_t* cache_row(const scalar_t* cache,
                                     const int32_t* block_table, int position,
                                     int kv_head, int num_kv_heads,
                                     int block_size, int head_dim) {
  const int64_t block_id = block_table[position / block_size];
  const int offset = position % block_size;
  return cache + ((block_id * num_kv_heads + kv_head) * block_size + offset) *
                     head_dim;
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

// ---------------------------------------------------------------- write_kv
//
// A scatter, and the simplest kernel in the repo: one grid-stride loop, one
// index decomposition, no reduction and no shared memory.

template <typename scalar_t>
__global__ void write_kv_kernel(
    scalar_t* __restrict__ key_cache,     // (num_blocks, kv_heads, block_size, D)
    scalar_t* __restrict__ value_cache,
    const scalar_t* __restrict__ key,     // (num_tokens, kv_heads, D)
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ slots,    // (num_tokens,)
    const int num_tokens, const int num_kv_heads, const int head_dim,
    const int block_size) {
  const int64_t num_values = (int64_t)num_tokens * num_kv_heads * head_dim;
  for (int64_t index = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       index < num_values; index += (int64_t)gridDim.x * blockDim.x) {
    const int dim = index % head_dim;
    const int kv_head = (index / head_dim) % num_kv_heads;
    const int token = index / (head_dim * num_kv_heads);

    const int64_t slot = slots[token];
    if (slot < 0) continue;               // padding: not a real token
    const int64_t block_id = slot / block_size;
    const int64_t offset = slot % block_size;

    const int64_t target =
        ((block_id * num_kv_heads + kv_head) * block_size + offset) * head_dim +
        dim;
    key_cache[target] = key[index];
    value_cache[target] = value[index];
  }
}

// ------------------------------------------------------------- paged_attn
//
// The online softmax, as in FlashAttention, over tiles of 128 positions:
//
//     new_max     = max(running_max, max(scores))
//     rescale     = exp(running_max - new_max)   what the old sums are worth
//     running_sum = running_sum * rescale + sum(p)          the denominator
//     accumulator = accumulator * rescale + sum(p * V)      the numerator
//
// The full score row is never stored. At a context of 2048 it is 8 KB for
// each (seq, head) of HBM traffic that does nothing.

template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) paged_attn_v1(
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
  const int kv_head = head / (num_heads / num_kv_heads);  // GQA
  const int thread = threadIdx.x;
  const int context_len = context_lens[seq];
  const int32_t* block_table = block_tables + seq * max_blocks_per_seq;

  extern __shared__ float shared[];
  float* scaled_query = shared;                     // head_dim
  float* tile_probs = scaled_query + head_dim;      // kThreads
  float* accumulator = tile_probs + kThreads;       // head_dim
  float* scratch = accumulator + head_dim;          // kThreads

  // The block reads the query one time, and every position uses it. So put
  // the scale into it here, not into every score.
  const int64_t query_offset = ((int64_t)seq * num_heads + head) * head_dim;
  for (int dim = thread; dim < head_dim; dim += kThreads) {
    scaled_query[dim] = static_cast<float>(query[query_offset + dim]) * scale;
    accumulator[dim] = 0.f;
  }
  __syncthreads();

  float running_max = -INFINITY;   // every thread holds the same copy
  float running_sum = 0.f;
  for (int tile_start = 0; tile_start < context_len; tile_start += kThreads) {
    const int position = tile_start + thread;
    const bool in_context = position < context_len;

    // The score: one thread, one position, a serial walk down head_dim.
    float score = -INFINITY;
    if (in_context) {
      const scalar_t* key_row = cache_row(key_cache, block_table, position,
                                          kv_head, num_kv_heads, block_size,
                                          head_dim);
      score = 0.f;
      for (int dim = 0; dim < head_dim; ++dim)
        score += scaled_query[dim] * static_cast<float>(key_row[dim]);
    }

    const float new_max = fmaxf(running_max, block_max(score, scratch));
    const float rescale = __expf(running_max - new_max);
    const float prob = in_context ? __expf(score - new_max) : 0.f;
    tile_probs[thread] = prob;
    running_sum = running_sum * rescale + block_sum(prob, scratch);
    running_max = new_max;

    // The numerator. The threads change roles: each thread now owns a part
    // of head_dim and walks every position of the tile. This read of V down
    // a column is as uncoalesced as the read of K above.
    const int tile_len = min(kThreads, context_len - tile_start);
    for (int dim = thread; dim < head_dim; dim += kThreads) {
      float weighted_sum = 0.f;
      for (int j = 0; j < tile_len; ++j) {
        const scalar_t* value_row = cache_row(value_cache, block_table,
                                              tile_start + j, kv_head,
                                              num_kv_heads, block_size,
                                              head_dim);
        weighted_sum += tile_probs[j] * static_cast<float>(value_row[dim]);
      }
      accumulator[dim] = accumulator[dim] * rescale + weighted_sum;
    }
    __syncthreads();
  }

  for (int dim = thread; dim < head_dim; dim += kThreads) {
    const float result = context_len > 0 ? accumulator[dim] / running_sum : 0.f;
    output[query_offset + dim] = static_cast<scalar_t>(result);
  }
}

}  // namespace

// ------------------------------------------------------------------- host

void write_kv(torch::Tensor key_cache, torch::Tensor value_cache,
              torch::Tensor key, torch::Tensor value,
              torch::Tensor slot_indices) {
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be on the GPU");
  TORCH_CHECK(key_cache.is_contiguous() && key.is_contiguous(),
              "write_kv needs contiguous tensors");
  const int num_tokens = key.size(0), num_kv_heads = key.size(1);
  const int head_dim = key.size(2), block_size = key_cache.size(2);
  auto slots = slot_indices.to(torch::kLong).contiguous();

  const int64_t num_values = (int64_t)num_tokens * num_kv_heads * head_dim;
  if (num_values == 0) return;
  const int num_blocks = (int)std::min<int64_t>(
      (num_values + kWriteThreads - 1) / kWriteThreads, kMaxWriteBlocks);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, key.scalar_type(), "write_kv", [&] {
        write_kv_kernel<scalar_t><<<num_blocks, kWriteThreads, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(
            key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            slots.data_ptr<int64_t>(), num_tokens, num_kv_heads, head_dim,
            block_size);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale) {
  TORCH_CHECK(query.is_cuda(), "query must be on the GPU");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
              "the KV cache must be contiguous");

  const int num_seqs = query.size(0), num_heads = query.size(1);
  const int head_dim = query.size(2);
  const int num_kv_heads = key_cache.size(1), block_size = key_cache.size(2);
  const int max_blocks_per_seq = block_tables.size(1);
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be a multiple of num_kv_heads");

  auto output = torch::empty_like(query);
  auto tables = block_tables.to(torch::kInt).contiguous();
  auto lens = context_lens.to(torch::kInt).contiguous();

  const size_t shared_bytes = (2 * head_dim + 2 * kThreads) * sizeof(float);
  const dim3 grid(num_seqs, num_heads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, query.scalar_type(), "paged_attn", [&] {
        paged_attn_v1<scalar_t><<<grid, kThreads, shared_bytes,
                                  at::cuda::getCurrentCUDAStream()>>>(
            output.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
            tables.data_ptr<int32_t>(), lens.data_ptr<int32_t>(),
            (float)scale, num_heads, num_kv_heads, head_dim, block_size,
            max_blocks_per_seq);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "paged decode attention (CUDA)");
  m.def("write_kv", &write_kv, "scatter K and V into the paged cache (CUDA)");
}
