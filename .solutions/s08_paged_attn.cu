// Reference solution, stage 08 - paged decode attention in CUDA.
//
// The mapping is the whole lesson here:
//
//     grid  = (num_seqs, num_heads)     one block per (sequence, query head)
//     block = 128 threads               they share one output vector
//
// One thread per context position, walking head_dim serially. It is correct
// and it is slow, and the reason it is slow is visible in one line below:
// thread t reads kp[0..D), thread t+1 reads an address D elements further on.
// The warp touches 32 different cache lines to get 32 dot products started.
// Stage 08b fixes exactly that.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;

// ---------------------------------------------------------------- write_kv
//
// A scatter, and the simplest kernel in the repo: one grid-stride loop, one
// index decomposition, no reduction and no shared memory.

template <typename scalar_t>
__global__ void write_kv_kernel(
    scalar_t* __restrict__ key_cache,          // (NB, KVH, BS, D)
    scalar_t* __restrict__ value_cache,
    const scalar_t* __restrict__ key,          // (T, KVH, D)
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ slots,         // (T,)
    const int T, const int KVH, const int D, const int BS) {
  const int64_t total = (int64_t)T * KVH * D;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int d = i % D;
    const int h = (i / D) % KVH;
    const int t = i / (D * KVH);

    const int64_t slot = slots[t];
    if (slot < 0) continue;                    // padding: not a real token
    const int64_t block = slot / BS;
    const int64_t off = slot % BS;

    const int64_t dst = ((block * KVH + h) * BS + off) * D + d;
    key_cache[dst] = key[i];
    value_cache[dst] = value[i];
  }
}

// ------------------------------------------------------------- paged_attn
//
// Online softmax, the FlashAttention rescaling, over tiles of 128 positions:
//
//     m_new = max(m_old, max(scores))      running maximum
//     alpha = exp(m_old - m_new)           what the old accumulator is worth
//     l     = l * alpha + sum(p)           running denominator
//     acc   = acc * alpha + sum(p * V)     running numerator
//
// The full score row is never written anywhere. That is the point: at 2048
// context it would be 8KB per (seq, head) of pure HBM traffic.

template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) paged_attn_v1(
    scalar_t* __restrict__ out,                // (S, H, D)
    const scalar_t* __restrict__ q,            // (S, H, D)
    const scalar_t* __restrict__ kc,           // (NB, KVH, BS, D)
    const scalar_t* __restrict__ vc,
    const int32_t* __restrict__ bt,            // (S, MBS)
    const int32_t* __restrict__ ctx,           // (S,)
    const float scale,
    const int H, const int KVH, const int D, const int BS, const int MBS) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int kvh = h / (H / KVH);               // GQA: which KV head do I read?
  const int tid = threadIdx.x;
  const int n = ctx[s];

  extern __shared__ float smem[];
  float* q_sh = smem;                          // D
  float* p_sh = q_sh + D;                      // kThreads
  float* acc_sh = p_sh + kThreads;             // D
  float* red = acc_sh + D;                     // kThreads
  __shared__ float m_i, l_i, m_new_sh, alpha_sh;

  // The query vector is read once per block and used by every position, so
  // fold the scale into it here rather than into every score.
  for (int d = tid; d < D; d += kThreads) {
    q_sh[d] = static_cast<float>(q[((int64_t)s * H + h) * D + d]) * scale;
    acc_sh[d] = 0.f;
  }
  if (tid == 0) {
    m_i = -INFINITY;
    l_i = 0.f;
  }
  __syncthreads();

  for (int base = 0; base < n; base += kThreads) {
    const int pos = base + tid;

    // --- score: one thread, one position, a serial walk down head_dim ---
    float sc = -INFINITY;
    if (pos < n) {
      const int phys = bt[s * MBS + (pos / BS)];
      const scalar_t* kp =
          kc + (((int64_t)phys * KVH + kvh) * BS + (pos % BS)) * D;
      float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += q_sh[d] * static_cast<float>(kp[d]);
      sc = dot;
    }

    // --- running maximum over the tile ---
    red[tid] = sc;
    __syncthreads();
    for (int o = kThreads / 2; o > 0; o >>= 1) {
      if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
      __syncthreads();
    }
    if (tid == 0) {
      const float mn = fmaxf(m_i, red[0]);
      alpha_sh = __expf(m_i - mn);
      m_new_sh = mn;
    }
    __syncthreads();
    const float alpha = alpha_sh;

    // --- probabilities, and the running denominator ---
    const float p = (pos < n) ? __expf(sc - m_new_sh) : 0.f;
    p_sh[tid] = p;
    red[tid] = p;
    __syncthreads();
    for (int o = kThreads / 2; o > 0; o >>= 1) {
      if (tid < o) red[tid] += red[tid + o];
      __syncthreads();
    }
    if (tid == 0) {
      l_i = l_i * alpha + red[0];
      m_i = m_new_sh;
    }

    // --- the numerator. Threads switch roles: now each owns a slice of
    //     head_dim and walks every position in the tile. Reading V down a
    //     column like this is as uncoalesced as the K read above. ---
    const int tile = min(kThreads, n - base);
    __syncthreads();
    for (int d = tid; d < D; d += kThreads) {
      float a = 0.f;
      for (int j = 0; j < tile; ++j) {
        const int pj = base + j;
        const int physj = bt[s * MBS + (pj / BS)];
        const scalar_t* vp =
            vc + (((int64_t)physj * KVH + kvh) * BS + (pj % BS)) * D;
        a += p_sh[j] * static_cast<float>(vp[d]);
      }
      acc_sh[d] = acc_sh[d] * alpha + a;
    }
    __syncthreads();
  }

  const float denom = (n > 0) ? l_i : 1.f;
  for (int d = tid; d < D; d += kThreads)
    out[((int64_t)s * H + h) * D + d] =
        static_cast<scalar_t>((n > 0) ? acc_sh[d] / denom : 0.f);
}

}  // namespace

// ------------------------------------------------------------------- host

void write_kv(torch::Tensor key_cache, torch::Tensor value_cache,
              torch::Tensor key, torch::Tensor value,
              torch::Tensor slot_indices) {
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be on the GPU");
  TORCH_CHECK(key_cache.is_contiguous() && key.is_contiguous(),
              "write_kv needs contiguous tensors");
  const int T = key.size(0), KVH = key.size(1), D = key.size(2);
  const int BS = key_cache.size(2);
  auto slots = slot_indices.to(torch::kLong).contiguous();

  const int64_t total = (int64_t)T * KVH * D;
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>((total + threads - 1) / threads, 4096);
  if (total == 0) return;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, key.scalar_type(), "write_kv", [&] {
        write_kv_kernel<scalar_t><<<blocks, threads,
                                    0, at::cuda::getCurrentCUDAStream()>>>(
            key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            slots.data_ptr<int64_t>(), T, KVH, D, BS);
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

  const int S = query.size(0), H = query.size(1), D = query.size(2);
  const int KVH = key_cache.size(1), BS = key_cache.size(2);
  const int MBS = block_tables.size(1);
  TORCH_CHECK(H % KVH == 0, "num_heads must be a multiple of num_kv_heads");

  auto out = torch::empty_like(query);
  auto bt = block_tables.to(torch::kInt).contiguous();
  auto ctx = context_lens.to(torch::kInt).contiguous();

  const size_t shared = (2 * D + 2 * kThreads) * sizeof(float);
  const dim3 grid(S, H);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, query.scalar_type(), "paged_attn", [&] {
        paged_attn_v1<scalar_t><<<grid, kThreads, shared,
                                  at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
            bt.data_ptr<int32_t>(), ctx.data_ptr<int32_t>(), (float)scale,
            H, KVH, D, BS, MBS);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "paged decode attention (CUDA)");
  m.def("write_kv", &write_kv, "scatter K/V into the paged cache (CUDA)");
}
