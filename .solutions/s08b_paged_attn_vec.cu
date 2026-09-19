// Reference solution, stage 08b - the same kernel, fed properly.
//
// Stage 08 gave one thread one position and walked head_dim serially. Nothing
// about the arithmetic changes here. What changes is which thread touches
// which byte:
//
//     stage 08   thread t reads k[pos_t][0..D)     addresses D apart
//     stage 08b  thread t reads k[pos][t*VEC..]    addresses 16 bytes apart
//
// Threads now cooperate ALONG head_dim. LPR = D/VEC of them share one row,
// and each one moves 16 bytes. So a warp issues a few full 128-byte
// transactions, and not 32 separate ones. The other ROWS = 128/LPR groups
// work on different positions at the same time.
//
// The accumulator stays in registers. Each thread owns VEC elements of the
// output, for the positions that its group walks. The online-softmax rescale
// applies equally to every partial sum. So the sum across groups can wait
// until the end. That is one shared-memory reduction for each block, and not
// one for each tile.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 8;

// 16 bytes is the widest load a thread can issue, and the width at which a
// warp's requests coalesce into whole 128-byte transactions. Everything below
// exists to hit it.
template <typename scalar_t, int VEC>
__device__ __forceinline__ void load_vec(float* dst, const scalar_t* src) {
  if constexpr (VEC * sizeof(scalar_t) == 16) {
    const float4 raw = *reinterpret_cast<const float4*>(src);
    const scalar_t* s = reinterpret_cast<const scalar_t*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC; ++i) dst[i] = static_cast<float>(s[i]);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) dst[i] = static_cast<float>(src[i]);
  }
}

template <typename scalar_t, int VEC>
__global__ void __launch_bounds__(kThreads) paged_attn_vec(
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
  const int kvh = h / (H / KVH);
  const int tid = threadIdx.x;
  const int n = ctx[s];

  const int LPR = D / VEC;                     // threads per K/V row
  const int ROWS = kThreads / LPR;             // positions in flight
  const int lane = tid % LPR;                  // which slice of head_dim
  const int row = tid / LPR;                   // which position in the tile

  extern __shared__ float smem[];
  float* q_sh = smem;                          // D
  float* s_sh = q_sh + D;                      // ROWS, one score per position
  float* red = s_sh + kThreads;                // kThreads, reduction scratch

  __shared__ float m_i, l_i, m_new_sh, alpha_sh;

  for (int d = tid; d < D; d += kThreads)
    q_sh[d] = static_cast<float>(q[((int64_t)s * H + h) * D + d]) * scale;
  if (tid == 0) {
    m_i = -INFINITY;
    l_i = 0.f;
  }
  __syncthreads();

  float qv[kMaxVec];                           // this thread's slice of q
#pragma unroll
  for (int i = 0; i < VEC; ++i) qv[i] = q_sh[lane * VEC + i];

  float acc[kMaxVec];
#pragma unroll
  for (int i = 0; i < VEC; ++i) acc[i] = 0.f;

  for (int base = 0; base < n; base += ROWS) {
    const int pos = base + row;
    const bool live = pos < n;

    // --- score. LPR lanes read one row between them, 16 bytes each. ---
    float part = 0.f;
    const scalar_t* kp = nullptr;
    const scalar_t* vp = nullptr;
    if (live) {
      const int phys = bt[s * MBS + (pos / BS)];
      const int64_t off = (((int64_t)phys * KVH + kvh) * BS + (pos % BS)) * D;
      kp = kc + off + lane * VEC;
      vp = vc + off + lane * VEC;
      float kv[kMaxVec];
      load_vec<scalar_t, VEC>(kv, kp);
#pragma unroll
      for (int i = 0; i < VEC; ++i) part += qv[i] * kv[i];
    }

    // --- sum the LPR partials down to one score per row ---
    red[tid] = part;
    __syncthreads();
    for (int o = LPR / 2; o > 0; o >>= 1) {
      if (lane < o) red[tid] += red[tid + o];
      __syncthreads();
    }
    if (lane == 0) s_sh[row] = live ? red[tid] : -INFINITY;
    __syncthreads();

    // --- running maximum and denominator over the ROWS scores ---
    red[tid] = (tid < ROWS) ? s_sh[tid] : -INFINITY;
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
    const float p = live ? __expf(s_sh[row] - m_new_sh) : 0.f;

    red[tid] = (lane == 0) ? p : 0.f;
    __syncthreads();
    for (int o = kThreads / 2; o > 0; o >>= 1) {
      if (tid < o) red[tid] += red[tid + o];
      __syncthreads();
    }
    if (tid == 0) {
      l_i = l_i * alpha + red[0];
      m_i = m_new_sh;
    }

    // --- numerator, in registers, same coalesced slice of V ---
    float vv[kMaxVec];
    if (live) load_vec<scalar_t, VEC>(vv, vp);
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      acc[i] = acc[i] * alpha + (live ? p * vv[i] : 0.f);
    __syncthreads();
  }

  // --- one reduction across the ROWS groups, once, at the end ---
  float* abuf = smem;                          // reuse: q_sh is finished with
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC; ++i) abuf[tid * kMaxVec + i] = acc[i];
  __syncthreads();

  if (row == 0) {
    for (int r = 1; r < ROWS; ++r)
#pragma unroll
      for (int i = 0; i < VEC; ++i)
        acc[i] += abuf[(r * LPR + lane) * kMaxVec + i];

    const float inv = (n > 0) ? 1.f / l_i : 0.f;
    scalar_t* op = out + ((int64_t)s * H + h) * D + lane * VEC;
#pragma unroll
    for (int i = 0; i < VEC; ++i) op[i] = static_cast<scalar_t>(acc[i] * inv);
  }
}


// The launcher picks the vector width. The target is 16 bytes for each
// thread. A head_dim that does not divide by it uses scalar loads, and never
// reads past the end of a row. Alignment is a precondition, not a hope.
template <typename scalar_t, int VEC>
void launch_impl(torch::Tensor& out, const torch::Tensor& q,
                 const torch::Tensor& kc, const torch::Tensor& vc,
                 const torch::Tensor& bt, const torch::Tensor& ctx,
                 float scale, dim3 grid, size_t shared,
                 int H, int KVH, int D, int BS, int MBS) {
  paged_attn_vec<scalar_t, VEC>
      <<<grid, kThreads, shared, at::cuda::getCurrentCUDAStream()>>>(
          out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(),
          kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(),
          bt.data_ptr<int32_t>(), ctx.data_ptr<int32_t>(), scale,
          H, KVH, D, BS, MBS);
}

template <typename scalar_t, int WIDE>
void launch(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& kc,
            const torch::Tensor& vc, const torch::Tensor& bt,
            const torch::Tensor& ctx, float scale, dim3 grid, size_t shared,
            int H, int KVH, int D, int BS, int MBS) {
  if (D % WIDE == 0 && D / WIDE <= kThreads) {
    launch_impl<scalar_t, WIDE>(out, q, kc, vc, bt, ctx, scale, grid, shared,
                                H, KVH, D, BS, MBS);
  } else {
    TORCH_CHECK(D <= kThreads,
                "head_dim ", D, " needs more than ", kThreads,
                " threads per row; raise kThreads");
    launch_impl<scalar_t, 1>(out, q, kc, vc, bt, ctx, scale, grid, shared,
                             H, KVH, D, BS, MBS);
  }
}

}  // namespace

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale) {
  TORCH_CHECK(query.is_cuda() && query.is_contiguous(),
              "query must be contiguous and on the GPU");
  TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
              "the KV cache must be contiguous");

  const int S = query.size(0), H = query.size(1), D = query.size(2);
  const int KVH = key_cache.size(1), BS = key_cache.size(2);
  const int MBS = block_tables.size(1);
  TORCH_CHECK(H % KVH == 0, "num_heads must be a multiple of num_kv_heads");

  auto out = torch::empty_like(query);
  auto bt = block_tables.to(torch::kInt).contiguous();
  auto ctx = context_lens.to(torch::kInt).contiguous();
  const dim3 grid(S, H);
  const size_t shared =
      std::max<size_t>(D + 2 * kThreads, kThreads * kMaxVec) * sizeof(float);

  AT_DISPATCH_SWITCH(
      query.scalar_type(), "paged_attn",
      AT_DISPATCH_CASE(at::kFloat,
                       [&] {
                         launch<scalar_t, 4>(out, query, key_cache, value_cache,
                                             bt, ctx, (float)scale, grid,
                                             shared, H, KVH, D, BS, MBS);
                       })
      AT_DISPATCH_CASE(at::kHalf,
                       [&] {
                         launch<scalar_t, 8>(out, query, key_cache, value_cache,
                                             bt, ctx, (float)scale, grid,
                                             shared, H, KVH, D, BS, MBS);
                       })
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        launch<scalar_t, 8>(out, query, key_cache, value_cache, bt, ctx,
                            (float)scale, grid, shared, H, KVH, D, BS, MBS);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "coalesced paged decode attention (CUDA)");
}
