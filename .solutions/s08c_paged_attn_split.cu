// Reference solution, stage 08c - warp primitives and split-K.
//
// Stage 08b reads memory about as well as this kernel can. At num_seqs = 64
// it runs near the measured streaming bandwidth of the card. At num_seqs = 1
// it runs at seven percent of it, and no amount of coalescing will fix that,
// because the problem is no longer the memory system:
//
//     grid = (num_seqs, num_heads) = (1, 16) = 16 blocks
//     this GPU has ~60 SMs
//
// Three quarters of the machine has no work. That is the entire content of
// this stage. Two changes:
//
//   1. The reductions move from shared memory into the warp. __shfl_xor_sync
//      exchanges a register between lanes directly, with no round trip
//      through shared memory and no __syncthreads. A butterfly over LPR
//      lanes leaves the answer in every lane of the group.
//
//   2. Split-K, also called flash-decoding. Cut the context into SPLITS
//      chunks and give each chunk its own block. Every block produces a
//      PARTIAL softmax state (m, l, acc), and a second kernel merges them
//      with the same rescaling rule the online softmax already uses. The
//      grid becomes (num_seqs, num_heads, SPLITS), and the machine fills up.
//
// The merge is exact, not an approximation. exp(m_j - M) is the same alpha
// from stage 08, applied across blocks instead of across tiles.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 8;
constexpr int kWarp = 32;
constexpr unsigned kFull = 0xffffffffu;

// A split shorter than this is not worth a block: the launch and the merge
// cost more than the tokens save.
constexpr int kMinChunk = 128;
constexpr int kMaxSplits = 32;

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

// --- warp primitives -------------------------------------------------------
//
// xor is a butterfly: after the loop every lane in the group holds the
// result, so there is no broadcast step. shfl_down would leave it in lane 0.

__device__ __forceinline__ float warp_sum(float v, int width) {
  for (int o = width / 2; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v, int width) {
  for (int o = width / 2; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
  return v;
}

// Both of these end with TWO barriers, and the second one is not decoration.
// Every thread reads smem[0] to get the answer, and the next call to either
// helper overwrites smem[0] with its own first-round write. Without a barrier
// between the read and that write, a warp that gets ahead clobbers the result
// a slower warp has not collected yet. The bug is intermittent, it depends on
// the data, and it looks like bad arithmetic.
__device__ __forceinline__ float block_max(float v, float* smem, int tid) {
  v = warp_max(v, kWarp);
  if (tid % kWarp == 0) smem[tid / kWarp] = v;
  __syncthreads();
  constexpr int NW = kThreads / kWarp;
  v = (tid < NW) ? smem[tid] : -INFINITY;
  if (tid < kWarp) v = warp_max(v, NW);
  if (tid == 0) smem[0] = v;
  __syncthreads();
  const float result = smem[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_sum(float v, float* smem, int tid) {
  v = warp_sum(v, kWarp);
  if (tid % kWarp == 0) smem[tid / kWarp] = v;
  __syncthreads();
  constexpr int NW = kThreads / kWarp;
  v = (tid < NW) ? smem[tid] : 0.f;
  if (tid < kWarp) v = warp_sum(v, NW);
  if (tid == 0) smem[0] = v;
  __syncthreads();
  const float result = smem[0];
  __syncthreads();
  return result;
}

// --- the split kernel ------------------------------------------------------
//
// There is not one __syncthreads in the loop below, and that is the point.
//
// Every lane of a row group ends the butterfly holding the same score, so
// every lane of a group runs the same online softmax and arrives at the same
// m and l without being told. Each group is therefore an INDEPENDENT partial
// softmax over its own stride of the context, exactly like a split is.
//
// So the block-wide max and sum that stage 08b did once per tile are not
// needed at all. They are replaced by one merge at the end, over ROWS partial
// states, with the same exp(m_r - M) rescale used to merge the splits. The
// same trick, at three levels: tile, group, block.

template <typename scalar_t, int VEC, bool SINGLE>
__global__ void __launch_bounds__(kThreads) paged_attn_split(
    scalar_t* __restrict__ out,                // (S, H, D), SINGLE only
    float* __restrict__ acc_part,              // (S, H, SPLITS, D)
    float* __restrict__ m_part,                // (S, H, SPLITS)
    float* __restrict__ l_part,
    const scalar_t* __restrict__ q,            // (S, H, D)
    const scalar_t* __restrict__ kc,           // (NB, KVH, BS, D)
    const scalar_t* __restrict__ vc,
    const int32_t* __restrict__ bt,            // (S, MBS)
    const int32_t* __restrict__ ctx,           // (S,)
    const float scale,
    const int H, const int KVH, const int D, const int BS, const int MBS,
    const int SPLITS) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int split = blockIdx.z;
  const int kvh = h / (H / KVH);
  const int tid = threadIdx.x;

  const int n = ctx[s];
  const int chunk = (n + SPLITS - 1) / SPLITS;
  const int lo = split * chunk;
  const int hi = min(n, lo + chunk);

  const int LPR = D / VEC;
  const int ROWS = kThreads / LPR;
  const int lane = tid % LPR;
  const int row = tid / LPR;

  extern __shared__ float smem[];
  float* q_sh = smem;                          // D, then reused as abuf
  float* sh_m = smem + kThreads * kMaxVec;     // ROWS
  float* sh_l = sh_m + kThreads;               // ROWS

  const int64_t part_base = ((int64_t)s * H + h) * SPLITS + split;

  for (int d = tid; d < D; d += kThreads)
    q_sh[d] = static_cast<float>(q[((int64_t)s * H + h) * D + d]) * scale;
  __syncthreads();

  if (lo >= hi) {                              // an empty split still reports
    if constexpr (SINGLE) {
      for (int d = tid; d < D; d += kThreads)
        out[((int64_t)s * H + h) * D + d] = static_cast<scalar_t>(0.f);
    } else {
      if (tid == 0) {
        m_part[part_base] = -INFINITY;
        l_part[part_base] = 0.f;
      }
      for (int d = tid; d < D; d += kThreads)
        acc_part[part_base * D + d] = 0.f;
    }
    return;
  }

  float qv[kMaxVec], acc[kMaxVec];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    qv[i] = q_sh[lane * VEC + i];
    acc[i] = 0.f;
  }
  float m_i = -INFINITY, l_i = 0.f;

  // Every group walks the same NUMBER of positions, so no lane of a warp can
  // leave the loop early and strand the others at a shuffle.
  const int steps = (hi - lo + ROWS - 1) / ROWS;
  __syncthreads();                             // q_sh is finished with

  for (int t = 0; t < steps; ++t) {
    const int pos = lo + t * ROWS + row;
    const bool live = pos < hi;

    const scalar_t* vp = nullptr;
    float part = 0.f;
    if (live) {
      const int phys = bt[s * MBS + (pos / BS)];
      const int64_t off = (((int64_t)phys * KVH + kvh) * BS + (pos % BS)) * D;
      vp = vc + off + lane * VEC;
      float kv[kMaxVec];
      load_vec<scalar_t, VEC>(kv, kc + off + lane * VEC);
#pragma unroll
      for (int i = 0; i < VEC; ++i) part += qv[i] * kv[i];
    }
    // Every lane of the warp must reach the shuffle, live or not. A _sync
    // primitive whose mask names a lane that never arrives does not return.
    // This is the trap that replaces __syncthreads when you go warp-level.
    const float summed = warp_sum(part, LPR);
    const float sc = live ? summed : -INFINITY;

    const float m_new = fmaxf(m_i, sc);
    // A whole group can be empty: more groups than positions left. Both
    // maxima are then -inf, and -inf minus -inf is NaN, which poisons the
    // merge for everybody. Stage 08b could not hit this, because its maximum
    // came from the whole block and something in the block was always live.
    const float alpha = (m_new == -INFINITY) ? 0.f : __expf(m_i - m_new);
    const float p = live ? __expf(sc - m_new) : 0.f;
    l_i = l_i * alpha + p;
    m_i = m_new;

    float vv[kMaxVec];
    if (live) load_vec<scalar_t, VEC>(vv, vp);
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      acc[i] = acc[i] * alpha + (live ? p * vv[i] : 0.f);
  }

  // --- merge the ROWS partial states. Once, not once per tile. ---
  float* abuf = smem;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VEC; ++i) abuf[tid * kMaxVec + i] = acc[i];
  if (lane == 0) {
    sh_m[row] = m_i;
    sh_l[row] = l_i;
  }
  __syncthreads();

  if (row == 0) {
    float M = -INFINITY;
    for (int r = 0; r < ROWS; ++r) M = fmaxf(M, sh_m[r]);

    float L = 0.f;
    float out_acc[kMaxVec];
#pragma unroll
    for (int i = 0; i < VEC; ++i) out_acc[i] = 0.f;

    for (int r = 0; r < ROWS; ++r) {
      const float w = __expf(sh_m[r] - M);
      L += sh_l[r] * w;
#pragma unroll
      for (int i = 0; i < VEC; ++i)
        out_acc[i] += abuf[(r * LPR + lane) * kMaxVec + i] * w;
    }

    // One split means nothing to merge. Normalise here and write the answer,
    // instead of writing a partial state out to memory for a second kernel
    // to read straight back. That round trip is small next to the K/V reads,
    // but the launch and the allocations behind it are not, and at a full
    // grid it is all cost and no benefit.
    if constexpr (SINGLE) {
      const float inv = (L > 0.f) ? 1.f / L : 0.f;
      scalar_t* op = out + ((int64_t)s * H + h) * D + lane * VEC;
#pragma unroll
      for (int i = 0; i < VEC; ++i)
        op[i] = static_cast<scalar_t>(out_acc[i] * inv);
    } else {
#pragma unroll
      for (int i = 0; i < VEC; ++i)
        acc_part[part_base * D + lane * VEC + i] = out_acc[i];
      if (lane == 0) {
        m_part[part_base] = M;
        l_part[part_base] = L;
      }
    }
  }
}

// --- the merge -------------------------------------------------------------
//
// Exactly the online-softmax rescale, one level up: each split is a tile.

template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) combine_splits(
    scalar_t* __restrict__ out,                // (S, H, D)
    const float* __restrict__ acc_part,        // (S, H, SPLITS, D)
    const float* __restrict__ m_part,          // (S, H, SPLITS)
    const float* __restrict__ l_part,
    const int H, const int D, const int SPLITS) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t base = ((int64_t)s * H + h) * SPLITS;

  extern __shared__ float smem[];
  float* scratch = smem;                       // kThreads / kWarp
  float* w = smem + kThreads / kWarp;          // SPLITS

  float mine = (tid < SPLITS) ? m_part[base + tid] : -INFINITY;
  const float M = block_max(mine, scratch, tid);

  if (tid < SPLITS) w[tid] = __expf(m_part[base + tid] - M);
  __syncthreads();

  float denom = block_sum(
      (tid < SPLITS) ? l_part[base + tid] * w[tid] : 0.f, scratch, tid);
  const float inv = (denom > 0.f) ? 1.f / denom : 0.f;

  for (int d = tid; d < D; d += kThreads) {
    float a = 0.f;
    for (int j = 0; j < SPLITS; ++j) a += acc_part[(base + j) * D + d] * w[j];
    out[((int64_t)s * H + h) * D + d] = static_cast<scalar_t>(a * inv);
  }
}

int choose_splits(int S, int H, int max_ctx) {
  // Enough blocks to give every SM something, and never so many that a split
  // is shorter than kMinChunk.
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int want = (2 * sms + S * H - 1) / (S * H);
  const int by_ctx = std::max(1, max_ctx / kMinChunk);
  return std::max(1, std::min({want, by_ctx, kMaxSplits}));
}

template <typename scalar_t, int VEC>
void launch_impl(torch::Tensor& out, const torch::Tensor& q,
                 const torch::Tensor& kc, const torch::Tensor& vc,
                 const torch::Tensor& bt, const torch::Tensor& ctx,
                 float scale, int H, int KVH, int D, int BS, int MBS,
                 int S, int splits) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_split =
      (std::max<size_t>(D, kThreads * kMaxVec) + 2 * kThreads) * sizeof(float);

  if (splits == 1) {
    paged_attn_split<scalar_t, VEC, true>
        <<<dim3(S, H, 1), kThreads, shared_split, stream>>>(
            out.data_ptr<scalar_t>(), nullptr, nullptr, nullptr,
            q.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
            vc.data_ptr<scalar_t>(), bt.data_ptr<int32_t>(),
            ctx.data_ptr<int32_t>(), scale, H, KVH, D, BS, MBS, 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  auto fopt = q.options().dtype(torch::kFloat);
  auto acc_part = torch::empty({S, H, splits, D}, fopt);
  auto m_part = torch::empty({S, H, splits}, fopt);
  auto l_part = torch::empty({S, H, splits}, fopt);

  paged_attn_split<scalar_t, VEC, false>
      <<<dim3(S, H, splits), kThreads, shared_split, stream>>>(
          nullptr, acc_part.data_ptr<float>(), m_part.data_ptr<float>(),
          l_part.data_ptr<float>(), q.data_ptr<scalar_t>(),
          kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(),
          bt.data_ptr<int32_t>(), ctx.data_ptr<int32_t>(), scale,
          H, KVH, D, BS, MBS, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t shared_comb = (kThreads / kWarp + splits) * sizeof(float);
  combine_splits<scalar_t><<<dim3(S, H), kThreads, shared_comb, stream>>>(
      out.data_ptr<scalar_t>(), acc_part.data_ptr<float>(),
      m_part.data_ptr<float>(), l_part.data_ptr<float>(), H, D, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int WIDE>
void launch(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& kc,
            const torch::Tensor& vc, const torch::Tensor& bt,
            const torch::Tensor& ctx, float scale, int H, int KVH, int D,
            int BS, int MBS, int S, int splits) {
  if (D % WIDE == 0 && D / WIDE <= kThreads)
    launch_impl<scalar_t, WIDE>(out, q, kc, vc, bt, ctx, scale, H, KVH, D, BS,
                                MBS, S, splits);
  else {
    TORCH_CHECK(D <= kThreads, "head_dim ", D, " is too large for ", kThreads,
                " threads");
    launch_impl<scalar_t, 1>(out, q, kc, vc, bt, ctx, scale, H, KVH, D, BS, MBS,
                             S, splits);
  }
}

}  // namespace

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale,
                         int64_t splits) {
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

  // The upper bound on context comes from the width of the block table, which
  // is a host-side size. Reading context_lens.max() would need a real value
  // off the device, and that is a full pipeline stall on every single call.
  int nsplits = (int)splits;
  if (nsplits <= 0) nsplits = choose_splits(S, H, MBS * BS);

  AT_DISPATCH_SWITCH(
      query.scalar_type(), "paged_attn",
      AT_DISPATCH_CASE(at::kFloat,
                       [&] {
                         launch<scalar_t, 4>(out, query, key_cache, value_cache,
                                             bt, ctx, (float)scale, H, KVH, D,
                                             BS, MBS, S, nsplits);
                       })
      AT_DISPATCH_CASE(at::kHalf,
                       [&] {
                         launch<scalar_t, 8>(out, query, key_cache, value_cache,
                                             bt, ctx, (float)scale, H, KVH, D,
                                             BS, MBS, S, nsplits);
                       })
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        launch<scalar_t, 8>(out, query, key_cache, value_cache, bt, ctx,
                            (float)scale, H, KVH, D, BS, MBS, S, nsplits);
      }));
  return out;
}

int64_t splits_for(int64_t S, int64_t H, int64_t max_ctx) {
  return choose_splits((int)S, (int)H, (int)max_ctx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "split-K paged decode attention (CUDA)",
        pybind11::arg("query"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_tables"),
        pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("splits") = 0);
  m.def("splits_for", &splits_for, "how many splits this shape would use");
}
