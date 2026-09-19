// Reference solution, stage 18b - INT8 weight-only GEMV, dequantized in the
// epilogue.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kThreads / kWarp;
constexpr unsigned kFull = 0xffffffffu;
constexpr int kSharedBytes = 32768;             // activation staging budget

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  return v;
}

// One warp per output row, every row of the batch at once.
//
// Two separate reuse decisions, and they point in opposite directions:
//
//   The kernel reads each WEIGHT one time, and never again. A copy into
//   shared memory has no purpose. The weights go directly from global memory
//   into registers, 16 bytes for each lane.
//
//   Every output row in the block reads the ACTIVATIONS, and each batch row
//   has only K of them. They go to shared memory one time. The kernel then
//   reads them from there N/kWarpsPerBlock times.
//
// MAXB accumulators then live in registers. The 16 weights that a lane loaded
// serve every row of the batch before the lane drops them. A kernel with one
// block for each (row, output) reads the whole matrix B times. That is the
// roofline mistake from stage 03, written in CUDA.
template <typename scalar_t, bool WIDE, int MAXB>
__global__ void __launch_bounds__(kThreads) gemv_int8(
    scalar_t* __restrict__ out,                // (B, N)
    const scalar_t* __restrict__ x,            // (B, K)
    const int8_t* __restrict__ w,              // (N, K)
    const float* __restrict__ scales,          // (N)
    const int B, const int N, const int K, const int TILE) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int n = blockIdx.x * kWarpsPerBlock + warp;
  const bool live = n < N;                     // no early return: the
                                               // __syncthreads below are
                                               // block-wide
  extern __shared__ char raw_sh[];
  scalar_t* x_sh = reinterpret_cast<scalar_t*>(raw_sh);

  const int8_t* row = w + (int64_t)(live ? n : 0) * K;
  float acc[MAXB];
#pragma unroll
  for (int b = 0; b < MAXB; ++b) acc[b] = 0.f;

  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int tile = min(TILE, K - k0);

    __syncthreads();
    for (int i = threadIdx.x; i < tile * B; i += kThreads) {
      const int b = i / tile, t = i % tile;
      x_sh[b * tile + t] = x[(int64_t)b * K + k0 + t];
    }
    __syncthreads();

    if (!live) continue;

    if constexpr (WIDE) {
      for (int s = lane; s < tile / 16; s += kWarp) {
        const int4 packed = *reinterpret_cast<const int4*>(row + k0 + s * 16);
        const int8_t* q = reinterpret_cast<const int8_t*>(&packed);
#pragma unroll
        for (int b = 0; b < MAXB; ++b) {
          if (b >= B) break;
          const scalar_t* xr = x_sh + b * tile + s * 16;
          float a = 0.f;
#pragma unroll
          for (int i = 0; i < 16; ++i)
            a += static_cast<float>(xr[i]) * static_cast<float>(q[i]);
          acc[b] += a;
        }
      }
    } else {
      for (int t = lane; t < tile; t += kWarp) {
        const float wv = static_cast<float>(row[k0 + t]);
#pragma unroll
        for (int b = 0; b < MAXB; ++b) {
          if (b >= B) break;
          acc[b] += static_cast<float>(x_sh[b * tile + t]) * wv;
        }
      }
    }
  }

  if (!live) return;

  // The epilogue. One multiply, on a value already in a register.
  // A dequantize of the WEIGHTS writes K bf16 values out to memory and reads
  // them back. This stage exists to avoid that traffic.
  const float sc = scales[n];
#pragma unroll
  for (int b = 0; b < MAXB; ++b) {
    if (b >= B) break;
    const float total = warp_sum(acc[b]);
    if (lane == 0) out[(int64_t)b * N + n] = static_cast<scalar_t>(total * sc);
  }
}

template <typename scalar_t, int MAXB>
void launch(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w,
            const torch::Tensor& sc, int B, int N, int K) {
  int tile = kSharedBytes / (MAXB * (int)sizeof(scalar_t));
  tile = (tile / 16) * 16;                     // keep every chunk 16-wide
  tile = std::min(tile, K);
  const size_t shared = (size_t)tile * B * sizeof(scalar_t);
  const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (K % 16 == 0)
    gemv_int8<scalar_t, true, MAXB><<<grid, kThreads, shared, stream>>>(
        out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), w.data_ptr<int8_t>(),
        sc.data_ptr<float>(), B, N, K, tile);
  else
    gemv_int8<scalar_t, false, MAXB><<<grid, kThreads, shared, stream>>>(
        out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), w.data_ptr<int8_t>(),
        sc.data_ptr<float>(), B, N, K, tile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_b(torch::Tensor& out, const torch::Tensor& x,
                const torch::Tensor& w, const torch::Tensor& sc, int B, int N,
                int K) {
  // MAXB is a compile-time bound, so the accumulators can be registers.
  if (B == 1) launch<scalar_t, 1>(out, x, w, sc, B, N, K);
  else if (B <= 2) launch<scalar_t, 2>(out, x, w, sc, B, N, K);
  else if (B <= 4) launch<scalar_t, 4>(out, x, w, sc, B, N, K);
  else launch<scalar_t, 8>(out, x, w, sc, B, N, K);
}

}  // namespace

// The largest batch for this kernel. Above it, the batch spreads the weight
// read far enough that a real GEMM wins. Call one.
constexpr int kMaxBatch = 8;

torch::Tensor gemv_int8_cuda(torch::Tensor x, torch::Tensor w,
                             torch::Tensor scales) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "everything must be on the GPU");
  TORCH_CHECK(w.dtype() == torch::kChar, "weights must be int8");
  TORCH_CHECK(w.is_contiguous(), "weights must be contiguous");

  const auto xf = (x.dim() == 1 ? x.unsqueeze(0) : x).contiguous();
  const int B = xf.size(0), K = xf.size(1);
  const int N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "shape mismatch: w is (", N, ",", w.size(1),
              ") but x is (", B, ",", K, ")");
  TORCH_CHECK(scales.numel() == N, "one scale per output channel");
  TORCH_CHECK(B <= kMaxBatch, "this kernel is a decode path: at ", B,
              " rows the weight read is already amortised, so call a GEMM");

  auto out = torch::empty({B, N}, xf.options());
  auto sc = scales.to(torch::kFloat).contiguous();

  AT_DISPATCH_SWITCH(
      xf.scalar_type(), "gemv_int8",
      AT_DISPATCH_CASE(at::kFloat,
                       [&] { dispatch_b<scalar_t>(out, xf, w, sc, B, N, K); })
      AT_DISPATCH_CASE(at::kHalf,
                       [&] { dispatch_b<scalar_t>(out, xf, w, sc, B, N, K); })
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        dispatch_b<scalar_t>(out, xf, w, sc, B, N, K);
      }));
  return x.dim() == 1 ? out.squeeze(0) : out;
}

int64_t max_batch() { return kMaxBatch; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_int8", &gemv_int8_cuda, "int8 weight-only GEMV with a fused "
                                      "dequantize epilogue");
  m.def("max_batch", &max_batch, "the largest batch this kernel handles");
}
