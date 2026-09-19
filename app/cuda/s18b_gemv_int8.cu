// Stage 18b - INT8 weight-only GEMV, dequantized in the epilogue.
//
// `./vc lore 18b`. `./vc test 18b`.
//
// Stage 18 made the weight bytes half as many. It then gave the multiply to
// PyTorch, and PyTorch had to build a bf16 copy of the weight first. Time
// that, and the quantization is a LOSS. You wrote a full-size matrix out to
// memory and read it back, and you also read the int8 one.
//
// Weight-only quantization only pays if the dequantize happens inside the
// kernel, on a value that is already in a register. That is all "fused
// epilogue" means, and this stage is where you write one.
//
//     y[b, n] = sum_k x[b, k] * (w[n, k] * scale[n])
//             = scale[n] * sum_k x[b, k] * w[n, k]
//
// The scale is per output channel, so it comes out of the sum: K multiplies
// become one, at the very end.
//
// ---------------------------------------------------------------------------
// TWO REUSE DECISIONS, POINTING OPPOSITE WAYS
// ---------------------------------------------------------------------------
//
// Shared memory holds data that the kernel reads more than one time. Ask that
// question about each input separately, and you get two different answers:
//
//   WEIGHTS       One warp reads each weight one time, and never again. To
//                 stage them in shared memory is a copy with no purpose. Go
//                 from global memory into registers, at 16 int8 for each lane
//                 for each load.
//
//   ACTIVATIONS   K values for each batch row. Every output row in the block
//                 reads them. Stage them one time, and read them N/warps
//                 times.
//
// A K of 4096 and a batch of 8 will not fit in shared memory at once, so the
// activations come through in tiles.
//
// ---------------------------------------------------------------------------
// AND ONE DECISION ABOUT THE BATCH
// ---------------------------------------------------------------------------
//
// Give each (batch row, output row) pair its own block, and the kernel reads
// the whole weight matrix B times. That is the roofline mistake from stage
// 03, written in CUDA.
//
// Keep MAXB accumulators in registers instead. The 16 weights that a lane
// loaded then serve every row of the batch before the lane drops them.
//
// Then measure against torch. The kernel wins at one row and loses badly at
// four rows. That crossover is the lesson. Decode and prefill are different
// machines, and a GEMV is not a small GEMM.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
//
//   - Accumulate in float. int8 * bf16 summed over 4096 terms in bf16 loses
//     the low bits and the perplexity guard notices.
//   - `if (n >= N) return;` before a __syncthreads() hangs the block. Carry a
//     flag instead.
//   - A 16-byte load of int8 needs the row to start 16-byte aligned, which
//     needs K % 16 == 0. Fall back to scalar loads when it does not.
//   - __shfl_xor_sync has the same rule it had in stage 08c: every lane in
//     the mask must arrive.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kThreads / kWarp;
constexpr unsigned kFull = 0xffffffffu;
constexpr int kSharedBytes = 32768;             // activation staging budget

// TODO: __device__ float warp_sum(float v)     -- stage 08c had this one
// TODO: __global__ void gemv_int8(...)         -- one warp per output row
// TODO: a launcher that picks the vector path and the compile-time MAXB

}  // namespace

// The largest batch for this kernel. Above it, call a GEMM.
constexpr int kMaxBatch = 8;

torch::Tensor gemv_int8_cuda(torch::Tensor x, torch::Tensor w,
                             torch::Tensor scales) {
  // x       (B, K) or (K,)      bf16, fp16 or fp32
  // w       (N, K)              int8, contiguous
  // scales  (N,)                one per output channel
  // ->      (B, N) or (N,)      same dtype as x
  TORCH_CHECK(false, "stage 18b: implement gemv_int8_cuda");
}

int64_t max_batch() { return kMaxBatch; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_int8", &gemv_int8_cuda, "int8 weight-only GEMV with a fused "
                                      "dequantize epilogue");
  m.def("max_batch", &max_batch, "the largest batch this kernel handles");
}
