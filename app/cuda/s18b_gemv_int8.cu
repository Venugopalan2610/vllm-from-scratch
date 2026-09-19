// Stage 18b - int8 weight-only GEMV, dequantized after the dot product.
//
// `./vc lore 18b`. `./vc test 18b`.
//
// Stage 18 made the weight bytes half as many. Then it gave the multiply to
// PyTorch, and PyTorch made a bf16 copy of the weight first. Time that, and
// the quantization is a LOSS: you wrote a full-size matrix to memory, read it
// back, and also read the int8 matrix.
//
// Weight-only quantization pays only if the dequantize is inside the kernel,
// on a value that is already in a register. That is the meaning of a "fused
// epilogue", and in this stage you write one.
//
//     output[b, n] = sum_k input[b, k] * (weight[n, k] * scale[n])
//                  = scale[n] * sum_k input[b, k] * weight[n, k]
//
// There is one scale for each output channel, so it comes out of the sum:
// in_features multiplies become one, at the end.
//
// ---------------------------------------------------------------------------
// TWO REUSE DECISIONS, IN OPPOSITE DIRECTIONS
// ---------------------------------------------------------------------------
//
// Shared memory holds data that the kernel reads more than one time. Ask that
// question about each input, and you get two different answers:
//
//   WEIGHTS       One warp reads each weight one time, and never again. A copy
//                 into shared memory has no purpose. Load from global memory
//                 into registers, 16 int8 values for each lane in each load.
//
//   ACTIVATIONS   in_features values for each batch row. Every output row
//                 reads them. Stage them in shared memory one time FOR EACH
//                 BLOCK.
//
// So a block must own MANY output rows. With one output row for each warp
// and 4 warps, every block stages the whole activation tile for only 4 rows.
// At a batch of 8, that activation traffic is four times the weight traffic,
// and the kernel is slower than cuBLAS at four rows. Give each warp 4 output
// rows. A block then owns 16 rows, and the activations cost a quarter as
// much.
//
// An in_features of 4096 and a batch of 8 do not fit in shared memory at one
// time, so the activations come in tiles.
//
// ---------------------------------------------------------------------------
// ONE DECISION ABOUT THE BATCH
// ---------------------------------------------------------------------------
//
// Give each (batch row, output row) pair its own block, and the kernel reads
// the whole weight matrix batch_size times. That is the roofline mistake of
// stage 03, in CUDA.
//
// Keep kRowsPerWarp x MAX_BATCH sums in registers. For each 16-byte chunk,
// load the 16 weights of all the rows of the warp first. Then, for each batch
// row, load its 16 activations from shared memory ONE time, with 16-byte
// vector loads. Use them for all the weight rows.
//
// Then measure against torch. The kernel is much faster at one row, and the
// margin decreases as the rows increase. That crossover is the lesson.
// Decode and prefill are different machines, and a GEMV is not a small GEMM.
// Stage 24 measures the crossover again in the real model, where it selects
// the path of each matmul.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
//
//   - Accumulate in float. A bf16 sum of 4096 int8 * bf16 products loses the
//     low bits, and the perplexity guard finds it.
//   - `if (row >= out_features) return;` before a __syncthreads() stops the
//     block. Keep a flag.
//   - A 16-byte load of int8 needs a row that starts 16-byte aligned, so
//     in_features % 16 == 0. Use scalar loads when it is not.
//   - __shfl_xor_sync has the same rule as in stage 08c: every lane in the
//     mask must arrive.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kThreads / kWarp;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kSharedBytes = 32768;       // the budget for the staged activations
constexpr int kRowsPerWarp = 4;

// TODO: __device__ float warp_sum(float value)   -- stage 08c had this one
// TODO: __global__ void gemv_int8(...)           -- kRowsPerWarp rows for each warp
// TODO: a launcher that selects the vector path and MAX_BATCH at compile time

}  // namespace

// The largest batch for this kernel. Above it, call a GEMM.
constexpr int kMaxBatch = 8;

torch::Tensor gemv_int8_cuda(torch::Tensor inputs, torch::Tensor weight,
                             torch::Tensor scales) {
  // inputs  (batch, in_features) or (in_features,)   bf16, fp16 or fp32
  // weight  (out_features, in_features)             int8, contiguous
  // scales  (out_features,)                         one for each output channel
  // ->      (batch, out_features) or (out_features,) the dtype of inputs
  TORCH_CHECK(false, "stage 18b: implement gemv_int8_cuda");
}

int64_t max_batch() { return kMaxBatch; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_int8", &gemv_int8_cuda,
        "int8 weight-only GEMV, dequantized after the dot product");
  m.def("max_batch", &max_batch, "the largest batch that this kernel accepts");
}
