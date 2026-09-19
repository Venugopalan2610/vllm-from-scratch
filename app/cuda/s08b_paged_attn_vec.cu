// Stage 08b - the same kernel, fed properly.
//
// `./vc lore 8b`. `./vc test 8b`.
//
// Start from your stage 08 kernel. The arithmetic does not change at all.
// Nothing about the online softmax changes. What changes is which thread
// touches which byte.
//
// ---------------------------------------------------------------------------
// WHAT IS WRONG WITH STAGE 08
// ---------------------------------------------------------------------------
//
// In stage 08, thread t owned position t and walked head_dim itself:
//
//     thread 0 reads k[pos_0][0], k[pos_0][1], ... k[pos_0][127]
//     thread 1 reads k[pos_1][0], k[pos_1][1], ... k[pos_1][127]
//
// At any one instruction, the 32 threads of a warp are reading 32 addresses
// that are head_dim elements apart. The memory system serves a warp in
// 128-byte transactions, so it has to issue 32 of them where a coalesced
// read would have issued 2. The bytes you needed all arrive eventually, but
// you spent an order of magnitude more requests getting them, and a
// bandwidth-bound kernel pays for requests.
//
// ---------------------------------------------------------------------------
// THE FIX: THREADS COOPERATE ALONG head_dim
// ---------------------------------------------------------------------------
//
//     VEC  = 16 / sizeof(scalar_t)      elements per thread: 8 half, 4 float
//     LPR  = head_dim / VEC             threads that share one K row
//     ROWS = blockDim.x / LPR           positions in flight at once
//
//     thread (row, lane) reads k[pos_row][lane*VEC ... lane*VEC+VEC)
//
// Consecutive lanes now read consecutive 16-byte chunks of the same row, and
// the requests of the warp merge into whole 128-byte transactions. 16 bytes
// is the widest load that one thread can issue. A narrower load wastes some
// of the width of the memory system.
//
// Two consequences to work through:
//
//   1. LPR threads now SHARE the dot product. Each one holds a partial sum,
//      and the kernel must add them. In this stage that reduction goes
//      through shared memory, as in stage 08. Stage 08c removes it.
//
//   2. The accumulator can stay in REGISTERS. Thread (row, lane) owns VEC
//      elements of the output, for the positions that its group walks.
//      The online softmax rescale alpha is the same for every thread in the
//      block. So each thread rescales its own partial sum, and the kernel
//      adds them one time, at the end. That is one shared-memory reduction
//      for each block, and not one for each tile.
//
// ---------------------------------------------------------------------------
// ALIGNMENT IS A PRECONDITION, NOT A HOPE
// ---------------------------------------------------------------------------
//
// A 16-byte load needs a 16-byte aligned address. A row starts at a multiple
// of head_dim * sizeof(scalar_t), from an allocation that torch aligned for
// you. So the wide path is legal exactly when that product divides by 16.
// When it does not divide by 16, use scalar loads.
//
// Do not read past the end of a row and hope that the mask catches it. A
// misaligned 16-byte load is a fault. It is not a wrong number.
//
// Then check your register count:
//     VC_CUDA_VERBOSE=1 ./vc test 8b
// If ptxas reports spill stores, you asked for more registers than exist and
// the "registers" you added are really local memory, which is DRAM.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 8;

// TODO: a helper that loads VEC scalars into a float[VEC].
//
// When VEC * sizeof(scalar_t) == 16, reinterpret the pointer as float4, load
// once, and convert. Otherwise load element by element. `if constexpr` picks
// between them at compile time, so the wide path has no branch in it.

// TODO: __global__ void paged_attn_vec(...)
//
// Your stage 08 kernel with the mapping above.

}  // namespace

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale) {
  // The vector width is a compile-time constant inside the kernel and a
  // runtime decision out here. Dispatch on the dtype to pick it, and fall
  // back to VEC = 1 when head_dim does not divide by it.
  TORCH_CHECK(false, "stage 08b: implement paged_attn");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "coalesced paged decode attention (CUDA)");
}
