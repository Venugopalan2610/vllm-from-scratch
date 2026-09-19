// Stage 08b - the same kernel, with good memory access.
//
// `./vc lore 8b`. `./vc test 8b`.
//
// Start from your stage 08 kernel. The arithmetic does not change. The
// online softmax does not change. The change is which thread touches which
// byte.
//
// ---------------------------------------------------------------------------
// THE PROBLEM OF STAGE 08
// ---------------------------------------------------------------------------
//
// In stage 08, thread t owned position t and walked head_dim alone:
//
//     thread 0 reads key[position_0][0], key[position_0][1], ... [127]
//     thread 1 reads key[position_1][0], key[position_1][1], ... [127]
//
// At each instruction, the 32 threads of a warp read 32 addresses that are
// head_dim elements apart. The memory system serves a warp in 128-byte
// transactions, so it issues 32 of them where a coalesced read issues 2. All
// the bytes arrive, but with ten times more requests, and a bandwidth-bound
// kernel pays for requests.
//
// ---------------------------------------------------------------------------
// THE FIX: THE THREADS COOPERATE ALONG head_dim
// ---------------------------------------------------------------------------
//
//     VEC           = 16 / sizeof(scalar_t)      elements for each thread: 8 half, 4 float
//     lanes_per_row = head_dim / VEC             threads that share one K row
//     rows_per_tile = blockDim.x / lanes_per_row positions in flight at one time
//
//     thread (row, lane) reads key[position_row][lane*VEC ... lane*VEC+VEC)
//
// Adjacent lanes now read adjacent 16-byte parts of the same row, and the
// requests of the warp join into whole 128-byte transactions. 16 bytes is
// the widest load of one thread. A narrower load wastes some of the width of
// the memory system.
//
// Two results of this change:
//
//   1. lanes_per_row threads now SHARE the dot product. Each thread holds a
//      partial sum, and the kernel must add them. In this stage that
//      reduction goes through shared memory, as in stage 08. Stage 08c
//      removes it.
//
//   2. The accumulator can stay in REGISTERS. Thread (row, lane) owns VEC
//      elements of the output, for the positions that its group walks. The
//      rescale of the online softmax is the same for every thread of the
//      block. So each thread rescales its own partial sum, and the kernel
//      adds them one time, at the end. That is one shared-memory reduction
//      for each block, not one for each tile.
//
// ---------------------------------------------------------------------------
// ALIGNMENT IS A PRECONDITION, NOT A HOPE
// ---------------------------------------------------------------------------
//
// A 16-byte load needs a 16-byte aligned address. A row starts at a multiple
// of head_dim * sizeof(scalar_t), in an allocation that torch aligned. So
// the wide path is correct only when that product divides by 16. When it
// does not, use scalar loads.
//
// Do not read past the end of a row and hope that the mask catches it. A
// misaligned 16-byte load is a fault, not a wrong number.
//
// Then look at your register count:
//     VC_CUDA_VERBOSE=1 ./vc test 8b
// If ptxas reports spill stores, you asked for more registers than exist.
// The added "registers" are then local memory, which is DRAM.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 8;

// TODO: load_vec, a helper that loads VEC scalars into a float[VEC].
//
// When VEC * sizeof(scalar_t) == 16, cast the pointer to float4, load one
// time, and convert. If not, load one element at a time. `if constexpr`
// selects between them at compile time, so the wide path has no branch.

// TODO: __global__ void paged_attn_vec(...)
//
// Your stage 08 kernel with the mapping above.

}  // namespace

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale) {
  // The vector width is a constant at compile time inside the kernel, and a
  // decision at run time here. Dispatch on the dtype to select it. Use
  // VEC = 1 when head_dim does not divide by it.
  TORCH_CHECK(false, "stage 08b: implement paged_attn");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "coalesced paged decode attention (CUDA)");
}
