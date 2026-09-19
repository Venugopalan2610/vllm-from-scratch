// Stage 08 - paged decode attention, in CUDA.
//
// `./vc lore 8` for the insight. `./vc test 8` to check yourself.
//
// Stage 07 was correct and slow: a Python loop over the sequences, with one
// gather and one SDPA call for each. This file replaces all of it with two
// kernel launches.
//
// ---------------------------------------------------------------------------
// THE DECISION OF THIS STAGE
// ---------------------------------------------------------------------------
//
// The arithmetic is not the decision. It is the arithmetic of stage 07, with
// no change. The decision is WHAT ONE BLOCK OWNS, because all the rest
// follows from it:
//
//     grid  = (num_seqs, num_heads)     one block for each (sequence, query head)
//     block = 128 threads               they share one output vector
//
// One block computes one (head_dim,) output vector. For that, it needs the
// whole context of its sequence, so it reads the block table itself. Blocks
// never communicate, so this kernel has no global synchronization. That is
// the reason for this decomposition.
//
// Inside the block, stage 08 uses the simplest mapping:
//
//     one thread for each context position, with a serial walk down head_dim
//
// It is correct, and it is slow. Stage 08b is about the reason.
//
// ---------------------------------------------------------------------------
// THE ONLINE SOFTMAX
// ---------------------------------------------------------------------------
//
// The score row never goes to memory. At a context of 2048, that row is 8 KB
// for each (seq, head). A write and a read of it cost more traffic than the K
// that made it. Use the method of FlashAttention:
//
//     new_max     = max(running_max, max(scores))
//     rescale     = exp(running_max - new_max)     the value of the old sums
//     p           = exp(scores - new_max)
//     running_sum = running_sum * rescale + sum(p)       the denominator
//     accumulator = accumulator * rescale + sum(p * V)   the numerator
//     output      = accumulator / running_sum
//
// Process the context in tiles of blockDim.x positions, and add each tile.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
//
//   - Accumulate in float, always, also when the cache is half. A sum of 2048
//     half products in half loses several digits, and the tolerances find it.
//
//   - Every thread of the block must reach a __syncthreads(). One inside
//     `if (position < context_len)` stops the kernel. A kernel that stops
//     looks like a test that stops, not like a crash.
//
//   - Mask by position, not by block. The allocator reuses blocks, so a slot
//     after context_len holds the tokens of another request. A score on
//     those tokens gives an answer that looks correct and is wrong.
//
//   - GQA: query head h reads KV head h / (num_heads / num_kv_heads).
//
//   - The cache layout is (num_blocks, num_kv_heads, block_size, head_dim).
//     So the address of position `position` of sequence `seq` is
//         block_id = block_tables[seq][position / block_size]
//         row      = (block_id * num_kv_heads + kv_head) * block_size
//                    + position % block_size
//         key_row  = key_cache + row * head_dim
//     Use int64_t for that product. A large cache overflows an int.
//
//   - Check every launch. C10_CUDA_KERNEL_LAUNCH_CHECK() changes a silent
//     failure into an exception. Without it, a kernel that never ran looks
//     like a numerical bug.
//
// A build error shows the real line number in this file. To see the compiler
// command and the register counts:  VC_CUDA_VERBOSE=1 ./vc test 8

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;

// TODO: __global__ void write_kv_kernel(...)
//
// A scatter, and the easiest kernel of this course. One grid-stride loop
// over num_tokens * num_kv_heads * head_dim. For each element, find its
// cache slot:
//
//     block_id = slot / block_size
//     offset   = slot % block_size
//
// and copy. No reduction, no shared memory, no barriers.

// TODO: __global__ void paged_attn_v1(...)
//
// The shape above. One block for each (seq, head), 128 threads, one thread
// for each position of the tile, and the online softmax, one tile at a time.
//
// You need shared memory for four things:
//   - the query vector, which you read one time and every position uses,
//   - the probabilities of the tile,
//   - the accumulator,
//   - a scratch array for the tree reductions.
// Give the size as the third launch argument. Declare it as
// `extern __shared__ float shared[]`.

}  // namespace

// ---------------------------------------------------------------------------
// The host side. app/s08_paged_cuda.py calls these two functions. The launch
// configuration is yours: it is part of the kernel.
// ---------------------------------------------------------------------------

void write_kv(torch::Tensor key_cache, torch::Tensor value_cache,
              torch::Tensor key, torch::Tensor value,
              torch::Tensor slot_indices) {
  TORCH_CHECK(false, "stage 08: implement write_kv");
}

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale) {
  TORCH_CHECK(false, "stage 08: implement paged_attn");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "paged decode attention (CUDA)");
  m.def("write_kv", &write_kv, "scatter K and V into the paged cache (CUDA)");
}
