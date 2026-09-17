// Stage 08 - paged decode attention, in CUDA.
//
// `./vc lore 8` for the insight. `./vc test 8` to check yourself.
//
// Stage 07 was correct and slow: a Python loop over sequences, one gather and
// one SDPA call each. This file replaces all of it with two kernel launches.
//
// ---------------------------------------------------------------------------
// THE DECISION THIS STAGE IS ABOUT
// ---------------------------------------------------------------------------
//
// Not the arithmetic. The arithmetic is stage 07's, unchanged. The decision
// is WHAT ONE BLOCK OWNS, because everything else follows from it:
//
//     grid  = (num_seqs, num_heads)     one block per (sequence, query head)
//     block = 128 threads               they cooperate on one output vector
//
// One block computes one (head_dim,) output vector. It needs the whole
// context for its sequence to do that, so it walks the block table itself.
// Blocks never talk to each other, so there is no global synchronisation
// anywhere in this kernel. That is not an accident, it is the reason this
// decomposition was chosen.
//
// Inside the block, stage 08 uses the simplest possible mapping:
//
//     one thread per context position, walking head_dim serially
//
// It is correct, and it is slow, and stage 08b is about exactly why.
//
// ---------------------------------------------------------------------------
// ONLINE SOFTMAX
// ---------------------------------------------------------------------------
//
// The score row is never materialised. At 2048 context that row is 8KB per
// (seq, head), and writing it out and reading it back is more traffic than
// the K it came from. The same trick as FlashAttention:
//
//     m_new = max(m_old, max(scores))      running maximum
//     alpha = exp(m_old - m_new)           what the old accumulator is worth
//     p     = exp(scores - m_new)
//     l     = l * alpha + sum(p)           running denominator
//     acc   = acc * alpha + sum(p * V)     running numerator
//     out   = acc / l
//
// Process the context in tiles of blockDim.x positions and fold each tile in.
//
// ---------------------------------------------------------------------------
// WHAT WILL SAVE YOU AN AFTERNOON
// ---------------------------------------------------------------------------
//
//   - Accumulate in float, always, even when the cache is half. Summing 2048
//     half products in half loses several digits and the tolerances catch it.
//
//   - __syncthreads() must be reached by every thread in the block. Putting
//     one inside `if (pos < n)` hangs the kernel, and a hung kernel looks
//     like a hung test, not like a crash.
//
//   - Mask by position, not by block. Blocks are recycled, so slots past
//     context_len hold some other request's tokens. Scoring them gives a
//     plausible, wrong answer.
//
//   - GQA: query head h reads KV head h / (num_heads / num_kv_heads).
//
//   - Cache layout is (num_blocks, num_kv_heads, block_size, head_dim), so
//     the address of position `pos` of sequence `s` is
//         phys = block_tables[s][pos / block_size]
//         row  = ((phys * num_kv_heads + kvh) * block_size + pos % block_size)
//         k    = key_cache + row * head_dim
//     Use int64_t for that product. A big cache overflows int.
//
//   - Check every launch. C10_CUDA_KERNEL_LAUNCH_CHECK() turns a silent
//     failure into an exception; without it a kernel that never ran reads as
//     a numerical bug.
//
// Build errors point at real line numbers in this file. To see the compiler
// command and the register counts:  VC_CUDA_VERBOSE=1 ./vc test 8

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;

// TODO: __global__ void write_kv_kernel(...)
//
// A scatter, and the easiest kernel you will write here. One grid-stride loop
// over num_tokens * num_kv_heads * head_dim. For each element work out which
// cache slot it belongs to:
//
//     block = slot / block_size
//     off   = slot % block_size
//
// and copy. No reduction, no shared memory, no barriers.

// TODO: __global__ void paged_attn_v1(...)
//
// The shape above. One block per (seq, head), 128 threads, one thread per
// position in the tile, online softmax folded tile by tile.
//
// You will need shared memory for: the query vector (read once, used by every
// position), the tile's probabilities, the running accumulator, and a
// scratch array for the tree reductions. Pass the size as the third launch
// argument and declare it `extern __shared__ float smem[]`.

}  // namespace

// ---------------------------------------------------------------------------
// The host side. These two functions are what app/s08_paged_cuda.py calls.
// The launch configuration is yours: it is part of the kernel, not plumbing.
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
  m.def("write_kv", &write_kv, "scatter K/V into the paged cache (CUDA)");
}
