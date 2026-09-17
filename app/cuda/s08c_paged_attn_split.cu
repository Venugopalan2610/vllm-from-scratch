// Stage 08c - warp primitives, occupancy, and split-K.
//
// `./vc lore 8c`. `./vc test 8c`.
//
// Stage 08b reads memory about as well as this kernel can. Run it at
// num_seqs = 64 and it sits near the streaming bandwidth of the card. Run it
// at num_seqs = 1 and it sits at a few percent of it, and no amount of
// coalescing will move that, because the memory system is no longer the
// problem:
//
//     grid = (num_seqs, num_heads) = (1, 16) = 16 blocks
//     ./vc info tells you how many SMs this GPU has
//
// Most of the machine has no work. Batch 1 is also the case every
// interactive request lives in. Two changes fix it.
//
// ---------------------------------------------------------------------------
// 1. WARP PRIMITIVES
// ---------------------------------------------------------------------------
//
// The LPR lanes that share a K row are contiguous threads, and LPR divides
// 32, so they are inside ONE warp. Lanes of a warp can exchange registers
// directly:
//
//     for (int o = width / 2; o > 0; o >>= 1)
//         v += __shfl_xor_sync(0xffffffff, v, o);
//
// No shared memory, no __syncthreads, and xor is a butterfly so every lane
// ends up with the answer rather than just lane 0.
//
// THE TRAP THAT REPLACES THE __syncthreads TRAP: every lane named in the
// mask must reach the instruction. Put a shuffle inside `if (pos < n)` and
// the lanes that fail the test never arrive, and the ones that did wait for
// them forever. Compute a harmless value in the dead lanes and let all 32
// reach the shuffle.
//
// Then look at what it cost you. ptxas prints registers per thread with
// VC_CUDA_VERBOSE=1, and `./vc info` prints the registers per SM. Divide.
// That is how many warps an SM can hold at once, which is how much memory
// latency it can hide. __launch_bounds__(128) lets ptxas trade registers for
// occupancy on purpose instead of by accident.
//
// ---------------------------------------------------------------------------
// 2. SPLIT-K, ALSO CALLED FLASH-DECODING
// ---------------------------------------------------------------------------
//
// If there are not enough (sequence, head) pairs to fill the GPU, make more
// blocks by cutting the CONTEXT:
//
//     grid = (num_seqs, num_heads, SPLITS)
//
// Block (s, h, j) attends over its own chunk of the context and writes a
// PARTIAL softmax state: the running maximum m, the running denominator l,
// and the un-normalised accumulator acc. A second kernel merges the splits:
//
//     M     = max_j m_j
//     w_j   = exp(m_j - M)
//     out   = sum_j acc_j * w_j / sum_j l_j * w_j
//
// which is the alpha rescale you already wrote, applied across blocks
// instead of across tiles. It is exact. There is no approximation here.
//
// How many splits: enough blocks to cover the SMs, and never so many that a
// split is shorter than a hundred or so tokens, because then the merge costs
// more than the split saved. Read the SM count with
// at::cuda::getCurrentDeviceProperties()->multiProcessorCount.
//
// DO NOT call .item() on a tensor to decide the grid. It copies from the
// device, which means a full pipeline stall, on every single call. The width
// of the block table is a host-side upper bound on context and costs nothing.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 8;
constexpr int kWarp = 32;
constexpr unsigned kFull = 0xffffffffu;

// TODO: warp_sum / warp_max over `width` lanes, with __shfl_xor_sync.
// TODO: block_max / block_sum: warp reduce, one value per warp into shared
//       memory, then reduce those in warp 0.
//
//       A block reduction that hands its answer back through shared memory
//       needs a barrier AFTER every thread has read it, not only before. The
//       next reduction reuses the same scratch, and a warp that runs ahead
//       overwrites a result a slower warp has not collected yet. The output
//       is then wrong for a few percent of the elements, on some inputs,
//       some of the time. compute-sanitizer --tool racecheck finds it every
//       time and names both lines; there is a check here that runs it.
//
//       Then consider not needing them at all. Every lane of a row group
//       leaves the butterfly holding the same score, so each group can carry
//       its OWN (m, l, acc) with no barrier anywhere in the loop, and the
//       groups merge at the end with the same rescale the splits use. Same
//       trick at three levels: tile, group, block.
// TODO: __global__ void paged_attn_split(...)  -> partial (m, l, acc)
// TODO: __global__ void combine_splits(...)    -> the merge above
// TODO: int choose_splits(int S, int H, int max_ctx)
//
//       One split means nothing to merge: write the answer straight out
//       rather than staging a partial state for a second kernel to read
//       back. An empty split, and an empty row group, both still have to
//       report something the merge can ignore -- and watch for -inf minus
//       -inf, which is NaN, and poisons every other partial state with it.

}  // namespace

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale,
                         int64_t splits) {
  // splits <= 0 means "you choose". The checks also force specific values, so
  // the merge has to be right for any split count, including 1.
  TORCH_CHECK(false, "stage 08c: implement paged_attn");
}

int64_t splits_for(int64_t S, int64_t H, int64_t max_ctx) {
  TORCH_CHECK(false, "stage 08c: implement choose_splits");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "split-K paged decode attention (CUDA)",
        pybind11::arg("query"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_tables"),
        pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("splits") = 0);
  m.def("splits_for", &splits_for, "how many splits this shape would use");
}
