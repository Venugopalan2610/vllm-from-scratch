// Stage 08c - warp primitives, occupancy, and split-K.
//
// `./vc lore 8c`. `./vc test 8c`.
//
// Stage 08b reads memory as well as this kernel can. At num_seqs = 64, it
// stays near the streaming bandwidth of the card. At num_seqs = 1, it gets a
// few percent of that bandwidth.
//
// Coalescing cannot change that number. The memory system is not the
// problem now:
//
//     grid = (num_seqs, num_heads) = (1, 16) = 16 blocks
//     ./vc info tells you the number of SMs of this GPU
//
// Most of the machine has no work. Every interactive request is at batch 1.
// Two changes fix it.
//
// ---------------------------------------------------------------------------
// 1. WARP PRIMITIVES
// ---------------------------------------------------------------------------
//
// The lanes_per_row lanes that share a K row are adjacent threads, and
// lanes_per_row divides 32, so they are inside ONE warp. The lanes of a warp
// can exchange registers directly:
//
//     for (int stride = width / 2; stride > 0; stride >>= 1)
//         value += __shfl_xor_sync(0xffffffff, value, stride);
//
// No shared memory, no __syncthreads. xor is a butterfly, so every lane gets
// the answer, not only lane 0.
//
// THE TRAP THAT REPLACES THE __syncthreads TRAP: every lane in the mask must
// reach the instruction. Put a shuffle inside `if (position < context_len)`,
// and the lanes that fail the test never arrive. The other lanes then wait
// for ever. Compute a harmless value in the lanes with no position, and let
// all 32 lanes reach the shuffle.
//
// Then look at the cost. ptxas prints the registers for each thread with
// VC_CUDA_VERBOSE=1, and `./vc info` prints the registers for each SM.
// Divide. That is the number of warps that an SM holds at one time, which is
// how much memory latency it can hide. __launch_bounds__(128) lets ptxas
// exchange registers for occupancy on purpose, not by accident.
//
// ---------------------------------------------------------------------------
// 2. SPLIT-K, ALSO CALLED FLASH-DECODING
// ---------------------------------------------------------------------------
//
// If there are not enough (sequence, head) pairs to fill the GPU, make more
// blocks: cut the CONTEXT.
//
//     grid = (num_seqs, num_heads, num_splits)
//
// Block (seq, head, split) attends over its own chunk of the context. Then it
// writes a PARTIAL softmax state: the running maximum, the running sum, and
// the accumulator, which is not normalized. A second kernel merges the
// splits:
//
//     global_max = max_j maximum_j
//     weight_j   = exp(maximum_j - global_max)
//     output     = sum_j accumulator_j * weight_j / sum_j sum_j * weight_j
//
// That is the rescale that you already wrote, across blocks, not across
// tiles. It is exact. It is not an approximation.
//
// How many splits? Enough blocks to cover the SMs. Never so many that a split
// is shorter than about one hundred tokens. Below that length the merge
// costs more than the split saves.
//
// Read the number of SMs with
// at::cuda::getCurrentDeviceProperties()->multiProcessorCount.
//
// DO NOT call .item() on a tensor to select the grid. It copies from the
// device, and that stops the pipeline on every call. The width of the block
// table is an upper limit of the context on the host, and it costs nothing.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxVec = 8;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// TODO: warp_sum and warp_max over `width` lanes, with __shfl_xor_sync.
// TODO: block_max and block_sum: a warp reduction, one value for each warp
//       into shared memory, then a reduction of those values in warp 0.
//
//       A block reduction that gives its answer through shared memory needs
//       a barrier AFTER every thread reads the answer, not only before. The
//       next reduction uses the same scratch, and a fast warp overwrites a
//       result before a slow warp reads it. The output is then wrong for a
//       few percent of the elements, on some inputs, some of the time.
//       compute-sanitizer --tool racecheck finds it every time and names
//       both lines. A check here runs it.
//
//       Then think about how to need no block reduction. Every lane of a row
//       group gets the same score from the butterfly. So each group can keep
//       its OWN (maximum, sum, accumulator), with no barrier in the loop. The
//       groups merge at the end, with the same rescale that merges the
//       splits. That is the same method at three levels: tile, group and
//       block.
// TODO: __global__ void paged_attn_split(...)  -> a partial state
// TODO: __global__ void combine_splits(...)    -> the merge above
// TODO: int choose_splits(int num_seqs, int num_heads, int max_context)
//
//       With one split there is nothing to merge. Write the answer directly.
//       Do not write a partial state for a second kernel to read back. An
//       empty split and an empty row group must both report a state that the
//       merge can ignore. Look for -inf minus -inf. That is NaN, and it
//       poisons all the other partial states.

}  // namespace

torch::Tensor paged_attn(torch::Tensor query, torch::Tensor key_cache,
                         torch::Tensor value_cache, torch::Tensor block_tables,
                         torch::Tensor context_lens, double scale,
                         int64_t splits) {
  // splits <= 0 means "you select". The checks also force specific values,
  // so the merge must be correct for every number of splits, also 1.
  TORCH_CHECK(false, "stage 08c: implement paged_attn");
}

int64_t splits_for(int64_t num_seqs, int64_t num_heads, int64_t max_context) {
  TORCH_CHECK(false, "stage 08c: implement choose_splits");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attn", &paged_attn, "split-K paged decode attention (CUDA)",
        pybind11::arg("query"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_tables"),
        pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("splits") = 0);
  m.def("splits_for", &splits_for, "the number of splits for this shape");
}
