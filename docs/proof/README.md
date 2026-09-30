# The proof log

The course says that a measurement proves each stage. This folder is that
measurement for the course itself: every check run against the reference
solutions, on one real card. `dev/proof.sh` makes all of it again, on your
card. It took about 15 minutes on this one.

## The machine

From [`machine.txt`](machine.txt):

| | |
|---|---|
| Date | 2026-09-30 |
| Commit | `3328639` |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB |
| Software | PyTorch 2.13.0 (CUDA 13.0), nvcc 13.2, transformers 5.14.1 |
| Model of the checks | Qwen3-0.6B |

## The results

**Every check of the torch track** ([`checks.log`](checks.log)): **904 passed, 0
failed, 2 skipped**, in 5 minutes 31 seconds. The two skipped checks read the
GPU performance counters, and this machine does not permit it without extra
rights (stages 08b and 08c, the Nsight Compute check).

**The live engine** ([`vc_run.log`](vc_run.log), [`vc_run_given.log`](vc_run_given.log)):
24 requests on the same workload.

| | Given parts only | Every reference part, 06 to 19 |
|---|---|---|
| Tokens per second | 139.5 | 1,718.4 (12.3x) |
| Requests that ran together, at most | 3 | 24 |
| Differences from the oracle that are not a rounding flip | 0 | 0 |

**The four lines of each stage** ([`four_lines.log`](four_lines.log)): the
measured time divided by the floor that the bytes give. Lower is closer to
the hardware.

| The work | Stage | Measured / floor |
|---|---|---|
| Paged attention, 8 sequences | 07, PyTorch gather | 12.35 |
| | 08, CUDA kernel | 4.61 |
| | 08b, coalesced loads | 2.64 |
| Paged attention, 2 sequences | 08c, split-K | 1.58 |
| An int8 layer, 16384 x 8192, one row | 18, PyTorch | 9.71 |
| | 18b, int8 GEMV kernel | 1.41 |
| A decode step of the model, 2 sequences | 21, eager | 1.92 |
| | 23, captured graph | 1.58 |

From the same run, `./vc run` also measured:

- Speculative decoding (stage 17): 3.2 tokens for each forward pass, 3.02x faster on a copy task.
- Int8 weights in every matmul (stage 18b): a batch-1 step 1.33x faster, with a KL from bf16 of 0.0045 nats.

## What this does not prove

- **One card.** The milliseconds are true only on this laptop. The ratios move
  less from card to card, but they do move. Run `dev/proof.sh` on yours.
- **The JAX track did not run.** Its 115 checks were not selected.
- **No GPU runs in CI.** CI runs the stages that need no GPU on each push. This
  log is the proof of the others, at one commit.
- **The reference solutions, not yours.** This log proves that each check can
  pass and that the course code works. Your own numbers come from `./vc test`.

## To make it again

```bash
dev/proof.sh      # writes this folder: checks.log, four_lines.log, vc_run*.log, machine.txt
```
