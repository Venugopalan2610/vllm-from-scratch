# Capstone Practicum I: read the timeline

> *You do not guess where latency comes from. You measure the machine.*

## The notebook

Open [`part8_nsys_1_CCreadTheTimeline_helper.ipynb`](part8_nsys_1_CCreadTheTimeline_helper.ipynb).
It profiles a real decode loop of Qwen3-1.7B with Nsight Systems (`nsys`), and
you read the trace with plain SQL. You find which side limits a step, the CPU
or the GPU, and you predict what CPU work costs in two loops: one that waits
for the GPU and then works, and one that works while the GPU runs. The checks
at the end test the machine and your model. Nsight Systems needs no special
permission. The notebook takes about ten minutes.

## Then your engine (after stage 28)

    ./vc nsys
    nsys export --type sqlite .cudacache/engine_profile.nsys-rep

The profile marks each step of the engine with an NVTX range named
`engine_step_<n>`. Use the functions of the notebook on it, and answer:

1. What fraction of the window is the GPU idle between the steps?
2. Which side is slower at the load of the profile? What would change the
   answer?
3. Where do the copies of the block tables and the slot mapping run? Do they
   block the kernels, and does the CPU wait for them? (Stage 23 stages them in
   pinned buffers.)

Open the report in the GUI to see the same data as a picture:

    nsys-ui .cudacache/engine_profile.nsys-rep
