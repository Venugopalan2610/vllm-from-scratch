"""Build the notebooks of Capstone Practicums I (Nsight Systems) and J
(Nsight Compute), each as a challenge and a solution.

    python dev/build_practicums.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from break_it.common import ROOT, code, md, save  # noqa: E402


def header(section, lecture):
    return md(f'''
|<h2>Course:</h2>|<h1><a href="https://derivingsystems.com/course.html" target="_blank">Build your own vLLM: inference engines from the memory system up</a></h1>|
|-|:-:|
|<h2>Part 8:</h2>|<h1>The Capstone<h1>|
|<h2>Section:</h2>|<h1>{section}<h1>|
|<h2>Lecture:</h2>|<h1><b>{lecture}<b></h1>|

<br>

<h5><b>Course repo:</b> <a href="https://github.com/Venugopalan2610/vllm-from-scratch" target="_blank">github.com/Venugopalan2610/vllm-from-scratch</a></h5>
<h5><b>The derivations:</b> <a href="https://derivingsystems.com" target="_blank">derivingsystems.com</a></h5>
<i>The notebooks build the intuition. The ladder in app/ makes you build the thing.</i>
''')


def build(cells, helper_path, solution_path, section, title):
    """cells: a list of ('md', text) or ('code', helper_text, solution_text)."""
    helper = [header(section, f'CodeChallenge HELPER: {title}')]
    solution = [header(section, f'CodeChallenge: {title}')]
    for cell in cells:
        if cell[0] == 'md':
            helper.append(md(cell[1]))
            solution.append(md(cell[1]))
        elif cell[0] == 'code':
            helper.append(code(cell[1]))
            solution.append(code(cell[2] if len(cell) > 2 else cell[1]))
        elif cell[0] == 'solution':                      # only in the solution
            solution.append(md(cell[1]))
    save(ROOT / helper_path, helper)
    save(ROOT / solution_path, solution)


# ============================================================ Practicum I

DRIVER = r'''
DRIVER = r"""
import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE, CPU_MS, BATCH, TOKENS = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), 40
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', dtype=torch.bfloat16).cuda().eval()
nvtx = torch.cuda.nvtx

def cpu_work():                          # the scheduler and the detokenizer: busy Python
    end = time.perf_counter() + CPU_MS / 1000
    while time.perf_counter() < end:
        pass

with torch.inference_mode():
    ids = tokenizer(['The three largest cities in Japan are'] * BATCH, return_tensors='pt').input_ids.cuda()
    out = model(ids, use_cache=True)
    cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
    for _ in range(5):                   # warm up, outside the window of the profile
        out = model(token, past_key_values=cache, use_cache=True)
        token = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.cudart().cudaProfilerStart()
    for step in range(TOKENS):
        nvtx.range_push('step')
        nvtx.range_push('forward')
        out = model(token, past_key_values=cache, use_cache=True)   # the CPU launches, the GPU runs later
        token = out.logits[:, -1:].argmax(-1)
        nvtx.range_pop()
        if MODE == 'series':
            token.tolist()               # wait for the GPU, THEN do the CPU work
        nvtx.range_push('cpu_work')
        cpu_work()
        nvtx.range_pop()
        nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print('STEP_MS', (time.perf_counter() - start) / TOKENS * 1000)
"""
(TRACES / 'decode_loop.py').write_text(DRIVER)

def run(mode, cpu_ms, batch, profiled=True):
    """Run the loop. -> (the path of the SQLite trace or None, the step in ms)."""
    command = [sys.executable, str(TRACES / 'decode_loop.py'), mode, str(cpu_ms), str(batch)]
    trace = TRACES / f'{mode}_{batch}'
    if profiled:
        command = ['nsys', 'profile', '--trace=cuda,nvtx', '--capture-range=cudaProfilerApi',
                   '--capture-range-end=stop',            # let the program finish, and print
                   '--export=sqlite', '--force-overwrite=true', '-o', str(trace)] + command
    result = subprocess.run(command, capture_output=True, text=True, cwd=TRACES)
    step = [float(line.split()[1]) for line in result.stdout.splitlines() if line.startswith('STEP_MS')]
    if not step:
        raise RuntimeError(result.stdout[-1500:] + result.stderr[-1500:])
    return (Path(f'{trace}.sqlite') if profiled else None), step[0]

def query(db, sql, *arguments):
    return sqlite3.connect(db).execute(sql, arguments).fetchall()
'''

SETUP_I = '''
# find the repo root. The directory you start from does not matter.
import sys, sqlite3, statistics, subprocess, shutil
from pathlib import Path
ROOT = next(folder for folder in [Path.cwd(), *Path.cwd().parents]
            if (folder/'cudalib').is_dir())
sys.path.insert(0, str(ROOT))
TRACES = ROOT / '.cudacache' / 'practicum_i'
TRACES.mkdir(parents=True, exist_ok=True)
print('nsys:', shutil.which('nsys') or 'NOT FOUND. It comes with the CUDA toolkit, in /usr/local/cuda/bin.')
'''

EX1_HELPER = '''
def kernel_intervals(db):
    """-> a sorted list of (start_ms, end_ms), one for each GPU kernel.
    The table is CUPTI_ACTIVITY_KIND_KERNEL. Its times are in nanoseconds."""
    ...

def merge(intervals):
    """-> the same time, with the overlaps joined. Two kernels on two streams
    can overlap, and a sum of their durations counts that time twice."""
    ...

def ranges(db, name):
    """-> the (start_ms, end_ms) of each NVTX range with this text, in order.
    The table is NVTX_EVENTS. A range that never ended has end NULL."""
    ...

db, _ = run('none', 0.0, 1)
print(len(kernel_intervals(db)), 'kernels,', len(ranges(db, 'step')), 'steps')
'''

EX1_SOLUTION = '''
def kernel_intervals(db):
    return sorted((start / 1e6, end / 1e6)
                  for start, end in query(db, 'SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL'))

def merge(intervals):
    joined = []
    for start, end in sorted(intervals):
        if joined and start <= joined[-1][1]:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return joined

def ranges(db, name):
    return [(start / 1e6, end / 1e6) for start, end in
            query(db, 'SELECT start, end FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL ORDER BY start', name)]

db, _ = run('none', 0.0, 1)
print(len(kernel_intervals(db)), 'kernels,', len(ranges(db, 'step')), 'steps')
'''

EX2_HELPER = '''
def step_profile(db):
    \"\"\"-> a dict for the window from the first 'step' to the end of the last:
         step_ms         the mean time of one step
         gpu_ms          the mean GPU busy time of one step (merged kernels)
         cpu_forward_ms  the mean duration of the 'forward' range
         lag_ms          the mean time from the end of a 'forward' range to the
                         end of the last kernel that it launched
         idle            the fraction of the window with no kernel running
    A kernel and the CPU call that launched it share a correlationId. The
    calls are in the table CUPTI_ACTIVITY_KIND_RUNTIME.\"\"\"
    ...

profiles = {}
for batch in (1, 256):
    db, _ = run('none', 0.0, batch)
    profiles[batch] = step_profile(db)
    print(f'batch {batch:3d}:', {key: round(value, 2) for key, value in profiles[batch].items()})
'''

EX2_SOLUTION = '''
def step_profile(db):
    steps = ranges(db, 'step')
    start, end = steps[0][0], steps[-1][1]
    inside = [(max(s, start), min(e, end)) for s, e in kernel_intervals(db) if e > start and s < end]
    busy = sum(e - s for s, e in merge(inside))
    forwards = ranges(db, 'forward')
    launches = [(launch / 1e6, done / 1e6) for launch, done in query(
        db, 'SELECT r.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k '
            'JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId')]
    lags = []
    for s, e in forwards:
        ends = [done for launch, done in launches if s <= launch <= e]
        if ends:
            lags.append(max(ends) - e)
    return dict(step_ms=(end - start) / len(steps), gpu_ms=busy / len(steps),
                cpu_forward_ms=statistics.mean(e - s for s, e in forwards),
                lag_ms=statistics.mean(lags), idle=1 - busy / (end - start))

profiles = {}
for batch in (1, 256):
    db, _ = run('none', 0.0, batch)
    profiles[batch] = step_profile(db)
    print(f'batch {batch:3d}:', {key: round(value, 2) for key, value in profiles[batch].items()})
'''

EX3_HELPER = '''
CPU_MS = 8.0

def slower_side(profile):
    \"\"\"-> 'CPU' or 'GPU', from the profile of the loop with no work.\"\"\"
    ...

def predict_extra(profile, cpu_ms, mode):
    \"\"\"The time in ms that `cpu_ms` of CPU work ADDS to one step, over the
    loop with no work.
      'series':  the loop waits for the GPU, and then it works.
      'overlap': the loop works while the GPU runs the step that it launched.\"\"\"
    ...

rows = []
for batch in (1, 256):
    for mode in ('series', 'overlap'):
        db, _ = run(mode, CPU_MS, batch)
        extra = step_profile(db)['step_ms'] - profiles[batch]['step_ms']
        predicted = predict_extra(profiles[batch], CPU_MS, mode)
        rows.append((batch, mode, predicted, extra))
        print(f'batch {batch:3d} ({slower_side(profiles[batch])} slower) {mode:8s}: '
              f'predicted +{predicted:4.1f} ms, measured +{extra:4.1f} ms')
'''

EX3_SOLUTION = '''
CPU_MS = 8.0

def slower_side(profile):
    return 'CPU' if profile['idle'] > 0.2 else 'GPU'     # an idle GPU waits for the CPU

def predict_extra(profile, cpu_ms, mode):
    cpu_slower = slower_side(profile) == 'CPU'
    if mode == 'overlap':
        return cpu_ms if cpu_slower else 0.0       # hidden only while the GPU is busy anyway
    tail = profile['lag_ms'] if cpu_slower else 0.0
    return cpu_ms + tail                            # wait for the last kernels, then work

rows = []
for batch in (1, 256):
    for mode in ('series', 'overlap'):
        db, _ = run(mode, CPU_MS, batch)
        extra = step_profile(db)['step_ms'] - profiles[batch]['step_ms']
        predicted = predict_extra(profiles[batch], CPU_MS, mode)
        rows.append((batch, mode, predicted, extra))
        print(f'batch {batch:3d} ({slower_side(profiles[batch])} slower) {mode:8s}: '
              f'predicted +{predicted:4.1f} ms, measured +{extra:4.1f} ms')
'''

CHECK_I = '''
### THE CHECKS. Do not edit this cell.

assert len(profiles) == 2 and len(rows) == 4, 'run Exercises 2 and 3 first: a check with no data proves nothing'
assert slower_side(profiles[1]) == 'CPU' and slower_side(profiles[256]) == 'GPU', \\
    'at batch 1 the CPU is the slower side here, and at batch 256 the GPU'
measured = {(batch, mode): extra for batch, mode, _, extra in rows}
predicted = {(batch, mode): value for batch, mode, value, _ in rows}
# the machine: the facts that every run shows
assert measured[(1, 'overlap')] > 0.7 * CPU_MS, 'with the CPU slower, the overlap must still pay the CPU work'
assert measured[(256, 'overlap')] < 0.6 * CPU_MS, 'with the GPU slower, the overlap must hide most of the CPU work'
for batch in (1, 256):
    assert 0.8 * CPU_MS < measured[(batch, 'series')] < 1.5 * CPU_MS, 'the series loop pays the CPU work, plus a little'
# your model: the same facts, predicted
assert predicted[(256, 'overlap')] < predicted[(1, 'overlap')], 'your model must hide more work when the GPU is slower'
for batch in (1, 256):
    assert abs(predicted[(batch, 'series')] - measured[(batch, 'series')]) < 0.35 * CPU_MS, \\
        f'batch {batch} series: predicted +{predicted[(batch, "series")]:.1f} ms, measured +{measured[(batch, "series")]:.1f} ms'
print('All the checks pass: you found the slower side, and what the CPU work costs on each side.')
'''

EX4_CODE = '''
_, plain = run('none', 0.0, 1, profiled=False)
print(f'batch 1, no profiler: {plain:.1f} ms for each step')
print(f'batch 1, under nsys : {profiles[1]["step_ms"]:.1f} ms for each step')
print(f'the profiler adds {profiles[1]["step_ms"] / plain - 1:.0%}')
'''

PRACTICUM_I = [
    ('md', '''
# Read the timeline

A timeline shows two machines side by side: the CPU, which launches work, and
the GPU, which runs it. A step is as slow as the slower of the two, plus every
moment where one waits for the other. You find the slower side first. Every
optimization depends on the answer.

This notebook profiles a real decode loop of Qwen3-1.7B with **Nsight
Systems** (`nsys`). The loop marks its phases with NVTX ranges: `step`,
`forward` (the CPU time to launch one forward pass) and `cpu_work` (8 ms of
busy Python, as a scheduler and a detokenizer cost). You read the trace from
its SQLite export with plain SQL.

The loop runs in three versions:

- `none`: no CPU work.
- `series`: after each step, the loop waits for the GPU, and then it works.
  This is the loop of Ticket 3 of the incident file.
- `overlap`: the loop works while the GPU runs the step that it launched.

Nsight Systems needs no special permission. Each profile loads the model, so
each run takes about a minute, and the whole notebook about ten.
'''),
    ('code', DRIVER),
    ('md', '''
# Exercise 1: read the kernels and the ranges

Write three functions. `merge` matters: two kernels on two streams can run at
the same time, and a sum of their durations would count that time twice.
'''),
    ('code', EX1_HELPER, EX1_SOLUTION),
    ('md', '''
# Exercise 2: which side is slower?

For the loop with no CPU work, at batch 1 and at batch 256, compute the step
time, the GPU busy time of one step, the duration of the `forward` range, the
lag, and the idle fraction of the GPU.

Before you run it, predict: at batch 1, is the CPU or the GPU the slower side?
And at batch 256?

Look closely at `cpu_forward_ms` when the GPU is the slower side. It is not the
time to launch. When the GPU falls behind, the queue of launches fills, and
each new launch waits for a free place. So the CPU seems slow because it waits
for the GPU.
'''),
    ('code', EX2_HELPER, EX2_SOLUTION),
    ('md', '''
# Exercise 3: what does 8 ms of CPU work cost?

Write `slower_side` and `predict_extra`. The second one predicts how much 8 ms
of CPU work adds to each step of the `series` loop and of the `overlap` loop,
at each batch size. Think about which side waits for which. When can the CPU
work hide behind the GPU? When the series loop waits for the GPU, what is it
still waiting for? Then the cell measures both loops at both batch sizes.
'''),
    ('code', EX3_HELPER, EX3_SOLUTION),
    ('md', '''
### The checks

The checks test two things. First, the machine: the facts that every run
shows. Second, your model: it must name the slower side at each batch size,
predict the series loop within 2.8 ms (35% of the CPU work), and hide more work
when the GPU is the slower side. Two profiled runs of the same loop can differ
by a few milliseconds, so the checks test the effect, not the third digit.

If your model predicts that the overlap at batch 256 hides **all** of the CPU
work, compare with the measurement, and read the `forward` range of each loop.
The difference is the most interesting number of this notebook.
'''),
    ('code', CHECK_I),
    ('md', '''
# Exercise 4: the profiler changes what it measures

Run the loop at batch 1 with no profiler, and compare the step time.
'''),
    ('code', EX4_CODE),
    ('md', '''
# Exercise 5: your engine (after stage 28)

`./vc nsys` profiles your capstone engine and writes
`.cudacache/engine_profile.nsys-rep`. Export it to SQLite:

    nsys export --type sqlite .cudacache/engine_profile.nsys-rep

Each step of the engine is an NVTX range named `engine_step_<n>`. Use your
functions of Exercise 1, with `text LIKE 'engine_step_%'`, and answer:

1. What fraction of the window is the GPU idle between the steps?
2. Is the CPU or the GPU the slower side of your engine at the load of the
   profile? What would change the answer?
3. Stage 23 captures the decode step as a graph. What does the graph remove
   from the timeline, and which side does it speed up?

### Before you open the solution

1. In Exercise 3, why does the overlap loop still pay for the CPU work when
   the CPU is the slower side?
2. A GPU that is idle 30% of a step is a clue, not a diagnosis. Name two
   different causes that give the same idle fraction, and the evidence on the
   timeline that separates them.
3. The profiler slowed the loop down. Is a conclusion from a profiled run
   still valid? Which conclusions survive, and which do not?
'''),
    ('solution', '''
### What I measured

On an RTX 4080 Laptop GPU, under Nsight Systems, with 8 ms of CPU work, in two runs:

| batch | the slower side | GPU idle, no work | series loop | overlap loop | `forward`: none / series / overlap |
|---|---|---|---|---|---|
| 1 | CPU | 36% to 38% | +8.0 to +8.8 ms | +6.8 to +7.8 ms | 17.3 / 15.7 / 17.1 ms |
| 256 | GPU | 4% | +9.5 to +9.7 ms | +3.1 to +3.2 ms | 35.7 / 17.3 / 30.9 ms |

- **At batch 1 the CPU is the slower side.** One step launches about 1,600
  kernels, and the CPU needs longer to launch them than the GPU needs to run
  them. The overlap cannot hide the CPU work, because the CPU is the critical
  path. It pays most of the 8 ms, like the series loop.
- **At batch 256 the GPU is the slower side**, and the overlap hides 60% of the
  work. Not all of it. The `forward` column shows why. The pure launch time is
  17.3 ms: the series loop shows it, because it launches into an empty queue.
  In the loop with no work and in the overlap loop, `forward` takes 30.9 to
  35.7 ms, because the launches block: the queue of launched kernels is full.
  A full queue holds a limited amount of GPU work. During the 8 ms of CPU work,
  the GPU finishes the queue and then waits for about 3 ms. A CPU that runs
  ahead can hide work only up to the depth of that queue.
- **The series loop pays the CPU work plus a small tail**: at batch 1, the
  last kernels of the step (the lag); at both sizes, the copy of the tokens to
  the CPU.
- **The profiler costs 35% to 36%** of the step at batch 1 (Exercise 4). It adds work
  to every launch, so it makes the CPU side look slower than it is. Compare a
  trace with a trace, and a clock with a clock.

The cure for a CPU-bound decode step is stage 23: a CUDA graph replaces the
1,600 launches with one. Then the CPU side almost disappears, and the queue
problem with it.
'''),
]


# ============================================================ Practicum J

SETUP_J = '''
# find the repo root. The directory you start from does not matter.
import sys, time, shutil
from pathlib import Path
ROOT = next(folder for folder in [Path.cwd(), *Path.cwd().parents]
            if (folder/'cudalib').is_dir())
sys.path.insert(0, str(ROOT))
import torch
import cudalib
WORK = ROOT / '.cudacache' / 'practicum_j'
WORK.mkdir(parents=True, exist_ok=True)
'''

KERNELS_J = r'''
KERNELS = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

// A: one thread for each row of 128 bf16 values. Each thread walks its row.
__global__ void rows_by_thread(const __nv_bfloat16* keys, float* out, long rows) {
  long row = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.f;
  for (int d = 0; d < 128; ++d) sum += __bfloat162float(keys[row * 128 + d]);
  out[row] = sum;
}

// B: 16 threads for each row, 16 bytes each.
__global__ void rows_by_16_threads(const __nv_bfloat16* keys, float* out, long rows) {
  long lane = threadIdx.x % 16;
  for (long row = (blockIdx.x * (long)blockDim.x + threadIdx.x) / 16; row < rows;
       row += (long)gridDim.x * blockDim.x / 16) {
    uint4 chunk = reinterpret_cast<const uint4*>(keys + row * 128)[lane];
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&chunk);
    float sum = 0.f;
    for (int i = 0; i < 8; ++i) sum += __bfloat162float(v[i]);
    for (int offset = 8; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset, 16);
    if (lane == 0) out[row] = sum;
  }
}

// C: a tile of 32 rows in shared memory, and each lane of a warp reads its own row, one column.
template <int WIDTH>
__device__ void tile_columns(float* out, int repeats) {
  __shared__ float tile[32][WIDTH];
  int lane = threadIdx.x;
  for (int c = 0; c < 128; ++c) tile[lane][c] = lane * 0.5f + c;
  __syncwarp();
  float sum = 0.f;
  for (int r = 0; r < repeats; ++r)
    for (int c = 0; c < 128; ++c) sum += tile[lane][(c + r) & 127];
  out[blockIdx.x * 32 + lane] = sum;
}
__global__ void tile_rows_128(float* out, int repeats) { tile_columns<128>(out, repeats); }
__global__ void tile_rows_129(float* out, int repeats) { tile_columns<129>(out, repeats); }

void run_a(torch::Tensor keys, torch::Tensor out) {
  long rows = keys.size(0);
  rows_by_thread<<<(rows + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)keys.data_ptr(), out.data_ptr<float>(), rows);
}
void run_b(torch::Tensor keys, torch::Tensor out) {
  rows_by_16_threads<<<1 << 16, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)keys.data_ptr(), out.data_ptr<float>(), keys.size(0));
}
void run_c(torch::Tensor out, int64_t width, int64_t repeats) {
  int blocks = out.numel() / 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (width == 128) tile_rows_128<<<blocks, 32, 0, stream>>>(out.data_ptr<float>(), repeats);
  else tile_rows_129<<<blocks, 32, 0, stream>>>(out.data_ptr<float>(), repeats);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_a", &run_a); m.def("run_b", &run_b); m.def("run_c", &run_c);
}
"""
kernels = cudalib.build_source('practicum_j', KERNELS)
keys = torch.randn(1 << 20, 128, device='cuda', dtype=torch.bfloat16)     # 268 MB, far more than L2
out = torch.empty(1 << 20, device='cuda')
tile_out = torch.empty(4096 * 32, device='cuda')
PEAK = cudalib.peak_bandwidth()
print(f'the kernels are built. The streaming copy of this card: {PEAK:.0f} GB/s')
'''

EX1J_HELPER = '''
def gbs(fn, num_bytes):
    """The GB/s of `fn`, which reads `num_bytes`. Use cudalib.bench_ms."""
    ...

a = gbs(lambda: kernels.run_a(keys, out), keys.numel() * 2)
b = gbs(lambda: kernels.run_b(keys, out), keys.numel() * 2)
c128 = cudalib.bench_ms(lambda: kernels.run_c(tile_out, 128, 64))
c129 = cudalib.bench_ms(lambda: kernels.run_c(tile_out, 129, 64))
print(f'A {a:.0f} GB/s, B {b:.0f} GB/s ({b / a:.2f}x);  C with rows of 128: {c128:.2f} ms, of 129: {c129:.2f} ms ({c128 / c129:.1f}x)')
'''

EX1J_SOLUTION = '''
def gbs(fn, num_bytes):
    return num_bytes / (cudalib.bench_ms(fn) / 1000) / 1e9

a = gbs(lambda: kernels.run_a(keys, out), keys.numel() * 2)
b = gbs(lambda: kernels.run_b(keys, out), keys.numel() * 2)
c128 = cudalib.bench_ms(lambda: kernels.run_c(tile_out, 128, 64))
c129 = cudalib.bench_ms(lambda: kernels.run_c(tile_out, 129, 64))
print(f'A {a:.0f} GB/s, B {b:.0f} GB/s ({b / a:.2f}x);  C with rows of 128: {c128:.2f} ms, of 129: {c129:.2f} ms ({c128 / c129:.1f}x)')
'''

PROBE_J = '''
METRICS = {
    'dram__bytes.sum.per_second': 'bytes out of DRAM each second',
    'smsp__average_data_bytes_per_sector_mem_global_op_ld.pct': 'of each 32-byte sector fetched, the % used',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum': 'shared-memory bank conflicts on loads',
    'smsp__inst_executed_op_shared_ld.sum': 'shared-memory load instructions',
    'sm__warps_active.avg.pct_of_peak_sustained_active': 'achieved occupancy',
}
DRIVER = WORK / 'driver.py'
DRIVER.write_text(f"""
import sys, torch
sys.path.insert(0, {str(ROOT)!r})
import cudalib
KERNELS = {KERNELS!r}
kernels = cudalib.build_source('practicum_j', KERNELS)
keys = torch.randn(1 << 20, 128, device='cuda', dtype=torch.bfloat16)
out = torch.empty(1 << 20, device='cuda')
tile_out = torch.empty(4096 * 32, device='cuda')
kernels.run_a(keys, out); kernels.run_b(keys, out)
kernels.run_c(tile_out, 128, 64); kernels.run_c(tile_out, 129, 64)
torch.cuda.synchronize()
""")

def counters(kernel_regex):
    """-> {metric: float} for the first kernel that matches, or None."""
    values, output = cudalib.run_ncu(DRIVER, list(METRICS), kernel=f'regex:{kernel_regex}')
    if not values:
        return None, output
    return {name: float(str(value).replace(',', '')) for name, (value, _unit) in values.items()}, output

first, output = counters('rows_by_thread')
HAVE_COUNTERS = first is not None
if not HAVE_COUNTERS:
    print('Nsight Compute cannot read the counters here.')
    if cudalib.probe.NO_COUNTER_PERMISSION in output:
        print("The driver lets only an administrator read them. To change it:\\n"
              "    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf\\n"
              "    sudo update-initramfs -u     # then reboot\\n"
              "Until then, the checks of this notebook use the clock only.")
    else:
        print(output[-1500:])
else:
    print('Nsight Compute can read the counters.')
'''

EX2J_HELPER = '''
if HAVE_COUNTERS:
    results = {name: counters(regex)[0] for name, regex in
               [('A', 'rows_by_thread'), ('B', 'rows_by_16_threads'),
                ('C128', 'tile_rows_128'), ('C129', 'tile_rows_129')]}
    for name, values in results.items():
        print(name, {metric.split('__')[1][:40]: round(value, 2) for metric, value in values.items()})

def sector_use(values):
    """The % of each fetched sector that the kernel used."""
    ...

def conflicts_per_load(values):
    """The bank conflicts for each shared-memory load instruction."""
    ...

def dram_fraction(values):
    """The DRAM bytes/s of the kernel, against PEAK (GB/s)."""
    ...
'''

EX2J_SOLUTION = '''
if HAVE_COUNTERS:
    results = {name: counters(regex)[0] for name, regex in
               [('A', 'rows_by_thread'), ('B', 'rows_by_16_threads'),
                ('C128', 'tile_rows_128'), ('C129', 'tile_rows_129')]}
    for name, values in results.items():
        print(name, {metric.split('__')[1][:40]: round(value, 2) for metric, value in values.items()})

def sector_use(values):
    return values['smsp__average_data_bytes_per_sector_mem_global_op_ld.pct']

def conflicts_per_load(values):
    loads = values['smsp__inst_executed_op_shared_ld.sum']
    return values['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'] / max(loads, 1)

def dram_fraction(values):
    return values['dram__bytes.sum.per_second'] / (PEAK * 1e9)
'''

CHECK_J = '''
### THE CHECKS. Do not edit this cell.

assert all(value > 0 for value in (a, b, c128, c129)), 'run Exercise 1 first'
assert b > 1.25 * a, f'B must stream faster than A: {b:.0f} against {a:.0f} GB/s'
assert c128 > 4 * c129, f'rows of 128 floats must be much slower: only {c128 / c129:.1f}x'
if HAVE_COUNTERS:
    assert sector_use(results['A']) < 40, 'A uses a small part of each sector'
    assert sector_use(results['B']) > 90, 'B uses almost all of each sector'
    assert conflicts_per_load(results['C128']) > 20, 'rows of 128: about 31 conflicts per load'
    assert conflicts_per_load(results['C129']) < 1, 'rows of 129: no conflict'
    assert dram_fraction(results['B']) > dram_fraction(results['A'])
    print('All the checks pass, with the counters and with the clock.')
else:
    print('The checks with the clock pass. The counter checks did not run: '
          'this machine does not let this user read the counters.')
'''

PRACTICUM_J = [
    ('md', '''
# Read the counters

A clock tells you that a kernel is slow. The hardware counters tell you
**why**: the bytes that the memory delivered, the part of each fetched sector
that the kernel used, the bank conflicts in [shared memory](../../GLOSSARY.md#shared-memory), and the warps that
were resident.

This notebook compiles three small kernels that you already met in the
incidents of Parts 3 and 8:

- **A**: one thread for each row of 128 bf16 values (the uncoalesced read of
  stage 08);
- **B**: 16 threads for each row, 16 bytes each (the coalesced read of stage
  08b);
- **C**: a tile of 32 x 128 floats in shared memory, where the lanes of a warp
  read one column, and the same with rows of 129 floats.

First you measure them with a clock, which always works. Then you read their
counters with **Nsight Compute** (`ncu`), which needs a permission that many
Linux machines do not give a normal user. The notebook tells you if yours does
not, and how to change it. The checks at the end always check the clock, and
they check the counters when they can.
'''),
    ('code', KERNELS_J),
    ('md', '''
# Exercise 1: the clock

Measure A and B in GB/s, and the two versions of C in ms. Before you run it,
predict the ratio B / A and the ratio of C with rows of 128 to C with rows of
129.
'''),
    ('code', EX1J_HELPER, EX1J_SOLUTION),
    ('md', '''
# Exercise 2: can this machine read the counters?

Run this cell as it is. It writes a small script that runs the four kernels,
and runs Nsight Compute on it.
'''),
    ('code', PROBE_J),
    ('md', '''
# Exercise 3: the counters, and what they mean

Write three functions that turn the raw counters into the three numbers that
explain Exercise 1. Before you run it, predict each number for A, B and the two
versions of C.
'''),
    ('code', EX2J_HELPER, EX2J_SOLUTION),
    ('md', '''
### The checks

The clock checks always run. The counter checks run when Nsight Compute can
read the counters.
'''),
    ('code', CHECK_J),
    ('md', '''
# Exercise 4: your kernels (after stage 08b)

`./vc ncu 8 64` and `./vc ncu 8b 64` print the same counters for your paged
attention kernels. Answer:

1. What part of each sector does your stage 08 kernel use, and your stage 08b
   kernel? Does the ratio match the gain that stage 08b measured?
2. Is your stage 08b kernel near the bandwidth of the card? If not, which
   counter tells you why?

### Before you open the solution

1. A reads 2 bytes of each 32-byte sector, 6.25%. Why is A not 16 times slower
   than B?
2. Kernel C with rows of 128 floats has about 31 conflicts for each load. Why
   31, and not 32?
3. Why does the notebook check the clock even when the counters work?
'''),
    ('solution', '''
### What I measured

On an RTX 4080 Laptop GPU with the clock: A about 210 GB/s, B about 320 GB/s
(1.5x); C with rows of 128 about 8 ms, with rows of 129 about 0.5 ms (15x). This
machine does not let a normal user read the counters, so I could not run the
counter cells here. With the counters, expect a sector use near 6% for A (the L1
cache serves the rest of each sector to the next loads of the same thread, which
is why A is not 16x slower), near 100% for B, about 31 conflicts for each load
for C with rows of 128 (32 accesses to one bank: one of them is not a conflict),
and 0 with rows of 129.
'''),
]


if __name__ == '__main__':
    build([('code', SETUP_I)] + PRACTICUM_I,
          'course/Part8_TheCapstone/4_nsys/part8_nsys_1_CCreadTheTimeline_helper.ipynb',
          'course/Part8_TheCapstone/4_nsys/solutions/part8_nsys_1_CCreadTheTimeline.ipynb',
          'Timeline profiling', 'read the timeline')
    build([('code', SETUP_J)] + PRACTICUM_J,
          'course/Part8_TheCapstone/5_ncu/part8_ncu_1_CCreadTheCounters_helper.ipynb',
          'course/Part8_TheCapstone/5_ncu/solutions/part8_ncu_1_CCreadTheCounters.ipynb',
          'Kernel counters', 'read the counters')
    print('practicums I and J built')
