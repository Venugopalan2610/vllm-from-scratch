"""Part 6, break it on purpose: the server and its metrics."""

PART = 6
NAME = 'The Server'
FOLDER = 'course/Part6_TheServer/3_incidents'
IMPORTS = '''import asyncio, random, statistics
from transformers import AutoTokenizer'''

INTRO = '''
In the incident file you went from a symptom to a cause. Here you go the other
way. You put one fault into a working event loop, stream, proxy or metric, and
you watch what it does.

The routine for each exercise is the same:

1. Read the fault.
2. **Write your prediction in the cell.** Answer the four questions.
3. Run the cell.
4. Write down where your prediction was wrong. This line is the one that
   teaches you.

The four questions:

- **Crash?** Does it raise an error, or does it run?
- **When?** Which stream, which moment, which load?
- **What?** What does the wrong result look like: a stall, a leak, a number
  that lies?
- **Which guard?** Which check would catch it?

`lab.FakeEngine` runs on the event loop like the engine of stage 15, with no
GPU: each step sleeps for the step time, then gives one token to each running
request. `lab.client` consumes a stream and records when each token arrives.
The notebook runs real `asyncio` code, so the stalls are real stalls.

This notebook needs no GPU.
'''

LOAD = '''
### run this cell

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
DOCUMENT = 'The committee met on Tuesday to review the budget, the hiring plan and the new office lease. ' * 5000
print('the document has', len(tokenizer.encode(DOCUMENT)), 'tokens')
'''

DRILLS = [
    ('d1', '''
# Exercise 1: tokenize in the event loop

Eight streams run on the fake engine, with a step of 22 ms. After one second,
a request with a long document arrives. The handler tokenizes the document
with a plain call, in the event loop. Then run the same with the call in a
worker thread.

This is Ticket 1 of the incident file. Predict the worst gap between two
tokens of each stream, in each case.
''', '''
async def run(offload):
  engine = lab.FakeEngine(slots=16, step_ms=22)
  engine.start()
  record = {}
  streams = [asyncio.create_task(lab.client(engine, i, 120, record)) for i in range(8)]
  await asyncio.sleep(1.0)
  start = time.perf_counter()
  if offload:
    ids = await asyncio.get_running_loop().run_in_executor(None, tokenizer.encode, DOCUMENT)
  else:
    ids = tokenizer.encode(DOCUMENT)                                    # THE FAULT: blocks the loop
  took = time.perf_counter() - start
  await asyncio.gather(*streams)
  await engine.stop()
  worst = [max(lab.gaps(times)) * 1000 for times in record.values()]
  print(f'{"worker thread" if offload else "event loop   "}: tokenize {took * 1000:4.0f} ms, '
        f'the worst gap of each stream {min(worst):4.0f} to {max(worst):4.0f} ms')

await run(offload=False)
await run(offload=True)
'''),

    ('d2', '''
# Exercise 2: the user presses stop

100 requests on an engine with 32 slots. 70% of the answers end after 40
tokens. 30% ramble to 400 tokens, and their users press stop after 5 tokens.
The stream handler closes, and nobody tells the engine. Then run the same
with `engine.abort` when the stream closes.

This is Ticket 2. Predict the ratio of the delivered tokens to the generated
tokens, and how long the engine stays busy.
''', '''
async def run(abort):
  engine = lab.FakeEngine(slots=32, step_ms=2)
  engine.start()
  rng = random.Random(0)
  record, clients = {}, []
  start = time.perf_counter()
  for rid in range(100):
    rambles = rng.random() < 0.3
    clients.append(lab.client(engine, rid, 400 if rambles else 40, record,
                              stop_after=5 if rambles else None,
                              on_close=engine.abort if abort else None))   # THE FAULT when None
  await asyncio.gather(*clients)
  users_done = time.perf_counter() - start
  while engine.running or engine.waiting:
    await asyncio.sleep(0.01)
  engine_done = time.perf_counter() - start
  await engine.stop()
  print(f'abort {str(abort):5s}: delivered {engine.delivered} of {engine.generated} generated '
        f'({engine.delivered / engine.generated:.0%}); the users were done after {users_done:.2f} s, '
        f'the engine after {engine_done:.2f} s')

await run(abort=False)
await run(abort=True)
'''),

    ('d3', '''
# Exercise 3: start the clock at admission

40 requests arrive at once on an engine with 8 slots and a step of 10 ms.
Each answer has 50 tokens. Measure the time to the first token from the
arrival, as the user sees it, and from the admission, as the code of Ticket 3
does.

Predict both medians.
''', '''
engine = lab.FakeEngine(slots=8, step_ms=10)
engine.start()
record, arrival = {}, time.perf_counter()
await asyncio.gather(*[lab.client(engine, rid, 50, record) for rid in range(40)])
await engine.stop()
from_arrival = sorted(times[0] for times in record.values())
from_admission = sorted(arrival + record[rid][0] - engine.admitted_at[rid] for rid in record)   # THE FAULT
print(f'TTFT from the arrival  : median {statistics.median(from_arrival) * 1000:5.0f} ms, max {from_arrival[-1] * 1000:5.0f} ms')
print(f'TTFT from the admission: median {statistics.median(from_admission) * 1000:5.0f} ms, max {from_admission[-1] * 1000:5.0f} ms')
'''),

    ('d4', '''
# Exercise 4: the average of the p99s

Ten pods. Nine are healthy. One has a GPU that throttles, and its requests
take ten times longer. Each pod serves 10% of the requests. The dashboard
shows the average of the ten p99 values.

This is Ticket 4. Predict the dashboard p99 and the true p99 of all the
requests.
''', '''
rng = random.Random(0)
pods = []
for pod in range(10):
  scale = 10 if pod == 9 else 1
  pods.append([scale * rng.lognormvariate(math.log(0.12), 0.25) for _ in range(10_000)])

def p99(values):
  return sorted(values)[int(0.99 * (len(values) - 1))]

dashboard = sum(p99(latencies) for latencies in pods) / len(pods)          # THE FAULT
everything = [x for latencies in pods for x in latencies]
print('p99 of each pod:', [round(p99(latencies) * 1000) for latencies in pods], 'ms')
print(f'the dashboard: {dashboard * 1000:.0f} ms    the true p99: {p99(everything) * 1000:.0f} ms')
print(f'the p90 of the bad pod: {sorted(pods[9])[int(0.9 * 9999)] * 1000:.0f} ms')
'''),

    ('d5', '''
# Exercise 5: a proxy that buffers

The engine sends an answer of 400 tokens as Server-Sent Events, one event of
about 60 bytes every 25 ms. A proxy between the server and the user collects
the bytes, and sends them on only when its buffer is full or the response
ends. Simulate buffers of 0 (no buffering), 4 KB and 32 KB.

This is Ticket 5. Predict the time to the first token and the time of the
whole answer that the user sees, for each buffer.
''', '''
events = [(i * 0.025 + 0.15, len(f'data: {{"id": "chunk-{i}", "text": " word{i}"}}\\n\\n')) for i in range(400)]
print(f'{len(events)} events, {sum(size for _, size in events):,} bytes')

def through_proxy(events, buffer_bytes):
  delivered, held = [], 0
  for t, size in events:
    held += size
    if held >= buffer_bytes:                                            # THE FAULT when the buffer is large
      delivered.append(t)
      held = 0
  if held:
    delivered.append(events[-1][0])
  return delivered

for buffer in (0, 4096, 32768):
  out = through_proxy(events, buffer)
  print(f'buffer {buffer:6,} B: first bytes at {out[0]:5.2f} s, the last at {out[-1]:5.2f} s, '
        f'{len(out)} deliveries')
'''),

    ('d6', '''
# Exercise 6: a load test with no think time

A simulated engine with 64 slots and a step of 13 ms + 0.25 ms for each
running request. Each answer has 200 tokens. First, a load test: 50 users
that send the next request as soon as the answer ends. Then production: the
same 50 users, who read for 20 s on average between two requests.

This is Ticket 7. Predict the tokens/s of each, and the mean number of
running requests.
''', '''
for name, think in [('load test, no think time', 0.0), ('production, 20 s to read', 20.0)]:
  rate, running, duration = lab.simulate_users(users=50, think_s=think)
  print(f'{name:26s}: {rate:6.0f} tok/s, {running:4.1f} requests running on average, '
        f'{duration:4.1f} s for each request')
print('Little: running = users x duration / (duration + think)')
'''),
]

MYSTERY = '''
# Exercise 7: three mystery metrics

The module `mystery.py` holds three metric functions. Each takes a log: a
list of dicts with `arrival`, `first` (the first token), `finish`, `tokens`
(the output tokens) and `done` (False for a request that timed out). Each one
has one fault. **Do not open the file.**

- `tpot(log)`: the mean time for each output token after the first.
- `ttft_p99(log)`: the p99 of the time to the first token.
- `throughput(log)`: output tokens per second of wall time.

`lab` has no reference for these. You know the definitions, so you can build
a log where you know every answer.

The cell below runs them on a simple log: one request at a time, no queue,
nothing times out. All three look plausible.

For each function:

1. Build a small log where you can compute the true value by hand, and that
   reaches the fault.
2. Write your diagnosis: the fault, and the log that proved it.
3. Only then, open `mystery.py` and check.

A hint about the method: one fault needs a request that waited before its
first token. One fault needs requests that did not finish. One fault needs
requests that overlap in time.
'''

MYSTERY_CODE = '''
from mystery import tpot, ttft_p99, throughput

simple = []
for i in range(200):
  start = i * 2.55                                                      # back to back
  simple.append(dict(arrival=start, first=start + 0.05, finish=start + 2.55, tokens=101, done=True))
print(f'tpot {tpot(simple) * 1000:.1f} ms,  ttft_p99 {ttft_p99(simple) * 1000:.0f} ms,  '
      f'throughput {throughput(simple):.1f} tok/s')
'''

MYSTERY_NAMES = ['tpot', 'ttft_p99', 'throughput']

FINGERPRINT_ROWS = ['tokenize in the event loop', 'no abort when the user stops',
                    'the TTFT clock starts at admission', 'the average of the p99s',
                    'a proxy that buffers', 'a load test with no think time']

SOLUTIONS = {
    'd1': '''
### What happens

| where the tokenizer runs | tokenize | the worst gap of each stream |
|---|---|---|
| in the event loop | 177 ms | 188 ms, in all 8 streams |
| in a worker thread | 238 ms | 23 ms, one normal step |

- **Crash?** No.
- **When?** At the moment the document arrives, in **every** stream at once.
- **What?** The engine loop is a task on the same event loop, so it cannot
  start the next step either. Every stream freezes for the time of the
  tokenization. The engine itself did nothing wrong.

In a thread, the tokenization takes a little longer, because it shares the
CPU, and nobody waits for it. It works because the Rust tokenizer releases
the GIL. A pure-Python function in a thread would still block the loop, and it
needs another process.

**The fingerprint.** All streams stall together, for the time of one CPU call.
**The guard.** A monitor of the event loop lag.
''',
    'd2': '''
### What happens

| abort on close | delivered / generated | the users are done | the engine is done |
|---|---|---|---|
| no | 3,370 / 10,480 = **32%** | 0.48 s | 1.30 s |
| yes | 3,370 / 3,370 = 100% | 0.30 s | 0.30 s |

Without the abort, 68% of the work goes to users who left. The engine stays
busy 2.7 times longer after the users finished, and even the users who stayed
are slower (0.48 s against 0.30 s), because the abandoned requests hold their
slots.

**The fingerprint.** Delivered / generated far below 1, and an engine that is
busy with nobody connected. **The guard.** That ratio on the dashboard, and
a test that disconnects and checks the slots.
''',
    'd3': '''
### What happens

- From the arrival: median **1,067 ms**, max 2,115 ms.
- From the admission: median **10 ms**, max 10 ms.

With 40 requests and 8 slots, the requests wait in 5 rounds of 50 steps. The
clock that starts at admission sees only the first step of each request, and
says 10 ms for everyone. The users see the queue.

**The fingerprint.** A TTFT that equals one step, whatever the load.
''',
    'd4': '''
### What happens

- The p99 of the healthy pods: 212 to 216 ms. The bad pod: 2,125 ms.
- The dashboard, the average of the ten: **405 ms**.
- The true p99 of all the requests: **1,651 ms**, 4x the dashboard.
- The p90 of the bad pod: **1,651 ms**, exactly the true p99.

The slowest 1% of the fleet is the slowest 10% of the bad pod, because that
pod has 10% of the requests. That is why the fleet p99 equals its p90.

**The guard.** Merge the histograms of the pods before you take a percentile.
''',
    'd5': '''
### What happens

The answer is 400 events and 18,580 bytes.

| buffer | first bytes at | last bytes at | deliveries |
|---|---|---|---|
| none | 0.15 s | 10.13 s | 400 |
| 4 KB | 2.42 s | 10.13 s | 5 |
| 32 KB | **10.13 s** | 10.13 s | 1 |

A buffer larger than the whole answer turns a stream into one response at the
end: TTFT = the total time. A smaller buffer gives a stream that arrives in
bursts of about 2 seconds.

**The fingerprint.** TTFT equals the total time. **The guard.** A probe
through the public path that compares the two.
''',
    'd6': '''
### What happens

| | tokens/s | running on average | time of one request |
|---|---|---|---|
| load test, no think time | 1,961 | 50.0 | 5.1 s |
| production, 20 s to read | 445 | 6.6 | 3.0 s |

Little's law predicts the production row: 50 x 3.0 / (3.0 + 20) = 6.5
requests running. The engine is not slower in production. Each request is
**faster** (3.0 s against 5.1 s), because it shares the engine with fewer
others. The throughput is lower because the users offer less work.

The ratio here is 23%, not the third of Ticket 7: the ratio depends on the
think time and on the time of a request, and the time of a request itself
depends on the load. A closed-loop load test measures what 50 impatient robots
can do, not what 50 people will ask for.
''',
}

MYSTERY_SOL = '''
### The three mysteries

**`tpot`: it divides the whole time of the request by all its tokens.** The
correct TPOT is `(finish - first) / (tokens - 1)`. On the simple log, the
difference is 1%: 25.2 ms against 25.0 ms, easy to miss. The log that proves
it: one request that waited 5 s in a queue, `arrival=0, first=5.0,
finish=7.5, tokens=101`. The true TPOT is 25 ms. The function says **74 ms**.
The queue leaks into a metric that must not contain it, and every dashboard
that uses this TPOT blames the decode for a queue.

**`ttft_p99`: it ignores the requests that did not finish.** The log that
proves it: 95 requests with a TTFT of 50 ms and 5 requests that timed out
with no first token. The function says 50 ms. The true p99 is at least the
timeout. The worst requests are exactly the ones that a filter on `done`
removes. This is survivorship bias.

**`throughput`: it divides by the sum of the request times, not by the wall
time.** With one request at a time and no idle time, the two are equal, so the
simple log hides it. The log that proves it: 10 requests that run at the same
time, each 2.55 s with 101 tokens. The true throughput is 1,010 / 2.55 = 396
tok/s. The function says 39.6, ten times less: it measures the speed of one
request, not of the server.

### The method

| Mystery | The log to build |
|---|---|
| tpot | a request that waited before its first token |
| ttft_p99 | requests that never finished |
| throughput | requests that overlap in time |

A metric function is code, and it needs tests with known answers like any
other code. The simple log was the one case where all three faults are
invisible.
'''

FINGERPRINTS_SOL = '''
# The fingerprint table

| Fault | Crash? | When it shows | What it looks like | The guard |
|---|---|---|---|---|
| tokenize in the event loop | no | a long prompt arrives | all streams stall together | the event loop lag |
| no abort when the user stops | no | users press stop | 32% delivered; a busy engine with nobody | delivered / generated |
| the TTFT clock starts at admission | no | under a queue | a TTFT of one step, always | a probe from the outside |
| the average of the p99s | no | one bad pod | 405 ms instead of 1,651 ms | merge the histograms |
| a proxy that buffers | no | behind the proxy only | TTFT = total time | a probe through the public path |
| a load test with no think time | no | always | 4x the real load | report the offered load |

Four of these six faults are in the measurement, not in the engine. Before you
fix the engine, check the number.
'''
