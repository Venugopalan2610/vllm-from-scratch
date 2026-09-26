# Part 6, The Server

Stages: stages 15-16.

Open the sections in the order of their numbers. In each section, open the
notebooks in the order of their numbers. The last notebook of a section is
its challenge (`CC..._helper`). Its solution is in `solutions/`.

| | |
|---|---|
| [`1_async/`](1_async/) | One loop, two clocks, and what happens when a client disconnects. Then build the engine loop. **No GPU.** |
| [`2_metrics/`](2_metrics/) | Six numbers, a Pareto frontier, and why throughput is not the answer. Then find which of three projects your p99 asks for. **No GPU.** |
| [`3_incidents/`](3_incidents/) | Seven tickets around the engine: the event loop, disconnects, proxies, and metrics that lie. Then break working code on purpose, and find three hidden faults from their behaviour. Do this section after stage 16. **No GPU.** |

## Systems Architecture: Resilient Serving & Zero-Copy IPC

### 1. The Python GIL & Multiprocessing Architecture
In Python, the Global Interpreter Lock (GIL) prevents concurrent CPU execution. If
FastAPI JSON deserialization, detokenization, and the engine step loop run in the
same process, detokenizing a large batch blocks the thread from launching the next
GPU step, stalling the GPU.
- **ProcessEngineWorker (Stage 27):** Isolates the engine into a separate OS process,
  protecting the GPU execution loop from web server overhead.

### 2. The Bottleneck of `mp.Queue` vs. Zero-Copy Shared Memory
Standard Python `multiprocessing.Queue` serializes every token event with `pickle`, and
sends it through a pipe. Measure before you optimize: on a laptop CPU, a pickle round trip
of a small token event took 1.1 µs, and an `mp.Queue` put and get took 2.4 µs. At 5,000
events each second that is about 1% of one core. The cost grows with the size of the event
(for example, log-probabilities for each token), and with the number of events.
- **SharedMemoryEventRing (Stage 27):** a POSIX shared-memory ring of fixed binary records
  (`struct.Struct("32sii64s")`). No pickle and no pipe. Exercise F of the course README
  asks you to measure it against `mp.Queue` on your machine.

### 3. Cache Line Alignment (`alignas(64)`) & False Sharing
In multi-threaded schedulers or CPU worker rings, multiple cores update adjacent
slots concurrently. If two variables share the same 64-byte L1 cache line, the CPU
cores invalidate each other's L1 cache line on every write (cache thrashing).
Systems code aligns slot structures to 64 bytes (`alignas(64)`) with explicit padding.

### 4. Client Disconnect Detection & KV Block Leak Prevention
When an end-user hits `Ctrl+C` or a mobile network drops mid-stream, naive servers
continue generating tokens until `max_tokens` is reached.
- **Disconnect Abort (Stage 27):** Streaming handlers poll `await request.is_disconnected()`.
  On disconnection, the server immediately triggers `ServingEngine.abort(rid)`,
  releasing all allocated KV blocks back to the pool instantly.

**When you finish this Part:** [Part 7, Modern vLLM](../Part7_ModernVLLM/).

**When it gets hard:** [four things to try](../README.md#when-it-gets-hard).
Do not stop. Continue. Be better than before.
