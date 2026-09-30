"""./vc run - your engine so far, on one fixed workload.

The live engine (tvllm/live.py) holds your part in each slot that you have
finished, and a given, naive part in the others. This script runs one
workload on it two times: with your newest part, and without it. The
difference is what your newest stage bought.

    ./vc run              with and without your newest part
    ./vc run --once       only with it (faster)
    ./vc run --blocks N   the size of the KV pool, in blocks of 16 tokens
    ./vc run --stages 06,07,09   use exactly these stages (to explore)
"""

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.style import paint  # noqa: E402

ORACLE_FILE = ROOT / ".live_oracle.json"
NEAR_TIE = 0.25          # two scores this close can swap from rounding alone

SYSTEM = ("You are a careful assistant for a team of software engineers. Answer in "
          "plain English, in short sentences. Give one example when it helps. Do not "
          "use jargon. If you do not know the answer, say so. ")
QUESTIONS = [
    "What is a hash map?", "Why is a cache useful?", "What does a scheduler do?",
    "Explain a mutex.", "What is a page table?", "Why do tests matter?", "What is latency?",
    "What is a queue?", "What is a deadlock?", "Why do we use version control?",
    "What is an index in a database?", "What is a race condition?",
    "Explain the difference between a process and a thread.", "What is a load balancer?",
    "Why is logging important?", "What is technical debt?", "What is a memory leak?",
    "What does an operating system do?", "What is recursion?", "What is a REST API?",
]
DOCUMENT = ("The team moved the billing service to a new cluster last week. The move "
            "took four days. On the second day the database connections ran out, "
            "because the new pool had a smaller limit. On the third day a cache "
            "returned stale prices for two hours. ") * 7
UNICODE = ["Write the word 'thank you' in Japanese, Chinese, Hindi and Korean:",
           "Give three emoji for good weather, and say what each one means:"]
MAX_TOKENS = [32, 48, 64, 96]


def workload(tokenizer):
    """-> [(rid, prompt ids, max_tokens)]: questions with one shared system
    prompt (for the prefix cache), two long documents (for chunked prefill),
    and two answers with characters of several bytes (for the detokenizer)."""
    jobs = [(f"q{i}", SYSTEM + question, MAX_TOKENS[i % 4]) for i, question in enumerate(QUESTIONS)]
    jobs += [("doc0", DOCUMENT + "Summarize the problems in one sentence.", 48),
             ("doc1", DOCUMENT + "What should the team check before the next move?", 64)]
    jobs += [(f"uni{i}", text, 48) for i, text in enumerate(UNICODE)]
    return [(rid, tokenizer(text).input_ids, max_tokens) for rid, text, max_tokens in jobs]


# ---------------------------------------------------------------- progress


def completed_stages():
    progress_file = ROOT / ".progress.json"
    if not progress_file.exists():
        return [], "torch"
    progress = json.loads(progress_file.read_text())
    return progress.get("tracks", {}).get("torch", progress.get("completed", [])), progress.get("backend", "torch")


def newest_live_stage(yours):
    from tvllm.live import LIVE_STAGES
    live = [stage for stage in LIVE_STAGES if stage in yours]
    return live[-1] if live else None


# ---------------------------------------------------------------- the oracle


def oracles(model, jobs):
    """The trusted tokens of each request, alone, with the given dense
    attention. Kept in .live_oracle.json, because they never change."""
    from tvllm.live import oracle_greedy
    key = hashlib.sha256(json.dumps([model.config.name, jobs]).encode()).hexdigest()[:16]
    cached = json.loads(ORACLE_FILE.read_text()) if ORACLE_FILE.exists() else {}
    if cached.get("key") == key:
        return cached["oracle"]
    print(paint("  computing the oracle: each request alone, once (it is kept)...", "dim"))
    result = {rid: oracle_greedy(model, ids, n) for rid, ids, n in jobs}
    ORACLE_FILE.write_text(json.dumps({"key": key, "oracle": result}))
    return result


def compare_with_oracle(outputs, oracle):
    """-> (identical, near ties, real differences [(rid, position, gap)])."""
    identical, ties, real = 0, 0, []
    for rid, tokens in outputs.items():
        expected, gaps = oracle[rid]
        first = next((i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b), None)
        if first is None:
            identical += 1
        elif gaps[first] <= NEAR_TIE:
            ties += 1
        else:
            real.append((rid, first, gaps[first]))
    return identical, ties, real


# ---------------------------------------------------------------- one run


def run_workload(model, parts, jobs, args):
    from tvllm.live import LiveEngine
    # A warm-up on a throwaway engine: the first calls load your kernels, cuBLAS
    # and the graphs. Then no step of the timed run has to be left out.
    warm_up = LiveEngine(model, parts, num_blocks=args.blocks, ignore_eos=True,
                         max_num_seqs=args.max_seqs, token_budget=args.token_budget)
    for rid, ids, _ in jobs[:3]:
        warm_up.add_request(rid, ids, 4)
    warm_up.run_to_completion()
    del warm_up
    engine = LiveEngine(model, parts, num_blocks=args.blocks, ignore_eos=True,
                        max_num_seqs=args.max_seqs, token_budget=args.token_budget)
    streamed = {rid: "" for rid, _, _ in jobs}
    for rid, ids, max_tokens in jobs:
        engine.add_request(rid, ids, max_tokens)
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    while engine.has_work():
        for rid, delta, _ in engine.step():
            streamed[rid] += delta
        if len(engine.stats["step_ms"]) > 20000:
            raise RuntimeError("more than 20000 steps: the engine does not finish")
    seconds = time.perf_counter() - start
    outputs = {rid: request.outputs for rid, request in engine.requests.items()}
    text_ok = sum(streamed[rid] == model.tokenizer.decode(outputs[rid]) for rid in outputs)
    return {"engine": engine, "outputs": outputs, "seconds": seconds, "text_ok": text_ok}


def summary(result):
    stats = result["engine"].stats
    warm = stats["step_ms"]
    busy = sum(warm) / 1e3                           # the steps only: not the graph capture
    return {
        "tokens/s": stats["tokens"] / busy,
        "seconds in steps": busy,
        "steps": stats["steps"],
        "most requests at once": stats["peak_running"],
        "peak KV blocks": stats["peak_blocks"],
        "prompt tokens computed": stats["prefill_computed"],
        "prompt tokens from cache": stats["prefill_cached"],
        "preemptions": stats["preemptions"],
        "tokens computed again": stats["recomputed"],
        "median step (ms)": statistics.median(warm) if warm else 0,
        "longest step (ms)": max(warm) if warm else 0,
        "graph replays": stats["graph_steps"],
        "streams = final text": f"{result['text_ok']}/{len(result['outputs'])}",
    }


# ---------------------------------------------------------------- printing


def show_parts(parts):
    print(paint("\n  YOUR ENGINE SO FAR", "bold"))
    for slot, whose, what in parts.table():
        color = "green" if whose == "yours" else "dim"
        print(f"  {slot:<10} " + paint(f"{whose:<6}", color) + f" {what}")


def fmt(value):
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def show_table(columns):
    """columns: [(title, summary dict)]."""
    keys = list(columns[0][1])
    width = max(len(k) for k in keys) + 2
    print("\n  " + " " * width + "".join(f"{title:>20}" for title, _ in columns))
    for key in keys:
        values = [fmt(column[key]) for _, column in columns]
        changed = len(set(values)) > 1
        line = f"  {key:<{width}}" + "".join(f"{v:>20}" for v in values)
        print(paint(line, "bold") if changed else line)


def show_oracle(result, oracle):
    identical, ties, real = compare_with_oracle(result["outputs"], oracle)
    total = len(result["outputs"])
    print(paint("\n  THE ORACLE", "bold") + paint("  (move 6: each request alone, with the given attention)", "dim"))
    print(f"  identical: {identical}/{total}.  "
          f"different from a near tie (a rounding flip, Part 0 assumption 6): {ties}.  "
          f"real differences: {len(real)}.")
    for rid, position, gap in real[:5]:
        print(paint(f"  {rid}: different from token {position}, where the oracle's best two "
                    f"scores differ by {gap:.2f}. That is not rounding. Look at your newest part.",
                    "red"))


HEADLINES = {
    "06-block-allocator": ("most requests at once", "Memory, not arithmetic, set how many requests ran together."),
    "07-paged-attention": ("tokens/s", "Attention now reads through your block tables, and the gather costs time. That is the bill that stage 08 pays."),
    "08-paged-attention-cuda": ("median step (ms)", "Your kernel, against the attention before it."),
    "08b-cuda-memory": ("median step (ms)", "Coalesced loads, against your stage 08 kernel."),
    "08c-cuda-warps": ("median step (ms)", "Split-K, against your stage 08b kernel."),
    "09-prefix-sharing": ("prompt tokens computed", "The shared system prompt is computed once, then read from your cache."),
    "10-admission-preemption": ("most requests at once", "Your scheduler admits by what a request needs now, and preempts when it must."),
    "11-chunked-prefill": ("longest step (ms)", "A long prompt no longer fills one step alone. Look at tokens/s too: that is the bill."),
    "12-cuda-graphs": ("median step (ms)", "Decode steps replay a graph, with no Python between the kernels."),
    "13-sampling": ("tokens/s", "Every row goes through your sampler, in one pass."),
    "14-detokenization": ("streams = final text", "What the client sees is now what the model wrote."),
}


# ---------------------------------------------------------------- sections


def section_server(model, parts, args):
    """Stage 15: your server, around the live engine."""
    from tvllm.live import LiveEngine
    server_module = __import__("app.s15_server", fromlist=["AsyncLLMEngine"])
    engine = LiveEngine(model, parts, num_blocks=args.blocks, max_num_seqs=args.max_seqs)
    base = engine.used_blocks()
    server = server_module.AsyncLLMEngine(engine)
    events = []

    async def consume(name, prompt, limit=None):
        count = 0
        async for _ in server.generate(prompt, 40, rid=name):
            events.append(name)
            count += 1
            if limit and count >= limit:
                break                          # the client leaves: your finally must abort
        return count

    async def main():
        await server.start()
        results = await asyncio.gather(consume("a", SYSTEM + QUESTIONS[0]),
                                       consume("b", SYSTEM + QUESTIONS[1]),
                                       consume("c", SYSTEM + QUESTIONS[2], limit=4))
        for _ in range(200):
            if not engine.has_work():
                break
            await asyncio.sleep(0.01)
        await server.stop()
        return results

    results = asyncio.run(main())
    switches = sum(1 for x, y in zip(events, events[1:]) if x != y)
    print(paint("\n  STAGE 15: YOUR SERVER, AROUND THE LIVE ENGINE", "bold"))
    print(f"  three streams, one engine: {results[0]} and {results[1]} deltas, and one client "
          f"that left after {results[2]}.")
    print(f"  the streams took turns {switches} times: they ran together, not one after the other.")
    left = engine.used_blocks() - base
    aborted = engine.requests["c"].aborted
    print(f"  the client that left was aborted: {aborted}.  blocks still in use after the "
          f"end: {left} (the prefix cache can keep some on purpose).")


def section_speculation(model, parts, args):
    from tvllm.live import speculative_greedy
    paragraph = ("A block table maps each logical position of a sequence to a physical block. "
                 "The engine reads the table on every step, so it must be small and exact. ")
    prompt = model.tokenizer("Copy this text exactly, twice.\n\n" + paragraph + "\n\n"
                             + paragraph[:60]).input_ids
    plain, plain_calls, plain_s, gaps = speculative_greedy(model, parts, prompt, 64, num_draft=0)
    fast, calls, seconds, _ = speculative_greedy(model, parts, prompt, 64, num_draft=4)
    print(paint("\n  STAGE 17: YOUR N-GRAM DRAFTS, ON A COPY TASK", "bold"))
    print(f"  plain decode:      {plain_calls} forward passes, {plain_s*1e3:7.0f} ms")
    print(f"  with your drafts:  {calls} forward passes, {seconds*1e3:7.0f} ms   "
          f"-> {len(fast)/calls:.2f} tokens for each pass, {plain_s/seconds:.2f}x")
    first = next((i for i, (a, b) in enumerate(zip(fast, plain)) if a != b), None)
    if first is None:
        print("  the same tokens as the plain decode (move 6): yes, all of them")
    elif gaps[first] <= NEAR_TIE:
        print(f"  the same tokens as the plain decode up to token {first}. There the plain "
              f"decode's two best scores differ by {gaps[first]:.3f}: a near tie. A verify pass "
              "has more rows, so it adds in a different order (Part 0, assumption 6).")
    else:
        print(paint(f"  different from the plain decode at token {first}, where its two best "
                    f"scores differ by {gaps[first]:.2f}. That is not rounding: check how you "
                    "accept a draft.", "red"))


def section_quantization(parts, args, reference_model):
    import torch
    from tvllm.live import LiveEngine, quantize_weights_
    from tvllm.model import load_model
    model = load_model()
    name = quantize_weights_(model, parts)

    def batch_one(some_model):
        engine = LiveEngine(some_model, parts, num_blocks=args.blocks)
        prompt = some_model.tokenizer(SYSTEM + QUESTIONS[4]).input_ids
        engine.add_request("x", prompt, 64)
        engine.ignore_eos = True
        engine.run_to_completion()
        steps = sorted(engine.stats["step_ms"][1:])
        return engine.requests["x"].outputs, steps[len(steps) // 2]

    reference_tokens, reference_ms = batch_one(reference_model)
    tokens, ms = batch_one(model)

    @torch.inference_mode()
    def logits_at(some_model, ids):
        from tvllm.model import DenseReference
        n = len(ids)
        return some_model.forward(torch.tensor(ids, device="cuda"), torch.arange(n, device="cuda"),
                                  DenseReference(some_model.config.num_layers),
                                  torch.arange(n, device="cuda"))
    prompt = reference_model.tokenizer(SYSTEM + QUESTIONS[4]).input_ids
    ids = prompt + reference_tokens
    rows = slice(len(prompt) - 1, len(ids) - 1)          # the positions that chose each token
    reference_logits, logits = logits_at(reference_model, ids)[rows], logits_at(model, ids)[rows]
    agreement = (reference_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
    kl = torch.nn.functional.kl_div(logits.log_softmax(-1), reference_logits.log_softmax(-1),
                                    log_target=True, reduction="batchmean").item()
    print(paint(f"\n  STAGE 18: {name.upper()}, IN EVERY MATMUL", "bold"))
    print(f"  weight bytes of one decode step: {reference_model.weight_bytes()/1e9:.2f} GB bf16 -> "
          f"{model.weight_bytes()/1e9:.2f} GB")
    print(f"  median batch-1 step: {reference_ms:.2f} ms bf16 -> {ms:.2f} ms  ({reference_ms/ms:.2f}x)")
    print(f"  at the {len(reference_tokens)} tokens that bf16 chose (move 6): the same top token "
          f"{agreement:.0%} of the time, KL from bf16 {kl:.4f} nats (the stage 18 gate is 0.015)")
    del model
    torch.cuda.empty_cache()


def section_guided(model, parts, args):
    from tests.helpers import json_prefix_state
    from tvllm.live import LiveEngine
    guided_module = __import__("app.s19_guided", fromlist=["GuidedDecoder"])
    tokenizer = model.tokenizer
    # The candidates: every token of some real JSON text, so that the model
    # can write JSON with the pieces that it prefers ('{\n', '": ', ...).
    people = [{"name": "Alice", "age": 30, "city": "Paris"},
              {"name": "Bob Smith", "age": 41, "city": "London", "active": True, "tags": []}]
    samples = [space + json.dumps(p, indent=indent) for p in people for indent in (None, 2, 4)
               for space in ("", " ")]
    samples += [str(n) for n in range(100)]
    candidates = sorted({t for text in samples for t in tokenizer.encode(text, add_special_tokens=False)})
    eos = min(model.eos_ids)
    prompt = "Return a JSON object about a person, with a name, an age and a city.\nJSON: "

    def generate(guided):
        engine = LiveEngine(model, parts, num_blocks=args.blocks)
        engine.add_request("j", prompt, 40, guided=guided)
        engine.run_to_completion()
        return tokenizer.decode(engine.requests["j"].outputs, skip_special_tokens=True).strip()

    decoder = guided_module.GuidedDecoder(tokenizer, json_prefix_state, candidates + [eos], eos_id=eos,
                                          vocab_size=model.config.vocab_size, device=str(model.device))
    free, forced = generate(None), generate(decoder)

    def parses(text):
        try:
            json.loads(text)
            return True
        except ValueError:
            return False
    print(paint("\n  STAGE 19: YOUR GUIDED DECODER, ON ONE JSON REQUEST", "bold"))
    print(f"  with no mask:    {free[:70]!r}   parses: {parses(free)}")
    print(f"  with your mask:  {forced[:70]!r}   parses: {parses(forced)}")
    print(paint(f"  (the mask allows {len(candidates)} tokens: the pieces of some real JSON text)", "dim"))
    cache = decoder.cache
    print(f"  your mask cache: {cache.hits} hits, {cache.misses} misses")


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(prog="./vc run")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--blocks", type=int, default=192)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--stages", default=None)
    args = parser.parse_args()

    from tvllm.live import LIVE_STAGES, Parts
    completed, backend = completed_stages()
    if args.stages:
        wanted = {s.strip().zfill(2) for s in args.stages.split(",")}
        completed = [stage for stage in LIVE_STAGES if stage.split("-")[0] in wanted]
    if backend != "torch" and not args.stages:
        print("\n  The live engine runs on the torch track. `./vc backend torch`, then `./vc run`.\n")
        return 1
    yours = [stage for stage in completed if stage in LIVE_STAGES]
    newest = newest_live_stage(yours)
    parts = Parts(yours)
    show_parts(parts)
    if newest is None:
        print(paint("\n  You have no part in the engine yet. Stage 06 puts the first one in. "
                    "This run shows the engine that your parts will replace.", "dim"))

    import torch
    from tvllm.model import load_model
    model = load_model()
    jobs = workload(model.tokenizer)
    oracle = oracles(model, jobs)
    print(paint(f"\n  the workload: {len(jobs)} requests, {sum(n for _, _, n in jobs):,} tokens to "
                f"generate, a KV pool of {args.blocks} blocks of 16 tokens", "dim"))

    compare = newest in HEADLINES and not args.once
    with_part = run_workload(model, parts, jobs, args)
    columns = [("your engine", summary(with_part))]
    if compare:
        without = run_workload(model, Parts([s for s in yours if s != newest]), jobs, args)
        columns.insert(0, (f"without {newest.split('-')[0]}", summary(without)))
    show_table(columns)
    if compare:
        key, sentence = HEADLINES[newest]
        print(paint(f"\n  stage {newest.split('-')[0]}: {sentence}", "cyan"))
        for shown in dict.fromkeys([key, "tokens/s"]):     # the number, then its bill or its gain
            print(paint(f"  {shown}: {fmt(columns[0][1][shown])} -> {fmt(columns[1][1][shown])}",
                        "cyan", "bold" if shown == key else "cyan"))
    show_oracle(with_part, oracle)

    metrics = with_part["engine"].metrics
    if metrics is not None:
        snap = metrics.snapshot()
        print(paint("\n  STAGE 16: YOUR METRICS, ON THIS RUN", "bold"))
        print("  " + "   ".join(f"{k} {snap[k]*1e3:.0f} ms" for k in
                                ("ttft_p50", "ttft_p99", "itl_p50", "itl_p99") if k in snap)
              + f"   kv_utilization_max {snap.get('kv_utilization_max', 0):.0%}")
    for note in with_part["engine"].notes:
        print(paint(f"\n  note: {note}", "yellow"))
    if with_part["engine"].graphs_off:
        print(paint(f"\n  graphs: off. {with_part['engine'].graphs_off}", "yellow"))

    sections = [("15-async-engine", lambda: section_server(model, parts, args)),
                ("17-speculative-decoding", lambda: section_speculation(model, parts, args)),
                ("19-guided-decoding", lambda: section_guided(model, parts, args)),
                ("18-quantization", lambda: section_quantization(parts, args, model))]
    for stage, run in sections:
        if stage in yours and parts.attention():
            try:
                run()
            except NotImplementedError as error:
                print(paint(f"\n  {stage}: a part is not implemented yet ({error})", "yellow"))
    if "20-tensor-parallel" in completed:
        print(paint("\n  stage 20 does not run here: tensor parallelism needs two GPUs to be real.", "dim"))
    print()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
