"""Add the four lines of docs/METHOD.md to each stage that has a natural
subject for them: a stub in app/, a check in tests/, and a reference solution.

    python dev/build_four_lines.py            stubs and checks (skips a stage that has them)
    python dev/build_four_lines.py --solutions DIR   also append the solutions to DIR/*.py

The four lines are the same in every stage: predict the floor, measure
honestly, divide, double. Only the subject changes, and so the floor: which
bytes (or which count) this stage's work needs at the least. The hints get
shorter from stage to stage, as docs/METHOD.md plans.

A stage with no natural subject has none: 14, 15, 16, 19, 20, 22, 25 to 31.
Detokenization, HTTP and metrics have no hardware floor. Stage 20 runs on CPU
ranks. Stage 28 is itself the four lines of the whole engine.
"""
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARK = "# ---------------------------------------------------------------- the four lines"

STAGES = [
    dict(stage="04", module="s04_static_batch", test="stage_04_static_batching/test_stage.py",
         kind="time", subject="batch_step", n=8, fixtures="hf, device",
         what="one decode step of a batch of n rows",
         facts=["facts.weight_bytes   the bytes of every weight of the model"],
         hint="all the weights, one time, for all the rows together:\n"
              "                                    facts.weight_bytes / facts.bandwidth_bytes_per_s, in ms",
         floor="facts.weight_bytes",
         meaning="above 1 is the Python and the launches around the weight read. A doubling near 1: the rows are almost free",
         body='''
    model, _ = hf
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    facts = cudalib.card_facts(weight_bytes=weight_bytes)
    floor_ms = weight_bytes / facts.bandwidth_bytes_per_s * 1e3

    def step_at(num_rows):
        token_ids = torch.randint(0, 1000, (num_rows, 1), device=device)
        return torch.inference_mode()(lambda: model(token_ids, use_cache=False))
'''),
    dict(stage="05", module="s05_continuous", test="stage_05_continuous_batching/test_stage.py",
         kind="time", subject="engine_step", n=8, fixtures="hf, device",
         what="one decode step of n running requests, each with a KV cache",
         facts=["facts.weight_bytes         the bytes of every weight of the model",
                "facts.kv_bytes_per_token   the KV cache bytes of one token",
                "facts.context_len          the tokens in the cache of each request"],
         hint="the weights one time, plus the KV cache of every row:\n"
              "                                    n x facts.context_len x facts.kv_bytes_per_token",
         floor="facts.weight_bytes + n * facts.context_len * facts.kv_bytes_per_token",
         meaning="the weights are read one time for all the rows, and the KV cache grows with them, so the doubling is above 1",
         body='''
    model, _ = hf
    config = model.config
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_bytes_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * 2
    context_len = 512
    facts = cudalib.card_facts(weight_bytes=weight_bytes, kv_bytes_per_token=kv_bytes_per_token,
                               context_len=context_len)
    floor_ms = (weight_bytes + 8 * context_len * kv_bytes_per_token) / facts.bandwidth_bytes_per_s * 1e3

    @torch.inference_mode()
    def step_at(num_rows):
        prompt_ids = torch.randint(0, 1000, (num_rows, context_len), device=device)
        cache = model(prompt_ids, use_cache=True).past_key_values
        next_ids = torch.randint(0, 1000, (num_rows, 1), device=device)
        return torch.inference_mode()(lambda: model(next_ids, past_key_values=cache, use_cache=True))
'''),
    dict(stage="06", module="s06_blocks", test="stage_06_block_allocator/test_stage.py",
         kind="count", subject="blocks", n=100, fixtures="",
         what="the blocks that one BlockTable holds after n tokens",
         facts=["facts.block_size   the tokens in one block"],
         hint="the blocks that n tokens need, at the least:\n"
              "                                    n / facts.block_size, rounded up",
         floor="math.ceil(n / facts.block_size)",
         meaning="1.0: a new block comes only when the last one is full",
         body='''
    from app.s06_blocks import BlockAllocator, BlockTable

    facts = Facts(block_size=16)

    def count_at(num_tokens):
        table = BlockTable(BlockAllocator(1000, facts.block_size))
        for _ in range(num_tokens):
            table.append_token()
        return len(table.blocks)

    predicted = math.ceil(100 / facts.block_size)
'''),
    dict(stage="07", module="s07_paged_attn", test="stage_07_paged_attention/test_stage.py",
         kind="time", subject="attention_call", n=8, fixtures="device",
         what="one call of your paged_attention, for n sequences",
         facts=["facts.context_len    the tokens in the context of each sequence",
                "facts.num_kv_heads   the KV heads",
                "facts.head_dim       the numbers in one head",
                "facts.value_bytes    the bytes of one number (fp16: 2)"],
         hint="the K and V that the call must read, for n sequences.\n"
              "                                    Use facts.context_len, facts.num_kv_heads, facts.head_dim\n"
              "                                    and facts.value_bytes.",
         floor="n * facts.context_len * 2 * facts.num_kv_heads * facts.head_dim * facts.value_bytes",
         meaning="how far the gather is from the bytes that it must read",
         function="paged_attention", import_line="from app.s07_paged_attn import paged_attention",
         body=None),
    dict(stage="08", module="s08_paged_cuda", test="stage_08_paged_attention_cuda/test_stage.py",
         kind="time", subject="attention_call", n=8, fixtures="nvcc, device",
         what="one call of your CUDA paged attention, for n sequences",
         facts=None, hint="the same bytes as in stage 07",
         floor=None, meaning="against stage 07: the same bytes, a new kernel",
         function="paged_attention_cuda", import_line="from app.s08_paged_cuda import paged_attention_cuda",
         body=None),
    dict(stage="08b", module="s08b_cuda_memory", test="stage_08b_cuda_memory/test_cuda.py",
         kind="time", subject="attention_call", n=8, fixtures="nvcc, device",
         what="one call of your coalesced paged attention, for n sequences",
         facts=None, hint="the same bytes as in stage 07",
         floor=None, meaning="against stage 08: the same bytes, better loads",
         function="paged_attention_vec", import_line="from app.s08b_cuda_memory import paged_attention_vec",
         body=None),
    dict(stage="08c", module="s08c_cuda_warps", test="stage_08c_cuda_warps/test_cuda.py",
         kind="time", subject="attention_call", n=2, fixtures="nvcc, device", context_len=16384,
         what="one call of your split-K paged attention, for n sequences",
         facts=None, hint="the same bytes as in stage 07",
         floor=None, meaning="at few sequences: near 1 means that your split gives the whole card work",
         function="paged_attention_split", import_line="from app.s08c_cuda_warps import paged_attention_split",
         body=None),
    dict(stage="09", module="s09_prefix", test="stage_09_prefix_sharing/test_stage.py",
         kind="count", subject="prompt_tokens_computed", n=10, fixtures="",
         what="the prompt tokens that n requests compute, when they share one system prompt",
         facts=["facts.system_len     the tokens of the system prompt that every request starts with",
                "facts.question_len   the tokens of the question of each request, all different",
                "facts.block_size     the tokens in one block"],
         hint="the prompt tokens if the shared system prompt were computed only one time.\n"
              "                                    Use facts.system_len and facts.question_len.",
         floor="facts.system_len + n * facts.question_len",
         meaning="above 1: the end of the system prompt shares a block with each question, so it is computed again",
         body='''
    import math

    from app.s09_prefix import PrefixCache, RefCountedAllocator, block_hashes

    facts = Facts(system_len=100, question_len=20, block_size=16)
    system_ids = list(range(1000, 1000 + facts.system_len))

    def count_at(num_requests):
        allocator = RefCountedAllocator(4096, facts.block_size)
        cache = PrefixCache(allocator)
        computed = 0
        for index in range(num_requests):
            first_id = 5000 + index * facts.question_len
            token_ids = system_ids + list(range(first_id, first_id + facts.question_len))
            hashes = block_hashes(token_ids, facts.block_size)
            hits = cache.lookup(hashes)
            computed += len(token_ids) - len(hits) * facts.block_size
            blocks = hits + allocator.allocate(math.ceil(len(token_ids) / facts.block_size) - len(hits))
            for position, block_hash in enumerate(hashes):
                if position >= len(hits):
                    cache.insert(block_hash, blocks[position])
        return computed

    predicted = facts.system_len + 10 * facts.question_len
'''),
    dict(stage="10", module="s10_scheduler", test="stage_10_admission_preemption/test_stage.py",
         kind="count", subject="steps", n=8, fixtures="",
         what="the steps that your scheduler needs to finish n requests, with a large pool",
         facts=["facts.max_tokens   the output tokens of each request"],
         hint="the steps if every request runs in every step: one prefill step, then\n"
              "                                    one step for each output token. Use facts.max_tokens.",
         floor="facts.max_tokens + 1",
         meaning="near 1: the requests share each step",
         body='''
    facts = Facts(max_tokens=20)

    def count_at(num_requests):
        scheduler = Scheduler(BlockAllocator(100_000, 16), max_num_seqs=64)
        for rid in range(num_requests):
            scheduler.add_request(rid, 32, facts.max_tokens)
        scheduler.run_to_completion()
        return scheduler.steps

    predicted = facts.max_tokens + 1
'''),
    dict(stage="11", module="s11_chunked", test="stage_11_chunked_prefill/test_stage.py",
         kind="count", subject="prefill_steps", n=200, fixtures="",
         what="the steps that one prompt of n tokens needs, under your token budget",
         facts=["facts.token_budget   the most tokens that one step may process"],
         hint="the steps for n prompt tokens, at most facts.token_budget in each step",
         floor="math.ceil(n / facts.token_budget)",
         meaning="1.0: each step is full until the last chunk",
         body='''
    import math

    facts = Facts(token_budget=64)

    def count_at(prompt_len):
        scheduler = ChunkedScheduler(BlockAllocator(10_000, 16), max_num_seqs=8,
                                     token_budget=facts.token_budget)
        scheduler.add_request(0, prompt_len, 1)
        steps = 0
        while scheduler.has_work():
            scheduler.step()
            steps += 1
        return steps

    predicted = math.ceil(200 / facts.token_budget)
'''),
    dict(stage="12", module="s12_cudagraph", test="stage_12_cuda_graphs/test_stage.py",
         kind="time", subject="graph_replay", n=4, fixtures="runner, device",
         what="one replay of your captured graph, for a batch of n rows",
         facts=["facts.weight_bytes   the bytes of the weights of the captured layers"],
         hint="which bytes must one replay read, at the least? Use facts.weight_bytes.",
         floor="facts.weight_bytes",
         meaning="above 1: the weights are small, so the launches and the copies set the time",
         body='''
    weight_bytes = 32 * (WIDTH * WIDTH * 2 + WIDTH * 2)
    facts = cudalib.card_facts(weight_bytes=weight_bytes)
    floor_ms = weight_bytes / facts.bandwidth_bytes_per_s * 1e3

    def step_at(batch_size):
        inputs = _random_batch(batch_size, device)
        return lambda: runner.run(inputs)
'''),
    dict(stage="13", module="s13_sampler", test="stage_13_sampling/test_stage.py",
         kind="time", subject="sampler_call", n=32, fixtures="device",
         what="one call of your sample() on n rows of logits",
         facts=["facts.vocab_size    the logits in one row",
                "facts.logit_bytes   the bytes of one logit (float32: 4)"],
         hint="which bytes must one call read, at the least? Use facts.vocab_size and\n"
              "                                    facts.logit_bytes.",
         floor="n * facts.vocab_size * facts.logit_bytes",
         meaning="about the passes over the logits, or more if your sampler sorts them",
         body='''
    from app.s13_sampler import SamplingParams, sample

    vocab_size = 151_936
    facts = cudalib.card_facts(vocab_size=vocab_size, logit_bytes=4)
    floor_ms = 32 * vocab_size * 4 / facts.bandwidth_bytes_per_s * 1e3

    def step_at(num_rows):
        logits = torch.randn(num_rows, vocab_size, device=device)
        params = [SamplingParams(temperature=0.8, top_p=0.9, seed=row) for row in range(num_rows)]
        return lambda: sample(logits, params)
'''),
    dict(stage="17", module="s17_speculative", test="stage_17_speculative_decoding/test_stage.py",
         kind="count", subject="tokens", n=1000, fixtures="",
         what="the tokens that n verify passes give, when each draft is accepted with a fixed chance",
         facts=["facts.acceptance_rate   the chance that one draft token is accepted",
                "facts.num_draft         the draft tokens of each pass"],
         hint="the tokens that n passes give, from the formula of this stage.\n"
              "                                    Use facts.acceptance_rate and facts.num_draft.",
         floor="n * (1 - facts.acceptance_rate ** (facts.num_draft + 1)) / (1 - facts.acceptance_rate)",
         meaning="near 1: the formula predicts what the drafts give",
         body='''
    facts = Facts(acceptance_rate=0.6, num_draft=4)

    def count_at(num_passes):
        rng = random.Random(0)
        produced = 0
        for _ in range(num_passes):
            accepted = 0
            while accepted < facts.num_draft and rng.random() < facts.acceptance_rate:
                accepted += 1
            produced += accepted + 1
        return produced

    predicted = 1000 * (1 - facts.acceptance_rate ** (facts.num_draft + 1)) / (1 - facts.acceptance_rate)
'''),
    dict(stage="18", module="s18_quantization", test="stage_18_quantization/test_stage.py",
         kind="time", subject="int8_linear_call", n=1, fixtures="device",
         what="one call of your QuantizedLinear, 16384 inputs x 8192 outputs, on n rows",
         facts=["facts.in_features       the inputs of the layer",
                "facts.out_features      the outputs of the layer",
                "facts.bytes_per_weight  the bytes of one int8 weight",
                "facts.bytes_per_scale   the bytes of one scale (one for each output)"],
         hint="which bytes must one call read, at the least?",
         floor="facts.out_features * facts.in_features * facts.bytes_per_weight + facts.out_features * facts.bytes_per_scale",
         meaning="far above 1: PyTorch builds a bf16 copy of the weight first. Stage 18b removes the copy",
         function="QuantizedLinear", import_line="from app.s18_quantization import QuantizedLinear",
         body=None),
    dict(stage="18b", module="s18b_gemv_cuda", test="stage_18b_int8_gemv_cuda/test_cuda.py",
         kind="time", subject="int8_linear_call", n=1, fixtures="nvcc, device",
         what="one call of your QuantizedLinearCUDA, 16384 inputs x 8192 outputs, on n rows",
         facts=None, hint="which bytes must one call read, at the least?",
         floor=None, meaning="against stage 18: near 1 at one row, because the kernel reads each int8 weight one time, with no copy",
         function="QuantizedLinearCUDA", import_line="from app.s18b_gemv_cuda import QuantizedLinearCUDA",
         body=None),
    dict(stage="21", module="s21_paged_runner", test="stage_21_paged_model/test_cuda.py",
         kind="time", subject="decode_step", n=2, fixtures="nvcc, tmodel",
         what="one eager decode step of your ModelRunner, for n sequences",
         facts=["facts.weight_bytes         the bytes that one step reads of the weights",
                "facts.kv_bytes_per_token   the KV cache bytes of one token",
                "facts.context_len          the tokens in the context of each sequence"],
         hint="which bytes must this step read, at the least?",
         floor="facts.weight_bytes + n * facts.context_len * facts.kv_bytes_per_token",
         meaning="how far your eager step is from the bytes that it must read",
         runner="ModelRunner(model, 16 * num_seqs + 16)", capture=False, model="tmodel", body=None),
    dict(stage="23", module="s23_graphs", test="stage_23_graphs/test_cuda.py",
         kind="time", subject="decode_step", n=2, fixtures="nvcc, tmodel",
         what="one decode step of your GraphedModelRunner, for n sequences",
         facts=None, hint="which bytes must this step read, at the least?",
         floor=None, meaning="against stage 21: the same bytes, with the launch gaps gone",
         runner="GraphedModelRunner(model, 16 * num_seqs + 16, max_model_len=256, buckets=(num_seqs,))",
         capture=True, model="tmodel", body=None),
    dict(stage="24", module="s24_quantized", test="stage_24_quantized/test_cuda.py",
         kind="time", subject="decode_step", n=1, fixtures="nvcc, qmodel",
         what="one graphed decode step of your int8 model, for n sequences",
         facts=None, hint="which bytes must this step read, at the least?",
         floor=None, meaning="against stage 23: fewer weight bytes, so a lower floor",
         runner="GraphedModelRunner(model, 16 * num_seqs + 16, max_model_len=256, buckets=(num_seqs,))",
         capture=True, model="qmodel", body=None),
    dict(stage="24b", module="s24b_kv_fp8", test="stage_24b_fp8_kv_cache/test_cuda.py",
         kind="time", subject="decode_step", n=4, fixtures="nvcc, tmodel",
         what="one graphed decode step with your FP8 KV cache, at a long context, for n sequences",
         facts=["facts.weight_bytes          the bytes that one step reads of the weights",
                "facts.kv_values_per_token   the K and V numbers of one token, all layers",
                "facts.bytes_per_kv_value    the bytes of one number in your cache",
                "facts.context_len           the tokens in the context of each sequence"],
         hint="which bytes must this step read, at the least?",
         floor="facts.weight_bytes + n * facts.context_len * facts.kv_values_per_token * facts.bytes_per_kv_value",
         meaning="at a long context the KV cache is a large part of the floor, and FP8 halves it",
         body='''
    from app.s24b_kv_fp8 import Fp8GraphedModelRunner, calibrate_kv_scales

    model = tmodel
    config = model.config
    prompt_ids = _prose_ids(model, 1024)
    scales = calibrate_kv_scales(model, prompt_ids)
    context_len = len(prompt_ids) + 1
    kv_values_per_token = 2 * config.num_layers * config.num_kv_heads * config.head_dim
    facts = cudalib.card_facts(weight_bytes=model.weight_bytes(), kv_values_per_token=kv_values_per_token,
                               bytes_per_kv_value=1, context_len=context_len)
    floor_ms = ((facts.weight_bytes + 4 * context_len * kv_values_per_token)
                / facts.bandwidth_bytes_per_s * 1e3)

    def step_at(num_seqs):
        blocks_per_seq = context_len // 16 + 2
        runner = Fp8GraphedModelRunner(model, blocks_per_seq * num_seqs + 16, scales,
                                       max_model_len=blocks_per_seq * 16, buckets=(num_seqs,))
        runner.capture()
        block_lists = [list(range(1 + index * blocks_per_seq, 1 + (index + 1) * blocks_per_seq))
                       for index in range(num_seqs)]
        runner.execute([SeqChunk(prompt_ids, 0, blocks) for blocks in block_lists])
        decodes = [SeqChunk([11], len(prompt_ids), blocks) for blocks in block_lists]
        return lambda: runner.execute(decodes)
'''),
]

BY_STAGE = {spec["stage"]: spec for spec in STAGES}
ATTENTION_FACTS, ATTENTION_FLOOR = BY_STAGE["07"]["facts"], BY_STAGE["07"]["floor"]
DECODE_FACTS, DECODE_FLOOR = BY_STAGE["21"]["facts"], BY_STAGE["21"]["floor"]
LINEAR_FACTS, LINEAR_FLOOR = BY_STAGE["18"]["facts"], BY_STAGE["18"]["floor"]
for spec in STAGES:
    if spec["facts"] is None:
        if spec["subject"] == "attention_call":
            spec["facts"], spec["floor"] = ATTENTION_FACTS, ATTENTION_FLOOR
        elif spec["subject"] == "decode_step":
            spec["facts"], spec["floor"] = DECODE_FACTS, DECODE_FLOOR
        else:
            spec["facts"], spec["floor"] = LINEAR_FACTS, LINEAR_FLOOR


def attention_body(spec):
    return f'''
    {spec["import_line"]}
    from tests.helpers import paged_problem

    # Larger than any L2 cache: the floor is a read of main memory.
    context_len, num_heads, num_kv_heads, head_dim = {spec.get("context_len", 4096)}, 16, 8, 128
    facts = cudalib.card_facts(context_len=context_len, num_kv_heads=num_kv_heads, head_dim=head_dim,
                               value_bytes=2)
    floor_ms = ({spec["n"]} * context_len * 2 * num_kv_heads * head_dim * 2
                / facts.bandwidth_bytes_per_s * 1e3)

    def step_at(num_seqs):
        query, _, _, paged = paged_problem(num_seqs, num_heads, num_kv_heads, head_dim, context_len,
                                           16, device, dtype=torch.float16)
        return lambda: {spec["function"]}(query, *paged)
'''


def linear_body(spec):
    return f'''
    {spec["import_line"]}

    # 128 MB of int8 weights: larger than any L2 cache.
    in_features, out_features = 16384, 8192
    facts = cudalib.card_facts(in_features=in_features, out_features=out_features,
                               bytes_per_weight=1, bytes_per_scale=4)
    floor_ms = ((out_features * in_features + out_features * 4)
                / facts.bandwidth_bytes_per_s * 1e3)
    dense = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)
    layer = {spec["function"]}.from_linear(dense)

    def step_at(num_rows):
        inputs = torch.randn(num_rows, in_features, device=device, dtype=torch.bfloat16)
        return torch.inference_mode()(lambda: layer(inputs))
'''


def decode_body(spec):
    capture = "\n        runner.capture()" if spec["capture"] else ""
    return f'''
    from tests.helpers import CAPSTONE_PROMPTS

    model = {spec["model"]}
    prompt_ids = model.tokenizer(CAPSTONE_PROMPTS[0]).input_ids
    context_len = len(prompt_ids) + 1
    facts = cudalib.card_facts(weight_bytes=model.weight_bytes(),
                               kv_bytes_per_token=model.kv_bytes_per_token(), context_len=context_len)
    floor_ms = ((facts.weight_bytes + {spec["n"]} * context_len * facts.kv_bytes_per_token)
                / facts.bandwidth_bytes_per_s * 1e3)

    def step_at(num_seqs):
        runner = {spec["runner"]}{capture}
        block_lists = [list(range(1 + index * 16, 1 + (index + 1) * 16)) for index in range(num_seqs)]
        runner.execute([SeqChunk(prompt_ids, 0, blocks) for blocks in block_lists])
        decodes = [SeqChunk([11], len(prompt_ids), blocks) for blocks in block_lists]
        return lambda: runner.execute(decodes)
'''


def names(spec):
    s = spec["subject"]
    if spec["kind"] == "time":
        return [f"{s}_floor_ms", f"{s}_measured_ms", f"{s}_measured_over_floor", f"{s}_ms_2n_over_n"]
    return [f"{s}_predicted", f"{s}_measured", f"{s}_measured_over_predicted", f"{s}_2n_over_n"]


def stub(spec):
    first, second, third, fourth = names(spec)
    facts = "\n".join(f"            {line}" for line in spec["facts"])
    hint = " ".join(spec["hint"].split())
    hint = "\n".join(textwrap.wrap("predict: " + hint, 64, initial_indent="            ",
                                   subsequent_indent="            "))
    wrap = lambda text, indent: "\n".join(textwrap.wrap(text, 72 - len(indent), initial_indent=indent,
                                                        subsequent_indent=indent))
    subject = wrap(f"The subject: {spec['what']}.", "    ").lstrip()
    if spec["kind"] == "time":
        runs = wrap(f"-> a function with no arguments. It runs {spec['what']}.", "            ")
        given = f'''        step_at(n)
{runs}
        n
            the size to measure at
        facts
            facts.bandwidth_bytes_per_s   the read bandwidth of this card
{facts}
        cudalib.bench_ms(function)
            -> the milliseconds of one call: warmed up, waited for, repeated'''
        measure, double = "measure: the time of step_at(n)", "double: the time at 2n over the time at n"
        argument, imports = "step_at", "    import cudalib\n\n"
    else:
        given = f'''        count_at(n)
{wrap("-> " + spec["what"], "            ")}
        n
            the size to count at
        facts
{facts}'''
        measure, double = "measure: count_at(n)", "double: count_at(2n) over count_at(n)"
        argument, imports = "count_at", ""
    return f'''

{MARK}


def four_lines({argument}, n, facts):
    """The four moves of docs/METHOD.md: predict, measure, divide, double.
    {subject}

    GIVEN
{given}

    ASKED: one line for each number
        {first}
{hint}
        {second}
            {measure}
        {third}
            divide
        {fourth}
            {double}
    """
{imports}    {first} = ...
    {second} = ...
    {third} = ...
    {fourth} = ...
    return {{
        "{first}": {first},
        "{second}": {second},
        "{third}": {third},
        "{fourth}": {fourth},
    }}
'''


def solution(spec):
    first, second, third, fourth = names(spec)
    if spec["kind"] == "time":
        return f'''

{MARK}


def four_lines(step_at, n, facts):
    """The four moves of docs/METHOD.md. Reference solution."""
    import cudalib

    {first} = ({spec["floor"]}) / facts.bandwidth_bytes_per_s * 1e3
    {second} = cudalib.bench_ms(step_at(n))
    {third} = {second} / {first}
    {fourth} = cudalib.bench_ms(step_at(2 * n)) / {second}
    return {{"{first}": {first}, "{second}": {second},
            "{third}": {third}, "{fourth}": {fourth}}}
'''
    return f'''

{MARK}


def four_lines(count_at, n, facts):
    """The four moves of docs/METHOD.md. Reference solution."""
    import math

    {first} = {spec["floor"]}
    {second} = count_at(n)
    {third} = {second} / {first}
    {fourth} = count_at(2 * n) / {second}
    return {{"{first}": {first}, "{second}": {second},
            "{third}": {third}, "{fourth}": {fourth}}}
'''


def test(spec):
    body = spec["body"]
    if body is None:
        body = {"attention_call": attention_body, "decode_step": decode_body}.get(
            spec["subject"], linear_body)(spec)
    if spec["kind"] == "time":
        head = ("    import torch\n\n    import cudalib\n"
                f"    from app.{spec['module']} import four_lines\n    from tests.helpers import check_four_lines\n")
        tail = (f'    check_four_lines(four_lines, "{spec["subject"]}", step_at, {spec["n"]}, facts, floor_ms,\n'
                f'                     "{spec["meaning"]}")\n')
    else:
        head = ("    import math\n    import random\n\n    from cudalib import Facts\n"
                f"    from app.{spec['module']} import four_lines\n    from tests.helpers import check_four_count_lines\n")
        tail = (f'    check_four_count_lines(four_lines, "{spec["subject"]}", count_at, {spec["n"]}, facts, predicted,\n'
                f'                           "{spec["meaning"]}")\n')
    return f'''

def test_the_four_lines({spec["fixtures"]}):
    """docs/METHOD.md: predict the floor, measure honestly, divide, double.
    The same four lines as in every stage. Only the subject changes."""
{head}{body}
{tail}'''


def main():
    solutions = None
    if "--solutions" in sys.argv:
        solutions = Path(sys.argv[sys.argv.index("--solutions") + 1])
    for spec in STAGES:
        app_file = ROOT / "app" / f"{spec['module']}.py"
        test_file = ROOT / "tests" / spec["test"]
        if MARK not in app_file.read_text():
            app_file.write_text(app_file.read_text().rstrip("\n") + "\n" + stub(spec))
        if "def test_the_four_lines" not in test_file.read_text():
            test_file.write_text(test_file.read_text().rstrip("\n") + "\n" + test(spec))
        if solutions is not None:
            target = solutions / f"{spec['module']}.py"
            if MARK not in target.read_text():
                target.write_text(target.read_text().rstrip("\n") + "\n" + solution(spec))
        print(f"stage {spec['stage']}: {spec['subject']}")


if __name__ == "__main__":
    main()
