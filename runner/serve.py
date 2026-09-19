"""./vc serve [--port 8000] [--model NAME] [--no-int8] [--bf16-kv] [--no-spec]

Start YOUR capstone server (stage 27) on the real model, with every part that
you built. That is int8 weights (24), an FP8 KV cache (24b), speculative
decoding (25) and JSON mode (26). Each flag turns one part off, so that you
can measure what it buys. Then, from another terminal:

    curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \\
      -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
    curl localhost:8000/metrics
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MEMORY_RESERVE = 1.5e9          # bytes left free for activations and graphs
MAX_MODEL_LEN = 4096
NUM_SPECULATIVE = 4


def parse_args(argv):
    options = {"port": 8000, "model": None,
               "int8": "--no-int8" not in argv,
               "fp8_kv": "--bf16-kv" not in argv,
               "speculate": "--no-spec" not in argv}
    for index, arg in enumerate(argv[:-1]):
        if arg == "--port":
            options["port"] = int(argv[index + 1])
        if arg == "--model":
            options["model"] = argv[index + 1]
    return options


def pool_blocks(model, fp8_kv):
    import torch

    bytes_per_token = model.kv_bytes_per_token() // (2 if fp8_kv else 1)
    free_bytes, _ = torch.cuda.mem_get_info()
    return int((free_bytes - MEMORY_RESERVE) / (bytes_per_token * 16))


def build_runner(model, num_blocks, fp8_kv):
    from app.s23_graphs import GraphedModelRunner
    from app.s24b_kv_fp8 import Fp8GraphedModelRunner, calibrate_kv_scales

    if not fp8_kv:
        runner = GraphedModelRunner(model, num_blocks,
                                    max_model_len=MAX_MODEL_LEN)
    else:
        calibration = (ROOT / "LORE.md").read_text()[:4000]
        token_ids = model.tokenizer(calibration).input_ids[:512]
        runner = Fp8GraphedModelRunner(model, num_blocks,
                                       calibrate_kv_scales(model, token_ids),
                                       max_model_len=MAX_MODEL_LEN)
    runner.capture()
    return runner


def main(argv):
    import uvicorn

    from app.s16_metrics import MetricsCollector
    from app.s24_quantized import quantize_model
    from app.s26_guided import GuidedEngine
    from app.s27_serve import build_app
    from tvllm import load_model

    options = parse_args(argv)
    model = load_model(options["model"]) if options["model"] else load_model()
    if options["int8"]:
        quantize_model(model)
    num_blocks = pool_blocks(model, options["fp8_kv"])
    runner = build_runner(model, num_blocks, options["fp8_kv"])
    metrics = MetricsCollector()
    engine = GuidedEngine(model, num_blocks, runner=runner, metrics=metrics,
                          num_speculative=NUM_SPECULATIVE if options["speculate"]
                          else 0)
    app = build_app(engine, metrics, model_name=model.config.name)
    kv_format = "FP8" if options["fp8_kv"] else "bf16"
    print(f"{model.config.name}: {num_blocks * 16:,} tokens of {kv_format} KV. "
          f"http://localhost:{options['port']}/v1")
    uvicorn.run(app, host="127.0.0.1", port=options["port"], log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
