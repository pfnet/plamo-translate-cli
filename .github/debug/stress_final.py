"""DEBUG: repeat PLaMo 3's last load step (embedding and lm_head) under memory pressure, and whole-model controls."""

import os
import subprocess
import sys
import time

RAW = """
import json
import mlx.core as mx
from plamo_translate.servers.mlx.plamo3_runtime import checkpoint_path
path = checkpoint_path("pfnet/plamo-3-translate-2b")
index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
raw = {}
for name in ("model.embed_tokens.weight", "lm_head.weight"):
    raw[name] = mx.load(str(path / index[name]))[name]
embed, head = raw["model.embed_tokens.weight"], raw["lm_head.weight"]
"""

FINAL = {
    # main: the embedding and lm_head converted, and lm_head quantized, in one evaluation.
    "together": RAW + "mx.eval(embed.astype(mx.float16), *mx.quantize(head.astype(mx.float16), 64, 4))\n",
    # One evaluation per tensor.
    "split": RAW + "mx.eval(embed.astype(mx.float16))\nmx.eval(*mx.quantize(head.astype(mx.float16), 64, 4))\n",
    # Casts on the CPU, only the quantization on the GPU, one evaluation per tensor.
    "cpu-cast": RAW
    + "mx.eval(embed.astype(mx.float16, stream=mx.cpu))\n"
    + "mx.eval(*mx.quantize(head.astype(mx.float16, stream=mx.cpu), 64, 4))\n",
}

WHOLE = """
import mlx.core as mx
from plamo_translate.servers.mlx.plamo3 import Model, ModelArgs, load_config
from plamo_translate.servers.mlx.plamo3_runtime import checkpoint_path, quantize
path = checkpoint_path("pfnet/plamo-3-translate-2b")
model = Model(ModelArgs.from_dict(load_config(path)))
weights = {}
for shard in sorted(path.glob("model*.safetensors")):
    weights.update(mx.load(str(shard)))
model.load_weights([(name, array.astype(mx.float16)) for name, array in weights.items()])
del weights
quantize(model, "4bit")
mx.eval(model.parameters())
"""

BIG = {"MLX_MAX_OPS_PER_BUFFER": "4096", "MLX_MAX_MB_PER_BUFFER": "1000000"}
part = sys.argv[1]
rounds = int(sys.argv[2])
if part == "final":
    variants = {name: (script, BIG) for name, script in FINAL.items()}
else:
    variants = {"single-big": (WHOLE, BIG), "single-default": (WHOLE, {})}

results = {name: [] for name in variants}
for round_ in range(rounds):
    for name, (script, extra) in variants.items():
        env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_MAX_")} | extra
        start = time.monotonic()
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
        elapsed = time.monotonic() - start
        if result.returncode == 0:
            outcome = "ok"
        elif "Timeout" in result.stderr:
            outcome = "GPU timeout"
        else:
            outcome = f"error {result.returncode}: {result.stderr.strip().splitlines()[-1:]}"
        results[name].append(outcome)
        print(f"round {round_} {name}: {outcome} in {elapsed:.1f} s", flush=True)
for name, outcomes in results.items():
    print(f"SUMMARY {name}: {outcomes.count('GPU timeout')} GPU timeouts, {outcomes.count('ok')} ok of {len(outcomes)}")
