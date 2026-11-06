"""DEBUG: load PLaMo 3 under memory pressure; does the command-buffer size decide the GPU timeouts?"""

import os
import subprocess
import sys
import time

LOAD = """
import sys
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
def stage(name):
    print("stage", name, file=sys.stderr, flush=True)
"""

SINGLE = LOAD + """
stage("all")
mx.eval(model.parameters())
"""

LAYERWISE = LOAD + """
for i, layer in enumerate(model.model.layers.layers):
    stage(f"layer {i}")
    mx.eval(layer.parameters())
    mx.clear_cache()
stage("final")
mx.eval(model.parameters())
mx.clear_cache()
"""

BIG = {"MLX_MAX_OPS_PER_BUFFER": "4096", "MLX_MAX_MB_PER_BUFFER": "1000000"}
VARIANTS = {
    "single-big": (SINGLE, BIG),
    "layerwise-big": (LAYERWISE, BIG),
    "layerwise-default": (LAYERWISE, {}),
}

rounds = int(sys.argv[1])
results = {name: [] for name in VARIANTS}
for round_ in range(rounds):
    for name, (script, extra) in VARIANTS.items():
        env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_MAX_")} | extra
        start = time.monotonic()
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
        elapsed = time.monotonic() - start
        stages = [line for line in result.stderr.splitlines() if line.startswith("stage ")]
        if result.returncode == 0:
            outcome = "ok"
        elif "Timeout" in result.stderr:
            outcome = f"GPU timeout at {stages[-1] if stages else '?'}"
        else:
            outcome = f"error {result.returncode}: {result.stderr.strip().splitlines()[-1:]}"
        results[name].append(outcome)
        print(f"round {round_} {name}: {outcome} in {elapsed:.1f} s", flush=True)
for name, outcomes in results.items():
    timeouts = sum(o.startswith("GPU timeout") for o in outcomes)
    print(f"SUMMARY {name}: {timeouts} GPU timeouts, {outcomes.count('ok')} ok of {len(outcomes)}")
