"""DEBUG: load PLaMo 3 repeatedly from a cold page cache under memory pressure, comparing load strategies."""

import os
import subprocess
import sys
import time

VARIANTS = {
    # Before PR #38: one evaluation of the whole model.
    "single": """
mx.eval(model.parameters())
""",
    # PR #38 (main): one evaluation per layer, reading and converting together.
    "layerwise": """
for layer in model.model.layers.layers:
    mx.eval(layer.parameters())
    mx.clear_cache()
mx.eval(model.parameters())
mx.clear_cache()
""",
}

OLD = """
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
"""

NEW = """
import mlx.core as mx
from plamo_translate.servers.mlx.plamo3 import load_model
from plamo_translate.servers.mlx.plamo3_runtime import checkpoint_path, quantize
load_model(checkpoint_path("pfnet/plamo-3-translate-2b"), mx.float16, lambda model: quantize(model, "4bit"))
"""

SCRIPTS = {name: OLD + body for name, body in VARIANTS.items()} | {"prefetch": NEW}

env = {**os.environ, "MLX_MAX_OPS_PER_BUFFER": "4096", "MLX_MAX_MB_PER_BUFFER": "1000000"}
results = {name: [] for name in SCRIPTS}
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for round_ in range(rounds):
    order = list(SCRIPTS)[round_ % 3 :] + list(SCRIPTS)[: round_ % 3]
    for name in order:
        if os.environ.get("CI"):
            subprocess.run(["sudo", "purge"], check=False)
        start = time.monotonic()
        result = subprocess.run([sys.executable, "-c", SCRIPTS[name]], capture_output=True, text=True, env=env)
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
