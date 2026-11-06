"""DEBUG: Plamo3Translator from main and from the fix, alternately, under incompressible memory pressure."""

import os
import subprocess
import sys
import time

LOAD = "from plamo_translate.servers.mlx.plamo3_runtime import Plamo3Translator\nPlamo3Translator('pfnet/plamo-3-translate-2b')\n"
pythons = {"main": sys.argv[1], "fix": sys.argv[2]}
rounds = int(sys.argv[3])
env = os.environ | {"MLX_MAX_OPS_PER_BUFFER": "4096", "MLX_MAX_MB_PER_BUFFER": "1000000"}
results = {name: [] for name in pythons}
for round_ in range(rounds):
    for name in (list(pythons) if round_ % 2 == 0 else list(pythons)[::-1]):
        start = time.monotonic()
        result = subprocess.run([pythons[name], "-I", "-c", LOAD], capture_output=True, text=True, env=env)
        elapsed = time.monotonic() - start
        if result.returncode == 0:
            outcome = "ok"
        elif "Timeout" in result.stderr:
            where = [line.strip() for line in result.stderr.splitlines() if "File " in line][-1:]
            outcome = f"GPU timeout {where}"
        else:
            outcome = f"error {result.returncode}: {result.stderr.strip().splitlines()[-1:]}"
        results[name].append(outcome)
        print(f"round {round_} {name}: {outcome} in {elapsed:.1f} s", flush=True)
for name, outcomes in results.items():
    timeouts = sum(o.startswith("GPU timeout") for o in outcomes)
    print(f"SUMMARY {name}: {timeouts} GPU timeouts, {outcomes.count('ok')} ok of {len(outcomes)}")
