"""Validate saved model loading and complete streamed/non-streamed CLI output."""

import argparse
import asyncio
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sacrebleu.metrics import CHRF

from plamo_translate.servers.utils import verify_mcp_server_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path("tests/fixtures/translation.en.txt").read_text().strip()
    reference = Path("tests/fixtures/translation.ja.txt").read_text().strip()
    with tempfile.TemporaryDirectory(prefix="plamo-mlx-validation-") as temporary:
        env = os.environ.copy()
        env.update(
            TMPDIR=temporary,
            PLAMO_TRANSLATE_CLI_SERVER_START_PORT="31200",
            PLAMO_TRANSLATE_CLI_SERVER_END_PORT="31299",
            PLAMO_TRANSLATE_CLI_MODEL_NAME=args.model,
            PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER="0",
            PLAMO_TRANSLATE_CLI_OPTIMIZE="0",
            PLAMO_TRANSLATE_CLI_TEMP="0",
            PLAMO_MAX_TOKENS="2048",
        )
        for name in (
            "PLAMO_TRANSLATE_CLI_PRECISION",
            "PLAMO_TRANSLATE_CLI_REPETITION_PENALTY",
            "PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE",
        ):
            env.pop(name, None)
        command = [sys.executable, "-m", "plamo_translate.main"]
        with (args.output / "server.log").open("w") as log:
            started = time.perf_counter()
            server = subprocess.Popen(command + ["server", "--model", args.model], env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError("Server exited; see server.log")
                    path = Path(temporary) / "plamo-translate-config.json"
                    config = json.loads(path.read_text()) if path.exists() else {}
                    if config.get("port") and asyncio.run(verify_mcp_server_ready(config["port"])):
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Server startup timed out")
                results = {
                    "model": args.model,
                    "startup_seconds": time.perf_counter() - started,
                    "config": config,
                    "runs": [],
                }
                assert config["model_name"] == args.model
                assert config["precision"] == "4bit"
                for stream in (True, False):
                    started = time.perf_counter()
                    result = subprocess.run(
                        command + ["--from", "English", "--to", "Japanese"] + ([] if stream else ["--no-stream"]),
                        input=source,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=180,
                    )
                    assert result.returncode == 0, result.stderr
                    assert result.stdout == args.expected.read_text(), "CLI output differs from direct inference"
                    assert len(result.stdout.strip().split("\n\n")) == 8
                    for required in ("Preferred Networks", "Learn or Die", "西川", "岡野原"):
                        assert required in result.stdout
                    (args.output / f"translation-{'stream' if stream else 'complete'}.txt").write_text(result.stdout)
                    results["runs"].append(
                        {
                            "stream": stream,
                            "elapsed_seconds": time.perf_counter() - started,
                            "exact_match_direct": True,
                            "chrf": CHRF().sentence_score(result.stdout, [reference]).score,
                        }
                    )
                (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(json.dumps(results, ensure_ascii=False), flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)


if __name__ == "__main__":
    with open("/tmp/plamo-translate-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
