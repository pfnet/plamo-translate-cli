"""Run with: uv run --with sacrebleu python scripts/benchmark_llama_cpp.py --help."""

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sacrebleu.metrics import CHRF

from plamo_translate.servers.llama_cpp.server import (
    LlamaCppOptions,
    LlamaCppRuntime,
    completion_events,
    completion_payload,
)
from plamo_translate.servers.utils import Message, TranslateRequest


async def measure(runtime, request, reference, directory, label):
    payload = completion_payload(request, max_tokens=4096)
    chunks, events = [], []
    first_token = None
    start = time.perf_counter()
    async with httpx.AsyncClient(trust_env=False, timeout=600) as client:
        async for event in completion_events(client, runtime.url, payload):
            events.append(event)
            if event.get("content"):
                if first_token is None:
                    first_token = time.perf_counter() - start
                chunks.append(event["content"])
    elapsed = time.perf_counter() - start
    translation = "".join(chunks)
    metric = CHRF()
    score = metric.corpus_score([translation], [[reference]])
    result = {
        "label": label,
        "options": asdict(runtime.options),
        "wall_seconds": elapsed,
        "first_token_seconds": first_token,
        "chrf": score.score,
        "chrf_signature": str(metric.get_signature()),
        "paragraphs": len(translation.strip().split("\n\n")),
        "timings": events[-1].get("timings"),
        "stop_type": events[-1].get("stop_type"),
        "truncated": events[-1].get("truncated"),
        "contains_names": all(x in translation.replace(" ", "") for x in ["西川徹", "岡野原大輔"]),
        "contains_motto": "Learn or Die" in translation,
        "contains_founders_love": "心から愛" in translation,
        "title_mentions_together": "と共に" in translation.split("\n", 1)[0],
    }
    (directory / f"{label}.txt").write_text(translation, encoding="utf-8")
    (directory / f"{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / f"{label}.events.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare complete translations, cold prompts and warm model weights.")
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/translation.en.txt"))
    parser.add_argument("--reference", type=Path, default=Path("tests/fixtures/translation.ja.txt"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--flash-attn", choices=["on", "off"], default="on")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    args.output.mkdir(parents=True, exist_ok=False)
    text = args.input.read_text().strip()
    reference = args.reference.read_text().strip()
    request = TranslateRequest(
        messages=[Message(role="user", content=text)], source_language="English", target_language="Japanese"
    )
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "llama_version": subprocess.run([args.llama_server, "--version"], capture_output=True, text=True).stdout,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "payload": completion_payload(request, max_tokens=4096),
        "note": "Single supplied example; chrF measures reference overlap, not general translation accuracy.",
    }
    native_dir = Path(args.llama_server).resolve().parent
    metadata["runtime_files_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(args.llama_server).resolve(), *sorted(native_dir.glob("*.dylib"))]
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    results = []
    for model in args.models:
        options = LlamaCppOptions(
            model=str(Path(model).resolve()),
            executable=str(Path(args.llama_server).resolve()),
            ctx_size=args.ctx_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            threads=args.threads,
            flash_attn=args.flash_attn,
        )
        runtime = LlamaCppRuntime(options)
        print(f"START {Path(model).stem}", flush=True)
        try:
            runtime.start()
            warmup = TranslateRequest(
                messages=[Message(role="user", content="Hello.")], source_language="English", target_language="Japanese"
            )
            payload = completion_payload(warmup, max_tokens=16)
            payload["stream"] = False
            with httpx.Client(trust_env=False, timeout=600) as client:
                response = client.post(runtime.url + "/completion", json=payload)
                response.raise_for_status()
            for repeat in range(args.repeats):
                label = f"{Path(model).stem}-{repeat + 1}"
                result = asyncio.run(measure(runtime, request, reference, args.output, label))
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            if runtime.log is not None:
                runtime.log.seek(0)
                (args.output / f"{Path(model).stem}.log").write_bytes(runtime.log.read())
            runtime.close()
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
