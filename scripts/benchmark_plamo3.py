"""Reproducible PLaMo 3 translation benchmark; never run GPU trials concurrently."""

import os

# MLX reads these on import, so they must precede it.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "4096")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import mlx.core as mx  # noqa: E402
from sacrebleu.metrics import CHRF  # noqa: E402

from plamo_translate.servers.mlx.pfml3 import pfml3_prompt  # noqa: E402
from plamo_translate.servers.mlx.plamo3_runtime import Plamo3Translator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["4bit", "6bit", "8bit", "mixed", "bf16"], default="4bit")
    parser.add_argument("--plain", action="store_true", help="Use plain MLX ops instead of the fused kernels")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--english", type=Path, default=Path("tests/fixtures/translation.en.txt"))
    parser.add_argument("--japanese", type=Path, default=Path("tests/fixtures/translation.ja.txt"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    translator = Plamo3Translator(args.model, args.precision)
    load_seconds = time.perf_counter() - started
    if args.plain:
        translator.fast = None

    english, japanese = args.english.read_text().strip(), args.japanese.read_text().strip()
    cases = {
        "en-ja": (english, japanese, "English", "Japanese"),
        "ja-en": (japanese, english, "Japanese", "English"),
    }
    results = []
    for name, (source, reference, source_language, target_language) in cases.items():
        messages = [
            SimpleNamespace(content=f"input lang={source_language}\n{source}"),
            SimpleNamespace(content=f"output lang={target_language}\n"),
        ]
        prompt_tokens = len(translator.tokenizer.encode(pfml3_prompt(messages), add_special_tokens=False))
        for run in range(args.runs):
            start = time.perf_counter()
            first = None
            chunks = []
            for segment in translator.stream(messages, args.max_tokens):
                first = first or time.perf_counter()
                chunks.append(segment.text)
            end = time.perf_counter()
            # The final segment only carries the finish reason.
            tokens = len(chunks) - 1
            text = "".join(chunks)
            results.append(
                dict(
                    case=name,
                    run=run,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=tokens,
                    first_token_seconds=first - start,
                    decode_tps=(tokens - 1) / (end - first),
                    total_seconds=end - start,
                    finish_reason=segment.finish_reason,
                    peak_memory_gb=mx.get_peak_memory() / 1e9,
                    chrf=CHRF().sentence_score(text, [reference]).score,
                )
            )
            (args.output / f"translation-{name}-{run}.txt").write_text(text)
            print(json.dumps({k: v for k, v in results[-1].items() if k != "case"} | {"case": name}), flush=True)

    summary = {name: statistics.median(r["decode_tps"] for r in results if r["case"] == name) for name in cases}
    metadata = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        versions={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers", "sacrebleu")},
        platform=platform.platform(),
        device=mx.device_info(),
        kernels=translator.fast is not None,
        load_seconds=load_seconds,
        median_decode_tps=summary,
        results=results,
    )
    (args.output / "results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"load {load_seconds:.2f}s; kernels={translator.fast is not None}; median decode tok/s {summary}", flush=True)


if __name__ == "__main__":
    # One GPU trial per host, even if two benchmark commands are launched.
    with open("/tmp/plamo-translate-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
