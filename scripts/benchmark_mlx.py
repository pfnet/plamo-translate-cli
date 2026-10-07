"""Reproducible single-request benchmark; never run GPU trials concurrently."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.resources
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from sacrebleu.metrics import CHRF

from plamo_translate.servers.warnings import suppress_optional_gpu_dependency_warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=["stock", "corrected", "optimized"], default="corrected")
    parser.add_argument("--precision", choices=["source", "4bit", "6bit", "8bit", "mixed"], default="source")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["source", "float16"], default="source")
    parser.add_argument("--legacy-prompt", action="store_true", help="Reproduce the old CLI prompt without BOS")
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/translation.en.txt"))
    parser.add_argument("--reference", type=Path, default=Path("tests/fixtures/translation.ja.txt"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    processes_before = subprocess.check_output(["ps", "-axo", "pid,%cpu,%mem,etime,ucomm"], text=True)
    template = importlib.resources.files("plamo_translate.assets").joinpath("chat_template.jinja2").read_text()
    started = time.perf_counter()
    with suppress_optional_gpu_dependency_warnings():
        model, tokenizer, config = load(
            args.model,
            tokenizer_config={"trust_remote_code": True, "chat_template": template},
            lazy=True,
            return_config=True,
        )
    if args.implementation != "stock":
        from plamo_translate.servers.mlx.model import configure_model

        configure_model(model, config, optimize=args.implementation == "optimized")
    if args.dtype == "float16":
        model.set_dtype(mx.float16)
    if args.precision != "source":
        if config.get("quantization"):
            raise ValueError("Quantization trials require original, unquantized source weights")
        bits = 4 if args.precision == "mixed" else int(args.precision[:-3])

        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if args.precision == "mixed":
                if path.endswith(("bcdt_proj", "dt_proj")):
                    return False
                return {"bits": 4 if ".mlp." in path else 8, "group_size": args.group_size}
            return True

        nn.quantize(model, bits=bits, group_size=args.group_size, class_predicate=predicate)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - started
    tokenizer.add_eos_token("<|plamo:op|>")
    source = args.input.read_text().strip()
    reference = args.reference.read_text().strip()

    def prompt(text):
        ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"input lang=English\n{text}"},
                {"role": "user", "content": "output lang=Japanese\n"},
            ],
            add_generation_prompt=False,
        )
        return ids[1:] if args.legacy_prompt and ids[0] == tokenizer.bos_token_id else ids

    kwargs = dict(sampler=make_sampler(temp=0), prefill_step_size=args.prefill_step_size)
    print(f"Loaded in {load_seconds:.2f}s; warming up", flush=True)
    for _ in stream_generate(model, tokenizer, prompt("Translate this sentence."), max_tokens=16, **kwargs):
        pass
    mx.synchronize()
    mx.clear_cache()
    results = []
    for run in range(args.runs):
        mx.reset_peak_memory()
        start = time.perf_counter()
        chunks, tokens = [], []
        for segment in stream_generate(model, tokenizer, prompt(source), max_tokens=args.max_tokens, **kwargs):
            chunks.append(segment.text)
            tokens.append(segment.token)
        mx.synchronize()
        seconds = time.perf_counter() - start
        translation = "".join(chunks)
        result = dict(
            run=run,
            elapsed_seconds=seconds,
            load_seconds=load_seconds,
            prompt_tokens=segment.prompt_tokens,
            prompt_tps=segment.prompt_tps,
            generation_tokens=segment.generation_tokens,
            generation_tps=segment.generation_tps,
            peak_memory_gb=segment.peak_memory,
            finish_reason=segment.finish_reason,
            chrf=CHRF().sentence_score(translation, [reference]).score,
            paragraphs=len(translation.strip().split("\n\n")),
            translation=translation,
            tokens=tokens,
        )
        results.append(result)
        (args.output / f"translation-{run}.txt").write_text(translation)
        print(json.dumps({k: v for k, v in result.items() if k not in ("translation", "tokens")}), flush=True)
        metadata = dict(
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            versions={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers", "sacrebleu")},
            platform=platform.platform(),
            device=mx.device_info(),
            model_config=config,
            revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            processes_before=processes_before,
            processes_after=subprocess.check_output(["ps", "-axo", "pid,%cpu,%mem,etime,ucomm"], text=True),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            reference_sha256=hashlib.sha256(reference.encode()).hexdigest(),
            backend_sha256=hashlib.sha256(Path("src/plamo_translate/servers/mlx/model.py").read_bytes()).hexdigest(),
            prompt_ids=prompt(source),
            results=results,
        )
        (args.output / "results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    # Teacher-forced likelihood evaluates the same supplied reference for every
    # candidate, independently of which wording greedy generation chose.
    prompt_ids = prompt(source)
    target_ids = tokenizer.encode(reference, add_special_tokens=False)
    ids = mx.array(prompt_ids + target_ids[:-1])
    logits = model(ids[None])[:, len(prompt_ids) - 1 :, :].astype(mx.float32)
    selected = mx.take_along_axis(logits[0], mx.array(target_ids)[:, None], axis=-1).squeeze(-1)
    nll = mx.mean(mx.logsumexp(logits[0], axis=-1) - selected).item()
    accuracy = mx.mean(mx.argmax(logits[0], axis=-1) == mx.array(target_ids)).item()
    metadata["reference_nll"] = nll if math.isfinite(nll) else None
    metadata["reference_logits_finite"] = math.isfinite(nll)
    metadata["reference_token_accuracy"] = accuracy
    (args.output / "results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"reference_nll": nll, "reference_token_accuracy": accuracy}), flush=True)


if __name__ == "__main__":
    # One GPU trial per host, even if two benchmark commands are launched.
    with open("/tmp/plamo-translate-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
