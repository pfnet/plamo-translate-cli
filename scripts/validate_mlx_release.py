"""Validate a standalone release against same-precision direct inference."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from sacrebleu.metrics import CHRF

from plamo_translate.servers.mlx.release import file_digest, inference_files
from plamo_translate.servers.warnings import suppress_optional_gpu_dependency_warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", required=True, choices=["4bit", "8bit", "bf16"])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runs", default=2, type=int)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    files = inference_files(args.model)
    dtypes = Counter()
    for name in files:
        if name.endswith(".safetensors"):
            with (args.model / name).open("rb") as file:
                header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
            dtypes.update(value["dtype"] for key, value in header.items() if key != "__metadata__")
    assert set(dtypes) == ({"BF16"} if args.precision == "bf16" else {"BF16", "U32"}), dtypes
    config = json.loads((args.model / "config.json").read_text())
    assert config["plamo_translate_precision"] == args.precision
    assert config["eos_token_id"] == [1, 4]
    if args.precision == "bf16":
        assert not config.get("quantization")
    else:
        assert config["quantization"]["bits"] == int(args.precision[:-3])
        assert config["quantization"]["group_size"] == 64
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path("tests/fixtures/translation.en.txt").read_text().strip()
    reference = Path("tests/fixtures/translation.ja.txt").read_text().strip()
    expected = args.expected.read_text()
    # Deliberately bypass the CLI loader: model_file must supply every correction.
    with suppress_optional_gpu_dependency_warnings():
        model, tokenizer = load(str(args.model), tokenizer_config={"trust_remote_code": True})
    assert type(model).__module__ == "custom_model"
    assert model.layers[0].mixer._ssm.__func__.__module__ == "custom_model"
    assert model.config.rope_local_theta == 1000000
    assert tokenizer.eos_token_ids == {1, 4}
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": source}], tokenize=True, add_generation_prompt=True,
        source_language="English", target_language="Japanese",
    )
    structured = tokenizer.apply_chat_template([
        {"role": "user", "content": f"input lang=English\n{source}"},
        {"role": "user", "content": "output lang=Japanese\n"},
    ], tokenize=True, add_generation_prompt=False)
    assert prompt == structured and prompt[0] == tokenizer.bos_token_id
    kwargs = dict(sampler=make_sampler(temp=0), prefill_step_size=512)
    for _ in stream_generate(model, tokenizer, prompt[:32], max_tokens=16, **kwargs):
        pass
    mx.synchronize()
    runs = []
    for index in range(args.runs):
        mx.reset_peak_memory()
        start = time.perf_counter()
        chunks = []
        for segment in stream_generate(model, tokenizer, prompt, max_tokens=2048, **kwargs):
            chunks.append(segment.text)
        mx.synchronize()
        output = "".join(chunks)
        elapsed = time.perf_counter() - start
        (args.output / f"translation-{index}.txt").write_text(output)
        assert segment.finish_reason == "stop", "Incomplete translation"
        assert output == expected, "Standalone release differs from same-precision direct inference"
        assert len(output.strip().split("\n\n")) == 8
        assert all(term in output for term in ("Preferred Networks", "Learn or Die", "西川", "岡野原"))
        run = {
            "elapsed_seconds": elapsed, "generation_tps": segment.generation_tps,
            "generation_tokens": segment.generation_tokens, "prompt_tps": segment.prompt_tps,
            "peak_memory_gb": segment.peak_memory, "finish_reason": segment.finish_reason,
            "chrf": CHRF().sentence_score(output, [reference]).score,
            "paragraphs": 8, "full_output_exact_match": True,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }
        runs.append(run)
        print(json.dumps(run), flush=True)
    evaluation = {
        "source_model": "pfnet/plamo-2-translate", "source_revision": args.source_revision,
        "precision": args.precision, "quantization": config.get("quantization"),
        "floating_weight_dtype": "bfloat16", "recurrent_state_dtype": "float32",
        "tensor_dtype_counts": dict(dtypes),
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")},
        "platform": platform.platform(), "device": mx.device_info(),
        "scope": "One English-to-Japanese example; not a general translation-quality guarantee.",
        "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "generation": {"temperature": 0, "max_tokens": 2048, "prefill_step_size": 512, "prompt_tokens": len(prompt)},
        "standalone_model_class": type(model).__module__ + "." + type(model).__name__,
        "natural_and_structured_prompts_match": True, "eos_token_ids": [1, 4], "runs": runs,
    }
    (args.model / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    report = {**evaluation, "validated_files": {name: file_digest(args.model / name) for name in files}}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(files)} inference files", flush=True)


if __name__ == "__main__":
    with open("/tmp/plamo-translate-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
