"""Teacher-forced checks of the PLaMo 3 MLX-4bit release on the supplied fixtures.

Compares this CLI's loader with the release's bundled MLX runtime, and both 4bit variants
(the release and the CLI's own conversion of the original weights) with FP32 original weights.
"""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model as load_release_runtime
from mlx_lm.utils import load_tokenizer
from transformers import PreTrainedConfig

from plamo_translate.servers.mlx.pfml3 import pfml3_prompt
from plamo_translate.servers.mlx.plamo3 import load_model
from plamo_translate.servers.mlx.plamo3_metal import FastDecoder
from plamo_translate.servers.mlx.plamo3_runtime import checkpoint_path, quantize
from plamo_translate.servers.utils import DEFAULT_MODEL_NAME, ORIGINAL_MODEL_NAME


def teacher_forced_cases(tokenizer, english: str, japanese: str) -> dict:
    """Prompt plus reference translation, and where the reference starts, for each direction."""
    cases = {}
    for name, (source, reference, source_language, target_language) in {
        "en-ja": (english, japanese, "English", "Japanese"),
        "ja-en": (japanese, english, "Japanese", "English"),
    }.items():
        messages = [
            SimpleNamespace(content=f"input lang={source_language}\n{source}"),
            SimpleNamespace(content=f"output lang={target_language}\n"),
        ]
        prompt = tokenizer.encode(pfml3_prompt(messages), add_special_tokens=False)
        cases[name] = (prompt + tokenizer.encode(reference, add_special_tokens=False), len(prompt))
    return cases


def plain_logits(model, cases) -> dict:
    return {
        name: np.array(model(mx.array(ids)[None], model.make_cache(), last_only=False)[0, start - 1 : -1].astype(mx.float32))
        for name, (ids, start) in cases.items()
    }


def fused_logits(model, cases) -> dict:
    decoder = FastDecoder(model)
    results = {}
    for name, (ids, start) in cases.items():
        caches = model.make_cache()
        model.prefill(ids[: start - 1], caches)
        rows = []
        for token in ids[start - 1 : -1]:
            step = decoder.logits(mx.array([token], dtype=mx.uint32), caches)
            mx.eval(step, [(cache.keys, cache.values) for cache in caches])
            rows.append(np.array(step))
        results[name] = np.stack(rows)
    return results


def runtime_logits(model, cases) -> dict:
    return {
        name: np.array(model(mx.array(ids)[None], cache=make_prompt_cache(model))[0, start - 1 : -1].astype(mx.float32))
        for name, (ids, start) in cases.items()
    }


def difference(actual: dict, expected: dict) -> dict:
    return {
        name: dict(
            top1=float((actual[name].argmax(-1) == expected[name].argmax(-1)).mean()),
            max_abs=float(np.abs(actual[name] - expected[name]).max()),
            rmse=float(np.sqrt(((actual[name] - expected[name]) ** 2).mean())),
        )
        for name in expected
    }


def fidelity(reference: dict, test: dict, cases) -> dict:
    """Top-1 agreement and KL(reference || test) per token, and NLL of the reference translation."""
    results = {}
    for name, (ids, start) in cases.items():
        log_p = reference[name] - np.logaddexp.reduce(reference[name], axis=-1, keepdims=True)
        log_q = test[name] - np.logaddexp.reduce(test[name], axis=-1, keepdims=True)
        target = np.array(ids[start:])
        results[name] = dict(
            top1=float((log_p.argmax(-1) == log_q.argmax(-1)).mean()),
            kl=float((np.exp(log_p) * (log_p - log_q)).sum(-1).mean()),
            nll=float(-log_q[np.arange(len(target)), target].mean()),
        )
    return results


def free_memory():
    gc.collect()
    mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--original", default=ORIGINAL_MODEL_NAME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--english", type=Path, default=Path("tests/fixtures/translation.en.txt"))
    parser.add_argument("--japanese", type=Path, default=Path("tests/fixtures/translation.ja.txt"))
    args = parser.parse_args()
    release, original = checkpoint_path(args.release), checkpoint_path(args.original)
    # The release runtime is modeling_plamo_mlx.py, which inference does not otherwise download.
    if not (release / "modeling_plamo_mlx.py").is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(args.release, "modeling_plamo_mlx.py")
    tokenizer = load_tokenizer(release, tokenizer_config_extra={"trust_remote_code": True, "config": PreTrainedConfig()})
    cases = teacher_forced_cases(tokenizer, args.english.read_text().strip(), args.japanese.read_text().strip())

    results = {"reference_tokens": {name: len(ids) - start for name, (ids, start) in cases.items()}}
    model, _ = load_model(release, mx.float16)
    release_plain = plain_logits(model, cases)
    results["release: fused kernels vs plain MLX"] = difference(fused_logits(model, cases), release_plain)
    del model
    free_memory()

    runtime, _ = load_release_runtime(release)
    shipped = runtime_logits(runtime, cases)
    results["release: this loader vs its runtime (BF16 activations)"] = difference(release_plain, shipped)
    runtime.update(
        nn.utils.tree_map(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a, runtime.parameters())
    )
    results["release: this loader vs its runtime cast to FP16"] = difference(release_plain, runtime_logits(runtime, cases))
    del runtime
    free_memory()

    model, _ = load_model(original, mx.float32)
    fp32 = plain_logits(model, cases)
    del model
    free_memory()
    model, _ = load_model(original, mx.float16, lazy=True)
    quantize(model, "4bit")
    mx.eval(model.parameters())
    converted = plain_logits(model, cases)
    del model
    free_memory()
    results["fidelity to FP32 original weights"] = {
        "CLI 4bit from original weights": fidelity(fp32, converted, cases),
        "MLX-4bit release, this loader": fidelity(fp32, release_plain, cases),
        "MLX-4bit release, its runtime": fidelity(fp32, shipped, cases),
        "FP32": fidelity(fp32, fp32, cases),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
