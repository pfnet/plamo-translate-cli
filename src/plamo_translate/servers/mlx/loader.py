"""Load original or MLX-quantized translation checkpoints."""

import importlib.resources
import contextlib
import sys

import mlx.core as mx
from mlx_lm.utils import load, quantize_model

from plamo_translate.servers.mlx.model import configure_model
from plamo_translate.servers.warnings import suppress_optional_gpu_dependency_warnings


def quantize_translation_model(model, config, precision: str):
    if precision == "bf16":
        if config.get("quantization"):
            raise ValueError("BF16 requires the original weights; quantized weights cannot recover BF16 accuracy")
        return model, config
    if precision not in ("4bit", "6bit", "8bit", "mixed"):
        raise ValueError(f"Unsupported precision: {precision}")
    if config.get("quantization"):
        existing = config.get("plamo_translate_precision", f"{config['quantization']['bits']}bit")
        if existing != precision:
            raise ValueError(
                f"Checkpoint is {existing}, but {precision} was requested. Use original weights to convert."
            )
        return model, config

    def predicate(path, module):
        if precision != "mixed":
            return True
        # Preserve small recurrent-state projections; compress the large MLPs.
        if path.endswith(("bcdt_proj", "dt_proj")):
            return False
        return {"bits": 4 if ".mlp." in path else 8, "group_size": 64, "mode": "affine"}

    with contextlib.redirect_stdout(sys.stderr):
        model, config = quantize_model(
            model,
            config,
            group_size=64,
            bits=4 if precision == "mixed" else int(precision[:-3]),
            quant_predicate=predicate,
        )
    config["plamo_translate_precision"] = precision
    return model, config


def load_translation_model(model_name: str, precision: str | None = None, *, optimize: bool = False):
    template = importlib.resources.files("plamo_translate.assets").joinpath("chat_template.jinja2").read_text()
    with suppress_optional_gpu_dependency_warnings():
        model, tokenizer, config = load(
            model_name,
            lazy=True,
            return_config=True,
            tokenizer_config={"trust_remote_code": True, "chat_template": template},
        )
    configure_model(model, config, optimize=optimize)
    # Existing quantization is authoritative unless explicitly overridden.
    if precision is not None or not config.get("quantization"):
        model, config = quantize_translation_model(model, config, precision or "4bit")
    mx.eval(model.parameters())
    tokenizer.add_eos_token("<|plamo:op|>")
    return model, tokenizer, config
