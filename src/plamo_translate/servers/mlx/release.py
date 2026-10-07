"""Portable inference assets and content hashes for MLX model releases.

This module does not import MLX, so packaging checks also run on Linux.
"""

import hashlib
import importlib.resources
import json
from pathlib import Path


def write_inference_assets(output: Path, config: dict, eos_token_ids: list[int]) -> None:
    assets = importlib.resources.files("plamo_translate.assets")
    template = assets.joinpath("mlx_chat_template.jinja").read_text()
    (output / "modeling_mlx_plamo2.py").write_text(assets.joinpath("modeling_mlx_plamo2.py").read_text())
    (output / "chat_template.jinja").write_text(template)
    config = {**config, "model_file": "modeling_mlx_plamo2.py", "eos_token_id": eos_token_ids}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer_path = output / "tokenizer_config.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    tokenizer["chat_template"] = template
    tokenizer_path.write_text(json.dumps(tokenizer, indent=2) + "\n")
    generation_path = output / "generation_config.json"
    generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    generation["eos_token_id"] = eos_token_ids
    generation_path.write_text(json.dumps(generation, indent=2) + "\n")


def file_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inference_files(model_path: Path) -> list[str]:
    """Check that the index and actual weight shards describe the same bundle."""
    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_file") != "modeling_mlx_plamo2.py":
        raise ValueError("Missing standalone PLaMo 2 model_file")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    expected = set(index["weight_map"].values())
    actual = {path.name for path in model_path.glob("model*.safetensors")}
    if actual != expected:
        raise ValueError(f"Weight shards differ from index: missing={expected - actual}, stale={actual - expected}")
    files = sorted(expected | {
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "modeling_mlx_plamo2.py", "modeling_plamo.py", "tokenization_plamo.py",
        "tokenizer.jsonl", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    })
    for name in files:
        if not (model_path / name).is_file():
            raise ValueError(f"Missing inference asset: {name}")
    return files
