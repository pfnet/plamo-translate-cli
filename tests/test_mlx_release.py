import json

import pytest
from jinja2 import Environment

from plamo_translate.servers.mlx.release import file_digest, inference_files, write_inference_assets


def make_bundle(tmp_path):
    names = [
        "model.safetensors", "modeling_plamo.py", "tokenization_plamo.py",
        "tokenizer.jsonl", "special_tokens_map.json",
    ]
    for name in names:
        (tmp_path / name).write_text("placeholder")
    (tmp_path / "tokenizer_config.json").write_text('{"bos_token": "<|plamo:bos|>"}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    config = {"model_type": "plamo2", "plamo_translate_precision": "bf16"}
    write_inference_assets(tmp_path, config, [1, 4])
    return tmp_path


def test_export_preserves_bf16_and_bundles_stop_tokens(tmp_path):
    make_bundle(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["plamo_translate_precision"] == "bf16" and "quantization" not in config
    assert config["model_file"] == "modeling_mlx_plamo2.py"
    assert config["eos_token_id"] == [1, 4]
    assert json.loads((tmp_path / "generation_config.json").read_text())["eos_token_id"] == [1, 4]
    assert (tmp_path / config["model_file"]).is_file()
    files = inference_files(tmp_path)
    assert "model.safetensors" in files and "chat_template.jinja" in files
    assert "plamo-translate-source.json" not in files


@pytest.mark.parametrize("text", ["hello", "output predictions", "input data"])
def test_exported_template_matches_structured_cli_prompt(tmp_path, text):
    make_bundle(tmp_path)
    template = (tmp_path / "chat_template.jinja").read_text()
    assert json.loads((tmp_path / "tokenizer_config.json").read_text())["chat_template"] == template
    render = Environment().from_string(template).render
    kwargs = {"bos_token": "<|plamo:bos|>", "source_language": "English", "target_language": "Japanese"}
    natural = render(messages=[{"role": "user", "content": text}], add_generation_prompt=True, **kwargs)
    structured = render(messages=[
        {"role": "user", "content": f"input lang=English\n{text}"},
        {"role": "user", "content": "output lang=Japanese\n"},
    ], add_generation_prompt=False, **kwargs)
    assert natural == structured
    assert natural.startswith("<|plamo:bos|>") and natural.endswith("output lang=Japanese\n")


def test_release_rejects_stale_or_missing_shards(tmp_path):
    make_bundle(tmp_path)
    stale = tmp_path / "model-00001-of-00002.safetensors"
    stale.touch()
    with pytest.raises(ValueError, match="stale="):
        inference_files(tmp_path)
    stale.unlink()
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing="):
        inference_files(tmp_path)


def test_manifest_detects_post_validation_changes(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"validated")
    before = file_digest(path)
    path.write_bytes(b"modified!")
    assert file_digest(path)["sha256"] != before["sha256"]
