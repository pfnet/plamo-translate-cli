import importlib.resources
import json
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
from jinja2 import Environment

from plamo_translate import main
from plamo_translate.servers import utils
from plamo_translate.servers.mlx.pfml3 import pfml3_prompt
from plamo_translate.servers.utils import Message, TranslateRequest, construct_llm_input


@pytest.fixture(autouse=True)
def isolated_environment():
    # main() exports the selected model and precision; keep them out of later tests.
    with mock.patch.dict(os.environ):
        yield


@pytest.mark.parametrize("precision", ["4bit", "6bit", "8bit", "mixed", "bf16"])
def test_explicit_model_takes_precedence_over_environment(monkeypatch, precision):
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", "old/model")
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_PRECISION", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["plamo-translate", "--model", "/local/source", "--precision", precision, "--input", "hello"]
    )
    monkeypatch.setattr(main, "update_config", lambda **kwargs: {})
    monkeypatch.setattr(main, "check_server_running", lambda: False)
    monkeypatch.setattr(main, "run_translate", lambda args: None)
    main.main()
    assert main.os.environ["PLAMO_TRANSLATE_CLI_MODEL_NAME"] == "/local/source"
    assert main.os.environ["PLAMO_TRANSLATE_CLI_PRECISION"] == precision


def test_prompt_has_exactly_one_bos_and_one_output_newline():
    template = importlib.resources.files("plamo_translate.assets").joinpath("chat_template.jinja2").read_text()
    prompt = Environment(trim_blocks=True, lstrip_blocks=True).from_string(template).render(
        bos_token="<|plamo:bos|>",
        messages=[
            {"role": "user", "content": "input lang=English\nhello"},
            {"role": "user", "content": "output lang=Japanese\n"},
        ],
    )
    assert prompt == (
        "<|plamo:bos|><|plamo:op|>dataset\ntranslation\n"
        "<|plamo:op|>input lang=English\nhello\n<|plamo:op|>output lang=Japanese\n"
    )


def test_failed_child_startup_does_not_wait_forever(monkeypatch):
    monkeypatch.setattr(main, "check_server_running", lambda: False)

    class DeadProcess:
        exitcode = 1

        def is_alive(self):
            return False

    with pytest.raises(RuntimeError, match="exited during model loading"):
        main.wait_for_server_ready(DeadProcess())


@pytest.mark.parametrize("precision", [None, "4bit", "6bit", "8bit", "mixed", "bf16"])
def test_every_precision_quantizes_plamo3_by_default(monkeypatch, precision):
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_PRECISION", raising=False)
    options = [] if precision is None else ["--precision", precision]
    monkeypatch.setattr(sys, "argv", ["plamo-translate", *options, "--input", "hello"])
    monkeypatch.setattr(main, "update_config", lambda **kwargs: {})
    monkeypatch.setattr(main, "check_server_running", lambda: False)
    monkeypatch.setattr(main, "run_translate", lambda args: None)
    main.main()
    assert main.os.environ["PLAMO_TRANSLATE_CLI_MODEL_NAME"] == "pfnet/plamo-3-translate-2b"
    assert main.os.environ.get("PLAMO_TRANSLATE_CLI_PRECISION") == precision


def test_environment_model_overrides_default(monkeypatch):
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", "mlx-community/plamo-2-translate")
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_PRECISION", raising=False)
    monkeypatch.setattr(sys, "argv", ["plamo-translate", "--precision", "4bit", "--input", "hello"])
    monkeypatch.setattr(main, "update_config", lambda **kwargs: {})
    monkeypatch.setattr(main, "check_server_running", lambda: False)
    monkeypatch.setattr(main, "run_translate", lambda args: None)
    main.main()
    assert main.os.environ["PLAMO_TRANSLATE_CLI_MODEL_NAME"] == "mlx-community/plamo-2-translate"


def test_running_server_with_another_model_is_named(monkeypatch, capsys):
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_PRECISION", raising=False)
    monkeypatch.setattr(sys, "argv", ["plamo-translate", "--precision", "4bit", "--input", "hello"])
    running = {"backend_type": "mlx", "model_name": "mlx-community/plamo-2-translate", "precision": "4bit"}
    monkeypatch.setattr(main, "update_config", lambda **kwargs: running)
    monkeypatch.setattr(main, "check_server_running", lambda: True)
    with pytest.raises(SystemExit):
        main.main()
    assert "mlx: mlx-community/plamo-2-translate; requested mlx: pfnet/plamo-3-translate-2b" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "source"),
    [
        ("Hello.", "<|plamo:tag|>input<|plamo:msg|>Hello."),
        ("input lang=English\nHello.", "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello."),
    ],
)
def test_mcp_request_with_only_target_language_builds_plamo3_prompt(content, source):
    request = TranslateRequest(messages=[Message(role="user", content=content)], target_language="Japanese")
    assert pfml3_prompt(construct_llm_input(request)) == (
        "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation"
        f"{source}<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
    )


def test_plamo3_prompt_without_bos_for_runtimes_that_add_it():
    messages = [Message(role="user", content="input\nHello."), Message(role="user", content="output lang=Japanese\n")]
    with_bos = pfml3_prompt(messages)
    assert with_bos.startswith("<|plamo:bos|><|plamo:tag|>dataset")
    assert pfml3_prompt(messages, bos=False) == with_bos.removeprefix("<|plamo:bos|>")


@pytest.mark.parametrize(
    ("text", "source", "target"),
    [
        # Japanese naming commands, products or variables used to be echoed back in Japanese.
        ("PLaMo 3 Translate を試してみました。", "English|Japanese", "en"),
        ("uv tool install plamo-translate でインストールできます。", "English|Japanese", "en"),
        ("PLAMO_TRANSLATE_CLI_MODEL_NAME を設定してください。", "English|Japanese", "en"),
        ("GitHub の README を読んでください。", "English|Japanese", "en"),
        ("こんにちは", "English|Japanese", "en"),
        ("Dockerのcontainerをrestartする", "English|Japanese", "en"),
        ("A, B, C, D, E のうちどれ", "English|Japanese", "en"),
        ("npm install --save-dev typescript eslint prettier を実行", "English|Japanese", "en"),
        ("東京", "English|Japanese", "en"),
        # English naming or quoting Japanese stays English.
        ("How do you say ありがとう?", "English|Japanese", "ja"),
        ("The word ラーメン means ramen.", "English|Japanese", "ja"),
        ("Meeting with 山田部長 and 佐藤課長 tomorrow", "English|Japanese", "ja"),
        ("Ship to 東京都千代田区丸の内1-9-2 by Friday", "English|Japanese", "ja"),
        ("In Japanese, 'ありがとうございます' means thank you very much.", "English|Japanese", "ja"),
        ("Good night.", "English|Japanese", "ja"),
        # A source language given with --from decides; --to defaults to the other of English and Japanese.
        ("`llama-server --model model.gguf` を実行", "Japanese", "en"),
        ("東京", "English", "ja"),
        # Other or unrecognized source languages leave it to the text, as before.
        ("今日はいい天気ですね。", "日本語", "en"),
        ("Bonjour tout le monde.", "French", "ja"),
    ],
)
def test_plamo3_target_is_the_other_of_english_and_japanese(text, source, target):
    messages = [Message(role="user", content=f"input lang={source}\n{text}"), Message(role="user", content="output\n")]
    assert pfml3_prompt(messages).endswith(f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target}<|plamo:msg|>")


def test_plamo3_checkpoint_downloads_before_the_server_starts(monkeypatch, capsys):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    cached = set()
    calls = []

    def snapshot_download(name, allow_patterns, local_files_only=False):
        calls.append((name, local_files_only))
        if local_files_only and name not in cached:
            raise LocalEntryNotFoundError("not cached")
        cached.add(name)
        return "/cache"

    configs = {
        "pfnet/plamo-3-translate-2b": {"model_type": "plamo3"},
        "pfnet/plamo-3-translate-2b-MLX-4bit": {"model_type": "plamo3", "quantization": {"bits": 4}},
        "mlx-community/plamo-2-translate": {"model_type": "plamo2"},
    }
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", raising=False)
    files = [("model-00001-of-00002.safetensors", 4_960_000_000), ("config.json", 1_000), ("README.md", 40_000)]

    class HfApi:
        def model_info(self, name, files_metadata):
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=size) for n, size in files])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", HfApi)
    monkeypatch.setattr(main, "checkpoint_config", configs.get)
    monkeypatch.setattr(main, "checkpoint_complete", lambda folder: True)
    main.fetch_mlx_checkpoint("pfnet/plamo-3-translate-2b")
    assert calls == [("pfnet/plamo-3-translate-2b", True), ("pfnet/plamo-3-translate-2b", False)]
    # The size counts the files that are downloaded, not the model card.
    assert "Downloading pfnet/plamo-3-translate-2b (5.0 GB) into the Hugging Face cache" in capsys.readouterr().err
    # Cached checkpoints are not looked up online, and other checkpoints are left to the server.
    calls.clear()
    for name in [
        "pfnet/plamo-3-translate-2b",
        "pfnet/plamo-3-translate-2b-MLX-4bit",
        "mlx-community/plamo-2-translate",
    ]:
        main.fetch_mlx_checkpoint(name)
    assert all(local for _, local in calls)
    assert not capsys.readouterr().err


def test_server_downloads_the_default_checkpoint_before_starting(monkeypatch):
    fetched = []
    monkeypatch.setattr(sys, "argv", ["plamo-translate", "server"])
    monkeypatch.setattr(main, "mlx_available", lambda: True)
    monkeypatch.setattr(main, "update_config", lambda **kwargs: {})
    monkeypatch.setattr(main, "check_server_running", lambda: False)
    monkeypatch.setattr(main, "fetch_mlx_checkpoint", fetched.append)

    def start_mcp_server(*args):
        assert fetched == [main.DEFAULT_MODEL_NAME]
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "start_mcp_server", start_mcp_server)
    main.main()
    assert fetched == [main.DEFAULT_MODEL_NAME]


def test_config_uses_a_private_directory_without_tmpdir(monkeypatch, tmp_path):
    # Linux usually leaves TMPDIR unset.
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(utils, "gettempdir", lambda: str(tmp_path))
    assert utils.update_config(port=1) == {"port": 1}
    directory = tmp_path / f"plamo-translate-{os.getuid()}"
    assert (directory / "plamo-translate-config.json").is_file()
    assert directory.stat().st_mode & 0o777 == 0o700
    assert utils.update_config()["port"] == 1
    # A world-writable directory of this user is tightened.
    directory.chmod(0o777)
    utils.update_config()
    assert directory.stat().st_mode & 0o777 == 0o700
    # Another user could have planted a symlink or a file at the shared path first.
    directory.rename(tmp_path / "elsewhere")
    directory.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="set TMPDIR"):
        utils.update_config()
    directory.unlink()
    directory.write_text("")
    with pytest.raises(RuntimeError, match="set TMPDIR"):
        utils.update_config()


def test_partial_or_mistyped_checkpoints_are_handled_before_the_server(monkeypatch, tmp_path, capsys):
    import huggingface_hub

    # A cache holding only config.json, which huggingface_hub cannot always tell from a complete snapshot.
    partial = tmp_path / "snapshot"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    calls = []

    def snapshot_download(name, allow_patterns, local_files_only=False):
        calls.append(local_files_only)
        return str(partial)

    class HfApi:
        def model_info(self, name, files_metadata):
            raise OSError("offline")

    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", HfApi)
    monkeypatch.setattr(main, "checkpoint_config", lambda name: {"model_type": "plamo3"})
    main.fetch_mlx_checkpoint("pfnet/plamo-3-translate-2b")
    assert calls == [True, False]
    assert "Downloading pfnet/plamo-3-translate-2b into" in capsys.readouterr().err
    # A path that does not exist is not looked up as a repository.
    monkeypatch.undo()
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", raising=False)
    with pytest.raises(SystemExit, match="neither a checkpoint directory nor a Hugging Face repository"):
        main.fetch_mlx_checkpoint(str(tmp_path / "missing" / "plamo-3"))


def test_checkpoint_complete_needs_every_shard(tmp_path):
    from plamo_translate.servers.mlx.pfml3 import checkpoint_complete

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.jsonl").write_text("")
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "model-00001-of-00002.safetensors").write_text("")
    assert not checkpoint_complete(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").write_text("")
    assert checkpoint_complete(tmp_path)
    (tmp_path / "tokenizer.jsonl").unlink()
    assert not checkpoint_complete(tmp_path)
