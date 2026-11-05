import importlib.resources
import os
import sys
from unittest import mock

import pytest
from jinja2 import Environment

from plamo_translate import main
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


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        (None, "pfnet/plamo-3-translate-2b-MLX-4bit"),
        ("4bit", "pfnet/plamo-3-translate-2b-MLX-4bit"),
        ("6bit", "pfnet/plamo-3-translate-2b"),
        ("8bit", "pfnet/plamo-3-translate-2b"),
        ("mixed", "pfnet/plamo-3-translate-2b"),
        ("bf16", "pfnet/plamo-3-translate-2b"),
    ],
)
def test_default_is_the_mlx_release_unless_another_precision_needs_original_weights(monkeypatch, precision, expected):
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_PRECISION", raising=False)
    options = [] if precision is None else ["--precision", precision]
    monkeypatch.setattr(sys, "argv", ["plamo-translate", *options, "--input", "hello"])
    monkeypatch.setattr(main, "update_config", lambda **kwargs: {})
    monkeypatch.setattr(main, "check_server_running", lambda: False)
    monkeypatch.setattr(main, "run_translate", lambda args: None)
    main.main()
    assert main.os.environ["PLAMO_TRANSLATE_CLI_MODEL_NAME"] == expected
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
    assert "mlx: mlx-community/plamo-2-translate; requested mlx: pfnet/plamo-3-translate-2b-MLX-4bit" in capsys.readouterr().err


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
