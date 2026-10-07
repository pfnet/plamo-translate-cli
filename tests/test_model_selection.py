import importlib.resources
import sys

import pytest
from jinja2 import Environment

from plamo_translate import main


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
