"""Real-model tests for the llama.cpp backend.

They download pfnet/plamo-3-translate-2b-GGUF (Q4_K_M, 1.8 GB) and the 0.8B draft into the Hugging Face cache, and
need llama-server on PATH or in PLAMO_TRANSLATE_CLI_LLAMA_SERVER.
"""

import os
import platform
import signal
import subprocess
import sys
import time

import pytest

from plamo_translate.main import check_server_running
from plamo_translate.servers.llama_cpp.model import DEFAULT_GGUF_MODEL
from plamo_translate.servers.utils import update_config

pytestmark = pytest.mark.skipif(
    os.environ.get("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER") == "1", reason="needs llama-server and the real model"
)

CLI_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_TIMEOUT_SECONDS", "900"))
SERVER_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_SERVER_STARTUP_TIMEOUT_SECONDS", "900"))
LLAMA_CPP = ["--backend-type", "llama.cpp"]


@pytest.fixture(autouse=True)
def integration_test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", "0")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    for name in [
        "PLAMO_TRANSLATE_CLI_MODEL_NAME",
        "PLAMO_TRANSLATE_CLI_TEMP",
        "PLAMO_TRANSLATE_CLI_REPETITION_PENALTY",
        "PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE",
    ]:
        monkeypatch.delenv(name, raising=False)


def translate(*arguments: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["plamo-translate", *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def stop(process: subprocess.Popen) -> None:
    # Ctrl+C, so that the server stops llama-server as it does for users.
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_one_shot_translation_with_default_gguf():
    output = translate("--input", "Proud, but humble", "--from", "English", "--to", "Japanese", *LLAMA_CPP)
    assert "誇り高" in output and "謙虚" in output, output
    # Issue #10: a single word must translate, not be continued.
    assert translate("--input", "completions", "--from", "English", "--to", "Japanese", *LLAMA_CPP) == "完了"
    # Standard input, with the English/Japanese direction chosen from the source.
    output = translate(*LLAMA_CPP, stdin="誇り高く、しかし謙虚に")
    assert "proud" in output.lower() and "humble" in output.lower(), output


def test_server_with_default_draft_serves_plain_clients(tmp_path):
    log_path = tmp_path / "server.log"
    with open(log_path, "w") as log:
        server = subprocess.Popen(["plamo-translate", "server", *LLAMA_CPP], stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_SECONDS
        while not check_server_running():
            assert server.poll() is None, log_path.read_text()
            assert time.monotonic() < deadline, "Timed out waiting for the MCP server to become ready."
            time.sleep(0.5)
        config = update_config()
        assert config.get("backend_type") == "llama.cpp"
        assert config.get("model_name") == DEFAULT_GGUF_MODEL

        # Clients find the running backend from the shared config without --backend-type.
        output = translate("--input", "Proud, but humble", "--from", "English", "--to", "Japanese")
        assert "誇り高" in output and "謙虚" in output, output
        output = translate(stdin="誇り高く、しかし謙虚に")
        assert "proud" in output.lower() and "humble" in output.lower(), output
    finally:
        stop(server)
    log = log_path.read_text()
    assert "Uvicorn running on" in log, log  # The server's standard error, where both messages below go.
    assert "Continuing without the speculative-decoding draft" not in log, log
    assert "did not enable the draft model" not in log, log


def test_interactive_mode_with_default_draft(tmp_path):
    # Keep readline history inside this test's temporary directory, including the write registered with atexit.
    entrypoint = tmp_path / "interactive_cli.py"
    entrypoint.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "import plamo_translate.main as cli\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    cli.Path = SimpleNamespace(home=lambda: Path(os.environ['TMPDIR']))\n"
        "    cli.main()\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(entrypoint), "--interactive", "--from", "English", "--to", "Japanese", *LLAMA_CPP],
        input="completions\n" * 3,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr
    responses = [line.removeprefix("> ").strip() for line in result.stdout.splitlines() if line.startswith("> ")]
    assert responses[:3] == ["完了"] * 3, result.stdout
    assert "Continuing without the speculative-decoding draft" not in result.stderr, result.stderr


@pytest.mark.skipif(sys.platform == "darwin" and platform.machine() == "arm64", reason="Apple Silicon defaults to MLX")
def test_machines_without_mlx_default_to_llama_cpp_without_tmpdir(monkeypatch, tmp_path):
    # Linux usually leaves TMPDIR unset; the CLI keeps its state in a private directory instead.
    monkeypatch.delenv("TMPDIR")
    output = translate("--input", "PLaMo 3 Translate を試してみました。")
    assert "PLaMo 3 Translate" in output and "tried" in output.lower(), output
