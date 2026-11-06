import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from plamo_translate.main import check_server_running
from plamo_translate.servers.utils import DEFAULT_MODEL_NAME, PLAMO_TRANSLATE_CLI_SERVER_START_PORT, update_config

CLI_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_TIMEOUT_SECONDS", "900"))
SERVER_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_SERVER_STARTUP_TIMEOUT_SECONDS", "900"))


@pytest.fixture(autouse=True)
def integration_test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER",
        os.environ.get("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", "0"),
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path))


def start_server(log_path: Path, *arguments: str) -> subprocess.Popen:
    """Write the server's output to a file: an unread pipe would block the server once it fills."""
    with open(log_path, "w") as log:
        return subprocess.Popen(
            ["plamo-translate", "server", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            # A timed-out server is aborted below: print where each thread was, and keep what it printed before.
            env={**os.environ, "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"},
        )


def wait_for_server_ready(
    process: subprocess.Popen, log_path: Path, timeout: int = SERVER_STARTUP_TIMEOUT_SECONDS
) -> None:
    deadline = time.monotonic() + timeout
    while not check_server_running():
        assert process.poll() is None, f"The server exited with {process.returncode}:\n{log_path.read_text()}"
        if time.monotonic() >= deadline:
            process.send_signal(signal.SIGABRT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise AssertionError(f"Timed out waiting for the MCP server to become ready:\n{log_path.read_text()}")
        time.sleep(0.5)


def stop_subprocess(process: subprocess.Popen | None) -> None:
    if process is None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_plamo_translate_server_roundtrip_with_real_model(monkeypatch, tmp_path):
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    log_path = tmp_path / "server.log"
    server_process = None
    try:
        server_process = start_server(log_path)
        wait_for_server_ready(server_process, log_path)

        config = update_config()
        assert config.get("port") == PLAMO_TRANSLATE_CLI_SERVER_START_PORT
        assert config.get("model_name") == DEFAULT_MODEL_NAME

        text_to_translate = "Proud, but humble"
        result = subprocess.run(
            ["plamo-translate", "--input", text_to_translate, "--from", "English", "--to", "Japanese"],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0
        assert "誇り高" in result.stdout and "謙虚" in result.stdout

        result = subprocess.run(
            ["plamo-translate", "--from", "English", "--to", "Japanese"],
            input=text_to_translate,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0
        assert "誇り高" in result.stdout and "謙虚" in result.stdout
    finally:
        stop_subprocess(server_process)


def test_plamo_translate_4bit_completions_issue_10(monkeypatch, tmp_path):
    """The reported short word must translate, including across interactive turns."""
    model_name = DEFAULT_MODEL_NAME
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", "0")
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", model_name)
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_TEMP", "0.0")
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_REPETITION_PENALTY", raising=False)
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE", raising=False)

    log_path = tmp_path / "server.log"
    server_process = None
    try:
        server_process = start_server(log_path, "--precision", "4bit")
        wait_for_server_ready(server_process, log_path)
        assert update_config().get("model_name") == model_name

        result = subprocess.run(
            ["plamo-translate", "--from", "English", "--to", "Japanese", "--input", "completions"],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "完了", result.stdout

        # Exercise the real CLI while keeping readline history inside this test's
        # temporary directory, including the history write registered with atexit.
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
            [sys.executable, str(entrypoint), "--interactive", "--from", "English", "--to", "Japanese"],
            input="completions\n" * 3,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, result.stderr
        responses = [line.removeprefix("> ").strip() for line in result.stdout.splitlines() if line.startswith("> ")]
        assert responses[:3] == ["完了"] * 3, result.stdout
        assert "Ctrl+D received. Exiting." in result.stdout, result.stdout
    finally:
        stop_subprocess(server_process)
