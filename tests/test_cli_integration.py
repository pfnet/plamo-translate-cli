import os
import subprocess
import sys
import time

import pytest

from plamo_translate.main import check_server_running
from plamo_translate.servers.utils import PLAMO_TRANSLATE_CLI_SERVER_START_PORT, update_config

CLI_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_TIMEOUT_SECONDS", "900"))
SERVER_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_SERVER_STARTUP_TIMEOUT_SECONDS", "900"))


@pytest.fixture(autouse=True)
def integration_test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER",
        os.environ.get("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", "0"),
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path))


def wait_for_server_ready(timeout: int = SERVER_STARTUP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_server_running():
            return
        time.sleep(0.5)

    raise AssertionError("Timed out waiting for the MCP server to become ready.")


def stop_subprocess(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_plamo_translate_server_roundtrip_with_real_model():
    server_process = None
    try:
        server_process = subprocess.Popen(
            ["plamo-translate", "server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_server_ready()

        config = update_config()
        assert config.get("port") == PLAMO_TRANSLATE_CLI_SERVER_START_PORT

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
    model_name = "mlx-community/plamo-2-translate"
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", "0")
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", model_name)
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_TEMP", "0.0")
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_REPETITION_PENALTY", raising=False)
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE", raising=False)

    server_process = None
    try:
        server_process = subprocess.Popen(
            ["plamo-translate", "server", "--precision", "4bit"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_server_ready()
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
