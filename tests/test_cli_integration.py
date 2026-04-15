import os
import subprocess
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
