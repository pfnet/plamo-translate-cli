import http.server
import multiprocessing
import os
import socket
import socketserver
import subprocess
import time

import pytest

from plamo_translate.main import check_server_running
from plamo_translate.servers.utils import PLAMO_TRANSLATE_CLI_SERVER_START_PORT, update_config

CLI_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_TIMEOUT_SECONDS", "20"))
SERVER_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("PLAMO_TRANSLATE_CLI_TEST_SERVER_STARTUP_TIMEOUT_SECONDS", "10"))


@pytest.fixture(autouse=True)
def isolated_test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER",
        os.environ.get("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER", "1"),
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path))


def wait_for_server_ready(timeout: int = SERVER_STARTUP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_server_running():
            return
        time.sleep(0.1)

    raise AssertionError("Timed out waiting for the MCP server to become ready.")


def wait_for_port_in_use(port: int, timeout: int = SERVER_STARTUP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)

    raise AssertionError(f"Timed out waiting for port {port} to start accepting connections.")


def test_update_config_without_kwargs_is_read_only(tmp_path):
    config_path = tmp_path / "plamo-translate-config.json"

    assert update_config() == {}
    assert not config_path.exists(), "Read-only access should not create the config file"

    initial_config = {"port": PLAMO_TRANSLATE_CLI_SERVER_START_PORT}
    update_config(**initial_config)
    initial_contents = config_path.read_text()

    assert update_config() == initial_config
    assert config_path.read_text() == initial_contents, "Read-only access should not rewrite the config file"


def stop_subprocess(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def stop_multiprocess(process: multiprocessing.Process | None) -> None:
    if process is None:
        return

    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def test_plamo_translate_without_server():
    text_to_translate = "Proud, but humble"
    command = ["plamo-translate", "--from", "English", "--to", "Japanese", "--input", text_to_translate]
    result = subprocess.run(command, capture_output=True, text=True, timeout=CLI_TIMEOUT_SECONDS)
    assert result.returncode == 0
    assert "誇り高" in result.stdout and "謙虚" in result.stdout


def test_plamo_translate_server_simple_use():
    first_process = None
    try:
        command = ["plamo-translate", "server"]
        first_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        wait_for_server_ready()

        config = update_config()
        print(f"Server started with config: {config}")
        assert "port" in config, "Server configuration should include a port"
        port = config["port"]
        assert port == PLAMO_TRANSLATE_CLI_SERVER_START_PORT, f"Expected server port to be 8000, got {port}"

        text_to_translate = "Proud, but humble"
        result = subprocess.run(
            ["plamo-translate", "--input", text_to_translate, "--from", "English", "--to", "Japanese"],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )
        assert "誇り高い" in result.stdout and "謙虚" in result.stdout

        result = subprocess.run(
            ["plamo-translate", "--from", "English", "--to", "Japanese"],
            input=text_to_translate,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )
        assert "誇り高い" in result.stdout and "謙虚" in result.stdout
    finally:
        stop_subprocess(first_process)


def test_plamo_translate_server_already_running():
    first_process = None
    second_process = None
    try:
        command = ["plamo-translate", "server"]
        first_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Starting first plamo-translate server process...")
        wait_for_server_ready()
        print("First server process started successfully.")

        # If the server is already running, the further call of `plamo-translate server` should not start a new server.
        second_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = second_process.communicate(timeout=CLI_TIMEOUT_SECONDS)
        print(stdout.strip())
        assert "MCP server is already running" in stdout
        config = update_config()
        print(f"Server started with config: {config}")
        assert "port" in config, "Server configuration should include a port"
        port = config["port"]
        assert port == PLAMO_TRANSLATE_CLI_SERVER_START_PORT, f"Expected server port to be 8000, got {port}"
    finally:
        stop_subprocess(first_process)
        stop_subprocess(second_process)


def start_http_server():
    port = PLAMO_TRANSLATE_CLI_SERVER_START_PORT
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        httpd.serve_forever()


def test_plamo_translate_server_find_new_port():
    http_server_process = None
    mcp_server_process = None
    try:
        http_server_process = multiprocessing.Process(target=start_http_server, daemon=True)
        http_server_process.start()
        print(f"HTTP server started on port {PLAMO_TRANSLATE_CLI_SERVER_START_PORT}")
        wait_for_port_in_use(PLAMO_TRANSLATE_CLI_SERVER_START_PORT)

        # The default port is used by the HTTP server, so the MCP server should use a different port
        command = ["plamo-translate", "server"]
        mcp_server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Starting plamo-translate server...")
        wait_for_server_ready()
        stop_subprocess(mcp_server_process)
        mcp_server_process = None

        config = update_config()
        print(f"Server started with config: {config}")
        assert "port" in config, "Server configuration should include a port"
        port = config["port"]
        assert port == PLAMO_TRANSLATE_CLI_SERVER_START_PORT + 1, (
            f"Expected server port to be {PLAMO_TRANSLATE_CLI_SERVER_START_PORT + 1}, got {port}"
        )
    finally:
        stop_multiprocess(http_server_process)
        stop_subprocess(mcp_server_process)


def test_plamo_translate_server_interactive():
    mcp_server_process = None
    client_process = None
    try:
        command = ["plamo-translate", "server"]
        mcp_server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        wait_for_server_ready()
        config = update_config()
        print(f"Server started with config: {config}")
        assert "port" in config, "Server configuration should include a port"
        port = config["port"]
        assert port == PLAMO_TRANSLATE_CLI_SERVER_START_PORT, (
            f"Expected server port to be {PLAMO_TRANSLATE_CLI_SERVER_START_PORT}, got {port}"
        )

        client_command = ["plamo-translate", "-i", "--from", "English", "--to", "Japanese"]
        client_process = subprocess.Popen(
            client_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        all_inputs = "\n".join(["Proud, but humble", "Boldly do what no one has done before"]) + "\n"

        stdout, stderr = client_process.communicate(input=all_inputs, timeout=CLI_TIMEOUT_SECONDS)
        assert "誇り高" in stdout and "謙虚" in stdout
        assert "大胆に" in stdout
    finally:
        stop_subprocess(mcp_server_process)
        stop_subprocess(client_process)
