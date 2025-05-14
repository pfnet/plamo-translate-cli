import http.server
import multiprocessing
import socketserver
import subprocess
import tempfile

from plamo_translate.servers.utils import PLAMO_TRANSLATE_CLI_SERVER_START_PORT, update_config


def test_plamo_translate_without_server():
    text_to_translate = "Proud, but humble"
    command = ["plamo-translate", "--from", "English", "--to", "Japanese", "--input", text_to_translate]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0  # This will likely fail without a model
    assert "誇り高" in result.stdout and "謙虚" in result.stdout


def test_plamo_translate_server_simple_use():
    first_process = None
    try:
        command = ["plamo-translate", "server"]
        first_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            if first_process.stderr is not None:
                line = first_process.stderr.readline()
                if "Application startup complete" in line.strip():
                    break

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
        )
        assert "誇り高い" in result.stdout and "謙虚" in result.stdout

        result = subprocess.run(
            ["plamo-translate", "--from", "English", "--to", "Japanese"],
            input=text_to_translate,
            capture_output=True,
            text=True,
        )
        assert "誇り高い" in result.stdout and "謙虚" in result.stdout
    finally:
        if first_process:
            first_process.terminate()


def test_plamo_translate_server_already_running():
    first_process = None
    second_process = None
    try:
        command = ["plamo-translate", "server"]
        first_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Starting first plamo-translate server process...")
        while True:
            if first_process.stderr is not None:
                line = first_process.stderr.readline()
                print(line.strip())
                if "Application startup complete" in line.strip():
                    break
        print("First server process started successfully.")

        # If the server is already running, the further call of `plamo-translate server` should not start a new server.
        second_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            if second_process.stdout is not None:
                line = second_process.stdout.readline()
                print(line.strip())
                if "MCP server is already running" in line.strip():
                    break
        config = update_config()
        print(f"Server started with config: {config}")
        assert "port" in config, "Server configuration should include a port"
        port = config["port"]
        assert port == PLAMO_TRANSLATE_CLI_SERVER_START_PORT, f"Expected server port to be 8000, got {port}"
    finally:
        if first_process:
            first_process.terminate()
        if second_process:
            second_process.terminate()


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

        # The default port is used by the HTTP server, so the MCP server should use a different port
        command = ["plamo-translate", "server"]
        mcp_server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Starting plamo-translate server...")
        while True:
            print("Waiting for plamo-translate server to start...")
            if mcp_server_process.stderr is not None:
                line = mcp_server_process.stderr.readline()
                print(line.strip())
                if "Application startup complete" in line.strip():
                    break
        mcp_server_process.terminate()

        config = update_config()
        print(f"Server started with config: {config}")
        assert "port" in config, "Server configuration should include a port"
        port = config["port"]
        assert port == PLAMO_TRANSLATE_CLI_SERVER_START_PORT + 1, (
            f"Expected server port to be {PLAMO_TRANSLATE_CLI_SERVER_START_PORT + 1}, got {port}"
        )
    finally:
        if http_server_process:
            http_server_process.terminate()
        if mcp_server_process:
            mcp_server_process.terminate()


def test_plamo_translate_server_interactive():
    mcp_server_process = None
    client_process = None
    try:
        command = ["plamo-translate", "server"]
        mcp_server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            if mcp_server_process.stderr is not None:
                line = mcp_server_process.stderr.readline()
                if "Application startup complete" in line.strip():
                    break
        config = update_config()
        print(f"Server started with config: {config}")
        assert "port" in config, "Server configuration should include a port"
        port = config["port"]
        assert port == PLAMO_TRANSLATE_CLI_SERVER_START_PORT, (
            f"Expected server port to be {PLAMO_TRANSLATE_CLI_SERVER_START_PORT}, got {port}"
        )

        client_command = ["plamo-translate", "--from", "English", "--to", "Japanese"]
        client_process = subprocess.Popen(
            client_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        all_inputs = "\n".join(["Proud, but humble", "Boldly do what no one has done before"]) + "\n"

        stdout, stderr = client_process.communicate(input=all_inputs)
        stdout_lines = stdout.strip().split("\n")
        assert "誇り高" in stdout_lines[0] and "謙虚" in stdout_lines[0]
        assert "大胆に" in stdout_lines[1]
    finally:
        if mcp_server_process:
            mcp_server_process.terminate()
        if client_process:
            client_process.terminate()
