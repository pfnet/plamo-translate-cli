import asyncio
import json
import subprocess
import sys
from unittest.mock import AsyncMock

import httpx
import pytest

from plamo_translate.main import print_translation, wait_for_server_ready
from plamo_translate.servers.llama_cpp.server import (
    LlamaCppOptions,
    LlamaCppRuntime,
    PLaMoTranslateServer,
    build_prompt,
    completion_events,
    completion_payload,
    validate_model_log,
)
from plamo_translate.servers.utils import Message, TranslateRequest


def request():
    return TranslateRequest(
        messages=[Message(role="user", content="Hello.")], source_language="English", target_language="Japanese"
    )


def test_prompt_preserves_template_and_does_not_mutate_request():
    original = request()
    expected = (
        "<|plamo:op|>dataset\ntranslation\n<|plamo:op|>input lang=English\nHello.\n<|plamo:op|>output lang=Japanese\n"
    )
    assert build_prompt(original) == expected
    assert build_prompt(original) == expected
    assert original.messages[0].content == "Hello."
    assert len(original.messages) == 1
    preformatted = TranslateRequest(
        messages=[
            Message(role="user", content="input lang=English\nHello."),
            Message(role="user", content="output lang=Japanese\n"),
        ]
    )
    assert build_prompt(preformatted) == expected
    payload = completion_payload(original)
    assert payload["temperature"] == 0
    assert payload["repeat_penalty"] == 1
    assert payload["cache_prompt"] is False
    assert "<|plamo:op|>" in payload["stop"]


def test_unpatched_rope_loader_is_rejected():
    with pytest.raises(RuntimeError, match="disables PLaMo 2 RoPE"):
        validate_model_log("print_info: n_rot                 = 0\n")
    validate_model_log("print_info: n_rot                 = 128\n")


def transport(events, status=200):
    content = "\n\n".join("data: " + json.dumps(event, ensure_ascii=False) for event in events) + "\n\n"
    return httpx.MockTransport(lambda req: httpx.Response(status, text=content))


def test_sse_unicode_and_terminal_content():
    async def run():
        async with httpx.AsyncClient(
            transport=transport(
                [
                    {"content": "こんにちは", "stop": False},
                    {"content": "。", "stop": True, "stop_type": "word", "truncated": False},
                ]
            )
        ) as client:
            return [e async for e in completion_events(client, "http://test", {})]

    assert "".join(e["content"] for e in asyncio.run(run())) == "こんにちは。"


@pytest.mark.parametrize(
    "events,status,error",
    [
        ([{"stop": True, "stop_type": "limit"}], 200, "truncated"),
        ([{"stop": True, "truncated": True}], 200, "truncated"),
        ([{"content": "partial", "stop": False}], 200, "disconnected"),
        ([{"error": {"message": "bad request"}}], 200, "bad request"),
        ([], 500, "HTTP 500"),
    ],
)
def test_completion_errors_are_not_success(events, status, error):
    async def run():
        async with httpx.AsyncClient(transport=transport(events, status)) as client:
            return [e async for e in completion_events(client, "http://test", {})]

    with pytest.raises(RuntimeError, match=error):
        asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_mcp_translation_stream_and_batch(monkeypatch, tmp_path, stream):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(LlamaCppRuntime, "start", lambda self: None)
    server = PLaMoTranslateServer("ERROR", LlamaCppOptions("model.gguf"))
    server.runtime.url = "http://test"
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "plamo_translate.servers.llama_cpp.server.httpx.AsyncClient",
        lambda **kwargs: client_type(
            transport=transport(
                [{"content": "こんにちは", "stop": False}, {"content": "。", "stop": True, "stop_type": "eos"}]
            )
        ),
    )
    context = AsyncMock()
    result = asyncio.run(server.translate(request(), stream, context))
    if stream:
        assert result == ""
        assert "".join(call.kwargs["message"] for call in context.report_progress.call_args_list) == "こんにちは。"
    else:
        assert result == "こんにちは。"
        context.report_progress.assert_not_called()


def test_runtime_reports_startup_failure_and_reaps_child(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    executable = tmp_path / "llama-server"
    executable.write_text("#!/bin/sh\necho 'invalid GGUF' >&2\nexit 23\n")
    executable.chmod(0o755)
    runtime = LlamaCppRuntime(LlamaCppOptions(str(model), str(executable), startup_timeout=2))
    with pytest.raises(RuntimeError, match="invalid GGUF"):
        runtime.start()
    assert runtime.process is None
    assert runtime.log is None


def test_runtime_shutdown_only_terminates_owned_child():
    runtime = LlamaCppRuntime(LlamaCppOptions("unused.gguf"))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    runtime.process = process
    runtime.close()
    runtime.close()
    assert process.poll() is not None


def test_mcp_port_selected_after_native_load_and_failure_cleans_up(monkeypatch):
    loaded = []
    closed = []
    monkeypatch.setattr(LlamaCppRuntime, "start", lambda self: loaded.append(True))
    monkeypatch.setattr(LlamaCppRuntime, "close", lambda self: closed.append(True))

    def no_port():
        assert loaded == [True]
        raise RuntimeError("No free MCP port")

    monkeypatch.setattr("plamo_translate.servers.llama_cpp.server.find_free_port", no_port)
    with pytest.raises(RuntimeError, match="No free MCP port"):
        PLaMoTranslateServer("ERROR", LlamaCppOptions("model.gguf"))
    assert closed == [True]


def test_dead_mcp_child_fails_without_waiting(monkeypatch):
    monkeypatch.setattr("plamo_translate.main.check_server_running", lambda: False)

    class DeadProcess:
        exitcode = 23

        def is_alive(self):
            return False

    with pytest.raises(RuntimeError, match="exit code 23"):
        wait_for_server_ready(DeadProcess(), timeout=30)


@pytest.mark.parametrize("stream", [True, False])
def test_interactive_history_retains_translation(capsys, stream):
    class Client:
        async def translate(self, messages):
            yield "こんにちは。"

    messages = [{"role": "user", "content": "output lang=Japanese\n"}]
    result = asyncio.run(print_translation(Client(), messages, stream))
    assert result[-1]["content"] == "output lang=Japanese\nこんにちは。"
    assert capsys.readouterr().out == "こんにちは。"


def test_cli_rejects_missing_gguf_without_hanging(tmp_path):
    import os

    env = dict(os.environ, TMPDIR=str(tmp_path))
    env.pop("PLAMO_TRANSLATE_CLI_MODEL_NAME", None)
    result = subprocess.run(
        [sys.executable, "-m", "plamo_translate.main", "server", "--backend-type", "llama.cpp"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "requires --model" in result.stderr
    assert not (tmp_path / "plamo-translate-config.json").exists()


def test_running_backend_identity_is_not_overwritten(monkeypatch, tmp_path):
    import importlib

    from plamo_translate.servers.utils import update_config

    main = importlib.import_module("plamo_translate.main")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    original = update_config(backend_type="mlx", model_name="existing", port=30000)
    monkeypatch.setattr(main, "check_server_running", lambda: True)
    monkeypatch.setattr(sys, "argv", ["plamo-translate", "--backend-type", "llama.cpp", "--model", "/tmp/model.gguf"])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2
    assert update_config() == original


def test_plain_client_reuses_running_llama_backend(monkeypatch, tmp_path):
    import importlib

    from plamo_translate.servers.utils import update_config

    main = importlib.import_module("plamo_translate.main")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    update_config(backend_type="llama.cpp", model_name="/tmp/model.gguf", port=30000)
    monkeypatch.setattr(main, "check_server_running", lambda: True)
    monkeypatch.setattr(sys, "argv", ["plamo-translate", "--input", "hello"])
    received = []
    monkeypatch.setattr(main, "run_translate", received.append)
    main.main()
    assert received[0].backend_type == "llama.cpp"


@pytest.mark.parametrize("stream", [False, True])
def test_mcp_tool_errors_reach_client(monkeypatch, tmp_path, stream):
    from contextlib import asynccontextmanager

    from mcp.types import CallToolResult, TextContent

    from plamo_translate.clients import translate
    from plamo_translate.servers.utils import update_config

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    update_config(port=30000)
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(isError=True, content=[TextContent(type="text", text="truncated")])

    @asynccontextmanager
    async def connection(*args, **kwargs):
        yield None, None, None

    @asynccontextmanager
    async def connect_session(*args, **kwargs):
        yield session

    monkeypatch.setattr(translate, "streamablehttp_client", connection)
    monkeypatch.setattr(translate, "ClientSession", connect_session)

    async def run():
        return [chunk async for chunk in translate.MCPClient(stream).translate([{"role": "user", "content": "hello"}])]

    with pytest.raises(RuntimeError, match="truncated"):
        asyncio.run(run())
