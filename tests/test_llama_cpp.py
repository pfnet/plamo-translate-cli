import asyncio
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock

import httpx
import pytest

from plamo_translate import main as cli
from plamo_translate.main import print_translation, wait_for_server_ready
from plamo_translate.servers.llama_cpp.model import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_GGUF_MODEL,
    canonical_model,
    gguf_architecture,
    resolve_model,
    same_model,
)
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

PLAMO3_PROMPT = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en"
    "<|plamo:msg|>Hello.<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
)


@pytest.fixture(autouse=True)
def isolated_environment():
    # main() exports the selected model; keep it out of later tests.
    with mock.patch.dict(os.environ):
        yield


def request(text="Hello.", source="English", target="Japanese"):
    return TranslateRequest(
        messages=[Message(role="user", content=text)], source_language=source, target_language=target
    )


def gguf_string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path, architecture="plamo3", before=()):
    """A header-only GGUF; `before` holds (key, value type, encoded value) entries placed ahead of the architecture."""
    entries = [*before, ("general.architecture", 8, gguf_string(architecture))]
    body = b"".join(gguf_string(key) + struct.pack("<I", kind) + value for key, kind, value in entries)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + body)
    return path


def test_prompt_preserves_template_and_does_not_mutate_request():
    original = request()
    expected = (
        "<|plamo:op|>dataset\ntranslation\n<|plamo:op|>input lang=English\nHello.\n<|plamo:op|>output lang=Japanese\n"
    )
    assert build_prompt(original, architecture="plamo2") == expected
    assert build_prompt(original, architecture="plamo2") == expected
    assert original.messages[0].content == "Hello."
    assert len(original.messages) == 1
    preformatted = TranslateRequest(
        messages=[
            Message(role="user", content="input lang=English\nHello."),
            Message(role="user", content="output lang=Japanese\n"),
        ]
    )
    assert build_prompt(preformatted, architecture="plamo2") == expected
    payload = completion_payload(original, architecture="plamo2")
    assert payload["temperature"] == 0
    assert payload["repeat_penalty"] == 1
    assert payload["cache_prompt"] is False
    assert "<|plamo:op|>" in payload["stop"]
    # Greedy decoding skips the CPU-side nucleus and min-p filters, which cannot change the argmax.
    assert payload["top_p"] == 1.0
    assert payload["min_p"] == 0.0


def test_plamo3_prompt_lets_llama_cpp_add_bos():
    original = request()
    assert build_prompt(original, architecture="plamo3") == PLAMO3_PROMPT
    assert original.messages[0].content == "Hello."
    payload = completion_payload(original, architecture="plamo3")
    assert payload["prompt"] == PLAMO3_PROMPT
    # The GGUF marks <|plamo:tag|>, <|plamo:bos|> and <|plamo:eos|> as end-of-generation tokens.
    assert payload["stop"] == []
    preformatted = TranslateRequest(
        messages=[
            Message(role="user", content="input lang=English\nHello."),
            Message(role="user", content="output lang=Japanese\n"),
        ]
    )
    assert build_prompt(preformatted, architecture="plamo3") == PLAMO3_PROMPT


def test_plamo3_prompt_chooses_direction_and_rejects_control_tokens():
    automatic = build_prompt(request("こんにちは。", "English|Japanese", ""), architecture="plamo3")
    assert automatic.endswith("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>")
    assert "<|plamo:tag|>input<|plamo:msg|>こんにちは。" in automatic
    with pytest.raises(ValueError, match="reserved"):
        build_prompt(request("a <|plamo:tag|> b"), architecture="plamo3")
    with pytest.raises(ValueError, match="Unsupported"):
        build_prompt(request(), architecture="llama")


def test_gguf_architecture_skips_every_value_type(tmp_path):
    before = [
        ("u8", 0, b"\x01"),
        ("i16", 3, struct.pack("<h", -2)),
        ("f32", 6, struct.pack("<f", 1.5)),
        ("bool", 7, b"\x01"),
        ("name", 8, gguf_string("general.architecture")),
        ("u64", 10, struct.pack("<Q", 7)),
        ("f64", 12, struct.pack("<d", 2.5)),
        ("tokens", 9, struct.pack("<IQ", 8, 2) + gguf_string("<|plamo:tag|>") + gguf_string("こんにちは")),
        ("scores", 9, struct.pack("<IQ", 6, 3) + struct.pack("<3f", 0, 1, 2)),
        ("nested", 9, struct.pack("<IQ", 9, 1) + struct.pack("<IQ", 4, 2) + struct.pack("<2I", 1, 2)),
    ]
    assert gguf_architecture(str(write_gguf(tmp_path / "plamo3.gguf", before=before))) == "plamo3"
    assert gguf_architecture(str(write_gguf(tmp_path / "plamo2.gguf", "plamo2"))) == "plamo2"


@pytest.mark.parametrize("content", [b"", b"GGML" + bytes(20), b"GGUF" + struct.pack("<IQQ", 1, 0, 0)])
def test_gguf_architecture_rejects_other_files(tmp_path, content):
    path = tmp_path / "model.gguf"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        gguf_architecture(str(path))


def test_gguf_architecture_rejects_truncated_header(tmp_path):
    path = write_gguf(tmp_path / "model.gguf")
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(ValueError, match="complete"):
        gguf_architecture(str(path))


def test_model_names_are_canonical_without_network(tmp_path):
    assert canonical_model("pfnet/plamo-3-translate-2b-GGUF") == DEFAULT_GGUF_MODEL
    assert canonical_model("pfnet/plamo-3-translate-2b-GGUF:q4_0") == "pfnet/plamo-3-translate-2b-GGUF:Q4_0"
    assert canonical_model("user/repo:Model-Q4_0.gguf") == "user/repo:Model-Q4_0.gguf"
    assert canonical_model("missing.gguf") == str(Path("missing.gguf").absolute())
    assert canonical_model("~/missing.gguf") == str(Path.home() / "missing.gguf")
    with pytest.raises(ValueError, match="neither"):
        canonical_model("not a model")
    assert same_model(DEFAULT_GGUF_MODEL, "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M")
    assert not same_model(DEFAULT_GGUF_MODEL, "pfnet/plamo-3-translate-2b-GGUF:Q4_0")


def test_hugging_face_cache_symlink_is_accepted(tmp_path):
    # Snapshot files link to blobs without a suffix, so the path must not be resolved before checking it.
    blob = write_gguf(tmp_path / "blob")
    link = tmp_path / "model-Q4_K_M.gguf"
    link.symlink_to(blob)
    assert canonical_model(str(link)) == str(link)
    command = LlamaCppOptions(str(link), sys.executable).command(8080)
    assert command[command.index("--model") + 1] == str(link)


def test_repository_quantization_selects_one_file(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ["m-Q4_0.gguf", "m-IQ4_0.gguf", "m-Q4_K_M.gguf", "m-Q8_0-00001-of-00002.gguf"]:
        (snapshot / name).touch()
    patterns = []

    def snapshot_download(repository, allow_patterns, local_files_only):
        patterns.append(allow_patterns)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert resolve_model("user/repo:q4_0") == str(snapshot / "m-Q4_0.gguf")
    assert patterns == [["*Q4_0.gguf"]]
    assert resolve_model("user/repo") == str(snapshot / "m-Q4_K_M.gguf")
    with pytest.raises(ValueError, match="found none"):
        resolve_model("user/repo:Q8_0")
    with pytest.raises(FileNotFoundError):
        resolve_model("user/repo:Q2_K", local_only=True)
    assert resolve_model(str(snapshot / "m-Q4_0.gguf")) == str(snapshot / "m-Q4_0.gguf")


def test_command_tunes_server_and_adds_draft(tmp_path):
    model = write_gguf(tmp_path / "model.gguf")
    draft = write_gguf(tmp_path / "draft.gguf")
    command = LlamaCppOptions(str(model), sys.executable, draft_model=str(draft)).command(8080)
    options = dict(zip(command[1::2], command[2::2]))
    assert options["--threads"] == "2"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("--cache-ram") + 1] == "0"
    assert command[command.index("--ctx-checkpoints") + 1] == "0"
    assert command[command.index("--chat-template") + 1] == "chatml"
    assert command[command.index("--model-draft") + 1] == str(draft)
    assert command[command.index("--spec-type") + 1] == "draft-simple"
    assert command[command.index("--gpu-layers-draft") + 1] == "99"
    cpu = LlamaCppOptions(str(model), sys.executable, gpu_layers=0).command(8080)
    assert cpu[cpu.index("--threads") + 1] == "8"
    assert "--model-draft" not in cpu
    with pytest.raises(ValueError, match="--draft-model"):
        LlamaCppOptions(str(model), sys.executable, draft_model=str(tmp_path / "missing.gguf")).command(8080)


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


@pytest.mark.parametrize("architecture", ["plamo2", "plamo3"])
@pytest.mark.parametrize("stream", [False, True])
def test_mcp_translation_stream_and_batch(monkeypatch, tmp_path, stream, architecture):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(LlamaCppRuntime, "start", lambda self: None)
    server = PLaMoTranslateServer("ERROR", LlamaCppOptions("model.gguf"))
    server.runtime.url = "http://test"
    server.runtime.architecture = architecture
    sent = []
    events = [{"content": "こんにちは", "stop": False}, {"content": "。", "stop": True, "stop_type": "eos"}]
    content = "\n\n".join("data: " + json.dumps(event, ensure_ascii=False) for event in events) + "\n\n"

    def respond(http_request):
        sent.append(json.loads(http_request.content))
        return httpx.Response(200, text=content)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "plamo_translate.servers.llama_cpp.server.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond)),
    )
    context = AsyncMock()
    result = asyncio.run(server.translate(request(), stream, context))
    assert sent[0]["prompt"].startswith("<|plamo:tag|>dataset" if architecture == "plamo3" else "<|plamo:op|>dataset")
    if stream:
        assert result == ""
        assert "".join(call.kwargs["message"] for call in context.report_progress.call_args_list) == "こんにちは。"
    else:
        assert result == "こんにちは。"
        context.report_progress.assert_not_called()


def test_runtime_reports_startup_failure_and_reaps_child(tmp_path):
    model = write_gguf(tmp_path / "model.gguf")
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
@pytest.mark.parametrize("translation", ["こんにちは。", "こんにちは。\n"])
def test_interactive_history_retains_translation(capsys, stream, translation):
    class Client:
        async def translate(self, messages):
            yield translation

    messages = [{"role": "user", "content": "output lang=Japanese\n"}]
    result = asyncio.run(print_translation(Client(), messages, stream))
    assert result[-1]["content"] == "output lang=Japanese\n" + translation
    assert capsys.readouterr().out == "こんにちは。\n"


def test_runtime_rejects_other_architectures(tmp_path):
    runtime = LlamaCppRuntime(LlamaCppOptions(str(write_gguf(tmp_path / "model.gguf", "llama")), sys.executable))
    with pytest.raises(RuntimeError, match="Unsupported GGUF architecture llama"):
        runtime.start()
    assert runtime.process is None


def test_cli_checks_llama_server_before_downloading(tmp_path):
    env = dict(os.environ, TMPDIR=str(tmp_path), PLAMO_TRANSLATE_CLI_LLAMA_SERVER=str(tmp_path / "missing"))
    env.pop("PLAMO_TRANSLATE_CLI_MODEL_NAME", None)
    # Offline and with an empty cache, a download attempt would fail with a different message.
    env.update(HF_HUB_OFFLINE="1", HF_HOME=str(tmp_path / "hf"))
    result = subprocess.run(
        [sys.executable, "-m", "plamo_translate.main", "server", "--backend-type", "llama.cpp"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "llama-server not found" in result.stderr
    assert not (tmp_path / "plamo-translate-config.json").exists()


def start_cli(monkeypatch, tmp_path, *arguments):
    """Run main() up to the translation and return its arguments; downloads come from local fake GGUFs."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_PRECISION", raising=False)
    files = {
        DEFAULT_GGUF_MODEL: write_gguf(tmp_path / "default.gguf"),
        DEFAULT_DRAFT_MODEL: write_gguf(tmp_path / "draft.gguf"),
        "pfnet/plamo-3-translate-2b-GGUF:Q4_0": write_gguf(tmp_path / "q4_0.gguf"),
        "user/plamo2-GGUF:Q4_K_M": write_gguf(tmp_path / "plamo2.gguf", "plamo2"),
    }
    monkeypatch.setattr(cli, "resolve_model", lambda name, local_only=False: str(files.get(name, name)))
    monkeypatch.setattr(cli, "check_server_running", lambda: False)
    received = []
    monkeypatch.setattr(cli, "run_translate", received.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plamo-translate", "--backend-type", "llama.cpp", "--llama-server", sys.executable, *arguments],
    )
    cli.main()
    return received[0]


def test_cli_defaults_to_official_gguf_with_draft_for_sessions(monkeypatch, tmp_path):
    from plamo_translate.servers.utils import update_config

    args = start_cli(monkeypatch, tmp_path, "--input", "hello")
    assert args.llama_options["model"] == str(tmp_path / "default.gguf")
    # Loading the draft takes longer than it saves on one translation.
    assert args.llama_options["draft_model"] is None
    assert update_config()["model_name"] == DEFAULT_GGUF_MODEL
    args = start_cli(monkeypatch, tmp_path, "--interactive")
    assert args.llama_options["draft_model"] == str(tmp_path / "draft.gguf")


@pytest.mark.parametrize(
    "arguments,draft",
    [
        (["--model", "pfnet/plamo-3-translate-2b-GGUF:q4_0"], None),
        (["--model", "pfnet/plamo-3-translate-2b-GGUF:Q4_0", "--draft-model", DEFAULT_DRAFT_MODEL], "draft.gguf"),
        (["--draft-model", DEFAULT_DRAFT_MODEL], "draft.gguf"),
    ],
)
def test_cli_draft_is_explicit_with_other_models(monkeypatch, tmp_path, arguments, draft):
    args = start_cli(monkeypatch, tmp_path, *arguments, "--input", "hello")
    assert args.llama_options["draft_model"] == (str(tmp_path / draft) if draft else None)


def test_cli_draft_can_be_disabled_for_sessions(monkeypatch, tmp_path):
    args = start_cli(monkeypatch, tmp_path, "--interactive", "--draft-model", "none")
    assert args.llama_options["draft_model"] is None


@pytest.mark.parametrize(
    "arguments,error",
    [
        (["--model", "user/plamo2-GGUF", "--draft-model", DEFAULT_DRAFT_MODEL], "draft model must be a plamo2"),
        (["--model", "not a model"], "neither a GGUF file"),
    ],
)
def test_cli_rejects_invalid_llama_models(monkeypatch, tmp_path, capsys, arguments, error):
    with pytest.raises(SystemExit) as exit_info:
        start_cli(monkeypatch, tmp_path, *arguments, "--input", "hello")
    assert exit_info.value.code == 2
    assert error in capsys.readouterr().err


def test_cli_rejects_draft_for_mlx(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(cli, "check_server_running", lambda: False)
    monkeypatch.setattr(sys, "argv", ["plamo-translate", "--draft-model", "none", "--input", "hello"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "--draft-model is for llama.cpp" in capsys.readouterr().err


def test_running_repository_model_matches_its_aliases(monkeypatch, tmp_path):
    from plamo_translate.servers.utils import update_config

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", raising=False)
    update_config(backend_type="llama.cpp", model_name=DEFAULT_GGUF_MODEL, port=30000)
    monkeypatch.setattr(cli, "check_server_running", lambda: True)
    received = []
    monkeypatch.setattr(cli, "run_translate", received.append)
    for model in ["pfnet/plamo-3-translate-2b-GGUF", "pfnet/plamo-3-translate-2b-GGUF:q4_k_m"]:
        monkeypatch.setattr(sys, "argv", ["plamo-translate", "--model", model, "--input", "hello"])
        cli.main()
    assert len(received) == 2
    monkeypatch.setattr(sys, "argv", ["plamo-translate", "--model", "pfnet/plamo-3-translate-2b-GGUF:Q4_0"])
    with pytest.raises(SystemExit):
        cli.main()


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
    monkeypatch.setenv("PLAMO_TRANSLATE_CLI_PRECISION", "8bit")
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
