"""MCP translation backed by the native llama.cpp Metal/CUDA/CPU server."""

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from mcp.server.fastmcp import Context, FastMCP

from plamo_translate.servers.utils import (
    INSTRUCTION,
    PLAMO_MAX_TOKENS,
    PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE,
    PLAMO_TRANSLATE_CLI_REPETITION_PENALTY,
    PLAMO_TRANSLATE_CLI_TEMP,
    PLAMO_TRANSLATE_CLI_TOP_K,
    PLAMO_TRANSLATE_CLI_TOP_P,
    TranslateRequest,
    construct_llm_input,
    find_free_port,
)


@dataclass(frozen=True)
class LlamaCppOptions:
    model: str
    executable: str = "llama-server"
    ctx_size: int = 32768
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = 8
    gpu_layers: int = 99
    flash_attn: str = "on"
    startup_timeout: float = 180

    def command(self, port: int) -> list[str]:
        executable = shutil.which(os.path.expanduser(self.executable))
        if executable is None:
            raise ValueError(
                "llama-server not found. Install/build llama.cpp or pass --llama-server /path/to/llama-server."
            )
        model = Path(self.model).expanduser().resolve()
        if not model.is_file() or model.suffix.lower() != ".gguf":
            raise ValueError(
                "--model must be an existing GGUF file. Convert the Hugging Face directory first (see README)."
            )
        if min(self.ctx_size, self.batch_size, self.ubatch_size, self.threads, self.startup_timeout) <= 0:
            raise ValueError("Context, batch, microbatch, threads and startup timeout must be positive.")
        return [
            executable,
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(self.ctx_size),
            "--batch-size",
            str(self.batch_size),
            "--ubatch-size",
            str(self.ubatch_size),
            "--threads",
            str(self.threads),
            "--n-gpu-layers",
            str(self.gpu_layers),
            "--flash-attn",
            self.flash_attn,
            "--parallel",
            "1",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--no-context-shift",
            "--no-webui",
            "--log-verbosity",
            "4",
        ]


class LlamaCppRuntime:
    """Own only the llama-server child started by this instance."""

    def __init__(self, options: LlamaCppOptions):
        self.options = options
        self.process: subprocess.Popen | None = None
        self.log = None
        self.url = ""

    def start(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = self.options.command(port)
        self.url = f"http://127.0.0.1:{port}"
        self.log = tempfile.TemporaryFile(mode="w+b")
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log)
            deadline = time.monotonic() + self.options.startup_timeout
            with httpx.Client(trust_env=False, timeout=1) as client:
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"llama-server exited with code {self.process.returncode}")
                    try:
                        if client.get(self.url + "/health").status_code == 200:
                            # Old PLaMo 2 loaders silently disable RoPE when layer 0 is Mamba.
                            log = os.pread(self.log.fileno(), 1024 * 1024, 0).decode("utf-8", errors="replace")
                            validate_model_log(log)
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
            raise TimeoutError("Timed out loading the GGUF model")
        except BaseException as exc:
            size = os.fstat(self.log.fileno()).st_size
            detail = os.pread(self.log.fileno(), 8000, max(0, size - 8000)).decode("utf-8", errors="replace")
            self.close()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError(f"{exc}\n{detail}") from exc

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


def validate_model_log(log: str) -> None:
    if re.search(r"n_rot\s*=\s*0\b", log):
        raise RuntimeError("This llama.cpp build disables PLaMo 2 RoPE (n_rot = 0). Run scripts/build_llama_cpp.sh.")


def build_prompt(request: TranslateRequest) -> str:
    messages = construct_llm_input(request.model_copy(deep=True))
    # llama.cpp adds BOS according to the GGUF tokenizer metadata, exactly once.
    return "<|plamo:op|>dataset\ntranslation\n" + "\n".join("<|plamo:op|>" + m.content for m in messages)


def completion_payload(request: TranslateRequest, *, max_tokens: int = int(PLAMO_MAX_TOKENS)) -> dict:
    return {
        "prompt": build_prompt(request),
        "stream": True,
        "n_predict": max_tokens,
        "temperature": float(PLAMO_TRANSLATE_CLI_TEMP),
        "top_p": float(PLAMO_TRANSLATE_CLI_TOP_P),
        "top_k": int(PLAMO_TRANSLATE_CLI_TOP_K),
        "repeat_penalty": float(PLAMO_TRANSLATE_CLI_REPETITION_PENALTY or 1.0),
        "repeat_last_n": int(PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE or 0),
        "seed": 0,
        "stop": ["<|plamo:op|>", "<|plamo:bos|>", "<|plamo:eos|>"],
        "cache_prompt": False,
    }


async def completion_events(client: httpx.AsyncClient, url: str, payload: dict):
    finished = False
    async with client.stream("POST", url + "/completion", json=payload) as response:
        if response.is_error:
            detail = (await response.aread()).decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {response.status_code}: {detail}")
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"llama.cpp: {event['error']}")
            if event.get("stop"):
                finished = True
                if event.get("truncated") or event.get("stop_type") == "limit":
                    raise RuntimeError("Translation was truncated. Increase --ctx-size or PLAMO_MAX_TOKENS.")
            yield event
    if not finished:
        raise RuntimeError("llama.cpp disconnected before completing the translation")


class PLaMoTranslateServer(FastMCP):
    def __init__(self, log_level: str, options: LlamaCppOptions, show_progress: bool = False):
        self.runtime = LlamaCppRuntime(options)
        self.runtime.start()
        try:
            # Select the MCP port after model loading; loading can take several seconds.
            super().__init__(
                name="plamo-translate",
                instructions=INSTRUCTION,
                log_level=log_level,
                stateless_http=False,
                host="127.0.0.1",
                port=find_free_port(),
            )
            self.lock = asyncio.Lock()
            self.add_tool(fn=self.translate, name="plamo-translate", description=INSTRUCTION)
        except BaseException:
            self.runtime.close()
            raise

    async def translate(self, request: TranslateRequest, stream: bool, context: Context) -> str:
        async with self.lock, httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(600, connect=10)) as client:
            chunks = []
            async for event in completion_events(client, self.runtime.url, completion_payload(request)):
                text = event.get("content", "")
                if text:
                    chunks.append(text)
                    if stream:
                        await context.report_progress(progress=len(chunks), total=None, message=text)
            return "" if stream else "".join(chunks)
