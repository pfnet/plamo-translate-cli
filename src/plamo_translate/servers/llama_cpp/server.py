"""MCP translation backed by the native llama.cpp Metal/CUDA/CPU server."""

import asyncio
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from mcp.server.fastmcp import Context, FastMCP

from plamo_translate.servers.llama_cpp.model import SUPPORTED_ARCHITECTURES, gguf_architecture, gguf_metadata
from plamo_translate.servers.mlx.pfml3 import pfml3_prompt
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


def find_llama_server(executable: str) -> str:
    path = shutil.which(os.path.expanduser(executable))
    if path is None:
        raise ValueError("llama-server not found. Install llama.cpp or pass --llama-server /path/to/llama-server.")
    return path


def gguf_file(name: str, option: str) -> str:
    # Not resolve(): Hugging Face cache entries are symlinks to blobs without the .gguf suffix.
    path = Path(name).expanduser().absolute()
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError(f"{option} must be an existing GGUF file or a Hugging Face GGUF repository (see README).")
    return str(path)


def yarn_range_arguments(metadata: dict) -> list[str]:
    """llama-server options that bring a fractional YaRN correction range to its nearest whole dimensions.

    PLaMo 3 ramps between fractional dimensions (`truncate: false`; 13.96 to 30.02 for the 2B models), but no
    llama.cpp release honors that yet (ggml-org/llama.cpp#30206, #30337): floor and ceil widen the range to 13 to 31.
    The angle error grows with the distance between tokens: on source text, the KL divergence from the exact range
    reached 0.04 at 2,048 tokens and 0.66 at 16,384. Betas whose dimensions lie just inside the nearest whole numbers
    make floor and ceil land on 14 and 30, cutting it to 3e-4 and 5e-3. A llama.cpp with those changes would use
    14.001 to 29.999 instead of the exact range, so drop these options once a release honors `yarn_truncate`.
    """
    architecture = metadata.get("general.architecture")
    scaling = f"{architecture}.rope.scaling."
    if metadata.get(scaling + "type") != "yarn" or metadata.get(scaling + "yarn_truncate", True):
        return []
    dimensions = metadata.get(f"{architecture}.rope.dimension_count") or metadata.get(
        f"{architecture}.attention.key_length"
    )
    original = metadata.get(scaling + "original_context_length")
    base = metadata.get(f"{architecture}.rope.freq_base")
    if not dimensions or not original or not base:
        return []
    # ggml_rope_yarn_corr_dim and its inverse.
    scale = dimensions / (2 * math.log(base))
    low = round(scale * math.log(original / (metadata.get(scaling + "yarn_beta_fast", 32.0) * 2 * math.pi))) + 0.001
    high = round(scale * math.log(original / (metadata.get(scaling + "yarn_beta_slow", 1.0) * 2 * math.pi))) - 0.001
    if not 0 < low < high:
        return []
    beta = lambda dimension: original / (2 * math.pi * math.exp(dimension / scale))  # noqa: E731
    return ["--yarn-beta-fast", f"{beta(low):.8g}", "--yarn-beta-slow", f"{beta(high):.8g}"]


@dataclass(frozen=True)
class LlamaCppOptions:
    model: str
    executable: str = "llama-server"
    # A smaller model with the same vocabulary proposes tokens that the main model verifies in one batch.
    draft_model: str | None = None
    ctx_size: int = 32768
    batch_size: int = 2048
    ubatch_size: int = 512
    # None: 2 with Metal offload, where more polling threads only slowed decoding, otherwise 8.
    threads: int | None = None
    gpu_layers: int = 99
    flash_attn: str = "on"
    startup_timeout: float = 180

    def command(self, port: int) -> list[str]:
        executable = find_llama_server(self.executable)
        model = gguf_file(self.model, "--model")
        metal = sys.platform == "darwin" and platform.machine() == "arm64"
        threads = self.threads or (2 if self.gpu_layers and metal else 8)
        if min(self.ctx_size, self.batch_size, self.ubatch_size, threads, self.startup_timeout) <= 0:
            raise ValueError("Context, batch, microbatch, threads and startup timeout must be positive.")
        speculation = []
        if self.draft_model:
            speculation = [
                "--model-draft",
                gguf_file(self.draft_model, "--draft-model"),
                "--spec-type",
                "draft-simple",
                "--gpu-layers-draft",
                str(self.gpu_layers),
                # Fastest measured on M1 Max; drafting stops early when the draft is unsure.
                "--spec-draft-n-max",
                "6",
                "--spec-draft-p-min",
                "0.6",
            ]
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
            str(threads),
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
            # Requests send PFML prompts to /completion and never use a chat template. The template that the
            # b11960 converter embeds in PLaMo 3 GGUFs raises without a target language, and the server
            # renders an example at startup, so replace it with a built-in one.
            "--chat-template",
            "chatml",
            # -ngl is explicit, so fitting would only spend a second probing device memory.
            "--fit",
            "off",
            # Requests set cache_prompt to false, so nothing is reused: skip saving slot states and SWA
            # checkpoints for each one.
            "--cache-ram",
            "0",
            "--ctx-checkpoints",
            "0",
            # The n_rot line checked by validate_model_log is logged at this verbosity.
            "--log-verbosity",
            "4",
            *yarn_range_arguments(gguf_metadata(model)),
            *speculation,
        ]


class LlamaCppRuntime:
    """Own only the llama-server child started by this instance."""

    def __init__(self, options: LlamaCppOptions):
        self.options = options
        self.process: subprocess.Popen | None = None
        self.log = None
        self.url = ""
        self.architecture = ""

    def start(self) -> None:
        self.architecture = gguf_architecture(self.options.model)
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise RuntimeError(f"Unsupported GGUF architecture {self.architecture}; expected PLaMo 2 or PLaMo 3")
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
                            validate_model_log(log, draft=self.options.draft_model is not None)
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


def validate_model_log(log: str, draft: bool = False) -> None:
    if re.search(r"n_rot\s*=\s*0\b", log):
        raise RuntimeError("This llama.cpp build disables PLaMo 2 RoPE (n_rot = 0). Run scripts/build_llama_cpp.sh.")
    if draft and "speculative decoding context initialized" not in log:
        # llama-server keeps serving without the draft when it is incompatible, for example in vocabulary. Print
        # rather than log: the CLI raises the root logger to ERROR or CRITICAL before servers start.
        print(
            "llama-server did not enable the draft model; decoding runs without speculation",
            file=sys.stderr,
            flush=True,
        )


def build_prompt(request: TranslateRequest, *, architecture: str) -> str:
    messages = construct_llm_input(request.model_copy(deep=True))
    # llama.cpp adds BOS according to the GGUF tokenizer metadata, exactly once.
    if architecture == "plamo3":
        return pfml3_prompt(messages, bos=False)
    if architecture == "plamo2":
        return "<|plamo:op|>dataset\ntranslation\n" + "\n".join("<|plamo:op|>" + m.content for m in messages)
    raise ValueError(f"Unsupported GGUF architecture: {architecture}")


def completion_payload(
    request: TranslateRequest, *, architecture: str, max_tokens: int = int(PLAMO_MAX_TOKENS)
) -> dict:
    temperature = float(PLAMO_TRANSLATE_CLI_TEMP)
    return {
        "prompt": build_prompt(request, architecture=architecture),
        "stream": True,
        "n_predict": max_tokens,
        "temperature": temperature,
        # Greedy decoding needs no nucleus filter, which would sort the 100k-token vocabulary on the CPU
        # for every token. min_p is off, as in the MLX backend.
        "top_p": float(PLAMO_TRANSLATE_CLI_TOP_P) if temperature > 0 else 1.0,
        "min_p": 0.0,
        "top_k": int(PLAMO_TRANSLATE_CLI_TOP_K),
        "repeat_penalty": float(PLAMO_TRANSLATE_CLI_REPETITION_PENALTY or 1.0),
        "repeat_last_n": int(PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE or 0),
        "seed": 0,
        # PLaMo 3 GGUFs end generation at their <|plamo:tag|>, <|plamo:bos|> and <|plamo:eos|> tokens.
        "stop": ["<|plamo:op|>", "<|plamo:bos|>", "<|plamo:eos|>"] if architecture == "plamo2" else [],
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
            payload = completion_payload(request, architecture=self.runtime.architecture)
            async for event in completion_events(client, self.runtime.url, payload):
                text = event.get("content", "")
                if text:
                    chunks.append(text)
                    if stream:
                        await context.report_progress(progress=len(chunks), total=None, message=text)
            return "" if stream else "".join(chunks)
