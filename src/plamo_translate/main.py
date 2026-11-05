#!/usr/bin/env python3
import argparse
import asyncio
import atexit
import importlib.util
import json
import logging
import multiprocessing
import os
import platform
try:
    import readline
except ImportError:  # Windows, which main() refuses with a message
    readline = None
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from plamo_translate import __version__
from plamo_translate.clients import translate
from plamo_translate.servers.llama_cpp.model import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_GGUF_MODEL,
    SUPPORTED_ARCHITECTURES,
    canonical_model,
    gguf_architecture,
    resolve_model,
    same_model,
)
from plamo_translate.servers.mlx.pfml3 import (
    CHECKPOINT_FILES,
    checkpoint_complete,
    checkpoint_config,
    is_plamo3_checkpoint,
)
from plamo_translate.servers.utils import (
    DEFAULT_MODEL_NAME,
    PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE,
    PLAMO_TRANSLATE_CLI_REPETITION_PENALTY,
    SUPPORTED_LANGUAGES,
    update_config,
    verify_mcp_server_ready,
)

logger = logging.getLogger(__name__)


def start_mcp_server(backend_type: str, log_level: str, show_progress: bool = False, llama_options=None) -> None:
    # To avoid showing warnings related to resource_tracker
    signal.signal(signal.SIGTERM, lambda _signal_number, _frame: exit(0))
    if os.environ.get("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER") == "1":
        from plamo_translate.servers.mock import server as mock_server

        server = mock_server.PLaMoTranslateServer(log_level=log_level, show_progress=show_progress)
        try:
            server.run(transport="streamable-http")
        except Exception as e:
            print(f"Error during server running: {e}")
    elif backend_type == "mlx":
        if is_plamo3_checkpoint(os.environ.get("PLAMO_TRANSLATE_CLI_MODEL_NAME")):
            # MLX reads these on import. Every matrix would otherwise end a command buffer.
            os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "4096")
            os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
        from plamo_translate.servers.mlx import server as mlx_server

        server = mlx_server.PLaMoTranslateServer(log_level=log_level, show_progress=show_progress)
        try:
            server.run(transport="streamable-http")
        except Exception as e:
            print(f"Error during server running: {e}")
    elif backend_type == "llama.cpp":
        from plamo_translate.servers.llama_cpp.server import LlamaCppOptions, PLaMoTranslateServer

        server = PLaMoTranslateServer(log_level, LlamaCppOptions(**llama_options), show_progress)
        try:
            server.run(transport="streamable-http")
        finally:
            server.runtime.close()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def fetch_gguf(name: str) -> str:
    """Local path of a canonical GGUF model, downloading it on first use."""
    try:
        return resolve_model(name, local_only=True)
    except FileNotFoundError:
        print(f"Downloading {name} into the Hugging Face cache...", file=sys.stderr, flush=True)
        return resolve_model(name)


def fetch_mlx_checkpoint(name: str | None) -> None:
    """Download an original PLaMo 3 checkpoint before its server starts, with a message and progress bars.

    Inside the server the 5.8 GB download of the default model was silent and counted against the startup wait.
    PLaMo 2 checkpoints are still fetched by the MLX loader.
    """
    if not name or os.environ.get("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER") == "1" or Path(name).expanduser().is_dir():
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

    try:
        if checkpoint_complete(snapshot_download(name, allow_patterns=CHECKPOINT_FILES, local_files_only=True)):
            return
    except LocalEntryNotFoundError:
        pass
    except HFValidationError:
        raise SystemExit(f"{name} is neither a checkpoint directory nor a Hugging Face repository.") from None
    config = checkpoint_config(name)
    if config is None or config.get("model_type") != "plamo3" or config.get("quantization"):
        # The server reports missing and quantized checkpoints, and loads PLaMo 2 its own way.
        return
    print(f"Downloading {name}{download_size(name)} into the Hugging Face cache...", file=sys.stderr, flush=True)
    try:
        snapshot_download(name, allow_patterns=CHECKPOINT_FILES)
    except KeyboardInterrupt:
        # Exit at once rather than waiting for the download threads to finish their files.
        print("\nDownload interrupted; the next run resumes it.", file=sys.stderr, flush=True)
        os._exit(130)
    except Exception as exc:
        raise SystemExit(f"Could not download {name}: {exc}") from exc


def download_size(name: str) -> str:
    """" (5.8 GB)" for the checkpoint files of a repository; the progress bar only counts files."""
    from fnmatch import fnmatchcase

    from huggingface_hub import HfApi

    try:
        files = HfApi().model_info(name, files_metadata=True).siblings or []
    except Exception:
        return ""
    size = sum(f.size or 0 for f in files if any(fnmatchcase(f.rfilename, p) for p in CHECKPOINT_FILES))
    return f" ({size / 1e9:.1f} GB)" if size else ""


def mlx_available() -> bool:
    """MLX runs on Apple Silicon only, and is installed there only."""
    return sys.platform == "darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx") is not None


def check_server_running() -> bool:
    config = update_config()
    if "port" not in config:
        return False
    port = config["port"]
    tools = asyncio.run(verify_mcp_server_ready(port))
    if "plamo-translate" in tools:
        return True
    return False


def wait_for_server_ready(process: multiprocessing.Process | None = None, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if process is not None and not process.is_alive():
            raise RuntimeError(
                f"Translation server failed to start: exited during model loading (exit code {process.exitcode})."
            )
        if check_server_running():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the translation server.")
        time.sleep(0.1)


async def print_translation(
    client: translate.MCPClient, messages: List[Dict[str, str]], stream: bool
) -> List[Dict[str, str]]:
    async for result in client.translate(messages):
        messages[-1]["content"] += result
        print(result, end="", flush=True)
    if not messages[-1]["content"].endswith("\n"):
        # PLaMo 3 stops without PLaMo 2's trailing newline; end the line for the shell or the next prompt.
        print(flush=True)

    return messages


def run_translate(args: argparse.Namespace) -> None:
    from_lang = args.from_lang
    if from_lang != "":
        from_lang = f" lang={from_lang}"

    to = args.to
    if to != "":
        to = f" lang={to}"

    backend_type = args.backend_type
    stream = args.stream

    if args.input is None and not args.interactive:
        input_text = sys.stdin.read()
        args.input = input_text
    else:
        input_text = args.input

    messages: List[Dict[str, str]] = []

    server = None
    if not check_server_running():
        if backend_type == "mlx":
            fetch_mlx_checkpoint(os.environ.get("PLAMO_TRANSLATE_CLI_MODEL_NAME"))
        if args.interactive:
            show_progress = True
        else:
            show_progress = False
        server = multiprocessing.Process(
            target=start_mcp_server,
            args=(backend_type, "CRITICAL", show_progress, args.llama_options),
            daemon=True,
        )
        server.start()
        try:
            wait_for_server_ready(server)
        except BaseException:
            server.terminate()
            server.join(timeout=15)
            raise

    client = translate.MCPClient(stream=stream)

    try:
        if args.interactive:
            history_file = Path.home() / ".plamo_translate_history"
            if not history_file.exists():
                history_file.touch()
            try:
                readline.read_history_file(history_file)
                readline.set_history_length(-1)
            except Exception:
                print(f"History file {history_file} not found. Starting a new history file.")
            atexit.register(readline.write_history_file, history_file)
            print("Interactive mode enabled. Type your input below (Ctrl+D to exit).")

            while True:
                try:
                    input_text = input("> ")
                    if input_text.strip() == "":
                        continue

                    messages.append(
                        {
                            "role": "user",
                            "content": f"input{from_lang}\n{input_text}",
                        },
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"output{to}\n",
                        },
                    )
                    messages = asyncio.run(print_translation(client, messages, stream=args.stream))

                except KeyboardInterrupt:
                    print("\nTranslation interrupted by user (Ctrl+C).")
                    sys.exit(0)
                    break
                except EOFError:
                    print("\nCtrl+D received. Exiting.")
                    sys.exit(0)
                    break

        else:
            # Non-interactive mode: translate the input once
            messages.append(
                {
                    "role": "user",
                    "content": f"input{from_lang}\n{input_text}",
                },
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"output{to}\n",
                },
            )
            asyncio.run(print_translation(client, messages, stream=args.stream))

    finally:
        if server is not None:
            server.terminate()
            server.join(timeout=15)
            if server.is_alive():
                server.kill()
                server.join()


def main() -> None:
    if sys.platform == "win32":
        sys.exit("plamo-translate does not support Windows; run it in WSL, where it uses llama.cpp.")
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
        help="Show program's version number and exit.",
    )

    # Add arguments for the default command (translate)
    # These will be used if no subcommand is provided
    global_parser.add_argument("--input", type=str, help="Input text to translate", default=None)
    global_parser.add_argument(
        "--from",
        type=str,
        help="Input language for translation",
        default="English|Japanese",
        choices=SUPPORTED_LANGUAGES,
        dest="from_lang",
    )
    global_parser.add_argument(
        "--to",
        type=str,
        help="Output language for translation",
        default="",
        choices=SUPPORTED_LANGUAGES + [""],
    )
    global_parser.add_argument(
        "--backend-type",
        type=str,
        default=None,
        choices=["mlx", "llama.cpp"],
        help="Server backend (default: reuse a running server, otherwise mlx on Apple Silicon and llama.cpp elsewhere)",
    )
    global_parser.add_argument(
        "--precision",
        "-p",
        type=str,
        default=None,
        choices=["4bit", "6bit", "8bit", "mixed", "bf16"],
        help="MLX precision (default: 4bit for original weights; preserve exported checkpoints)",
    )
    global_parser.add_argument(
        "--model",
        help=(
            "Model path or Hugging Face repository. MLX: checkpoint (default: "
            f"{DEFAULT_MODEL_NAME}). llama.cpp: GGUF file or <user>/<repo>[:QUANT] (default: {DEFAULT_GGUF_MODEL})"
        ),
    )
    global_parser.add_argument(
        "--draft-model",
        help=(
            "llama.cpp draft GGUF or <user>/<repo>[:QUANT] for speculative decoding, or 'none' "
            f"(default for servers and interactive mode with the default model: {DEFAULT_DRAFT_MODEL})"
        ),
    )
    global_parser.add_argument(
        "--llama-server",
        default=os.environ.get("PLAMO_TRANSLATE_CLI_LLAMA_SERVER", "llama-server"),
        help="Path to the native llama-server executable",
    )
    global_parser.add_argument("--ctx-size", type=int, default=32768, help="llama.cpp context size")
    global_parser.add_argument("--batch-size", type=int, default=2048, help="llama.cpp prompt batch size")
    global_parser.add_argument("--ubatch-size", type=int, default=512, help="llama.cpp physical batch size")
    global_parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="llama.cpp CPU threads (default: 2 with GPU layers on Apple Silicon, otherwise 8)",
    )
    global_parser.add_argument("--gpu-layers", type=int, default=99, help="llama.cpp GPU layers (0 for CPU)")
    global_parser.add_argument("--flash-attn", choices=["on", "off", "auto"], default="on")
    global_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Enable batch processing mode for translation",
    )
    global_parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enable interactive mode for translation",
    )

    # Create the parser for the "server" command
    parser = argparse.ArgumentParser(description="PLaMo Translate CLI", parents=[global_parser])

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    _ = subparsers.add_parser("server", help="Run the server", parents=[global_parser])
    _ = subparsers.add_parser(
        "show-claude-config", help="Show the MCP server config for Claude Desktop", parents=[global_parser]
    )

    args = parser.parse_args()

    # Route to appropriate command handler
    if hasattr(args, "version") and args.version:
        # The version action should have already exited, but as a fallback:
        sys.exit(0)

    if PLAMO_TRANSLATE_CLI_REPETITION_PENALTY is not None and PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE is None:
        raise ValueError(
            "If PLAMO_TRANSLATE_CLI_REPETITION_PENALTY is set, "
            "PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE must also be set."
        )
    elif PLAMO_TRANSLATE_CLI_REPETITION_PENALTY is None and PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE is not None:
        raise ValueError(
            "If PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE is set, "
            "PLAMO_TRANSLATE_CLI_REPETITION_PENALTY must also be set."
        )

    if args.input is None and (args.interactive or sys.stdin.isatty()):
        args.interactive = True
        logging.basicConfig(level=logging.ERROR)
        os.environ["PLAMO_TRANSLATE_CLI_SERVER_LOG_LEVEL"] = "CRITICAL"
    else:
        args.interactive = False
        logging.basicConfig(level=logging.CRITICAL)
        os.environ["PLAMO_TRANSLATE_CLI_SERVER_LOG_LEVEL"] = "CRITICAL"

    args.stream = not args.no_stream
    config = update_config()
    running = check_server_running()
    mock = os.environ.get("PLAMO_TRANSLATE_CLI_USE_MOCK_SERVER") == "1"
    if not running and args.backend_type is None:
        # The mock server stands in for either backend; keep the MLX default so tests do not depend on the host.
        args.backend_type = "mlx" if mock or mlx_available() else "llama.cpp"
    args.backend_type = args.backend_type or config.get("backend_type", "mlx")
    if args.backend_type == "mlx" and not running and not mock and not mlx_available():
        parser.error(
            "MLX needs an Apple Silicon Mac with macOS 14 or later. Use --backend-type llama.cpp, which runs "
            "llama-server from llama.cpp (see README)."
        )
    if args.backend_type == "llama.cpp" and args.precision is not None:
        parser.error("For llama.cpp, select precision with the GGUF passed to --model; --precision is for MLX.")
    if args.backend_type != "llama.cpp" and args.draft_model is not None:
        parser.error("--draft-model is for llama.cpp; pass --backend-type llama.cpp.")
    precision = (args.precision or os.environ.get("PLAMO_TRANSLATE_CLI_PRECISION")) if args.backend_type == "mlx" else None
    requested_model = args.model or os.environ.get("PLAMO_TRANSLATE_CLI_MODEL_NAME")
    model_name = requested_model
    if model_name is None and running and precision is None:
        model_name = config.get("model_name")
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME if args.backend_type == "mlx" else DEFAULT_GGUF_MODEL
    # The mock server ignores the model and records its name as "mock".
    llama = args.backend_type == "llama.cpp" and not mock
    if llama:
        try:
            model_name = canonical_model(model_name)
        except ValueError as exc:
            parser.error(str(exc))
    if running and args.command != "show-claude-config":
        existing_model = config.get("model_name")
        if llama and config.get("backend_type") == "llama.cpp" and existing_model:
            try:
                existing_model = canonical_model(existing_model)
            except ValueError:
                pass
        if config.get("backend_type", "mlx") != args.backend_type or not same_model(existing_model, model_name):
            parser.error(
                f"A different backend/model is already running ({config.get('backend_type', 'mlx')}: {existing_model}; "
                f"requested {args.backend_type}: {model_name}). Stop it first, or use a separate TMPDIR."
            )
        if args.backend_type == "mlx" and precision is not None and config.get("precision") != precision:
            parser.error("A server with different precision is running; stop it before changing --precision.")
    if args.backend_type == "mlx" and precision is not None:
        os.environ["PLAMO_TRANSLATE_CLI_PRECISION"] = precision
    args.llama_options = None
    if llama and not running and args.command != "show-claude-config":
        from dataclasses import asdict

        from plamo_translate.servers.llama_cpp.server import LlamaCppOptions, find_llama_server, gguf_file

        draft_name = args.draft_model
        if draft_name is None and model_name == DEFAULT_GGUF_MODEL and (args.command == "server" or args.interactive):
            # On M1 Max the draft made decoding about 20% faster but took 0.8 s to load, which only pays off
            # after 420 to 560 generated tokens; persistent sessions recover it.
            draft_name = DEFAULT_DRAFT_MODEL
        if draft_name is not None and draft_name.lower() == "none":
            draft_name = None
        try:
            # Fail before downloading gigabytes.
            find_llama_server(args.llama_server)
            model_path = gguf_file(fetch_gguf(model_name), "--model")
            architecture = gguf_architecture(model_path)
            if architecture not in SUPPORTED_ARCHITECTURES:
                raise ValueError(f"{model_name} is a {architecture} model; expected PLaMo 2 or PLaMo 3")
            draft_path = None
            if draft_name is not None:
                try:
                    draft_path = gguf_file(fetch_gguf(canonical_model(draft_name)), "--draft-model")
                    if gguf_architecture(draft_path) != architecture:
                        raise ValueError(f"The draft model must be a {architecture} GGUF with the same vocabulary")
                except Exception as exc:
                    if args.draft_model is not None:
                        raise
                    # The default draft only speeds decoding up; offline without it in the cache, go on without it.
                    print(f"Continuing without the speculative-decoding draft: {exc}", file=sys.stderr, flush=True)
                    draft_path = None
            options = LlamaCppOptions(
                model=model_path,
                executable=args.llama_server,
                draft_model=draft_path,
                ctx_size=args.ctx_size,
                batch_size=args.batch_size,
                ubatch_size=args.ubatch_size,
                threads=args.threads,
                gpu_layers=args.gpu_layers,
                flash_attn=args.flash_attn,
            )
            options.command(0)
        except Exception as exc:
            parser.error(str(exc))
        args.llama_options = asdict(options)
    if not running and args.command != "show-claude-config":
        update_config(backend_type=args.backend_type, model_name=model_name)
    if model_name is not None:
        os.environ["PLAMO_TRANSLATE_CLI_MODEL_NAME"] = model_name

    if args.command == "server":
        logging.basicConfig(level=logging.INFO)
        if check_server_running():
            print("MCP server is already running. Skipping server start.")
            sys.exit(0)
        if args.backend_type == "mlx":
            fetch_mlx_checkpoint(model_name)
        while not check_server_running():
            try:
                logger.info("Starting server...")
                start_mcp_server(args.backend_type, "INFO", True, args.llama_options)
                logger.info("The server is running (Ctrl+C to stop)")
            except KeyboardInterrupt:
                logger.error("\nCtrl+C received. Exiting.")
                break
            except EOFError:
                logger.error("\nCtrl+D received. Exiting.")
                break
            except Exception as e:
                if args.backend_type == "llama.cpp":
                    parser.exit(1, f"Could not start llama.cpp: {e}\n")
                logger.error(f"Server failed: {e}")
                raise

    elif args.command == "show-claude-config":
        cmd = subprocess.run(["which", "npx"], check=True, capture_output=True, text=True)
        if cmd.returncode != 0:
            logger.error("npx command not found. Please install Node.js and npx.")
            exit(1)
        npx_path = cmd.stdout.strip()
        config = update_config()
        print(
            json.dumps(
                {
                    "mcpServers": {
                        "plamo-translate": {
                            "command": npx_path,
                            "args": [
                                "-y",
                                "mcp-remote",
                                f"http://localhost:{config['port']}/mcp",
                                "--allow-http",
                                "--transport",
                                "http-only",
                            ],
                            "env": {"PATH": os.environ["PATH"]},
                        }
                    }
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        run_translate(args)


if __name__ == "__main__":
    main()
