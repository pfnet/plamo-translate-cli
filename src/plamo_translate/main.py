#!/usr/bin/env python3
import argparse
import asyncio
import atexit
import json
import logging
import multiprocessing
import os
import readline
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from plamo_translate import __version__
from plamo_translate.clients import translate
from plamo_translate.servers.utils import (
    PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE,
    PLAMO_TRANSLATE_CLI_REPETITION_PENALTY,
    SUPPORTED_LANGUAGES,
    update_config,
    verify_mcp_server_ready,
)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

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
            raise RuntimeError(f"Translation server failed to start (exit code {process.exitcode}).")
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
        help="Server backend (default: reuse a running server, otherwise mlx)",
    )
    global_parser.add_argument(
        "--precision",
        "-p",
        type=str,
        default=None,
        choices=["4bit", "8bit", "bf16"],
        help="Model parameter's precision to use (default: 4bit)",
    )
    global_parser.add_argument("--model", help="Local GGUF for llama.cpp, or MLX model path/repository")
    global_parser.add_argument(
        "--llama-server",
        default=os.environ.get("PLAMO_TRANSLATE_CLI_LLAMA_SERVER", "llama-server"),
        help="Path to the native llama-server executable",
    )
    global_parser.add_argument("--ctx-size", type=int, default=32768, help="llama.cpp context size")
    global_parser.add_argument("--batch-size", type=int, default=2048, help="llama.cpp prompt batch size")
    global_parser.add_argument("--ubatch-size", type=int, default=512, help="llama.cpp physical batch size")
    global_parser.add_argument("--threads", type=int, default=8, help="llama.cpp CPU threads")
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
    args.backend_type = args.backend_type or (config.get("backend_type", "mlx") if running else "mlx")
    if args.backend_type == "llama.cpp" and args.precision is not None:
        parser.error("For llama.cpp, select precision with the GGUF passed to --model; --precision is for MLX.")
    model_name = args.model or os.environ.get("PLAMO_TRANSLATE_CLI_MODEL_NAME")
    if model_name is None and running and args.precision is None:
        model_name = config.get("model_name")
    if model_name is None and args.backend_type == "mlx":
        model_name = {
            "4bit": "mlx-community/plamo-2-translate",
            "8bit": "mlx-community/plamo-2-translate-8bit",
            "bf16": "mlx-community/plamo-2-translate-bf16",
        }[args.precision or "4bit"]
    if args.command != "show-claude-config" and model_name is None:
        parser.error("llama.cpp requires --model /path/to/model.gguf (or PLAMO_TRANSLATE_CLI_MODEL_NAME).")
    if args.backend_type == "llama.cpp" and model_name:
        model_name = str(Path(model_name).expanduser().resolve())
    if running and args.command != "show-claude-config":
        existing_model = config.get("model_name")
        if config.get("backend_type") == "llama.cpp" and existing_model:
            existing_model = str(Path(existing_model).expanduser().resolve())
        if config.get("backend_type", "mlx") != args.backend_type or existing_model != model_name:
            parser.error("A different backend/model is already running. Stop it first, or use a separate TMPDIR.")
    args.llama_options = None
    if args.backend_type == "llama.cpp" and not running and args.command != "show-claude-config":
        from dataclasses import asdict

        from plamo_translate.servers.llama_cpp.server import LlamaCppOptions

        options = LlamaCppOptions(
            model=model_name,
            executable=args.llama_server,
            ctx_size=args.ctx_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            threads=args.threads,
            gpu_layers=args.gpu_layers,
            flash_attn=args.flash_attn,
        )
        try:
            options.command(0)
        except ValueError as exc:
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
                logger.error(f"An error occurred: {str(e)}: {e}. Restarting server...")

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
