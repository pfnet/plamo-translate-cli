import asyncio
import json
import logging
import os
import socket
import textwrap
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = [
    "Japanese",
    "Japanese(easy)",
    "English",
    "Chinese",
    "Taiwanese",
    "Korean",
    "Arabic",
    "Italian",
    "Indonesian",
    "Dutch",
    "Spanish",
    "Thai",
    "German",
    "French",
    "Vietnamese",
    "Russian",
    "English|Japanese",
]

PLAMO_TRANSLATE_CLI_MODEL_NAME = os.environ.get("PLAMO_TRANSLATE_CLI_MODEL_NAME", "mlx-community/plamo-2-translate")
PLAMO_TRANSLATE_CLI_SERVER_START_PORT = int(os.environ.get("PLAMO_TRANSLATE_CLI_SERVER_START_PORT", 30000))
PLAMO_TRANSLATE_CLI_SERVER_END_PORT = int(os.environ.get("PLAMO_TRANSLATE_CLI_SERVER_END_PORT", 30099))
PLAMO_TRANSLATE_CLI_SERVER_LOG_LEVEL = os.environ.get("PLAMO_TRANSLATE_CLI_SERVER_LOG_LEVEL", "INFO")
PLAMO_TRANSLATE_CLI_TEMP = os.environ.get("PLAMO_TRANSLATE_CLI_TEMP", "0.0")
PLAMO_TRANSLATE_CLI_TOP_P = os.environ.get("PLAMO_TRANSLATE_CLI_TOP_P", "0.98")
PLAMO_TRANSLATE_CLI_TOP_K = os.environ.get("PLAMO_TRANSLATE_CLI_TOP_K", "0")
PLAMO_TRANSLATE_CLI_REPETITION_PENALTY = os.environ.get("PLAMO_TRANSLATE_CLI_REPETITION_PENALTY", None)
PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE = os.environ.get("PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE", None)
PLAMO_MAX_TOKENS = os.environ.get("PLAMO_MAX_TOKENS", "32768")
SUPPORTED_LANGUAGES_LIST_STR = "\n-".join(SUPPORTED_LANGUAGES)
INSTRUCTION = textwrap.dedent(
    f"""Use the `plamo-translate` tool to translate text between multiple languages.
    Supported languages include:

    - {SUPPORTED_LANGUAGES_LIST_STR}

    Use the tool by specifying the text and the source and target languages.
    """
)


async def verify_mcp_server_ready(port: int) -> List[str]:
    """Verify if the MCP server is ready to accept connections."""
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        async with streamablehttp_client(url) as (
            read_stream,
            write_stream,
            get_session_id_callback,
        ):
            async with ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
            ) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [tool.name for tool in tools.tools]
    except Exception:
        return []


def find_free_port(
    start_port: int = PLAMO_TRANSLATE_CLI_SERVER_START_PORT,
    end_port: int = PLAMO_TRANSLATE_CLI_SERVER_END_PORT,
) -> int:
    """
    Find a port in the range [start_port, end_port].
    """
    config = update_config()

    # Phase 1: Check for existing MCP server with 'plamo-translate' tool
    if "port" in config:
        port = config["port"]

        try:
            tools = asyncio.run(verify_mcp_server_ready(port))
        except Exception as e:
            logger.info(f"Failed to connect to MCP server on port {port}: {e}")
            tools = []

        if "plamo-translate" in tools:
            logger.info(f"Found existing MCP server with 'plamo-translate' tool on port {port}.")
            return port

        previous_port = port
    else:
        previous_port = None

    # Phase 2: If no suitable MCP server found, find any free port in the range
    for port in range(start_port, end_port + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.1)  # Short timeout for connection attempt
            try:
                result = sock.connect_ex(("127.0.0.1", port))
                if result != 0:
                    # If connection failed (errno != 0), port is likely free
                    logger.info(f"Found free port: {port}")
                    if previous_port is not None and previous_port != port:
                        logger.info(f"Updating MCP server port from {previous_port} to {port}.")
                    update_config(port=port)
                    return port
            except Exception:
                # This can happen if e.g. sock.connect_ex itself has issues, or port is restricted
                pass  # Try next port

    raise RuntimeError(
        "Could not find a suitable MCP server with 'plamo-translate' tool "
        f"or a free port in the range {start_port}-{end_port}."
    )


def update_config(**kwargs) -> Dict[str, Any]:
    tmp_dir = os.environ.get("TMPDIR", None)
    if tmp_dir is None:
        raise ValueError("TMPDIR environment variable is not set. Please set it to a valid directory.")
    tmp_config_path = Path(tmp_dir) / "plamo-translate-config.json"

    if not tmp_config_path.exists():
        with tmp_config_path.open("w") as f:
            json.dump(kwargs, f, indent=4)
        config = kwargs
        logger.info(
            f"Created new temporary config file at {tmp_config_path} with initial values: "
            f"{json.dumps(config, indent=4, ensure_ascii=False)}"
        )
    else:
        with tmp_config_path.open("r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Config file {tmp_config_path} is corrupted. Recreating it.")
                config = {}
        for key, value in kwargs.items():
            config[key] = value

        with tmp_config_path.open("w") as f:
            json.dump(config, f)

    return config


class Message(BaseModel):
    """Model for messages in translation request"""

    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")


class TranslateRequest(BaseModel):
    """Request model for translation"""

    messages: List[Message] = Field(..., description="List of messages for translation")
    source_language: Optional[str] = Field(
        "",
        description=(
            "Source language that is one of the followings: "
            f"{', '.join(SUPPORTED_LANGUAGES)}. "
            "Note that 'English|Japanese' is used to detect the input language automatically."
        ),
    )
    target_language: Optional[str] = Field(
        "",
        description=(
            "Target language that is one of the followings: "
            f"{', '.join(SUPPORTED_LANGUAGES)}. "
            "This can be empty when the source language is 'English|Japanese'."
        ),
    )


def construct_llm_input(request: TranslateRequest) -> List[Message]:
    """Construct the input for the LLM from messages and languages"""

    # If it has already been constructed messages with lang=* part, return it as is
    if request.source_language == "" and request.target_language == "":
        return request.messages

    if request.source_language != "":
        source_text = request.messages[-1].content.strip()
        request.messages[-1].content = f"input lang={request.source_language}\n" + source_text
    if request.target_language != "":
        request.messages.append(Message(role="user", content=f"output lang={request.target_language}\n"))
    else:
        request.messages.append(Message(role="user", content="output\n"))

    return request.messages
