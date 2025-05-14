import asyncio
import logging
from typing import AsyncGenerator, Dict, List
from urllib.parse import urlunparse

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.session import RequestResponder
from mcp.types import TextContent

from plamo_translate.servers.utils import Message, TranslateRequest, update_config

logger = logging.getLogger(__name__)


async def message_handler(
    message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
) -> None:
    if isinstance(message, Exception):
        logger.error("Error: %s", message)
        return


class MCPClient:
    def __init__(self, stream: bool) -> None:
        """Initialize the MCP client.

        Args:
            stream (bool): Whether to stream the translation results.
        """
        self.stream = stream
        self.config = update_config()

        port = self.config.get("port", None)
        if port is None:
            raise ValueError("Port is not set in the configuration. Please start the MCP server first.")
        self.url = urlunparse(("http", f"127.0.0.1:{port}", "mcp", "", "", ""))

    async def translate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Translate messages. If stream=True, yields chunks as they arrive."""
        async with streamablehttp_client(self.url) as (
            read_stream,
            write_stream,
            get_session_id_callback,
        ):
            async with ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                message_handler=message_handler,
            ) as session:
                await session.initialize()

                messages_obj = [Message(**message) for message in messages]
                request = TranslateRequest(messages=messages_obj, source_language="", target_language="")

                if self.stream:
                    # For streaming, we'll need to handle the response differently
                    # This will yield chunks as they arrive
                    async for chunk in self._translate_stream(session, request):
                        yield chunk
                else:
                    # The messages should already have source and target languages, so omit to specify them again
                    response = await session.call_tool(
                        "plamo-translate",
                        arguments={
                            "request": request,
                            "stream": False,
                        },
                    )

                    # Extract text from response content
                    if response.content and len(response.content) > 0:
                        content = response.content[0]
                        if isinstance(content, TextContent):
                            yield content.text
                        else:
                            raise ValueError(f"Unexpected content type: {type(content)}")
                    else:
                        raise ValueError("Empty response from translation tool")

    async def _translate_stream(self, session: ClientSession, request: TranslateRequest):
        """Handle streaming translation responses."""
        # Use a queue to pass messages from progress_handler to the generator
        message_queue: asyncio.Queue[str] = asyncio.Queue()
        call_complete = asyncio.Event()

        async def progress_handler(progress: float, total: float | None, message: str | None) -> None:
            """Handle progress updates which might contain partial translations."""
            if message:
                await message_queue.put(message)

        async def call_tool_wrapper():
            """Wrapper to call the tool and signal completion"""
            try:
                response = await session.call_tool(
                    "plamo-translate",
                    arguments={
                        "request": request,
                        "stream": True,
                    },
                    progress_callback=progress_handler,
                )
                # Put the final response in the queue if needed
                if response.content and len(response.content) > 0:
                    content = response.content[0]
                    if isinstance(content, TextContent):
                        await message_queue.put(content.text)
            finally:
                call_complete.set()

        # Start the tool call in the background
        asyncio.create_task(call_tool_wrapper())

        # Yield messages as they arrive
        chunks = []
        while not call_complete.is_set() or not message_queue.empty():
            try:
                message = await asyncio.wait_for(message_queue.get(), timeout=0.1)
                chunks.append(message)
                yield message
            except asyncio.TimeoutError:
                # No message available, continue waiting
                continue
