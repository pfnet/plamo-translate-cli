import asyncio
from types import SimpleNamespace

import pytest
from mcp.types import TextContent

from plamo_translate.clients.translate import MCPClient


@pytest.mark.parametrize("progress", [[], ["前半"], ["前半", "後半"]])
def test_streaming_final_response_recovers_missing_progress_without_duplicates(progress):
    class Session:
        async def call_tool(self, *args, progress_callback, **kwargs):
            for index, text in enumerate(progress):
                await progress_callback(index, None, text)
            return SimpleNamespace(isError=False, content=[TextContent(type="text", text="前半後半")])

    async def run():
        client = MCPClient.__new__(MCPClient)
        return "".join([chunk async for chunk in client._translate_stream(Session(), None)])

    assert asyncio.run(run()) == "前半後半"


def test_streaming_propagates_server_error():
    class Session:
        async def call_tool(self, *args, **kwargs):
            return SimpleNamespace(isError=True, content=[TextContent(type="text", text="token limit reached")])

    async def run():
        client = MCPClient.__new__(MCPClient)
        return [chunk async for chunk in client._translate_stream(Session(), None)]

    with pytest.raises(RuntimeError, match="token limit reached"):
        asyncio.run(run())
