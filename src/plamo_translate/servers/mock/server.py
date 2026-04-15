import asyncio
import logging
from typing import Dict

from mcp.server.fastmcp import Context, FastMCP

from plamo_translate.servers.utils import INSTRUCTION, TranslateRequest, find_free_port, update_config

logger = logging.getLogger(__name__)

MOCK_TRANSLATIONS: Dict[str, str] = {
    "Proud, but humble": "誇り高いが、謙虚です。",
    "Boldly do what no one has done before": "誰もしたことがないことを大胆にやりなさい。",
}


def _extract_input_text(request: TranslateRequest) -> str:
    for message in reversed(request.messages):
        if not message.content.startswith("input"):
            continue

        _, _, input_text = message.content.partition("\n")
        return input_text.strip().lstrip(">").strip()

    return request.messages[-1].content.strip().lstrip(">").strip()


class PLaMoTranslateServer(FastMCP):
    """Lightweight MCP server used by the test suite."""

    def __init__(self, log_level: str, show_progress: bool = False) -> None:
        super().__init__(
            name="plamo-translate",
            instructions=INSTRUCTION,
            log_level=log_level,
            stateless_http=False,
            host="127.0.0.1",
            port=find_free_port(),
        )
        update_config(model_name="mock")
        self.show_progress = show_progress
        self.add_tool(
            fn=self.translate,
            name="plamo-translate",
            description=INSTRUCTION,
        )

    async def translate(self, request: TranslateRequest, stream: bool, context: Context) -> str:
        input_text = _extract_input_text(request)
        translation = next(
            (candidate for source, candidate in MOCK_TRANSLATIONS.items() if source in input_text),
            f"[mock translation] {input_text}",
        )

        if not stream:
            return translation

        for index, chunk in enumerate([translation], start=1):
            await context.report_progress(progress=index, total=1, message=chunk)
            await asyncio.sleep(0)

        return ""
