import asyncio
import contextlib
import importlib.resources
import logging
import os
import subprocess
import sys
from typing import Callable, Tuple

import mlx.core as mx
import mlx.nn as nn
from mcp.server.fastmcp import Context, FastMCP
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

from plamo_translate.servers.utils import (
    INSTRUCTION,
    PLAMO_MAX_TOKENS,
    PLAMO_TRANSLATE_CLI_MODEL_NAME,
    PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE,
    PLAMO_TRANSLATE_CLI_REPETITION_PENALTY,
    PLAMO_TRANSLATE_CLI_TEMP,
    PLAMO_TRANSLATE_CLI_TOP_K,
    PLAMO_TRANSLATE_CLI_TOP_P,
    TranslateRequest,
    construct_llm_input,
    find_free_port,
    update_config,
)

logger = logging.getLogger(__name__)


class PLaMoTranslateServer(FastMCP):
    """PLaMo Translate Server using FastMCP."""

    def __init__(self, log_level: str, show_progress: bool = False) -> None:
        super().__init__(
            name="plamo-translate",
            instructions=INSTRUCTION,
            log_level=log_level,
            stateless_http=False,
            host="127.0.0.1",
            port=find_free_port(),
            lifespan=self.lifespan,
        )

        # Set environment variables to switch if it shows progress bars for loading models or not
        self.show_progress = show_progress

        model, tokenizer, sampler, logits_processors = self.load_model()
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.logits_processors = logits_processors

        self.add_tool(
            fn=self.translate,
            name="plamo-translate",
            description=INSTRUCTION,
        )

    @contextlib.asynccontextmanager
    async def lifespan(self, server: FastMCP):
        try:
            async with contextlib.AsyncExitStack() as stack:
                # Pre-processings before a request is processed
                yield
                # Post-processings after a request is processed
        except Exception as e:
            logger.error(f"Error during lifespan: {str(e)} {e}")
            await stack.aclose()

    def load_model(self) -> Tuple[nn.Module, TokenizerWrapper, Callable[..., mx.array], list]:
        """Load the MLX model if not already loaded."""
        try:
            ref = importlib.resources.files("plamo_translate.assets").joinpath("chat_template.jinja2")
            chat_template = ref.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise RuntimeError("chat_template.jinja2 not found in assets directory")

        model_name = os.getenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", PLAMO_TRANSLATE_CLI_MODEL_NAME)
        update_config(model_name=model_name)

        # Reload mlx_lm.utils here to refleect the environment variables for progress bars
        if self.show_progress:
            envs = os.environ
            envs["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
            subprocess.run(
                [sys.executable, "-m", "mlx_lm", "generate", "--model", model_name, "--max-tokens", "1"],
                env=envs,
                stdout=subprocess.DEVNULL,
            )

        model, tokenizer = load(
            model_name,
            model_config={"trust_remote_code": True},
            tokenizer_config={
                "trust_remote_code": True,
                "chat_template": chat_template,
            },
        )
        tokenizer.add_eos_token("<|plamo:op|>")

        sampler = make_sampler(
            temp=float(PLAMO_TRANSLATE_CLI_TEMP),
            top_p=float(PLAMO_TRANSLATE_CLI_TOP_P),
            top_k=int(PLAMO_TRANSLATE_CLI_TOP_K),
        )

        logits_processors = make_logits_processors(
            repetition_penalty=(
                float(PLAMO_TRANSLATE_CLI_REPETITION_PENALTY)
                if PLAMO_TRANSLATE_CLI_REPETITION_PENALTY is not None
                else None
            ),
            repetition_context_size=(
                int(PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE)
                if PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE is not None
                else None
            ),
        )

        return model, tokenizer, sampler, logits_processors

    async def translate(self, request: TranslateRequest, stream: bool, context: Context) -> str:
        """Run the translation tool"""
        logger.info(f"Received translation request: {context.request_id}")
        try:
            messages = construct_llm_input(request)
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)  # type:ignore[call-arg]

            # Generate translation
            translation = ""
            segments_count = 0

            for segment in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                sampler=self.sampler,
                logits_processors=self.logits_processors,
                max_tokens=int(PLAMO_MAX_TOKENS),
            ):
                translation += segment.text
                segments_count += 1

                if stream:
                    # Send progress notification with the new segment
                    await context.report_progress(
                        progress=segments_count,
                        total=None,  # We don't know the total in advance
                        message=segment.text,  # Send the segment as the message
                    )

                    # Small delay to ensure progress is sent
                    await asyncio.sleep(0)

            if not stream:
                return translation
            else:
                return ""

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise e
