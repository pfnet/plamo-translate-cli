import asyncio
import contextlib
import logging
import os
from typing import Callable, Tuple

import mlx.core as mx
import mlx.nn as nn
from mcp.server.fastmcp import Context, FastMCP
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from plamo_translate.servers.mlx.loader import load_translation_model

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
        self.generation_lock = asyncio.Lock()

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
        model_name = os.getenv("PLAMO_TRANSLATE_CLI_MODEL_NAME", PLAMO_TRANSLATE_CLI_MODEL_NAME)
        if self.show_progress:
            enable_progress_bars()
        else:
            disable_progress_bars()
        precision = os.getenv("PLAMO_TRANSLATE_CLI_PRECISION")
        model, tokenizer, config = load_translation_model(
            model_name,
            precision,
            optimize=os.getenv("PLAMO_TRANSLATE_CLI_OPTIMIZE", "0") == "1",
        )
        actual_precision = config.get("plamo_translate_precision")
        if actual_precision is None:
            actual_precision = f"{config['quantization']['bits']}bit" if config.get("quantization") else "bf16"
        update_config(model_name=model_name, precision=actual_precision)

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
        async with self.generation_lock:
            return await self._translate(request, stream, context)

    async def _translate(self, request: TranslateRequest, stream: bool, context: Context) -> str:
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
                prefill_step_size=int(os.getenv("PLAMO_TRANSLATE_CLI_PREFILL_STEP_SIZE", "512")),
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

            if segment.finish_reason == "length":
                raise RuntimeError("Translation reached PLAMO_MAX_TOKENS before EOS; increase the limit and retry")

            # The final result also carries the full text so clients can recover
            # any progress notifications that arrive late or are lost.
            return translation

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise e
