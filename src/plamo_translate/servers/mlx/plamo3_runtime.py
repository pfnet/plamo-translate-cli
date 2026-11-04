"""Load a local PLaMo 3 checkpoint and stream translations."""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import wired_limit
from mlx_lm.utils import load_tokenizer
from transformers import PreTrainedConfig

from plamo_translate.servers.mlx.pfml3 import BOS, MSG, TAG, pfml3_prompt
from plamo_translate.servers.mlx.plamo3 import load_model
from plamo_translate.servers.mlx.plamo3_metal import FastDecoder
from plamo_translate.servers.warnings import suppress_optional_gpu_dependency_warnings

GROUP_SIZE = 64
# Largest FP32 logit difference accepted between the kernels and the plain MLX ops.
SELF_CHECK_TOLERANCE = 0.25

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    text: str
    finish_reason: str | None = None


def quantize(model: nn.Module, precision: str) -> None:
    """Quantize every projection, including lm_head; embeddings are looked up, not streamed."""
    if precision == "bf16":
        return
    if precision not in ("4bit", "6bit", "8bit", "mixed"):
        raise ValueError(f"Unsupported precision: {precision}")

    def predicate(path: str, module: nn.Module) -> bool | dict:
        if not isinstance(module, nn.Linear):
            return False
        if precision == "mixed":
            return {"bits": 4 if ".mlp." in path else 8, "group_size": GROUP_SIZE}
        return True

    bits = 4 if precision == "mixed" else int(precision[:-3])
    nn.quantize(model, group_size=GROUP_SIZE, bits=bits, class_predicate=predicate)


class Plamo3Translator:
    def __init__(self, path: str, precision: str | None = None):
        self.path = Path(path).expanduser()
        self.precision = precision or "4bit"
        # FP16 keeps more mantissa bits than BF16; activations stayed far below its range on real text.
        self.model, self.config = load_model(self.path, mx.float16, lazy=True)
        quantize(self.model, self.precision)
        mx.eval(self.model.parameters())
        with suppress_optional_gpu_dependency_warnings():
            self.tokenizer = load_tokenizer(
                self.path,
                # See loader.py: AutoConfig would import the checkpoint's PyTorch model.
                tokenizer_config_extra={"trust_remote_code": True, "config": PreTrainedConfig()},
            )
        self.tokenizer.add_eos_token("<|plamo:tag|>")
        self.caches = self.model.make_cache()
        mx.eval([(cache.keys, cache.values) for cache in self.caches])
        self.fast = FastDecoder(self.model) if FastDecoder.supports(self.model) else None
        if self.fast is not None and not self._kernels_agree():
            logger.warning("Metal kernels disagree with MLX on this GPU; using the slower plain implementation")
            self.fast = None

    def _kernels_agree(self) -> bool:
        """The kernels assume GPU details (SIMD width, in-place writes) that may vary by chip; compare two steps."""
        prompt = self.tokenizer.encode(f"{BOS}{TAG}dataset{MSG}translation", add_special_tokens=False)
        results = []
        for logits in (self.fast.logits, lambda token, caches: self.model(token[None], caches)[0, -1]):
            for cache in self.caches:
                cache.reset()
            self.model.prefill(prompt[:-2], self.caches)
            for token in prompt[-2:]:
                step = logits(mx.array([token], dtype=mx.uint32), self.caches).astype(mx.float32)
                mx.eval(step, [(cache.keys, cache.values) for cache in self.caches])
            results.append(step)
        return mx.max(mx.abs(results[0] - results[1])).item() <= SELF_CHECK_TOLERANCE

    def _logits(self, token: mx.array) -> mx.array:
        if self.fast is not None:
            return self.fast.logits(token, self.caches)
        return self.model(token[None], self.caches)[0, -1].astype(mx.float32)

    def _greedy(self, token: mx.array, limit: int) -> Iterator[int]:
        # The next step is encoded while the GPU finishes this one, so it never idles.
        pending = None
        try:
            for _ in range(limit):
                if self.fast is not None:
                    following = self.fast.next_token(token, self.caches)
                else:
                    following = mx.argmax(self._logits(token), keepdims=True).astype(mx.uint32)
                mx.async_eval(following)
                if pending is not None:
                    yield pending.item()
                pending = token = following
            if pending is not None:
                yield pending.item()
        finally:
            # In-flight dispatches write the caches in place; finish them before any reuse.
            mx.eval(token)

    def _sampled(
        self, token: mx.array, prompt: list[int], limit: int, sampler: Callable, processors: list[Callable]
    ) -> Iterator[int]:
        history = list(prompt)
        for _ in range(limit):
            logits = self._logits(token)[None]
            for processor in processors:
                logits = processor(mx.array(history), logits)
            token = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(mx.uint32)
            mx.eval(token, [(cache.keys, cache.values) for cache in self.caches])
            history.append(token.item())
            yield history[-1]

    def stream(
        self,
        messages: list,
        max_tokens: int,
        prefill_step: int = 512,
        sampler: Callable | None = None,
        logits_processors: list[Callable] | None = None,
    ) -> Iterator[Segment]:
        """Yield text as it decodes. Without a sampler or processors, decode greedily on the GPU."""
        prompt = self.tokenizer.encode(pfml3_prompt(messages), add_special_tokens=False)
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        with wired_limit(self.model):
            for cache in self.caches:
                cache.reset()
            self.model.prefill(prompt[:-1], self.caches, prefill_step)
            start = mx.array(prompt[-1:], dtype=mx.uint32)
            if sampler is None and not logits_processors:
                tokens = self._greedy(start, max_tokens)
            else:
                tokens = self._sampled(start, prompt, max_tokens, sampler, logits_processors or [])
            finish_reason = "length"
            try:
                for token in tokens:
                    if token in self.tokenizer.eos_token_ids:
                        finish_reason = "stop"
                        break
                    detokenizer.add_token(token)
                    yield Segment(detokenizer.last_segment)
            finally:
                tokens.close()
            detokenizer.finalize()
            yield Segment(detokenizer.last_segment, finish_reason)
