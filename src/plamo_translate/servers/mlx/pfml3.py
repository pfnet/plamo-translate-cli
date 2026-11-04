"""Build PFML3 prompts from the CLI's request messages.

This module does not import MLX, so prompt checks also run on Linux.
"""

import json
import re
from pathlib import Path

BOS = "<|plamo:bos|>"
TAG = "<|plamo:tag|>"
KEY = "<|plamo:key|>"
VAL = "<|plamo:val|>"
MSG = "<|plamo:msg|>"
RESERVED = "<|plamo:"
AUTO_DIRECTION = "English|Japanese"

LANGUAGE_CODES = {
    "Japanese": "ja",
    "Japanese(easy)": "ja(easy)",
    "English": "en",
    "Chinese": "zh-hans",
    "Taiwanese": "zh-hant",
    "Korean": "ko",
    "Arabic": "ar",
    "Italian": "it",
    "Indonesian": "id",
    "Dutch": "nl",
    "Spanish": "es",
    "Thai": "th",
    "German": "de",
    "French": "fr",
    "Vietnamese": "vi",
    "Russian": "ru",
}

_MESSAGE = re.compile(r"(input|output)(?: lang=([^\n]*))?\n(.*)", re.DOTALL)
_JAPANESE = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
_ALPHABET = re.compile(r"[A-Za-z]")


def is_plamo3_checkpoint(name: str | None) -> bool:
    """Only local checkpoints are recognized; this never contacts a model hub."""
    if not name:
        return False
    config = Path(name).expanduser() / "config.json"
    return config.is_file() and json.loads(config.read_text()).get("model_type") == "plamo3"


def _code(language: str) -> str:
    return LANGUAGE_CODES.get(language, language)


def _opposite(text: str) -> str:
    """PLaMo 3 needs an explicit target, so choose the English/Japanese direction from the source."""
    return "en" if len(_JAPANESE.findall(text)) >= len(_ALPHABET.findall(text)) and _JAPANESE.search(text) else "ja"


def pfml3_prompt(messages: list) -> str:
    """Convert `input lang=X` / `output lang=Y` messages, ending with an empty output."""
    parts = [f"{BOS}{TAG}dataset{MSG}translation"]
    source = None
    kind = None
    for message in messages:
        match = _MESSAGE.fullmatch(message.content)
        if match is None:
            raise ValueError("PLaMo 3 messages must start with an 'input' or 'output' line")
        kind, language, text = match.groups()
        if RESERVED in text:
            raise ValueError(f"Text must not contain reserved PFML tokens ({RESERVED}...)")
        if kind == "input":
            if source is not None:
                raise ValueError("Alternate source text and translation; end with an empty translation")
            if not text.strip():
                raise ValueError("Source text must not be empty")
            attribute = f"{KEY}lang{VAL}{_code(language)}" if language and language != AUTO_DIRECTION else ""
            parts.append(f"{TAG}input{attribute}{MSG}{text}")
            source = text
        else:
            if source is None:
                raise ValueError("Each translation must follow its source text")
            target = _code(language) if language else _opposite(source)
            parts.append(f"{TAG}output{KEY}lang{VAL}{target}{MSG}{text}")
            source = None
    if kind != "output" or not parts[-1].endswith(MSG):
        raise ValueError("The last message must be an empty 'output' line")
    return "".join(parts)
