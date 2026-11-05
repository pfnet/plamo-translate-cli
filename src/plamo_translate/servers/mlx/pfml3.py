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
# Files of an original PLaMo 3 checkpoint that the MLX backend loads.
CHECKPOINT_FILES = ["*.json", "*.safetensors", "tokenizer.jsonl", "tokenization_plamo.py"]
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
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")
_ALPHABET = re.compile(r"[A-Za-z]")
_HIRAGANA = re.compile(r"[\u3041-\u309f]")
_WORD = re.compile(r"[A-Za-z]+")
# Quoted text, such as 'ありがとう' in an English question, says nothing about the language around it.
_QUOTED = re.compile(r"(?<![A-Za-z])'[^'\n]*'(?![A-Za-z])|\"[^\"\n]*\"|“[^”\n]*”|‘[^’\n]*’")
_ENGLISH_FUNCTION_WORDS = frozenset(
    """a an the of to in on at by for from with and or but if is are was were be been am do does did not no can could
    will would should may might must have has had i you he she it we they me him her us them my your his its our their
    this that these those there here what which who whom whose when where why how so as than then very too also just
    please""".split()
)


def checkpoint_config(name: str) -> dict | None:
    """config.json of a local checkpoint directory or a Hugging Face repository."""
    local = Path(name).expanduser()
    if local.is_dir():
        config = local / "config.json"
        return json.loads(config.read_text()) if config.is_file() else None
    try:
        from huggingface_hub import hf_hub_download

        config = hf_hub_download(name, "config.json")
    except Exception:
        # Not a repository, or offline without a cached copy: the PLaMo 2 loader reports it.
        return None
    return json.loads(Path(config).read_text())


def checkpoint_complete(folder: str | Path) -> bool:
    """Whether a checkpoint folder has its config, tokenizer and every weight shard.

    A cache can hold part of a snapshot, such as only config.json, and huggingface_hub cannot always tell.
    """
    folder = Path(folder)
    if not (folder / "config.json").is_file() or not (folder / "tokenizer.jsonl").is_file():
        return False
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        return bool(shards) and all((folder / shard).is_file() for shard in shards)
    return (folder / "model.safetensors").is_file()


def is_plamo3_checkpoint(name: str | None) -> bool:
    if not name:
        return False
    config = checkpoint_config(name)
    return config is not None and config.get("model_type") == "plamo3"


def _code(language: str) -> str:
    return LANGUAGE_CODES.get(language, language)


def _opposite(text: str, language: str | None) -> str:
    """PLaMo 3 needs an explicit target: the other of English and Japanese.

    A given English or Japanese source decides. Otherwise grammar decides: hiragana marks Japanese, English function
    words mark English, and names, commands and quotations mark neither. Comparing Japanese characters with Latin
    letters sent Japanese naming commands or products, such as "PLaMo 3 Translate を試してみました。", to Japanese,
    and the model echoed it. Without hiragana, as in Chinese or English with Japanese names, characters still decide.
    """
    source = _code(language) if language and language != AUTO_DIRECTION else None
    if source in ("ja", "ja(easy)"):
        return "en"
    if source == "en":
        return "ja"
    hiragana = len(_HIRAGANA.findall(_QUOTED.sub("", text)))
    if hiragana:
        function_words = sum(word.lower() in _ENGLISH_FUNCTION_WORDS for word in _WORD.findall(text))
        return "en" if hiragana > 2 * function_words else "ja"
    return "en" if len(_JAPANESE.findall(text)) >= len(_ALPHABET.findall(text)) and _JAPANESE.search(text) else "ja"


def pfml3_prompt(messages: list, bos: bool = True) -> str:
    """Convert `input lang=X` / `output lang=Y` messages, ending with an empty output.

    Pass `bos=False` when the runtime adds BOS itself, as llama.cpp does.
    """
    parts = [f"{BOS if bos else ''}{TAG}dataset{MSG}translation"]
    source = None
    source_language = None
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
            source, source_language = text, language
        else:
            if source is None:
                raise ValueError("Each translation must follow its source text")
            target = _code(language) if language else _opposite(source, source_language)
            parts.append(f"{TAG}output{KEY}lang{VAL}{target}{MSG}{text}")
            source = None
    if kind != "output" or not parts[-1].endswith(MSG):
        raise ValueError("The last message must be an empty 'output' line")
    return "".join(parts)
