"""Locate GGUF models for llama.cpp: local files or Hugging Face repositories.

This module imports neither MLX nor llama.cpp bindings, so it also runs on Linux.
"""

import os
import re
import struct
from pathlib import Path

# Official PLaMo 3 GGUFs; llama.cpp's own `-hf` option also defaults to Q4_K_M.
DEFAULT_GGUF_MODEL = "pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
# Same vocabulary, about a third of the 2B model's cost per token.
DEFAULT_DRAFT_MODEL = "pfnet/plamo-3-translate-0.8b-GGUF:Q4_0"
DEFAULT_QUANTIZATION = "Q4_K_M"
SUPPORTED_ARCHITECTURES = ("plamo2", "plamo3")

_REPOSITORY = re.compile(r"[\w.-]+/[\w.-]+")
_SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
# GGUF value types with a fixed size: struct format characters.
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING = 8
_ARRAY = 9


def canonical_model(name: str) -> str:
    """A local path made absolute, or `<user>/<repo>:<QUANT or file.gguf>`; never touches the network."""
    path = Path(name).expanduser()
    if path.exists() or name.lower().endswith(".gguf") and ":" not in name:
        # Not resolve(): Hugging Face cache entries are symlinks to blobs without the .gguf suffix.
        return str(path.absolute())
    repository, _, selector = name.partition(":")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError(f"{name} is neither a GGUF file nor a Hugging Face repository (<user>/<repo>[:QUANT])")
    selector = selector or DEFAULT_QUANTIZATION
    return f"{repository}:{selector if selector.lower().endswith('.gguf') else selector.upper()}"


def resolve_model(name: str, local_only: bool = False) -> str:
    """Path of the GGUF file for a canonical model, downloading it into the Hugging Face cache if needed.

    With `local_only`, raise FileNotFoundError instead of downloading.
    """
    if not _REPOSITORY.fullmatch(name.partition(":")[0]) or Path(name).exists():
        return name
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repository, _, selector = canonical_model(name).partition(":")
    try:
        if selector.lower().endswith(".gguf"):
            return hf_hub_download(repository, selector, local_files_only=local_only)
        # Hub patterns are case-sensitive; repositories spell quantizations both ways.
        pattern = "*" + "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in selector) + ".gguf"
        snapshot = Path(snapshot_download(repository, allow_patterns=[pattern], local_files_only=local_only))
    except LocalEntryNotFoundError as exc:
        if local_only:
            raise FileNotFoundError(f"{name} is not in the Hugging Face cache") from exc
        raise
    # Q4_0 must not match IQ4_0, and split GGUFs would need every shard.
    suffix = re.compile(rf"(^|[^A-Za-z0-9]){re.escape(selector)}\.gguf$", re.IGNORECASE)
    matches = sorted(p for p in snapshot.rglob("*.gguf") if suffix.search(p.name) and not _SHARD.search(p.name))
    if len(matches) == 1:
        return str(matches[0])
    if local_only and not matches:
        # A cached snapshot can hold another quantization only.
        raise FileNotFoundError(f"{name} is not in the Hugging Face cache")
    if not matches and not any(snapshot.rglob("*.gguf")):
        raise ValueError(
            f"{repository} has no {selector} GGUF; for MLX checkpoints such as pfnet/plamo-3-translate-2b, "
            "use the -GGUF repository or --backend-type mlx"
        )
    found = ", ".join(p.name for p in matches) or "none"
    raise ValueError(f"Expected one {selector} GGUF in {repository}, found {found}. Pass <repo>:<file>.gguf instead.")


def same_model(first: str | None, second: str | None) -> bool:
    if first is None or second is None:
        return first == second
    return os.path.realpath(first) == os.path.realpath(second) if Path(first).exists() else first == second


def gguf_architecture(path: str) -> str:
    """`general.architecture` from a GGUF header, read without loading any tensor."""
    metadata = gguf_metadata(path, until="general.architecture")
    if not isinstance(metadata.get("general.architecture"), str):
        raise ValueError(f"{path} does not declare general.architecture")
    return metadata["general.architecture"]


def gguf_metadata(path: str, until: str | None = None) -> dict:
    """Scalar and string values of a GGUF header, up to and including the key `until`; arrays are skipped."""
    metadata = {}
    with open(path, "rb") as stream:

        def read(form: str):
            size = struct.calcsize("<" + form)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError(f"{path} is not a complete GGUF file")
            return struct.unpack("<" + form, data)[0]

        def string() -> bytes:
            size = read("Q")
            data = stream.read(size)
            if len(data) != size:
                raise ValueError(f"{path} is not a complete GGUF file")
            return data

        def skip(kind: int) -> None:
            if kind in _SCALARS:
                stream.seek(struct.calcsize("<" + _SCALARS[kind]), os.SEEK_CUR)
            elif kind == _STRING:
                stream.seek(read("Q"), os.SEEK_CUR)
            elif kind == _ARRAY:
                item, count = read("I"), read("Q")
                if item in _SCALARS:
                    stream.seek(count * struct.calcsize("<" + _SCALARS[item]), os.SEEK_CUR)
                else:
                    for _ in range(count):
                        skip(item)
            else:
                raise ValueError(f"{path} has an unknown GGUF value type {kind}")

        if stream.read(4) != b"GGUF" or read("I") not in (2, 3):
            raise ValueError(f"{path} is not a GGUF version 2 or 3 file")
        read("Q")  # tensor count
        for _ in range(read("Q")):
            # llama.cpp does not require UTF-8 strings, so neither does this reader.
            key, kind = string().decode("utf-8", "replace"), read("I")
            if kind in _SCALARS:
                metadata[key] = read(_SCALARS[kind])
            elif kind == _STRING:
                metadata[key] = string().decode("utf-8", "replace")
            else:
                skip(kind)
            if key == until:
                break
    return metadata
