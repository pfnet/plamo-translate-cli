#!/usr/bin/env bash
set -euo pipefail

version="$(uv version --short)"
# PyPI accepts a new file name for an existing version, and files can never be replaced: bump the version first.
if curl -sf "https://pypi.org/pypi/plamo-translate/${version}/json" > /dev/null; then
    echo "plamo-translate ${version} is already on PyPI; bump the version in pyproject.toml and uv.lock." >&2
    exit 1
fi

rm -rf dist
# The package is pure Python: publish its py3-none-any wheel for every platform. Dependency markers install MLX on
# Apple Silicon only; elsewhere the CLI uses llama.cpp.
uv build --wheel

uv run twine upload dist/*
