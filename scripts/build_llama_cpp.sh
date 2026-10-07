#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_dir="${1:-$project_dir/.local/llama.cpp}"
revision=db33d3cb89d8b0d954df66047b13e9cc22df8f10
patch_file="$project_dir/scripts/patches/llama-cpp-plamo2-rope.patch"

if [[ ! -d "$llama_dir" ]]; then
    mkdir -p "$(dirname "$llama_dir")"
    git clone --depth 1 --branch b11318 https://github.com/ggml-org/llama.cpp.git "$llama_dir"
fi
if [[ "$(git -C "$llama_dir" rev-parse HEAD)" != "$revision" ]]; then
    echo "Expected llama.cpp $revision. Choose a new build directory; existing checkout was not changed." >&2
    exit 1
fi
if git -C "$llama_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "PLaMo 2 RoPE fix already applied."
else
    git -C "$llama_dir" apply --check "$patch_file"
    git -C "$llama_dir" apply "$patch_file"
fi
cmake -S "$llama_dir" -B "$llama_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$llama_dir/build" --config Release \
    -j "${CMAKE_BUILD_PARALLEL_LEVEL:-8}" --target llama-server llama-quantize llama-bench
echo "llama-server: $llama_dir/build/bin/llama-server"
