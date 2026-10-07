# plamo-translate-cli

A command-line interface for translation using the plamo-2-translate model with local execution.

## Features

- Translate text between 16+ languages including Japanese, English, Chinese, Korean, and more
- Simple command-line interface for easy integration into scripts and workflows
- MLX and native llama.cpp backends, including Metal acceleration on Apple Silicon

## Installation

### For macOS

`plamo-translate` currently installs on Python 3.10 through 3.14 on macOS.
No additional workaround is required for `sentencepiece` on Python 3.13 or 3.14 with current upstream releases.

```sh
pip install plamo-translate
```

#### [`uv tool`](https://docs.astral.sh/uv/concepts/tools/)

If you use [`uv`](https://github.com/astral-sh/uv) as a package manager rather than `pip`, you can install `plamo-translate` into an isolated environment:

```sh
uv tool install -p 3.14 plamo-translate
```

## Development

```sh
uv sync
source .venv/bin/activate
```

## Requirements

- Python 3.10 through 3.14
  - Common dependencies:
    - mcp[cli]
    - numba
  - On macOS:
    - mlx-lm

## Usage

### Basic usage

You can specify the input and output language by giving `--from` and `--to` options.
If you don't specify them, the input/output language will be automatically selected from English or Japanese.

#### Interactive mode

```sh
$ plamo-translate
Loading models...done!
Interactive mode enabled. Type your input below (Ctrl+D to exit).
> こんにちは、お元気ですか？
Hello, how are you?
> 「お腹減った〜何食べたい？」「私はうなぎ！」
"I'm hungry! What do you want to eat?" "I want eel!"
> You translate ambiguous expression in Japanese into English very well.
あなたは日本語の曖昧な表現を英語に翻訳するのがとても上手です。
```

#### Pipe mode

```sh
$ cat file.txt | plamo-translate
The virtual worlds of the internet have experienced remarkable technological advancement. Meanwhile, the real world still contains numerous areas where technology has yet to make significant inroads, with many inefficient manual tasks and dangerous work still requiring human intervention. This situation stems from the fact that conventional technology has struggled to adapt to the dynamic changes and diverse conditions of the real world.

PFN's core strengths lie in machine learning and deep learning technologies, which demonstrate exceptional flexibility in handling uncertainty and have the potential to create significant impact in the real world. For example, by applying deep learning technologies to robots that excel at repetitive tasks, we can enable them to make more human-like flexible judgments and perform complex tasks.

To create meaningful impact in the real world, it's essential to push the boundaries of cutting-edge technology and research application domains where technological innovation can create tangible change. For these purposes, PFN assembles a team of exceptionally talented professionals with diverse expertise.
```

#### Server mode

First, launch the server:

```sh
$ plamo-translate server
```

Then, use the client mode:

```sh
$ plamo-translate --input '家計は火の車だ'
Our household is in financial trouble.
```

You can also use the interactive mode with the server:

```sh
$ plamo-translate
Loading models...done!
Interactive mode enabled. Type your input below (Ctrl+D to exit).
> 家計は火の車だ
Our household is in financial trouble.
```

It can skip the loading time of the model, so it is useful when you want to use this tool frequently.

### Using from MCP Client

The `plamo-translate server` command starts an MCP (Model Context Protocol) server. This allows `plamo-translate` to be used as a tool in other applications that support MCP, such as Claude Desktop.

Here, we introduce how to use `plamo-translate` with Claude Desktop, which is a popular MCP client.

1.  Start the `plamo-translate` server:
    ```sh
    plamo-translate server
    ```
2.  In a new terminal, run the following command to display the MCP configuration for Claude Desktop:
    ```sh
    plamo-translate show-claude-config
    ```
    and you will see the configuration in JSON format as follows:
    ```json
    {
      "mcpServers": {
        "plamo-translate": {
          "command": "/Users/shunta/.linuxbrew/bin/npx",
          "args": [
            "-y",
            "mcp-remote",
            "http://localhost:8000/mcp",
            "--allow-http",
            "--transport",
            "http-only"
          ],
          "env": {
            "PATH": "[THE SAME STRING AS YOUR CURRENT PATH ENVIRONMENT VARIABLE]",
          }
        }
      }
    }
    ```
3.  Copy the outputted configuration.
4.  Paste this configuration into your Claude Desktop's MCP configuration file (on macOS, this is typically located at `~/Library/Application Support/Claude/claude_desktop_config.json`).

Once configured, you can use `plamo-translate` directly from Claude Desktop.

#### Select precision of the model weight

You can specify the precision of the model weight by giving a `--precision` option.

```sh
$ plamo-translate server --precision 8bit
```
## Supported Languages

- Japanese
- Japanese(easy)
- English

### Experimentally Supported Languages

- Chinese
- Taiwanese
- Korean
- Arabic
- Italian
- Indonesian
- Dutch
- Spanish
- Thai
- German
- French
- Vietnamese
- Russian

## Server Backends

- mlx: Optimized for macOS with Apple Silicon (default on macOS)
- llama.cpp: Local GGUF inference with the native `llama-server`, exposed through the same CLI and MCP tool

### llama.cpp on Apple Silicon

Build the tested runtime from this checkout (requires Git, CMake and Xcode Command Line Tools):

```sh
bash scripts/build_llama_cpp.sh
```

The build pins llama.cpp `b11318` (`db33d3cb89d8b0d954df66047b13e9cc22df8f10`) and applies
[`scripts/patches/llama-cpp-plamo2-rope.patch`](scripts/patches/llama-cpp-plamo2-rope.patch).
This fixes a PLaMo 2 loader bug: when layer 0 is Mamba, the common loader sets the rotary
dimension to zero, silently disabling RoPE. The patch restores the attention head dimension.
An unpatched runtime can produce fluent translations with missing paragraphs even with F16 weights.
The backend rejects a runtime that reports `n_rot = 0`. The pinned revision also includes the
[PLaMo BOS/EOS tokenizer fix](https://github.com/ggml-org/llama.cpp/pull/29734).

Convert the original local weights once; the original model directory is read only:

```sh
uv venv .local/convert-venv
uv pip install --python .local/convert-venv/bin/python \
  'torch==2.14.1' 'transformers==4.57.6' 'huggingface-hub<1' \
  sentencepiece numpy safetensors protobuf -e .local/llama.cpp/gguf-py
uv run python scripts/prepare_llama_cpp.py \
  --model-dir /Users/shunta/Models/pfnet--plamo-2-translate \
  --llama-cpp-dir .local/llama.cpp \
  --python .local/convert-venv/bin/python \
  --output-dir /Users/shunta/Models/pfnet--plamo-2-translate-gguf \
  --quantization Q4_0-ssm-f16
```

Conversion saves F16 plus the requested quantizations, logs, checksums, source revision and a
copy of the model license. It refuses to overwrite existing weights. F16 avoids low-bit weight
quantization; activation and normalization arithmetic differ from MLX, so numerical equivalence
is not assumed. Allow about 26 GB for F16 plus the recommended GGUF, in addition to the source.

The recommended `Q4_0-ssm-f16` recipe uses Q4_0 with all Mamba `ssm_out` matrices kept in F16.
On the tested M1 Max, this preserves content omitted by plain Q4_0 while keeping most of its speed.
See [the measured comparison and full translations](benchmarks/llama_cpp_20261007/README.md).
Other available recipes are Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q5_0, Q4_0 and Q4_1.

Start a persistent server, then translate using the existing client:

```sh
uv run plamo-translate server --backend-type llama.cpp \
  --llama-server .local/llama.cpp/build/bin/llama-server \
  --model /Users/shunta/Models/pfnet--plamo-2-translate-gguf/plamo-2-translate-Q4_0-ssm-f16.gguf

# In another terminal:
uv run plamo-translate --from English --to Japanese < tests/fixtures/translation.en.txt
```

For one-shot use, omit `server` and pass the backend, executable and model options with the input.
`--no-stream`, interactive translation, and `show-claude-config` work with either backend.
A plain client reuses the running backend. Explicitly requesting a different model/backend fails
without changing the running server's configuration. Use a separate `TMPDIR` to run both backends.
The native process binds to loopback and is terminated when its owning MCP server exits.

`--precision` selects MLX weights; for llama.cpp, choose the GGUF file with `--model`.
`PLAMO_TRANSLATE_CLI_MODEL_NAME` can also specify the GGUF path, and
`PLAMO_TRANSLATE_CLI_LLAMA_SERVER` can specify the executable.

The llama.cpp defaults are full GPU offload (`--gpu-layers 99`), Flash Attention, F16 KV cache,
one inference slot, a 32768-token context, 2048-token batches, 512-token microbatches, and 8 CPU threads.
Override them with `--ctx-size`, `--batch-size`, `--ubatch-size`, `--threads`, `--gpu-layers`, and
`--flash-attn on|off|auto`. Use `--gpu-layers 0` for CPU execution. Greedy decoding, disabled
repetition penalty, the translation-specific prompt, and the model's BOS/EOS settings preserve
the intended translation behavior. Prompt caching is disabled for repeatable independent requests.
Token/context exhaustion and interrupted responses are reported as errors, including in streaming mode.

Reproduce the supplied English/Japanese quality and speed comparison:

```sh
uv run --with sacrebleu python scripts/benchmark_llama_cpp.py \
  --llama-server .local/llama.cpp/build/bin/llama-server \
  --models /Users/shunta/Models/pfnet--plamo-2-translate-gguf/plamo-2-translate-{F16,Q4_0-ssm-f16}.gguf \
  --repeats 2 --output .local/benchmark-new-run
```

The benchmark saves full translations, raw streaming events, native logs, time to first token,
prompt/decode timings, and chrF against the supplied reference. It warms model weights with a
separate short prompt and measures complete translations without reusing the prompt cache.
Keep other GPU inference, quantization and media workloads idle while measuring.
chrF is reference overlap for this single example; review the actual translations for missing
content and do not interpret it as general translation accuracy.

## Options

- --input TEXT Input text to translate
- --from TEXT Input language for translation (default: English)
- --to TEXT Output language for translation (default: Japanese)
- --precision Model weight precision to use. You can select from: [4bit, 8bit, bf16] (default: 4bit)

## Configuration

You can configure the following parameters using environment variables:

- `PLAMO_TRANSLATE_CLI_SERVER_START_PORT`: Specifies the starting port number for the server.
- `PLAMO_TRANSLATE_CLI_SERVER_END_PORT`: Specifies the ending port number for the server.
- `PLAMO_TRANSLATE_CLI_TEMP`: Sets the temperature for text generation.
- `PLAMO_TRANSLATE_CLI_TOP_P`: Sets the top-p (nucleus) sampling probability.
- `PLAMO_TRANSLATE_CLI_TOP_K`: Sets the top-k sampling number.
- `PLAMO_TRANSLATE_CLI_REPETITION_PENALTY`: Sets the repetition penalty.
- `PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE`: Sets the context size for repetition penalty.

## Deploy

```sh
bash scripts/deploy.sh
```
