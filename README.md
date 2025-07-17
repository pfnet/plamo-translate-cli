[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/pfnet-plamo-translate-cli-badge.png)](https://mseep.ai/app/pfnet-plamo-translate-cli)

# plamo-translate-cli

A command-line interface for translation using the plamo-2-translate model with local execution.

## Features

- Translate text between 16+ languages including Japanese, English, Chinese, Korean, and more
- Simple command-line interface for easy integration into scripts and workflows
- Supports various server backends (MLX, with planned support for Ollama and vLLM)
  - Currently, optimized for macOS with Apple Silicon using MLX framework

## Installation

### For macOS

#### Python>=3.13

The issue arises because the currently distributed sentencepiece package on PyPI (latest version: 0.2.0) is not compatible with Python 3.13 or higher and CMake 4.0 or higher.
As a result, attempting to install sentencepiece as a dependency for this CLI tool package would cause build errors.

However, the latest commit in the main branch of the sentencepiece GitHub repository now supports Python 3.13 and CMake 4.0 or higher (though no release has yet been made).
Therefore, when installing sentencepiece in a Python 3.13 environment, you must first install sentencepiece directly from the GitHub repository.
**This step will likely be unnecessary once the next version of sentencepiece is released.**

```sh
brew install cmake
pip install git+https://github.com/google/sentencepiece.git@2734490#subdirectory=python
pip install plamo-translate
```

#### Python<3.13

```sh
pip install plamo-translate
```

#### [`uv tool`](https://docs.astral.sh/uv/concepts/tools/)

If you use [`uv`](https://github.com/astral-sh/uv) as a package manager rather than `pip`, you can install `plamo-translate` into an isolated environment:

```sh
uv tool install -p 3.12 plamo-translate
```

## Development

```sh
uv sync
source .venv/bin/activate
```

## Requirements

- Python 3.10 or higher
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
