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

#### Use original local weights with the optimized MLX backend

```sh
uv run plamo-translate server \
  --model /Users/shunta/Models/pfnet--plamo-2-translate --precision 4bit
uv run plamo-translate --from English --to Japanese < benchmarks/translation.en.txt
```

`--model` accepts a local checkpoint directory or a Hugging Face repository and
takes precedence over `PLAMO_TRANSLATE_CLI_MODEL_NAME`. Original weights are
quantized once during loading; the source files are never modified. By default,
original weights use 4bit and exported checkpoints retain their declared precision,
including unquantized BF16 exports.
Request `--precision bf16` to retain the original weights. Changing a quantized
checkpoint's precision requires loading the original weights again.

For faster subsequent startup, save a separate copy first:

```sh
uv run python scripts/prepare_mlx_model.py \
  --model /Users/shunta/Models/pfnet--plamo-2-translate \
  --output /Users/shunta/Models/pfnet--plamo-2-translate-mlx-4bit \
  --precision 4bit
uv run plamo-translate server --model /Users/shunta/Models/pfnet--plamo-2-translate-mlx-4bit
```

`mixed` uses 4bit MLPs, 8bit attention/large Mamba projections and embeddings,
and original precision for the small Mamba state projections. `4bit` and `6bit`
are also available. These settings change model numerics and may change translations.
Stop an existing server before changing its model or precision.

The backend honors the checkpoint's local/full-attention RoPE bases and PLaMo's
unclipped SSM time steps, keeps recurrent state in FP32, and preserves the original BF16 activation dtype.
An experimental fused decode convolution is available with
`PLAMO_TRANSLATE_CLI_OPTIMIZE=1`; it is disabled by default because measured speed
was indistinguishable from the upstream convolution and rounding changed some tokens.
The prompt includes the checkpoint's BOS token. GPU requests are serialized;
streaming also returns a final complete result to recover missing progress messages.

#### Reproduce the translation benchmark

```sh
uv sync --locked
uv run python scripts/benchmark_mlx.py \
  --model /Users/shunta/Models/pfnet--plamo-2-translate \
  --precision 8bit --runs 3 --output benchmarks/results/my-8bit
```

The supplied full English/Japanese example is in `benchmarks/translation.en.txt`
and `benchmarks/translation.ja.txt`. Each process runs a short warmup, then uses greedy
decoding with a fresh cache for every measured run, and records the full translation, token IDs, EOS/length
termination, load time, prompt/decode throughput, memory, package versions, backend
hash, and process snapshots. A host lock prevents concurrent runs of this script;
stop other GPU workloads while measuring. Timing excludes loading and warmup.

Quality checks include chrF against the supplied reference (higher is better) and
teacher-forced reference negative log likelihood (NLL, lower is better). They measure
this one example, not general translation accuracy; also inspect omissions, names,
paragraphs and repetition in the saved output. `--implementation corrected` (the default) uses
the production backend; `optimized` enables the experimental convolution; `stock --legacy-prompt`
reproduces the upstream model and old CLI prompt. The benchmark also supports
`--dtype float16` as an explicit numerical experiment.

#### Measured result (2026-10-07)

Apple M1 Max, 32 GPU cores, 64 GB; MLX 0.31.1 / mlx-lm 0.31.2,
Python 3.13.15. Source checkpoint revision:
`cae8da342a3e051ed69f90ce24c23eacff908732`. Two warm runs per setting,
batch size 1, temperature 0, prefill chunk 512, full supplied example.
Other llama.cpp workloads were stopped for the final comparison.

| Setting | Decode tok/s (median) | Full generation seconds (median) | chrF ↑ | Reference NLL ↓ | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old model implementation + old prompt, same source quantized to 4bit | 51.32 | 9.58 | 64.70 | 0.6030 | 6.56 |
| **Selected: corrected model + BOS, 4bit/BF16 activations** | **51.44** | **9.61** | **73.77** | **0.4595** | **6.57** |
| Corrected model + BOS, original BF16 weights | 15.38 | 28.49 | 73.46 | 0.4919 | 19.87 |
| Experimental fused convolution, 8bit/BF16 activations | 31.52 | 14.72 | 73.54 | 0.4936 | 11.11 |
| Experimental fused convolution, mixed/BF16 activations | 42.56 | 11.23 | 72.71 | 0.4861 | 8.05 |

The selected 4bit mode retains the old 4bit speed while improving this example's
quality. Quantization accounts for most of the speed advantage over original BF16
weights. BF16 timing varied (13.45–17.32 tok/s); selected 4bit runs were
51.41–51.46 tok/s. These are local measurements, not a claim of globally optimal
speed or unchanged accuracy on other documents/languages. The old-model comparison
uses the same original checkpoint quantized locally, not a different community revision.

The selected output has all eight paragraphs, the title's “Together with You”,
the partner/social-implementation paragraph, both founders, and “Learn or Die”.
The old output dropped the title and the separate partner paragraph. The selected
output is not identical to the reference: it still includes awkward wording such
as `真に価値ある価値`, and uses `あなたと共に` rather than the reference's `皆様と共に`.
The original BF16 output also omitted the title's “Together with You”.
One reference cannot establish general quality; use 8bit/BF16 when validating other
material if small quantization changes are unacceptable.

The experimental fused convolution was not selected: 51.66 versus 51.44 tok/s is
too small a difference to establish a win, and its output changed. Converting all
floating weights/activations to FP16 was rejected: reference logits became nonfinite,
the output repeated the unknown token and reached the 2048-token limit. No FP16
conversion is applied by the production loader.

Raw measurements and complete translations are in
[`benchmarks/results`](benchmarks/results); the selected output is
[`final-corrected-4bit/translation-0.txt`](benchmarks/results/final-corrected-4bit/translation-0.txt).
The saved 4bit checkpoint is 5.36 GB. Reloading that checkpoint through the MCP server
took 4.68 seconds; complete CLI translations took 10.64 seconds (streaming) and
10.23 seconds (non-streaming), and both matched direct inference byte-for-byte.
The selected weights are published at
[`mlx-community/plamo-2-translate`](https://huggingface.co/mlx-community/plamo-2-translate/tree/900f34ccc16d4e8592b1baa38889e2f53f6717e8).
That revision includes the corrected standalone MLX implementation and BOS/EOS
settings. Its output matched the selected translation exactly, and all 15 published
files were verified by SHA-256. This release uses ordinary affine quantization;
it replaces the previous DWQ-labeled release, whose quality was not compared here.
Reproduce this check with:

```sh
uv run python scripts/validate_mlx_cli.py \
  --model /Users/shunta/Models/pfnet--plamo-2-translate-mlx-4bit \
  --expected benchmarks/results/final-corrected-4bit/translation-0.txt \
  --output benchmarks/results/my-cli-roundtrip
```

Reference implementations: [PFN checkpoint](https://huggingface.co/pfnet/plamo-2-translate)
and [upstream MLX PLaMo 2](https://github.com/ml-explore/mlx-lm/blob/v0.31.2/mlx_lm/models/plamo2.py).

#### Standalone 8bit and BF16 releases

| Release | Published revision | Weight size | chrF | Verified files |
| --- | --- | ---: | ---: | ---: |
| 8bit | [`6b1851db`](https://huggingface.co/mlx-community/plamo-2-translate-8bit/tree/6b1851db9edd12c3b7a8b2c33f7505a87380a3ae) | 10.12 GB | 73.94 | 16 |
| BF16 | [`bcc519c8`](https://huggingface.co/mlx-community/plamo-2-translate-bf16/tree/bcc519c834c0eb0f980168d48d1d9239e3dae175) | 19.06 GB | 73.46 | 18 |

Both releases reproduce corrected direct inference at the same precision in two
standalone runs and both CLI modes. chrF uses the supplied example only; BF16 still
omits “Together with You” in the title. All published files match the validated
local bytes. Full outputs and publication hashes are in
[`benchmarks/releases/2026-10-07`](benchmarks/releases/2026-10-07).

`--precision 8bit` selects `mlx-community/plamo-2-translate-8bit` and
`--precision bf16` selects `mlx-community/plamo-2-translate-bf16` unless a model
path or repository was explicitly provided. The exporter supports both precisions:

```sh
uv run python scripts/prepare_mlx_model.py \
  --model /path/to/pfnet-plamo-2-translate --output /path/to/mlx-8bit --precision 8bit
uv run python scripts/prepare_mlx_model.py \
  --model /path/to/pfnet-plamo-2-translate --output /path/to/mlx-bf16 --precision bf16
```

Each export includes the standalone `modeling_mlx_plamo2.py`, a BOS-aware chat
template, and both EOS IDs. BF16 export preserves the source floating weights
without quantization; 8bit uses affine group-64 quantization while retaining BF16
floating parameters. No FP16 activation conversion or experimental convolution is
enabled. A saved BF16 export also retains its precision when loaded via `--model`
without `--precision`.

Validate against the full output of corrected direct inference at the same
precision, then validate both CLI delivery modes:

```sh
uv run python scripts/validate_mlx_release.py \
  --model /path/to/mlx-8bit --precision 8bit \
  --source-revision cae8da342a3e051ed69f90ce24c23eacff908732 \
  --expected /path/to/direct-8bit-translation.txt --output /path/to/validation-8bit
uv run python scripts/validate_mlx_cli.py \
  --model /path/to/mlx-8bit --expected-precision 8bit \
  --expected /path/to/direct-8bit-translation.txt --output /path/to/cli-8bit
```

Use `bf16` and the corresponding source-precision output for BF16 validation.
The standalone check calls `mlx_lm.load` directly, verifies identical structured
and natural-message prompts, requires EOS, compares the complete output, and hashes
every inference asset. Equality is with the same-precision implementation; it does
not imply equality between 8bit and BF16 or with the supplied human reference.

For publication, prepare the model card and license files, then run the local
preflight below. Add `--upload` to publish using existing Hugging Face credentials:

```sh
uv run python scripts/publish_mlx_model.py \
  --model /path/to/mlx-8bit --repo mlx-community/plamo-2-translate-8bit \
  --validation /path/to/validation-8bit/validation.json \
  --expected-head PREVIOUS_HUGGING_FACE_COMMIT --output /path/to/publication-8bit
```

Publication rejects modified inference files, guards against concurrent remote
updates, removes obsolete weight shards in the same commit, and verifies remote
SHA-256 values after upload. It uses an explicit file allowlist; local provenance
paths and supplied translation text are excluded from the model repository.

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
- --model Local model directory or Hugging Face repository
- --precision Model weight precision: [4bit, 6bit, 8bit, mixed, bf16] (default: 4bit for original weights)

## Configuration

You can configure the following parameters using environment variables:

- `PLAMO_TRANSLATE_CLI_SERVER_START_PORT`: Specifies the starting port number for the server.
- `PLAMO_TRANSLATE_CLI_SERVER_END_PORT`: Specifies the ending port number for the server.
- `PLAMO_TRANSLATE_CLI_MODEL_NAME`: Local model directory or Hugging Face repository.
- `PLAMO_TRANSLATE_CLI_PRECISION`: Explicit weight precision, equivalent to `--precision`.
- `PLAMO_TRANSLATE_CLI_OPTIMIZE`: Set to `1` to opt in to the experimental decode convolution (default: `0`; correctness fixes are always enabled).
- `PLAMO_TRANSLATE_CLI_PREFILL_STEP_SIZE`: Prompt chunk size (default: 512).
- `PLAMO_MAX_TOKENS`: Maximum generated tokens (default: 32768). Reaching this limit reports an incomplete translation error.
- `PLAMO_TRANSLATE_CLI_TEMP`: Sets the temperature for text generation.
- `PLAMO_TRANSLATE_CLI_TOP_P`: Sets the top-p (nucleus) sampling probability.
- `PLAMO_TRANSLATE_CLI_TOP_K`: Sets the top-k sampling number.
- `PLAMO_TRANSLATE_CLI_REPETITION_PENALTY`: Sets the repetition penalty.
- `PLAMO_TRANSLATE_CLI_REPETITION_CONTEXT_SIZE`: Sets the context size for repetition penalty.

## Deploy

```sh
bash scripts/deploy.sh
```
