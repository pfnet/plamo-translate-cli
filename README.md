# plamo-translate-cli

A command-line interface for local translation powered by plamo-3-translate-2b, a PFN™ translation model

Built with PLaMo. The example translations in this README are output of PLaMo Translate models; see [License](#license).

## Features

- Translate text between 16+ languages including Japanese, English, Chinese, Korean, and more
- Simple command-line interface for easy integration into scripts and workflows
- MLX and native llama.cpp backends, including Metal acceleration on Apple Silicon; both run PLaMo 3
- PLaMo 3 Translate 2B by default; PLaMo 2 Translate remains available with `--model`

## Installation

`plamo-translate` installs on Python 3.10 through 3.14:

```sh
pip install plamo-translate
```

If you use [`uv`](https://github.com/astral-sh/uv) as a package manager rather than `pip`, you can install
`plamo-translate` into an isolated environment with [`uv tool`](https://docs.astral.sh/uv/concepts/tools/):

```sh
uv tool install -p 3.14 plamo-translate
```

| Platform | Default backend | Notes |
| --- | --- | --- |
| Apple Silicon Mac, macOS 14 or later | MLX | Installs MLX. The default model needs about 2.6 GB of memory while it loads. |
| Linux, Intel Mac | llama.cpp | Installs without MLX; install `llama-server` as described below. |

MLX needs macOS 14 or later: on an Apple Silicon Mac with macOS 13, pip installs an old release without PLaMo 3
Translate. Windows is not supported; use WSL, where the CLI runs as on Linux.

### Linux and Intel Macs

These platforms translate with [llama.cpp](https://github.com/ggml-org/llama.cpp). Install `llama-server` b11960 or
newer, for example with [Homebrew](https://brew.sh) (`brew install llama.cpp`), or build the tested revision from a
checkout of this repository with `scripts/build_llama_cpp.sh`. CI builds that revision on Ubuntu and translates with
it on the CPU; Intel Macs and Homebrew on Linux have not been tested. Put `llama-server` on `PATH`, or point to it:

```sh
export PLAMO_TRANSLATE_CLI_LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
plamo-translate --from English --to Japanese --input "Hello"
```

The first run downloads the default GGUF (1.81 GB); see [PLaMo 3 on llama.cpp](#plamo-3-on-llamacpp).

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
  - On Apple Silicon Macs (macOS 14 or later):
    - mlx-lm
- For the llama.cpp backend: `llama-server` from llama.cpp b11960 (0.8.0) or newer, for example `brew install llama.cpp`;
  measurements used `scripts/build_llama_cpp.sh`

## Usage

### Model

Without `--model`, the MLX backend translates with
[`pfnet/plamo-3-translate-2b`](https://huggingface.co/pfnet/plamo-3-translate-2b).
The first run downloads its original weights (5.76 GB) into the Hugging Face cache, and each
load quantizes them to 4bit or to the precision given with `--precision`; `bf16` skips quantization
and computes in FP16. The model is distributed under the PLaMo Community License; see [License](#license).

With the llama.cpp backend (`--backend-type llama.cpp`, the default on Linux and Intel Macs), the default model is the
official GGUF
[`pfnet/plamo-3-translate-2b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF) at Q4_K_M (1.81 GB),
which loads without conversion. See [PLaMo 3 on llama.cpp](#plamo-3-on-llamacpp).

PLaMo 2 remains available. Pass its repository with `--model`, or set
`PLAMO_TRANSLATE_CLI_MODEL_NAME` to use it for every run:

```sh
plamo-translate --model mlx-community/plamo-2-translate
```

### Basic usage

You can specify the input and output language by giving `--from` and `--to` options.
If you don't specify them, the input/output language will be automatically selected from English or Japanese.

#### Interactive mode

```sh
$ plamo-translate
Interactive mode enabled. Type your input below (Ctrl+D to exit).
> こんにちは、お元気ですか？
Hello, how are you?
> 「お腹減った〜何食べたい？」「私はうなぎ！」
"I'm hungry~ What do you want to eat?" "I want eel!"
> You translate ambiguous expression in Japanese into English very well.
日本語の曖昧な表現を英語にうまく訳していますね。
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
The household finances are in dire straits.
```

You can also use the interactive mode with the server:

```sh
$ plamo-translate
Interactive mode enabled. Type your input below (Ctrl+D to exit).
> 家計は火の車だ
The household finances are in dire straits.
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
          "command": "/path/to/npx",
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

The default PLaMo 3 model converts its original weights to the requested precision on every load;
`bf16` keeps them unquantized and computes in FP16.

### PLaMo 2 checkpoints on MLX

This section and its benchmarks cover PLaMo 2. For the default model, see
[PLaMo 3 checkpoints on Apple Silicon](#plamo-3-checkpoints-on-apple-silicon).

#### Use original local weights with the optimized MLX backend

```sh
uv run plamo-translate server \
  --model ~/Models/pfnet--plamo-2-translate --precision 4bit
uv run plamo-translate --from English --to Japanese < tests/fixtures/translation.en.txt
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
  --model ~/Models/pfnet--plamo-2-translate \
  --output ~/Models/pfnet--plamo-2-translate-mlx-4bit \
  --precision 4bit
uv run plamo-translate server --model ~/Models/pfnet--plamo-2-translate-mlx-4bit
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
  --model ~/Models/pfnet--plamo-2-translate \
  --precision 4bit --runs 3 --output .local/benchmarks/my-4bit
```

The supplied full English/Japanese example is in `tests/fixtures/translation.en.txt`
and `tests/fixtures/translation.ja.txt`. Each process runs a short warmup, then uses greedy
decoding with a fresh cache for every measured run, and records the full translation, token IDs, EOS/length
termination, load time, prompt/decode throughput, memory, package versions, backend
hash. A host lock prevents concurrent runs of this script;
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

Only reviewed summaries are versioned under [`benchmarks/`](benchmarks/README.md).
Raw measurements, translations and logs stay in ignored `.local/` directories.
The benchmark does not collect a machine-wide process list.
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
  --model ~/Models/pfnet--plamo-2-translate-mlx-4bit \
  --expected .local/benchmarks/my-4bit/translation-0.txt \
  --output .local/benchmarks/my-cli-roundtrip
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
local bytes. A summary of the measurements and publication checks is in
[`benchmarks/releases/2026-10-07`](benchmarks/releases/2026-10-07).

Select them with `--model mlx-community/plamo-2-translate-8bit` or
`--model mlx-community/plamo-2-translate-bf16`. Without `--model` or
`PLAMO_TRANSLATE_CLI_MODEL_NAME`, `--precision` applies to the default PLaMo 3 model.
The exporter supports both precisions:

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

The CLI accepts the languages below. The PLaMo 3 Translate model card lists all of them among its
53 languages except `Japanese(easy)`, which the CLI passes to PLaMo 3 as `ja(easy)`. The supported
and experimental groups below describe PLaMo 2.

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

- mlx: Optimized for Apple Silicon (default there)
- llama.cpp: Local GGUF inference with the native `llama-server`, exposed through the same CLI and MCP tool
  (default on Linux and Intel Macs).
  It starts in about a second (two with the draft model that servers and interactive mode load), and with the
  draft it decodes PLaMo 3 at about 130 tok/s on M1 Max

### PLaMo 3 checkpoints on Apple Silicon

`pfnet/plamo-3-translate-2b` is the default model. `--model` also accepts the Hugging Face repository of another original PLaMo 3 translation checkpoint, or a local directory holding one. On first use the original weights are downloaded into the Hugging Face cache; they are converted and quantized each time they are loaded. The checkpoint type is read from its `config.json`:

```sh
plamo-translate --from English --to Japanese --input "Hello"
plamo-translate --model /path/to/plamo-3-translate-checkpoint --from English --to Japanese --input "Hello"
```

Already quantized checkpoints such as the MLX-4bit release are rejected before their weights are downloaded.

`--precision 4bit` (default) and `bf16` run on fused Metal kernels; `bf16` computes in FP16, which keeps more mantissa bits than BF16. Other precisions use plain MLX ops. On startup one decoded step is compared with plain MLX ops, and the server falls back to them with a warning if the kernels disagree. With `English|Japanese` the direction is chosen from the source text, since PLaMo 3 needs an explicit target language.

Measurements and limitations are in the [benchmark summary](benchmarks/plamo3_mlx_20261104/README.md). Reproduce them with:

```sh
uv run python scripts/benchmark_plamo3.py --model pfnet/plamo-3-translate-2b --output .local/benchmarks/plamo3-4bit
```

### PLaMo 3 on llama.cpp

Install llama.cpp b11960 (0.8.0) or newer, for example with `brew install llama.cpp`, or build the tested
revision as described [below](#llamacpp-runtime-and-plamo-2-ggufs). The measurements here used that build. Homebrew's
llama.cpp 0.8.0 is the same revision but links Homebrew's separately packaged ggml, and it was not measured.
Then select the backend:

```sh
plamo-translate --backend-type llama.cpp --from English --to Japanese --input "Hello"
```

The first run downloads `pfnet/plamo-3-translate-2b-GGUF` Q4_K_M (1.81 GB) into the Hugging Face cache. Later runs
start `llama-server` in about a second; the first launch after installing llama.cpp also compiles its Metal shaders,
which took about 8 seconds on M1 Max. `--model` accepts a local GGUF file, or a Hugging Face GGUF repository
followed by an optional quantization or file name:

```sh
plamo-translate --backend-type llama.cpp --model pfnet/plamo-3-translate-2b-GGUF:Q4_0 --input "Hello"
plamo-translate --backend-type llama.cpp --model /path/to/model.gguf --input "Hello"
```

`plamo-translate server --backend-type llama.cpp` and interactive mode also load
[`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) Q4_0 (0.55 GB) as a
draft model for speculative decoding: it proposes up to six tokens, and the 2B model checks them in one batch.
One-shot translations skip it: loading it took 0.8 s, which on M1 Max pays off only after about 420 (Japanese to
English) to 560 (English to Japanese) generated tokens, more than the fixtures' 350 to 390. Pass `--draft-model <GGUF or repository>` to use a draft
anyway, for example for long documents, or `--draft-model none` to disable it. Other models than the default get a
draft only with `--draft-model`. If the default draft cannot be downloaded, for example offline, translation
continues without it.

Measured on M1 Max with the English fixture ([summary](benchmarks/llama_cpp_plamo3_20261105/README.md)):

| GGUF | Draft | Decode tok/s | Translation s |
| --- | --- | ---: | ---: |
| Q4_K_M (default) | none | 111 | 3.82 |
| Q4_K_M | 0.8B Q4_0 | 132 | 3.26 |
| Q4_0 | none | 134 | 3.12 |
| Q4_0 | 0.8B Q4_0 | 141 | 2.99 |

The draft did not change any output. With the previous request and server settings, Q4_K_M decoded at 68 tok/s;
greedy decoding no longer runs the CPU-side top-p and min-p filters over the whole vocabulary. The fused MLX kernels still decode
faster (170 tok/s), but a complete one-shot CLI translation of a short sentence took 2.4 seconds with llama.cpp and
6 to 7 seconds with MLX, which quantizes the original weights on every load.

Q4_K_M is the default because the model card reports Q4_0 0.90 chrF++ lower than BF16 on FLORES+ translation between
Japanese and other languages, while both stay within 0.0011 of BF16 on its English–Japanese metrics.

A BF16 GGUF converted from the original weights translated both fixtures byte-identically to the MLX FP16 backend.
No llama.cpp release honors the GGUF's `yarn_truncate = false` yet
([#30206](https://github.com/ggml-org/llama.cpp/pull/30206), [#30337](https://github.com/ggml-org/llama.cpp/pull/30337)):
it widens the YaRN correction range of the three full-attention layers from 13.96–30.02 to 13–31. The error grows
with input length, to a KL divergence of 0.04 at 2,048 tokens and 0.66 at 16,384, so the backend passes
`--yarn-beta-fast` and `--yarn-beta-slow` values that make llama.cpp round to 14–30 instead, cutting it to 3e-4 and
5e-3. In an 8,000-token document, 66 of 315 lines had differed from the MLX backend, including a mistranslated term;
with the 14–30 range, 2 to 24 did ([summary](benchmarks/llama_cpp_plamo3_yarn_20261105/README.md)). These options
must be removed once a llama.cpp release honors `yarn_truncate`.

### llama.cpp runtime and PLaMo 2 GGUFs

Build the tested runtime from this checkout (requires Git, CMake and a C/C++ compiler: Xcode Command Line Tools on
macOS, for example `build-essential` on Ubuntu):

```sh
bash scripts/build_llama_cpp.sh
uv run plamo-translate --backend-type llama.cpp --llama-server .local/llama.cpp/build/bin/llama-server --input "Hello"
```

The build pins llama.cpp `b11960` (`8642541b255bfb5dc39cd351a178be09013a55d6`), the revision of Homebrew's
llama.cpp 0.8.0, and applies [`scripts/patches/llama-cpp-plamo2-rope.patch`](scripts/patches/llama-cpp-plamo2-rope.patch).
The patch is only needed for PLaMo 2. It fixes a loader bug: when layer 0 is Mamba, the common loader sets the rotary
dimension to zero, silently disabling RoPE. The patch restores the attention head dimension.
An unpatched runtime can produce fluent translations with missing paragraphs even with F16 weights.
The backend rejects a runtime that reports `n_rot = 0`, so use this build rather than Homebrew's for PLaMo 2.

Official PLaMo 3 GGUFs need no conversion. To convert other PLaMo 2 or PLaMo 3 checkpoints, convert the original
local weights once; the original model directory is read only:

```sh
uv venv .local/convert-venv
uv pip install --python .local/convert-venv/bin/python \
  'torch==2.11.0' 'transformers==4.57.6' 'huggingface-hub<1' \
  sentencepiece numpy safetensors protobuf -e .local/llama.cpp/gguf-py
uv run python scripts/prepare_llama_cpp.py \
  --model-dir ~/Models/pfnet--plamo-2-translate \
  --llama-cpp-dir .local/llama.cpp \
  --python .local/convert-venv/bin/python \
  --output-dir ~/Models/pfnet--plamo-2-translate-gguf \
  --quantization Q4_0-ssm-f16
```

Conversion saves an unquantized GGUF (F16 for PLaMo 2, BF16 for PLaMo 3) plus the requested quantizations, logs,
checksums, source revision and a copy of the model license. It refuses to overwrite existing weights.
Activation and normalization arithmetic differ from MLX, so numerical equivalence is not assumed. Allow about 26 GB
for PLaMo 2 F16 plus the recommended GGUF, in addition to the source. The official PLaMo 3 GGUFs were quantized with
an importance matrix; pass one from `llama-imatrix` with `--imatrix` to do the same.

For PLaMo 2, the recommended `Q4_0-ssm-f16` recipe uses Q4_0 with all Mamba `ssm_out` matrices kept in F16.
On the tested M1 Max, this preserves content omitted by plain Q4_0 while keeping most of its speed.
See [the measured comparison and quality summary](benchmarks/llama_cpp_20261007/README.md).
Other available recipes are Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q5_0, Q4_0, Q4_1 and Q3_K_M; PLaMo 3 defaults to Q4_K_M.

Start a persistent server, then translate using the existing client:

```sh
uv run plamo-translate server --backend-type llama.cpp \
  --llama-server .local/llama.cpp/build/bin/llama-server \
  --model ~/Models/pfnet--plamo-2-translate-gguf/plamo-2-translate-Q4_0-ssm-f16.gguf

# In another terminal:
uv run plamo-translate --from English --to Japanese < tests/fixtures/translation.en.txt
```

For one-shot use, omit `server` and pass the backend, executable and model options with the input.
`--no-stream`, interactive translation, and `show-claude-config` work with either backend.
A plain client reuses the running backend. Explicitly requesting a different model/backend fails
without changing the running server's configuration. Use a separate `TMPDIR` to run both backends.
The native process binds to loopback and is terminated when its owning MCP server exits.

`--precision` selects MLX weights; for llama.cpp, choose the GGUF with `--model`.
`PLAMO_TRANSLATE_CLI_MODEL_NAME` can also specify the GGUF path or repository, and
`PLAMO_TRANSLATE_CLI_LLAMA_SERVER` can specify the executable.

The llama.cpp defaults are full GPU offload (`--gpu-layers 99`), Flash Attention, F16 KV cache,
one inference slot, a 32768-token context, 2048-token batches, 512-token microbatches, and 2 CPU threads on Apple
Silicon with GPU layers (8 otherwise; set `--threads` for partial offload). Override them with `--ctx-size`, `--batch-size`, `--ubatch-size`, `--threads`,
`--gpu-layers`, and `--flash-attn on|off|auto`. Use `--gpu-layers 0` for CPU execution. Greedy decoding, disabled
repetition penalty, the translation-specific prompt, and the model's BOS/EOS settings preserve
the intended translation behavior. With `PLAMO_TRANSLATE_CLI_TEMP` above 0, sampling uses temperature, top-p and
top-k only, as in the MLX backend; llama.cpp's default min-p of 0.05 is not applied. Requests disable prompt caching,
so each one, including the history of an interactive session, is processed from the start; the server therefore
keeps no prompt cache or context checkpoints. It skips llama.cpp's memory fitting, which only probes when the GPU layers are given, and replaces the
GGUF's chat template, which this backend does not use. Token/context exhaustion and interrupted responses are
reported as errors, including in streaming mode.

Reproduce the supplied English/Japanese quality and speed comparison:

```sh
uv run --with sacrebleu python scripts/benchmark_llama_cpp.py \
  --llama-server .local/llama.cpp/build/bin/llama-server \
  --models pfnet/plamo-3-translate-2b-GGUF:Q4_K_M pfnet/plamo-3-translate-2b-GGUF:Q4_0 \
  --repeats 3 --output .local/benchmark-new-run
```

`--models` also accepts local GGUF files, and `--draft-model` adds a draft.
The benchmark saves full translations, raw streaming events, native logs, time to first token,
prompt/decode timings, and chrF against the supplied reference. It warms model weights with a
separate short prompt and measures complete translations without reusing the prompt cache.
Keep other GPU inference, quantization and media workloads idle while measuring.
chrF is reference overlap for this single example; review the actual translations for missing
content and do not interpret it as general translation accuracy.

## Options

- --input TEXT Input text to translate
- --from TEXT Input language for translation (default: English|Japanese, chosen from the text)
- --to TEXT Output language for translation (default: the other of English and Japanese)
- --backend-type Server backend: [mlx, llama.cpp] (default: reuse a running server, otherwise mlx on Apple Silicon
  and llama.cpp elsewhere)
- --model MLX: local model directory or Hugging Face repository (default: pfnet/plamo-3-translate-2b).
  llama.cpp: GGUF file or `<user>/<repo>[:QUANT or file.gguf]` (default: pfnet/plamo-3-translate-2b-GGUF:Q4_K_M)
- --draft-model llama.cpp draft GGUF or repository, or `none` (default: pfnet/plamo-3-translate-0.8b-GGUF:Q4_0
  for `server` and interactive mode with the default model; otherwise none)
- --precision MLX weight precision: [4bit, 6bit, 8bit, mixed, bf16] (default: 4bit for original weights)

## Configuration

You can configure the following parameters using environment variables:

- `PLAMO_TRANSLATE_CLI_SERVER_START_PORT`: Specifies the starting port number for the server.
- `PLAMO_TRANSLATE_CLI_SERVER_END_PORT`: Specifies the ending port number for the server.
- `PLAMO_TRANSLATE_CLI_MODEL_NAME`: Local model directory or Hugging Face repository; a GGUF file or repository for llama.cpp.
- `PLAMO_TRANSLATE_CLI_LLAMA_SERVER`: Path to `llama-server` (default: `llama-server` on `PATH`).
- `PLAMO_TRANSLATE_CLI_PRECISION`: Explicit weight precision, equivalent to `--precision`.
- `PLAMO_TRANSLATE_CLI_OPTIMIZE`: PLaMo 2 only. Set to `1` to opt in to the experimental decode convolution (default: `0`; correctness fixes are always enabled).
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

## License

The source code of `plamo-translate` is licensed under the [Apache License 2.0](LICENSE).

The models it downloads are distributed under the PLaMo Community License
([English](https://plamo.preferredai.jp/info/plamo-community-license-en),
[Japanese](https://plamo.preferredai.jp/info/plamo-community-license-ja)), not under the Apache License. This includes
[`pfnet/plamo-3-translate-2b`](https://huggingface.co/pfnet/plamo-3-translate-2b) (MLX default),
[`pfnet/plamo-3-translate-2b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF) (llama.cpp default),
[`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) (draft model that
llama.cpp servers and interactive mode download) and the PLaMo 2 Translate models. Read the license before using
them. In short, as summarized on the model cards:

- Personal and non-commercial use, including academic use, is free and needs no registration.
- Commercial use is free when the revenue or sales of the user or its affiliates in the most recent fiscal year were
  1 billion yen or less, after registering through the [PLaMo Commercial Use Registration Form](https://forms.gle/mTL8tBLrMYXKNZD56). The license
  counts use within an organization as commercial use. Above 1 billion yen, a separate agreement with PFN is needed
  ([inquiry page](https://translate.preferredai.jp/llm/contact/)).
- Translations that you publish must be clearly indicated as output of the model (Article 5).
- Providing the models or their output to others requires a copy of the license for the recipients, "Built with
  PLaMo" in related websites, interfaces or documentation, and "PLaMo" in the names of models you derive from them
  (Article 4).

The license text is authoritative; this summary is not.
