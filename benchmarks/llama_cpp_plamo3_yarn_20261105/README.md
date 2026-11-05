# PLaMo 3 YaRN range on llama.cpp, 2026-11-05

Built with PLaMo.

Apple M1 Max, 64 GB. llama.cpp b11960 built with `scripts/build_llama_cpp.sh`, as in the
[llama.cpp measurements](../llama_cpp_plamo3_20261105/README.md). Models: `pfnet/plamo-3-translate-2b` (original
weights for MLX; BF16 GGUF converted with `scripts/prepare_llama_cpp.py`) and the official Q4_K_M GGUF.

## Cause

PLaMo 3 applies YaRN in its three full-attention layers (7, 15 and 23) and keeps the correction range fractional
(`truncate: false`): dimension pairs 13.96 to 30.02 of 64 ramp from extrapolation to interpolation. No llama.cpp
release honors the GGUF's `yarn_truncate = false` yet
([#30206](https://github.com/ggml-org/llama.cpp/pull/30206) for ggml, CPU and CUDA only, and
[#30337](https://github.com/ggml-org/llama.cpp/pull/30337) for PLaMo 3 are open): floor and ceil widen the range to
13 to 31. The angle error of each changed pair grows linearly with the distance between tokens.

The backend now passes `--yarn-beta-fast 31.738495 --yarn-beta-slow 1.0040927`, computed from the GGUF metadata, so
that floor and ceil land on 14 and 30, the nearest whole dimensions. The measurements below used 31.676893 and
1.0060454, which give the same 14 and 30. A llama.cpp that keeps the range fractional would use 14.001 to 29.999 with
these options instead of the exact range, so they must go once a release honors `yarn_truncate`.

## Logits, MLX FP32

The FP32 MLX model with each range. Prompts are PFML3 English-to-Japanese translation requests whose source is English
Markdown prose (this repository's README and llama.cpp's `docs`, without code blocks and tables), three documents per
length. KL divergence from the exact range and top-1 agreement over the next-token distributions of the last 256
prompt positions; medians over the three documents.

| Prompt tokens | llama.cpp rounding (13–31) | With the options (14–30) | MLX 4bit, exact range |
| --- | --- | --- | --- |
| 545–598 | 1.8e-4 / 99.6% | 1.5e-7 / 100% | 0.16 / 78.9% |
| 2,033–2,083 | 0.038 / 96.5% | 3.0e-4 / 100% | 0.41 / 69.5% |
| 4,087–4,134 | 0.14 / 93.4% | 7.4e-4 / 99.6% | 0.51 / 70.7% |
| 8,205–8,241 | 0.23 / 83.2% | 1.6e-3 / 98.8% | 0.53 / 70.3% |
| 16,365–16,388 | 0.66 / 76.2% | 5.2e-3 / 98.0% | 0.43 / 72.3% |

The first output token moved little (KL at most 0.06 with the rounding), and the earlier check on the fixtures'
teacher-forced translations (mean KL at most 4e-5 up to 2,700 tokens) saw the same: confident predictions hide the
error that source positions carry into later layers.

## Logits, llama.cpp

`llama-perplexity` on the same prose as plain text, against the BF16 GGUF with the options. KL divergence over the
second half of each chunk: 4 chunks of 2,048 tokens, 2 of 8,192 and 1 of 16,384.

| Scored positions | BF16, rounding | Q4_K_M, options | Q4_K_M, rounding (previous default) |
| --- | --- | --- | --- |
| 1,024–2,047 | 0.026 / 96.9% | 0.26 / 81.3% | 0.28 / 81.2% |
| 4,096–8,191 | 0.30 / 86.2% | 0.53 / 74.8% | 0.67 / 72.3% |
| 8,192–16,383 | 0.85 / 71.6% | 0.38 / 77.4% | 1.14 / 64.2% |

Beyond about 8,000 tokens the rounding moves predictions more than Q4_K_M quantization does.

## Translations

Greedy English-to-Japanese translations of one document per length (the first of the three above), compared with the
fused MLX FP16 backend, which uses the exact range. Lines that differ from it, by `difflib`:

| Source tokens | Lines | MLX FP16, 13–31 | MLX FP16, 14–30 | llama-server BF16, rounding | llama-server BF16, options |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,064 | 75 | 9 | 1 | 9 | 1 |
| 4,115 | 208 | 17 | 14 | 18 | 9 |
| 8,186 | 315 | 66 | 24 | 66 | 2 |

chrF of `llama-server`'s output against MLX rose from 98.4, 98.6 and 96.5 to 99.8, 99.4 and 99.9 with the options.
Greedy outputs diverge after the first changed token, and with the same 14–30 range MLX and `llama-server` split at
character 7,724 of the 8,186-token document, so the gain varies between runs: 2 or 24 lines there.

Applying llama.cpp's rounding to the MLX model reproduced `llama-server`'s 8,186-token translation byte for byte; at
2,064 and 4,115 tokens the two agreed for the first 4,393 and 5,791 characters. Most differences from the exact range
were wording, such as グループ化 against 集約, but the rounding also translated "zero in-loop vector spills" as
ループ内量子化がゼロ ("zero in-loop quantization") and dropped the parenthetical "of 4 `k` each" (各グループ4個). No line was
dropped.

With the options, the BF16 GGUF still translated both fixtures byte-identically to MLX FP16, and Q4_K_M decoded the
English fixture at 112 tok/s (111 before) with the same output.

## Limitations

- Only M1 Max and English source text were measured, with one generated document per length, so the translation
  counts are examples rather than rates.
- Predicting Markdown prose is less certain than translating it, so every KL divergence here, including quantization,
  is larger than on translation outputs.
- The options approximate the exact range, and their error also grows with length. An exact range needs the upstream
  changes, and then these options must be removed; #30206 has no Metal kernel, so on Apple Silicon the operation would
  fall back to the CPU (not measured).
