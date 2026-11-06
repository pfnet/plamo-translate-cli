# PLaMo 3 Translate 2B on fused Metal kernels, 2026-11-04

Built with PLaMo.

Source: `pfnet/plamo-3-translate-2b` at `355637cc04db67195ba87a97da85eaf1d52a22e8`. The measured copy
has byte-identical weights, configuration and tokenizer. On 2026-11-05 the downloaded repository reproduced
the same outputs at 161 and 167 tok/s on the shared machine.

Apple M1 Max, 32 GPU cores, 64 GB; MLX 0.31.1, mlx-lm 0.31.2. Greedy decoding through
`Plamo3Translator.stream`, three runs per direction of the fixtures in `tests/fixtures`
(380 and 405 prompt tokens, about 350 to 390 generated tokens). Only the median is summarized.

| Precision | Implementation | Decode tok/s | First token (380-token prompt) | chrF en-ja / ja-en |
| --- | --- | ---: | ---: | --- |
| 4bit | Fused kernels | 170 | 0.31 s | 87.4 / 84.5 |
| 4bit | Plain MLX ops | 107 | 0.32 s | 87.4 / 84.8 |
| 8bit | Plain MLX ops | 72 | 0.40 s | 86.2 / 84.7 |
| FP16 (`bf16`) | Fused kernels | 55 | 0.30 s | 86.2 / 84.7 |
| FP16 (`bf16`) | Plain MLX ops | 52 | 0.31 s | 86.2 / 84.7 |

All rows use `MLX_MAX_OPS_PER_BUFFER=4096` and `MLX_MAX_MB_PER_BUFFER=1000000`. Plain 4bit measured
93 tok/s with the default limits, and the fused kernels varied between 116 and 166 tok/s.
At 2564 tokens of context, past the 2048-token sliding window, fused 4bit measured about 135 tok/s.

chrF compares each output with the single supplied reference text. It is reported only to expose
gross failures, and the small differences between rows are not evidence of a quality ranking.

## Correctness

The reference is the checkpoint's PyTorch implementation at FP32 on CPU.

| Check | Result |
| --- | --- |
| Plain MLX at FP32, first 16 steps of each case and 13 steps past the window | Largest logit difference below 0.0001 before the window; RMSE below 0.0001 past it |
| Fused FP16, teacher-forced and free-running, 384 tokens | All tokens match; logit RMSE 0.003 to 0.006 |
| Fused FP16 at 2552 to 2564 tokens of context, 13 steps | All tokens match; logit RMSE about 0.003 |
| Fused 4bit against plain 4bit | Largest logit difference about 0.02 |

Fused 4bit differs from the reference as much as plain 4bit does: logit RMSE is about 0.4, the first
differing token of the 384-token case is the ninth, and 97% of teacher-forced top-1 tokens match.

## Where the time goes

Each component was removed from a real 4bit token to measure the difference (5.75 ms at 380 tokens):

| Component | Time | Share |
| --- | ---: | ---: |
| 96 projections | 4.21 ms | 73% |
| 24 attention dispatches | 0.72 ms | 13% |
| `lm_head` and argmax | 0.47 ms | 8% |
| 48 normalization dispatches | 0.35 ms | 6% |

Seven dispatches per layer replace the plain implementation's many small ones. Matrix-vector products
reach about 650 GMAC/s, against 530 for `mx.quantized_matmul`, both in chained dispatches. A cached
32-row window runs equally fast, so 4bit projections are limited by ALU throughput, not by DRAM.
FP16 weights read about 340 GB/s at a quarter of that MAC rate, which suggests a memory limit and would
explain why kernels gain only 6% there. BF16 and FP16 `mx.matmul` ran at the same speed on this GPU.

## Quantization trials

Teacher-forced on both fixtures against FP32 (top-1 agreement en-ja / ja-en, KL divergence):

| Weights | Bytes excluding embeddings | Top-1 % | KL |
| --- | ---: | --- | --- |
| 4bit, min-max, group 64 | 1.46 GB | 97.7 / 97.2 | 0.018 / 0.017 |
| 4bit, MSE-optimized, group 64 | 1.46 GB | 95.6 / 95.5 | 0.015 / 0.011 |
| 4bit, min-max, group 32 | 1.63 GB | 96.1 / 97.5 | 0.013 / 0.013 |
| 3bit, min-max, group 64 | 1.14 GB | 93.5 / 92.4 | 0.082 / 0.111 |
| 5bit, min-max, group 64 | 1.79 GB | 97.9 / 99.4 | 0.004 / 0.003 |
| 8bit, MSE-optimized, group 64 | 2.77 GB | 99.7 / 100 | 0.0001 / 0.0000 |

Clipping outliers lowered KL but also top-1 agreement, so min-max stays the default.

## Limitations

- Only M1 Max was measured. The kernels assume 32-lane SIMD-groups and in-place buffer writes; the startup self-check guards this on other chips, but their speed is unknown.
- Quality was compared on two passages of a single document, not on a translation evaluation set.
- FP16 activations reached at most 3377 across these documents, repeated tokens and random tokens, 19 times below the FP16 limit. This is an observation, not a bound, and activations are not clamped.
- Precisions other than 4bit and `bf16` have no kernels. Prompts always run on plain MLX ops, at about 1200 tok/s.
