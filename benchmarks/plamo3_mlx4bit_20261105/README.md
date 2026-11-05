# PLaMo 3 Translate 2B MLX-4bit release, 2026-11-05

Release: `pfnet/plamo-3-translate-2b-MLX-4bit` at `63777ddc6dc80aacacd92a1bf6966105f2f6270c`. Its 96 projections
are 4bit affine with group size 64; the embeddings and `lm_head` stay in BF16 and are loaded as FP16. It is compared
with this CLI's own 4bit conversion of `pfnet/plamo-3-translate-2b` at `355637cc04db67195ba87a97da85eaf1d52a22e8`,
which uses min-max affine quantization with group size 64 for every projection, including `lm_head`.

Apple M1 Max, 32 GPU cores, 64 GB; MLX 0.31.1, mlx-lm 0.31.2. Both use the fixtures in `tests/fixtures`.

## Fidelity to the original weights

Teacher-forced on the reference translations (386 en-ja and 361 ja-en tokens) against the original weights at FP32
on plain MLX ops, which matched the PyTorch FP32 implementation below 1e-4. Reproduce with
`uv run python scripts/evaluate_plamo3_release.py --output .local/benchmarks/plamo3-fidelity`.

| Weights | Top-1 agreement en-ja / ja-en | KL divergence en-ja / ja-en | Reference NLL en-ja / ja-en |
| --- | --- | --- | --- |
| CLI 4bit from original weights | 97.9% / 97.2% | 0.019 / 0.028 | 0.220 / 0.808 |
| MLX-4bit release | 98.7% / 97.5% | 0.008 / 0.010 | 0.225 / 0.825 |
| FP32 original weights | 100% | 0 | 0.219 / 0.793 |

The release stays closer to the original model's predictions, with two to three times lower KL divergence, but
assigns slightly lower probability to this one human reference. Two passages of one document are not an evaluation
set. The model card reports the release within 0.0021 PFTB of BF16.

## Loader correctness

| Check | Result |
| --- | --- |
| This loader against the release's bundled `modeling_plamo_mlx.py`, both in FP16 | All top-1 tokens match; logit RMSE 0.004, largest difference 0.09 |
| The same, with the bundled runtime in BF16 as shipped | All top-1 tokens match; logit RMSE 0.03 |
| Fused kernels against plain MLX ops on the release | All top-1 tokens match; largest logit difference 0.07 |

## Speed and memory

Greedy decoding through `Plamo3Translator.stream` on the fused kernels, three runs per direction; medians. Reproduce
with `uv run python scripts/benchmark_plamo3.py --model pfnet/plamo-3-translate-2b-MLX-4bit --output .local/benchmarks/plamo3-release`.

| Weights | Download | Load | Decode tok/s en-ja / ja-en | chrF en-ja / ja-en | Peak memory |
| --- | ---: | ---: | --- | --- | ---: |
| MLX-4bit release | 2.41 GB | 3.6 s | 138 / 135 | 86.6 / 83.8 | 5.42 GB |
| CLI 4bit from original weights | 5.76 GB | 3.5 s | 162 / 162 | 87.4 / 84.5 | 7.22 GB |

The release decodes about 15% slower because its FP16 `lm_head` reads 550 MB per token, against about 155 MB at 4bit.
Its memory peaks during the 380-token prompt prefill (2.64 GB after loading); the CLI's own conversion peaks while
quantizing the original weights, and reaches 5.02 GB during the same prefill. chrF compares each output with the
single supplied reference and is reported only to expose gross failures; these differences are not a quality ranking.

## Limitations

- Only M1 Max was measured.
- Fidelity was measured on two passages of a single document, not on a translation evaluation set.
- The release is 4bit only. Other precisions are converted from the original weights.
