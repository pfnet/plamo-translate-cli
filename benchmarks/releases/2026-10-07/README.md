# Standalone MLX releases, 2026-10-07

Source: `pfnet/plamo-2-translate` at `cae8da342a3e051ed69f90ce24c23eacff908732`.
Apple M1 Max, 32 GPU cores, 64 GB; MLX 0.31.1 and mlx-lm 0.31.2.

Published and verified revisions:

| Release | Published revision | Weight size | chrF | Verified files |
| --- | --- | ---: | ---: | ---: |
| 8bit | [`6b1851db`](https://huggingface.co/mlx-community/plamo-2-translate-8bit/tree/6b1851db9edd12c3b7a8b2c33f7505a87380a3ae) | 10.12 GB | 73.94 | 16 |
| BF16 | [`bcc519c8`](https://huggingface.co/mlx-community/plamo-2-translate-bf16/tree/bcc519c834c0eb0f980168d48d1d9239e3dae175) | 19.06 GB | 73.46 | 18 |

| Release | Weight bytes | chrF | Tensor dtypes | Validation |
| --- | ---: | ---: | --- | --- |
| 8bit | 10,124,903,941 | 73.94 | 597 BF16, 162 U32 | Two standalone runs and both CLI modes match corrected direct 8bit inference |
| BF16 | 19,056,520,807 | 73.46 | 435 BF16 | Two standalone runs and both CLI modes match corrected direct BF16 inference |

The single English/Japanese example in `tests/fixtures/translation.*.txt` is used throughout.
Both releases finish at EOS, retain eight paragraphs, the founders' names, Preferred Networks and Learn or Die.
Equality refers to the same-precision direct implementation, not the human reference or another precision.
The BF16 title still omits “Together with You”; no claim of broad translation quality is made.

Standalone validation checked every inference asset's size and SHA-256 and compared
streaming and non-streaming CLI output with direct inference at the same precision.
The standalone test deliberately bypassed the CLI model loader. Only this summary
is versioned; detailed validation records and generated translations stay local.

Other tasks shared the machine, so timing values are observational and excluded from performance
comparisons. One initial BF16 attempt hit Metal out-of-memory while another approximately 35 GB
model was loaded; that trial produced no accepted output. After that model was released, both
standalone runs and both CLI modes passed.

Publication verified the remote inference files against their local SHA-256 hashes.
The model upload allowlist excludes supplied source/reference texts, translations
and local paths. Detailed publication records and model-card copies are not stored
in this directory.
