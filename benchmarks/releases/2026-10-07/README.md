# Standalone MLX releases, 2026-10-07

Source: `pfnet/plamo-2-translate` at `cae8da342a3e051ed69f90ce24c23eacff908732`.
Apple M1 Max, 32 GPU cores, 64 GB; MLX 0.31.1 and mlx-lm 0.31.2.

| Release | Weight bytes | chrF | Tensor dtypes | Validation |
| --- | ---: | ---: | --- | --- |
| 8bit | 10,124,903,941 | 73.94 | 597 BF16, 162 U32 | Two standalone runs and both CLI modes match corrected direct 8bit inference |
| BF16 | 19,056,520,807 | 73.46 | 435 BF16 | Two standalone runs and both CLI modes match corrected direct BF16 inference |

The single English/Japanese example in `tests/fixtures/translation.*.txt` is used throughout.
Both releases finish at EOS, retain eight paragraphs, the founders' names, Preferred Networks and Learn or Die.
Equality refers to the same-precision direct implementation, not the human reference or another precision.
The BF16 title still omits “Together with You”; no claim of broad translation quality is made.

The 8bit baseline is `../../results/release-8bit-baseline`; BF16 uses the already verified
`../../results/final-native-bf16` output. The `validation.json` files include every inference
asset's size and SHA-256; `cli/results.json` records streaming and non-streaming output equality
and actual served precision. The standalone test deliberately bypasses the CLI model loader.

Other tasks shared the machine, so timing values are observational and excluded from performance
comparisons. One initial BF16 attempt hit Metal out-of-memory while another approximately 35 GB
model was loaded; that trial produced no accepted output. After that model was released, both
standalone runs and both CLI modes passed. See `bf16/attempts.json`.

`model-card.md` is the model card submitted to Hugging Face. `publication.json` records the
resulting repository revision and verified remote file hashes. Supplied source/reference texts,
translations, and local paths are not included in the Hugging Face upload allowlist.

Upload transport: initial `hf-xet` 1.4.3 uploads repeatedly timed out while sharing
the uplink with other transfers, before either repository changed. They were restarted sequentially
with a process-local `uv run --with hf-xet==1.7.0` overlay (model/runtime dependencies
were not changed), `HF_XET_FIXED_UPLOAD_CONCURRENCY=2`,
`HF_XET_CLIENT_READ_TIMEOUT=600s`, `HF_XET_CLIENT_RETRY_MAX_DURATION=1800s`, and
`HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION=1`. These are documented in
[Hugging Face's Xet settings](https://huggingface.co/docs/hub/xet/using-xet-storage#environment-variables).
