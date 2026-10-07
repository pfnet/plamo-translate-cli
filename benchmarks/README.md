# Benchmark summaries

Only reviewed `README.md` summaries belong in this directory. Keep measurement
conditions, aggregate results and limitations; exclude individual logs, detailed
JSON, generated translations, absolute local paths and machine-wide process lists.
Raw output must stay in an ignored directory such as `.local/benchmarks/`.

- [MLX measurements](../README.md#measured-result-2026-10-07)
- [llama.cpp measurements](llama_cpp_20261007/README.md)
- [Standalone MLX release validation](releases/2026-10-07/README.md)

The shared source and reference text are regression fixtures in
[`tests/fixtures`](../tests/fixtures). Benchmark and validation scripts use those
fixtures directly. Reproduce the 4bit MLX baseline before validating CLI output:

```sh
uv run python scripts/benchmark_mlx.py \
  --model /path/to/plamo-2-translate --precision 4bit \
  --runs 2 --output .local/benchmarks/my-4bit
uv run python scripts/validate_mlx_cli.py \
  --model /path/to/plamo-2-translate-mlx-4bit \
  --expected .local/benchmarks/my-4bit/translation-0.txt \
  --output .local/benchmarks/my-cli-roundtrip
```

The build excludes this directory from package distributions. CI rejects tracked
benchmark files other than these summaries, including force-added ignored files.
