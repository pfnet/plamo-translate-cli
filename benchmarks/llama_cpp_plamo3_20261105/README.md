# PLaMo 3 Translate 2B on llama.cpp, 2026-11-05

Built with PLaMo.

Apple M1 Max, 32 GPU cores, 64 GB. llama.cpp b11960 (`8642541b255bfb5dc39cd351a178be09013a55d6`, the revision of
Homebrew's llama.cpp 0.8.0) built with `scripts/build_llama_cpp.sh`, Release, Metal. Models:
`pfnet/plamo-3-translate-2b-GGUF` at `f2b4599d` and the draft `pfnet/plamo-3-translate-0.8b-GGUF` at `03891e90`.
Greedy decoding of the fixtures in `tests/fixtures` (380 and 405 prompt tokens, 350 to 393 generated tokens),
one warm-up request per server, three runs per setting, medians only. Desktop applications stayed open.

## Result

| GGUF | Draft | Decode tok/s | Translation s | First token s |
| --- | --- | ---: | ---: | ---: |
| Q4_K_M (default) | none | 111 | 3.82 | 0.28 |
| Q4_K_M | 0.8B Q4_0 | 132 | 3.26 | 0.28 |
| Q4_0 | none | 134 | 3.12 | 0.24 |
| Q4_0 | 0.8B Q4_0 | 141 | 2.99 | 0.25 |

English to Japanese, streamed through `scripts/benchmark_llama_cpp.py` with the backend's own server options:

```sh
uv run python scripts/benchmark_llama_cpp.py --llama-server .local/llama.cpp/build/bin/llama-server \
  --models pfnet/plamo-3-translate-2b-GGUF:Q4_K_M pfnet/plamo-3-translate-2b-GGUF:Q4_0 \
  [--draft-model pfnet/plamo-3-translate-0.8b-GGUF:Q4_0] --repeats 3 --output .local/benchmarks/llama-plamo3
```

With the draft, every output was byte-identical to the output without it, in both directions and for both
quantizations. The draft proposed 368 tokens for the English source and 344 for the Japanese one; 85% were
accepted. Japanese to English gained more: 112 to 142 tok/s with Q4_K_M.

For comparison, the fused MLX kernels decode at 170 tok/s on this machine
([summary](../plamo3_mlx_20261104/README.md)). llama.cpp's own `llama-bench` measured 117 tok/s for Q4_K_M and
136 tok/s for Q4_0, so the stock Metal kernels, not the server, now bound single-stream decoding.

## Server options

Q4_K_M, decode tok/s English to Japanese / Japanese to English, all rows in one session. Each step keeps the
previous ones.

| Setting | tok/s | Output |
| --- | ---: | --- |
| Previous request and server settings: top-p 0.98 and server default min-p 0.05 at temperature 0, 8 threads | 68 / 67 | reference |
| Greedy without the CPU nucleus and min-p filters | 100 / 93 | identical |
| 2 CPU threads, no RAM prompt cache, no SWA checkpoints, no memory fitting | 111 / 112 | identical |
| 0.8B draft, at most 6 tokens, stopped below probability 0.6 | 132 / 142 | identical |

In other sessions the first row measured 76 / 68 and the second 105 / 90 and 109 / 108: the slower settings vary
with background load, while the last two rows stayed within 2.5 tok/s.

The filters sorted the 107,520-token vocabulary on the CPU for every token; they cannot change a greedy argmax.
With full GPU offload only the input embedding runs on the CPU, and fewer polling threads decoded faster.

Speculation settings, Q4_K_M with 2 threads, in a session where the draft-less setting gave 113 / 113: at most 8 tokens with probability 0.7 gave 126 / 134 tok/s and
10 tokens with 0.8 gave 122 / 132. A Q4_K_M draft gave 123 / 128. With 8 threads and no probability floor, 2, 3, 4
and 6 draft tokens gave 105, 119, 116 and 115 tok/s for English to Japanese. `--model-draft` alone does nothing in
this revision; `--spec-type draft-simple` enables it.

Not adopted:

- Backend (GPU) sampling: 83 / 85 tok/s, against 105 / 90 without it in the same session.
- Metal environment switches (concurrency, graph optimization, fusion, residency sets): within noise for Q4_0
  in `llama-bench`; disabling shared buffers dropped it to 55 tok/s.
- Microbatches of 256, 1024 and 2048 tokens: a 4096-token prompt processed at 1256, 1307 and 1226 tok/s, against
  1294 at the default 512.

Decode slows little with context: `llama-bench` measured 117, 111, 108 and 106 tok/s for Q4_K_M at depths of 0,
2048, 4096 and 8192 tokens.

## Startup and end-to-end time

| Step | Time |
| --- | ---: |
| `llama-server` ready, Q4_K_M, default memory fitting | 1.85 s |
| The same with `--fit off` (the backend passes `-ngl`, so fitting only probed) | 1.06 s |
| With the 0.8B draft, default fitting / `--fit off` | 3.53 / 1.89 s |
| First launch after building (Metal shader compilation, cached by macOS afterwards) | about 8.6 s |

Complete CLI invocations, including Python, the MCP server and model loading:

| Invocation | llama.cpp | MLX 4bit |
| --- | ---: | ---: |
| One-shot, "Proud, but humble" | 2.3 to 2.5 s | 6.2 to 7.1 s |
| One-shot, English fixture | 6.0 to 6.1 s | 8.6 to 8.7 s |
| Client of a running server, "Proud, but humble" | 0.55 s | 0.54 to 0.71 s |
| Client of a running server, English fixture | 3.7 to 3.8 s | 3.0 to 3.2 s |

One-shot runs used no draft. With it the English fixture took 6.5 s, because loading the draft costs more than it
saves on one translation; the CLI therefore loads it by default only for `plamo-translate server` and interactive
mode. The MLX one-shot time includes quantizing the original weights and Hugging Face metadata requests.

## Correctness

| Check | Result |
| --- | --- |
| Token IDs of four PFML3 prompts (both fixtures, automatic direction, repeated spaces and characters, emoji) | Identical to the Hugging Face tokenizer, with one BOS added by llama.cpp |
| BF16 GGUF converted with `scripts/prepare_llama_cpp.py` from a copy matching revision `355637cc` by SHA-256 | Translations byte-identical to the fused MLX FP16 output in both directions |
| Q4_K_M against the BF16 GGUF output (chrF) | 93.4 / 97.2, against 93.0 / 95.4 for MLX 4bit against MLX FP16 |

llama.cpp ignores the GGUF's `rope.scaling.yarn_truncate = false` and rounds the YaRN correction range of the three
full-attention layers to dimensions 13 to 31 instead of 13.96 to 30.02. Applying that rounding to the FP32 MLX model
changed teacher-forced logits by a mean KL divergence of 9e-6 and 1e-5 on the fixtures and 4e-5 on a 2700-token
sequence, with 100%, 100% and 99.91% top-1 agreement and reference NLL within 0.0014. 4bit quantization moves
logits by a KL divergence of about 0.018.

Correction: these short, partly repeated sequences hid the error, which grows with input length. Source-text
predictions moved by a KL divergence of 0.04 at 2,048 tokens and 0.66 at 16,384, and an 8,000-token document
translated differently in 66 of 315 lines. The backend now corrects most of it; see the
[YaRN range summary](../llama_cpp_plamo3_yarn_20261105/README.md).

chrF against the supplied references, English to Japanese / Japanese to English: BF16 GGUF 86.2 / 84.7,
Q4_K_M 83.0 / 85.6, Q4_0 88.0 / 84.8, MLX 4bit 87.4 / 84.5. One document per direction cannot rank these. The model
card's evaluation shows Q4_K_M within 0.02 chrF++ of BF16 on FLORES+ Japanese-multilingual pairs and Q4_0 0.90 lower,
with English–Japanese scores within 0.0011 of BF16 for both, so Q4_K_M stays the default.

## Limitations

- Only M1 Max was measured. Other chips may favor different quantizations or draft lengths.
- Speculative decoding with greedy acceptance is equivalent to plain greedy decoding, but verification batches use
  different Metal kernels, so near-tied tokens could differ on other inputs. None differed here.
- Homebrew's llama.cpp 0.8.0 is the measured llama.cpp revision, but it links Homebrew's separately packaged ggml,
  which can be newer than the bundled ggml 0.28.0, and its binary was not run.
