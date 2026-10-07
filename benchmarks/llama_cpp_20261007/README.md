# PLaMo 2 translate / llama.cpp 実測（2026-10-07）

推奨は **Q4_0 + Mambaの `ssm_out` 行列をF16で保持**する構成。Apple M1 Maxで生成 **40.29 token/秒**、提示された全文の翻訳 **11.19秒**。F16基準の16.67 token/秒、25.73秒に対し、生成速度2.42倍、全文所要時間2.30倍の改善だった。chrFはF16の72.96に対し73.81。これは提示された1例に対する結果であり、汎用的な品質維持や全ハードウェアでの最速を保証するものではない。

Built with PLaMo。隣接する訳文ファイルはPLaMoモデルの生成出力。

## 条件と再現

- Apple M1 Max、GPU 32コア、メモリ64 GiB。通常のデスクトップアプリは起動したまま。他のモデル推論・量子化処理との競合があった予備測定は速度集計から除外した。
- 元モデル: `/Users/shunta/Models/pfnet--plamo-2-translate`。ダウンロード記録のrevision: `cae8da342a3e051ed69f90ce24c23eacff908732`。ファイル単位のSHA-256は [conversion.json](conversion.json)。元ファイルは変更していない。
- llama.cpp: `db33d3cb89d8b0d954df66047b13e9cc22df8f10`（b11318）+ [RoPE修正](../../scripts/patches/llama-cpp-plamo2-rope.patch)。Release / AppleClang 21 / Metal。実行バイナリ・ライブラリのSHA-256は各 `*-metadata.json` に記録。
- 全33層をGPUへ配置、Flash Attention有効、F16 KV cache、context 32768、batch 2048、microbatch 512、CPU threads 8、parallel 1。
- temperature 0、seed 0、repeat penalty 1、prompt cacheなし。独立した短文でwarmup後、全文を各2回翻訳。ロード時間は表に含まない。全文時間はHTTPストリーム開始から終了までで、prompt処理も含む。
- [入力](../../tests/fixtures/translation.en.txt)・[参照訳](../../tests/fixtures/translation.ja.txt)はユーザー提供文。入力の両端空白を除去し、専用翻訳プロンプトを組み立てる。元トークナイザーとllama.cppの全364入力トークンが一致することを照合した。BOSは1個。
- chrF: sacreBLEU 2.6.0、`nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no`。表の時間と速度は2回の中央値。各構成の2回の出力は一致した。

```sh
bash scripts/build_llama_cpp.sh
uv run --with sacrebleu python scripts/benchmark_llama_cpp.py \
  --llama-server .local/llama.cpp/build/bin/llama-server \
  --models /Users/shunta/Models/pfnet--plamo-2-translate-gguf/plamo-2-translate-{F16,Q4_0-ssm-f16}.gguf \
  --repeats 2 --output .local/reproduce-llama
```

モデル変換は [README](../../README.md#llamacpp-on-apple-silicon) の手順で再現できる。推奨GGUFはQ4_0量子化時に `--tensor-type ssm_out=f16` を指定して作成した。モデルサイズは5.81 GiB。

## 全文での比較

| GGUF | サイズ GiB | 生成 token/秒 | 全文 秒 | chrF | 確認した品質差 |
|---|---:|---:|---:|---:|---|
| F16 | 17.75 | 16.67 | 25.73 | 72.96 | 見出しの Together with You を省略 |
| Q8_0 | 9.43 | 27.79 | 15.98 | 72.81 | 同上 |
| Q6_K | 7.28 | 28.06 | 16.03 | 73.63 | 同上 |
| Q5_K_M | 6.31 | 23.34 | 18.96 | 73.52 | 同上、Q8より遅い |
| Q4_K_M | 5.39 | 30.71 | 15.07 | 71.93 | 内容は概ね保持、Learn or Die に説明を追加 |
| Q4_0 | 5.10 | **42.29** | **10.86** | 72.25 | 創業者が計算機科学・技術を心から愛する、という内容を省略。推奨から除外 |
| Q5_0 | 6.16 | 33.14 | 13.87 | 70.86 | 内容は概ね保持、Learn or Die に説明を追加 |
| **Q4_0 + ssm_out F16** | **5.81** | **40.29** | **11.19** | **73.81** | 見出し・創業者の記述・Learn or Die・署名を保持。採用 |
| Q4_1 | 5.63 | 40.31 | 11.28 | 73.38 | Learn or Die の英語表記を残さず日本語に置換 |

[推奨構成の全文出力](plamo-2-translate-Q4_0-ssm-f16-1.txt)と[F16出力](plamo-2-translate-F16-1.txt)を確認できる。全構成とも見出し・署名を含め8段落で、EOS終了・生成上限未到達だった。段落数やEOS終了だけでは訳抜けを検出できず、Q4_0の省略は全文比較で見つかった。

推奨構成でも参照訳との完全一致ではない。例えば exponential growth の2回目を「爆発的な成長」と訳すなど、語彙や文体の差がある。今回の確認範囲では本文の主要内容の欠落は認めなかったが、chrFの上昇だけを根拠に量子化がモデル自体の品質を改善したとは解釈しない。保持する行列の選択も同じ1例で調整しており、独立した評価セットによる検証ではない。

## 精度不具合の修正

未修正llama.cppは、先頭のMamba層のhead数0を参照して `n_rot_full = 0` とし、その後PLaMo 2固有のloaderでhead幅128を読み込んでもRoPE次元を復元しなかった。RoPEの基数だけを変えても出力が同一だったため追跡し、[4行の修正](../../scripts/patches/llama-cpp-plamo2-rope.patch)で本来の128次元を設定した。[修正版のロードログ](plamo-2-translate-F16.log)には `n_rot = 128`、`freq_base = 1000000`、全33層のGPU配置が記録されている。

同じF16重み・入力で、修正前は6段落・chrF 61.06、修正後は8段落・72.96。[修正前の出力](unpatched-F16.txt)には本文の統合や省略があった。修正前の速度測定には量子化処理などの競合があるため速度比較には使用していない。バックエンドは起動ログに `n_rot = 0` があれば起動を拒否する。

このrevisionは[PLaMoのBOS/EOS修正](https://github.com/ggml-org/llama.cpp/pull/29734)も含む。チャット用テンプレートではなくモデル専用の翻訳プロンプトを使用し、`<|plamo:op|>` で停止する。

## 実行設定とCLI検証

| 推奨GGUFでの設定 | 生成 token/秒 | 全文 秒 | 判定 |
|---|---:|---:|---|
| threads 8 / microbatch 512 / FA on | 40.29 | 11.19 | 既定として採用 |
| threads 4 / microbatch 256 | 40.90 | 11.17 | 全文時間に実用的な差なし |
| FA off | 39.33 | 11.44 | 少し遅い |
| ngram-simple、n=3 / m=8 / draft最大8 | 35.07 | 12.70 | 112 draft中12 tokenしか採用されず遅い。不採用 |

これらの設定でも全文出力は推奨構成と一致した。追加設定の生値と実際の上書き引数は [tuning.json](tuning.json) に保存した。

実際のMCPサーバーを専用のTMPDIRと31300番台のポートで起動し、CLIのパイプ入力によるstreaming / non-streamingの全文が、直接生成の結果と完全一致した。CLIを含む時間は11.73秒と11.79秒。[実行記録](cli-results.json)、[出力](cli-stream.txt)、[サーバーログ](cli-server.log)を保存した。interactive / non-streamingでの2往復も確認した。

単体・mock CLIテストは29件成功。SSE切断、HTTPエラー、生成上限、起動失敗、子プロセスの終了、バックエンドの取り違え防止、一括処理時の履歴保持を検証している。実モデルのone-shot自動起動・終了と、`PLAMO_MAX_TOKENS=1` でエラーを返して終了コード1になることも確認した。[追加検証記録](additional-checks.json)。検証用に起動したnativeプロセスは全て終了済み。

各条件の生値と出力hashは [results.json](results.json)。より詳細な全SSEイベント・全サーバーログは、この作業環境の `.local/bench-final`、`.local/bench-classic`、`.local/bench-mixed`、`.local/tuning` に保持している。
