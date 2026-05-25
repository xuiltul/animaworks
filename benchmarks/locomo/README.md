# LoCoMo Benchmark for AnimaWorks

AnimaWorks の記憶システム（RAG + グラフ拡散活性化 + BM25/RRF統合）を [LoCoMo](https://github.com/snap-research/locomo)（ACL 2024）ベンチマークで評価するアダプター。

## セットアップ

### 1. データセット取得

```bash
mkdir -p benchmarks/locomo/data
cd benchmarks/locomo/data
# snap-research/locomo リポジトリからダウンロード
wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
cd ../../..
```

### 2. 依存パッケージ

AnimaWorks の RAG スタック（通常のインストールで含まれる）:

```bash
pip install chromadb sentence-transformers rank-bm25
```

回答生成・LLM Judge に LiteLLM + API キー:

```bash
export OPENAI_API_KEY="sk-..."
```

## 実行

```bash
# 全モード（vector / vector_graph / scope_all）、F1のみ
python -m benchmarks.locomo.runner

# 1会話のみテスト
python -m benchmarks.locomo.runner --conversations 1

# 特定モード
python -m benchmarks.locomo.runner --mode vector

# LLM Judge 有効
python -m benchmarks.locomo.runner --judge

# 回答モデル変更
python -m benchmarks.locomo.runner --answer-model gpt-4o
```

### オプション

| フラグ | デフォルト | 説明 |
|--------|----------|------|
| `--data PATH` | `benchmarks/locomo/data/locomo10.json` | データセットパス |
| `--mode MODE` | `all` | `vector` / `vector_graph` / `scope_all` / `all` |
| `--conversations N` | `10` | 処理する会話数 |
| `--top-k K` | `5` | 検索結果数 |
| `--judge` | off | LLM Judge 有効化 |
| `--judge-model` | `gpt-4o` | Judge モデル |
| `--answer-model` | `gpt-4o-mini` | 回答生成モデル |
| `--output DIR` | `benchmarks/locomo/results` | 結果出力先 |

## 検索モード

| モード | 内容 |
|-------|------|
| `vector` | ChromaDB ベクトル検索のみ（ベースライン） |
| `vector_graph` | ベクトル + NetworkX グラフ拡散活性化 |
| `scope_all` | ベクトル + グラフ + BM25 キーワード検索 + RRF 融合 |

## 評価メトリクス

- **Stemmed token F1**: 公式 LoCoMo メトリクス（カテゴリ別ルール適用）
- **LLM Judge**: GPT-4o による意味的正誤判定（`--judge` 有効時）

## カテゴリ

| ID | 名前 | 説明 |
|----|------|------|
| 1 | multi_hop | 複数事実の組み合わせ推論 |
| 2 | temporal | 時間に関する質問 |
| 3 | complex | 複雑な質問 |
| 4 | open_domain | 自由形式 |
| 5 | adversarial | 情報がない場合の棄権判定 |

## 結果

`benchmarks/locomo/results/` に JSON で保存:

```
results/
├── 2026-04-22T10-30-00_vector.json
├── 2026-04-22T10-30-00_vector_graph.json
└── 2026-04-22T10-30-00_scope_all.json
```

## 制約事項

- **スレッド安全性**: `AnimaWorksLoCoMoAdapter` はプロセスごとに1インスタンスのみ使用可能。`ANIMAWORKS_DATA_DIR` 環境変数を使用するため、マルチスレッドでの並行実行は不可。並列実行にはプロセス分離を使用すること。

## 設定情報

- **Embedding**: `intfloat/multilingual-e5-small` (384次元)
- **データ隔離**: 一時ディレクトリ（`ANIMAWORKS_DATA_DIR`）を使用、既存データに影響しない

## Legacy 回帰スモーク（Wave 1 harness）

固定ベースライン: `benchmarks/locomo/baselines/legacy_scope_all_20260522.json`  
（2026-05-22 計測: overall F1 **63.7%**, open_domain **70.8%**, 1 conv / top-k=10）

### 環境変数

| 変数 | 必須 | 説明 |
|------|------|------|
| `LOCOMO_ANSWER_MODEL` | No | 回答モデル（default: `deepseek-v4-flash`） |
| `LOCOMO_LLM_CREDENTIAL` | No | `~/.animaworks/config.json` の credential 名（default: `vllm-lb` → `http://localhost:4000/v1`） |
| `OPENAI_API_BASE` | No | 明示 override（未設定時は credential 解決） |
| `OPENAI_API_KEY` | 推奨 | API キー（ローカル LiteLLM では `dummy` 可） |

trserveru 標準構成（LiteLLM プロキシ）:

```bash
# 追加設定なしで vllm-lb (localhost:4000) を使用
./scripts/locomo_legacy_smoke.sh

# 明示指定する場合
export LOCOMO_ANSWER_MODEL="deepseek-v4-flash"
export LOCOMO_LLM_CREDENTIAL="vllm-lb"
```

旧 MacStudio 直叩きとの比較:

```bash
export OPENAI_API_BASE="http://100.72.124.21:8001/v1"
export LOCOMO_ANSWER_MODEL="openai/mlx-community/Qwen3.5-397B-A17B-4bit"
```

Neo4j 確認をスキップ: `./scripts/locomo_legacy_smoke.sh --skip-neo4j`

### pytest

```bash
# ベースライン JSON の存在確認のみ（LLM 不要）
pytest tests/integration/test_locomo_legacy_smoke.py::test_legacy_baseline_file_shape -m locomo

# 1 conv フルスモーク（LLM 必須）
pytest tests/integration/test_locomo_legacy_smoke.py -m locomo
```

### ベースライン更新手順

1. `./scripts/locomo_legacy_smoke.sh` で新結果を取得
2. overall / open_domain が意図的改善であることを確認
3. `benchmarks/locomo/baselines/legacy_scope_all_20260522.json` の `overall_f1` / `by_category` を更新（ファイル名は日付サフィックスで新規作成しても可）

