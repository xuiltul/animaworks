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
