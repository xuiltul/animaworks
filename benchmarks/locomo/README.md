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

# 標準プロトコル（10会話 / LLM Judge / cat5除外 / leakage path off）
python -m benchmarks.locomo.runner --standard-protocol

# 回答モデル変更
python -m benchmarks.locomo.runner --answer-model gpt-4o
```

### オプション

| フラグ | デフォルト | 説明 |
|--------|----------|------|
| `--data PATH` | `benchmarks/locomo/data/locomo10.json` | データセットパス |
| `--mode MODE` | `all` | `vector` / `vector_graph` / `scope_all` / `all` |
| `--conversations N` | `10` | 処理する会話数 |
| `--top-k K` | `10` | 検索結果数 |
| `--exclude-cat5` | off | adversarial（category 5）を評価集計から除外 |
| `--judge` | off | LLM Judge 有効化 |
| `--judge-model` | `gpt-4o` | Judge モデル |
| `--answer-model` | `gpt-4o-mini` | 回答生成モデル |
| `--enable-locomo-alias` | off | 旧LoCoMo alias mapを有効化。conv-26由来のgold answer leakageを含むため標準計測では禁止 |
| `--enable-locomo-category-branches` | off | gold category labelを使う検索gate/answer整形を有効化。標準計測では禁止 |
| `--standard-protocol` | off | `scope_all`、10会話、LLM Judge on、cat5除外、leakage flags off の標準条件を適用 |
| `--output DIR` | `benchmarks/locomo/results` | 結果出力先 |

## 標準計測プロトコル

2026-06以降の比較基準は以下の条件に固定する。

- データ: `locomo10.json` の10会話フルラン
- 標準モード: `scope_all`
- leakage path: `--enable-locomo-alias` と `--enable-locomo-category-branches` はどちらも未指定（off）
- 主指標: LLM Judge の `cat5_excluded.overall_judge`
- 副指標: token F1 の `overall_f1`（category 5込み、過去時系列との互換用）
- Judge モデル: 回答モデルと異なるモデルを使う（`judge_model != answer_model`）
- error扱い: retrieve/answer失敗は `status=error` として結果JSONに残し、F1=0として混ぜない。summaryの `error_rate` が2%を超えるrunは無効

標準run例:

```bash
python -m benchmarks.locomo.runner \
  --standard-protocol \
  --answer-model deepseek-v4-flash \
  --judge-model gpt-4o
```

結果JSONの `config` には `leakage_alias_map_enabled`、`category_branches_enabled`、
`category_dependent_normalization_enabled`、`protocol_version` が記録される。`summary.standard_protocol`
には主指標値、judge独立性、leakage-free判定、error率、valid判定が記録される。

2026-06-11 のローカル再基準化試行は `benchmarks/results/locomo_standard_protocol_baseline_20260611.json`
に記録した。2問smokeの段階で error率が100%となったため、有効な10会話baselineではない。

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

以下は1会話・token F1・category 5込み・旧条件のスモーク回帰用であり、Mem0/Zep等との比較PASS判定や
2026-06以降の標準baselineとしては無効。`benchmarks/locomo/baselines/*.json` にも
`invalid_for_standard_comparison` を記録している。

ベースライン（回答モデルで自動選択、`LOCOMO_BASELINE` で override 可）:

| モデル | ファイル | overall F1 | open_domain F1 |
|--------|----------|------------|----------------|
| `deepseek-v4-flash`（default） | `baselines/legacy_scope_all_deepseek_v4_flash_20260525.json` | **44.7%** | **40.3%** |
| `Qwen3.5-397B` 等 | `baselines/legacy_scope_all_20260522.json` | **63.7%** | **70.8%** |

計測条件: 1 conv / top-k=10 / scope_all

**2026-05-25 Phase 1–3 は棄却:** 短文回答プロンプト、category 別 gate 緩和、category 4 の年抽出は deepseek smoke を **44.7% → 37.0%** に下げたため、ベースラインへ反映しない。新しい prompt / gate 実験は、代表失敗ケースの unit test と 1 conv smoke の両方を通してから別ベースラインとして追加する。

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
3. adversarial が既存ベースラインから大きく悪化していないことを確認
4. 該当モデルの `benchmarks/locomo/baselines/legacy_scope_all_<model>_YYYYMMDD.json` を更新（または新規作成して `default_baseline_path()` のマッピングを更新）
