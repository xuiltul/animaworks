# 夜間consolidation効率化とcredentials vault移行 実装結果

## 実装概要

### 1. 矛盾ペア検証のバッチ化

- anima内の複数knowledgeターゲットから得た候補ペアを統合・重複排除し、最大20ペアを1回のLLM呼び出しで判定するようにした。
- 入力と出力は `pair_id` 付きJSON配列とし、件数・順序・型を検証する。
- バッチ応答を解析できない場合は、従来の1ペア単位判定へフォールバックする。
- `contradiction_batch_size=1` では従来の単一ペア経路を使う。
- 統計にLLM対象ペア数、実LLM呼び出し数、NLI足切り数を記録する。

### 2. NLI前段フィルタ

- 既存の `SharedNLIModel` を利用し、高信頼のentailmentまたはneutralをLLM判定前に除外する。
- 閾値を `contradiction_nli_prefilter_threshold` で設定し、`None` では無効化する。
- NLIモデルのロード・推論に失敗した場合は、候補を除外せずLLM判定へ継続する。
- animaごとの足切り件数とLLM対象件数をINFOログへ出力する。

### 3. 休眠animaのconsolidationスキップ

- 通常の設定解決層で `anima_defaults.consolidation_enabled` を読み、animaの `status.json` に値があれば優先して、`false` のanimaをスキップする。
- `activity_log/*.jsonl` のエントリ時刻を確認し、デフォルトでは直近7日間に活動がないanimaの日次・週次consolidationをスキップする。
- legacy mixin経路とsupervisor scheduler経路の両方に同じ判定を適用し、スキップ理由をINFOログへ出力する。

### 4. Upsert失敗ループの停止

- ファイル単位の連続Upsert失敗回数を `state/rag_upsert_failures.json` に永続化する。
- デフォルト3回の連続失敗で、原本は移動・削除せず、永続スキップリストへ登録して以後のindex対象から外す。
- Upsert成功直後に失敗カウンターをリセットし、スキップ履歴は同JSONで確認できる。登録時のWARNは1回だけ出力する。
- HTTP write circuitが開いた失敗や、同一実行中に同じcollectionの複数ファイルが失敗した場合はサービス全体の一時障害とみなし、ファイル別カウンターへ加算しない。
- 実ランタイムログと対象ファイルを読み取り調査した結果、`recovered_*.md` 固有の内容・chunk ID異常は確認できず、vector serviceのHTTP 500/503後に通常episodeも失敗していた。永続カウンター、非破壊スキップリスト、一時障害の除外で再試行ループと一括誤隔離を防ぐ方針とした。

### 5. credentials vault移行

- config読み込み時に `{"$vault": "KEY"}` を既存 `vault.json` の `shared` セクションから再帰的に解決する。
- 平文設定は従来どおり読み込み、参照先欠落・不正参照は秘密値を含めず設定エラーにする。
- config保存時は既存のvault参照を保持し、解決済み秘密値が平文で再保存されないようにした。設定API/CLIでcredentialを明示更新・クリアした場合は、参照を保ったままvault側の値を更新する。
- `scripts/migrate_credentials_to_vault.py` を追加した。デフォルトはdry-runで、`--apply` 時だけAWS/Azure等のcredentialをvaultへ移動する。適用前にconfig/vaultのバックアップを作成し、対象ファイルをmode 600にする。
- 実ランタイムに対して移行スクリプトは実行していない。テストはすべて一時ディレクトリ内のダミー値で実施した。

## 追加configキー

| キー | デフォルト | 用途 |
|---|---:|---|
| `anima_defaults.consolidation_enabled` / anima `status.json.consolidation_enabled` | `true` | anima単位のconsolidation有効化 |
| `consolidation.contradiction_batch_size` | `20` | 1回のLLM呼び出しにまとめる矛盾候補数 |
| `consolidation.contradiction_nli_prefilter_threshold` | `0.70` | entailment/neutralのNLI足切り閾値。`None` で無効 |
| `consolidation.inactivity_skip_enabled` | `true` | 休眠animaの自動スキップ |
| `consolidation.inactivity_days` | `7` | 休眠判定の活動確認日数 |
| `rag.upsert_quarantine_failure_threshold` | `3` | Upsert連続失敗から隔離までの回数 |

## 削減見込み

- バッチ化単体では、1 anima・1実行あたり20ペアの場合、LLMセッションを20回から1回へ減らせるため最大95%削減となる。
- 実際の削減率はanimaごとの候補数分布に依存するが、観測された矛盾判定1,616セッションの大部分をバッチ単位へ集約できる。
- NLI足切りと休眠animaスキップは、バッチ化後の呼び出し数をさらに削減する。足切り数と対象ペア数をログ化したため、運用後に実測可能である。

## 変更ファイル

- `core/config/io.py`
- `core/config/cli.py`
- `core/config/resolver.py`
- `core/config/schemas.py`
- `core/config/vault.py`
- `core/lifecycle/knowledge_correction.py`
- `core/lifecycle/system_consolidation.py`
- `core/memory/contradiction.py`
- `core/memory/rag/http_store.py`
- `core/memory/rag/indexer.py`
- `core/memory/rag/indexer_delete.py`
- `core/supervisor/_mgr_scheduler.py`
- `scripts/migrate_credentials_to_vault.py`
- `tests/unit/core/lifecycle/test_knowledge_correction.py`
- `tests/unit/core/lifecycle/test_weekly_memory_hygiene.py`
- `tests/unit/core/memory/test_contradiction.py`
- `tests/unit/core/memory/test_contradiction_batching.py`
- `tests/unit/core/memory/test_forgetting.py`
- `tests/unit/core/memory/test_indexer_deletion.py`
- `tests/unit/core/memory/test_indexer_upsert_quarantine.py`
- `tests/unit/core/supervisor/test_consolidation_targets.py`
- `tests/unit/core/test_consolidation_inactivity.py`
- `tests/unit/core/test_lifecycle_consolidation.py`
- `tests/unit/test_consolidation_retry.py`
- `tests/unit/test_config_vault_references.py`
- `docs/plans/20260712_consolidation効率化とvault移行_結果.md`

## 検証

- 計画書指定の対象選択テスト
- 変更モジュールに対応する既存テストと新規単体テスト
- 変更PythonファイルのRuff lint
- `system_consolidation`、`knowledge_correction`、`indexer` のimport smoke test
- migrationのdry-run非変更、表示差分、apply時のバックアップ・vault参照・暗号化・mode 600を一時ディレクトリで検証
