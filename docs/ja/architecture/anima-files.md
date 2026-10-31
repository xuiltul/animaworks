> 確認したコミット: b304b7dc

# Anima のファイル

各 Anima は `~/.animaworks/animas/{name}/` にディレクトリを持つ。個性や行動方針、実行設定はテキストと JSON に分けて保存し、セッション中に更新される情報は `state/` に置く。

## 定義と設定

| ファイル | 役割 |
|---|---|
| `IDENTITY.md` | 名前、基本プロフィール、話し方など、外部に提示する恒常的なアイデンティティを記述する。 |
| `SOUL.md` | 価値観、原則、内面の性格など、振る舞いの基底を記述する。 |
| `AGENTS.md` | 専門性、追加の行動指針、作業上の指示を記述し、system prompt に取り込む。旧 `injection.md`。 |
| `permissions.json` | 個別 Anima のツール、コマンド、ファイルアクセスに関する権限を表す。 |
| `status.json` | 有効状態、role、担当、モデル、認証、supervisor、heartbeat の有効化など、実行時に解決する per-anima 情報を保持する。 |
| `HEARTBEAT.md` | heartbeat の指示と活動時間帯を記述する。実行間隔は全体設定または `status.json` の値から決まり、このファイルの本文からは設定しない。 |
| `BOOTSTRAP.md` | 初回起動時の自己紹介や初期化の指示を記述する。 |
| `cron.md` | 定期実行するタスクを見出しごとに記述する。スケジュールと実行内容は supervisor が読み取る。 |

`status.json` は ModelConfig に関わる per-anima 値の正本である。対応キーと既定値の一覧は[設定リファレンス](../reference/config.md)を参照する。`permissions.json` と全体の `permissions.global.json` が権限の判断に使われる。権限の境界と評価方法は[セキュリティ](../security.md)を参照する。

## ワークスペースファイル

Anima の人格と作業指示は、OpenClaw / Hermes と同じ標準ファイル名で管理する。旧名（`identity.md`、`injection.md`、`heartbeat.md`、`bootstrap.md`）はサーバ起動時の migration で自動的に新名へ切り替わり、以後は新名だけを読み書きする。

| 用途 | AnimaWorks | OpenClaw | Hermes |
|---|---|---|---|
| 公開アイデンティティ | `IDENTITY.md` | `IDENTITY.md` | `IDENTITY.md`（取り込み） |
| 内面の価値観・原則 | `SOUL.md` | `SOUL.md` | `SOUL.md`（書き出し時は IDENTITY → SOUL の順で連結） |
| 作業ディレクトリの指示 | `AGENTS.md` | `AGENTS.md` | `AGENTS.md` |
| 定期巡回の指示 | `HEARTBEAT.md` | `HEARTBEAT.md` | `HEARTBEAT.md` |
| 初回起動の指示 | `BOOTSTRAP.md` | — | — |

旧 `identity.md` は migration 時に `split_legacy_identity()` で分割される。名前・プロフィール・話し方などの見出しは `IDENTITY.md`、価値観・性格・口調などの見出しは `SOUL.md` に入る。旧 `injection.md` は `AGENTS.md`、旧 `heartbeat.md` は `HEARTBEAT.md`、旧 `bootstrap.md` は `BOOTSTRAP.md` に対応する。

### 移行とロールバック

サーバ起動時、`core/infra/runtime_init.py` の自動 migration が anima の検出より前に全 anima を移行する。変更前の旧ファイルと書き換え前の参照ファイルは、anima ごとの `archive/standard_files_migration/<UTC時刻>/` に保存され、`manifest.json` に旧名→新名の対応、sha256、新ファイル一覧、衝突、実行時刻とバージョンが記録される。

```bash
# 戻す内容を確認する（何も変更しない）
animaworks migrate --rollback-standard-files --dry-run

# 全 anima を戻す
animaworks migrate --rollback-standard-files

# 特定の anima だけ戻す
animaworks migrate --rollback-standard-files --anima alice
```

ロールバックは新しい標準ファイルを `archive/standard_files_migration/<UTC時刻>/rolled_back_new/` に退避してから旧ファイルを復元し、`migration_state.json` から本ステップの適用記録を外す。移行後に新ファイルを編集していた場合は警告が表示されるが、編集内容は `rolled_back_new/` に残る。**旧版の AnimaWorks で起動する前に**このコマンドを実行すること。

### 書き出しと取り込み

```bash
# OpenClaw / Hermes 形式へ書き出す
animaworks anima export alice --format openclaw --out ./alice-openclaw
animaworks anima export alice --format hermes --out ./alice-hermes

# OpenClaw / Hermes のデータを取り込む
animaworks import openclaw --path ~/.openclaw --target-anima alice --apply
animaworks import hermes --path ~/.hermes --target-anima alice --apply
```

書き出しでは `permissions.json`、`status.json`、`cron.md` は標準形式に対応物がないため含めず、`ANIMAWORKS_EXPORT.md` に書き出したもの・書き出さなかったものを記録する。秘密情報は `redact_credentials()` で伏字化する。取り込み元に `SOUL.md`、`IDENTITY.md`、`AGENTS.md`、`HEARTBEAT.md` がある場合、同名の新ファイルが無ければ直接反映する。既存ファイルがある場合は `--replace` を付けない限り上書きせず、`state/migrations/drafts/` に下書きを残す。

## 実行状態

| パス | 役割 |
|---|---|
| `state/current_state.md` | 進行中の作業、判断、次の手順などを保持する作業状態である。prompt に必要な範囲で取り込まれ、更新は状態ファイル用の lock を介して保護される。 |
| `state/task_queue.jsonl` | 旧形式のタスクデータが残る場合の移行入力である。現在のタスクの正本は共有 SQLite TaskStore であり、このファイルを実行時の別キューとして扱わない。 |
| `state/background_tasks/` | `animaworks-tool submit` で開始したバックグラウンド処理の状態と結果を保存する。完了結果は会話や後続の背景実行から通知・参照される。 |
| `scratch/` | 使い捨ての作業場所である。子プロセスには `ANIMAWORKS_SCRATCH_DIR` としてタスク用の `scratch/task-<id>/` かセッション種類別の `scratch/<種類>/` が渡される。タスク用はタスクの完了・キャンセルで、それ以外は 24 時間放置で housekeeping が削除する。 |

Anima の長期記憶、会話履歴、スキルなどは用途別のサブディレクトリに分かれている。記憶の詳細は `docs/ja/memory/` の各章を参照する。

## 設計判断

- **記憶は Markdown ファイルで持つ。** AI が自然に読み書きでき、grep とも相性がよい。JSON は設定と状態に限る。
- **書庫型の記憶を採る。** 直近 N 件を prompt に詰める切り詰め型では記憶量に上限ができる。必要なものを検索して想起する書庫型なら、記憶は増え続けてよい。
- **権限は視野の制限である。** 知らないことがあるから他者に尋ねる。全員が全てを見られると組織は意味を失う。
