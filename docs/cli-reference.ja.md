# AnimaWorks CLI リファレンス

**[English version](cli-reference.md)**

> 最終更新: 2026-03-06
> 関連: [spec.ja.md](spec.ja.md), [api-reference.ja.md](api-reference.ja.md)

---

## 基本

```bash
animaworks <コマンド> [オプション]
# または
python3 -m main <コマンド> [オプション]
```

### グローバルオプション

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--data-dir` | path | ランタイムデータディレクトリ（デフォルト: `~/.animaworks` または `ANIMAWORKS_DATA_DIR`） |
| `--help`, `-h` | — | ヘルプ表示 |

---

## 目次

1. [初期化・セットアップ](#1-初期化セットアップ)
2. [サーバー管理](#2-サーバー管理)
3. [チャット・メッセージング](#3-チャットメッセージング)
4. [Board（共有チャネル）](#4-board共有チャネル)
5. [Anima 管理](#5-anima-管理)
6. [設定 (Config)](#6-設定-config)
7. [RAG インデックス](#7-rag-インデックス)
8. [モデル情報](#8-モデル情報)
9. [タスク管理](#9-タスク管理)
10. [ログ・コスト](#10-ログコスト)
11. [アセット管理](#11-アセット管理)

---

## 1. 初期化・セットアップ

### `init` — ランタイムディレクトリの初期化

テンプレートからランタイムディレクトリ（`~/.animaworks/`）を初期化する。

```bash
animaworks init                          # 対話式
animaworks init --from-md character.md   # MDファイルからAnima作成
animaworks init --template default       # テンプレートから作成
animaworks init --blank --name alice     # 空のAnima作成
animaworks init --skip-anima             # インフラのみ（Anima作成なし）
animaworks init --force                  # 不足テンプレートのマージ
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--force` | flag | 既存ランタイムに不足テンプレートのみマージ |
| `--template` | string | テンプレート名指定（非対話） |
| `--from-md` | path | MDファイルからAnima作成（非対話） |
| `--blank` | string | 空のAnima作成（非対話） |
| `--skip-anima` | flag | インフラのみ初期化 |
| `--name` | string | Anima名の上書き（`--from-md` と併用） |

**注意:** `--force`, `--template`, `--from-md`, `--blank`, `--skip-anima` は排他。

---

## 2. サーバー管理

### `start` — サーバー起動

```bash
animaworks start                     # バックグラウンド起動
animaworks start -f                  # フォアグラウンド（ログ表示）
animaworks start --host 0.0.0.0 --port 18500
```

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--host` | string | "0.0.0.0" | バインドホスト |
| `--port` | int | 18500 | ポート番号 |
| `-f`, `--foreground` | flag | — | フォアグラウンド実行 |

`serve` は `start` のエイリアス。

---

### `stop` — サーバー停止

```bash
animaworks stop           # 通常停止（SIGTERM）
animaworks stop --force   # 強制停止（SIGKILL + 孤立プロセス回収）
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--force` | flag | SIGKILL で強制終了、孤立 runner も終了 |

---

### `restart` — サーバー再起動

```bash
animaworks restart
animaworks restart --force   # 強制停止後に起動
```

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--host` | string | "0.0.0.0" | バインドホスト |
| `--port` | int | 18500 | ポート番号 |
| `-f`, `--foreground` | flag | — | フォアグラウンド実行 |
| `--force` | flag | — | 強制停止 |

---

### `reset` — ランタイム完全リセット

サーバー停止 → ランタイムディレクトリ削除 → 再初期化。

```bash
animaworks reset              # リセットのみ
animaworks reset --restart    # リセット後にサーバー起動
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--restart` | flag | リセット後にサーバーを起動 |

---

## 3. チャット・メッセージング

### `chat` — Anima とチャット

```bash
animaworks chat alice "こんにちは"
animaworks chat alice "タスクの進捗を教えて" --from admin
```

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `anima` | positional | 必須 | Anima名 |
| `message` | positional | 必須 | 送信メッセージ |

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--from` | string | "human" | 送信者名 |

---

### `heartbeat` — ハートビート手動実行

```bash
animaworks heartbeat alice
```

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `anima` | positional | 必須 | Anima名 |

---

### `send` — Anima間メッセージ送信

```bash
animaworks send alice bob "レポートです"
animaworks send alice bob "タスクの依頼" --intent delegation
```

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `from_person` | positional | 必須 | 送信者 |
| `to_person` | positional | 必須 | 受信者 |
| `message` | positional | 必須 | 本文 |

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--intent` | string | "" | delegation, report, question |
| `--thread-id` | string | null | スレッドID |
| `--reply-to` | string | null | 返信先メッセージID |

---

## 4. Board（共有チャネル）

### `board read` — チャネル読み取り

```bash
animaworks board read general
animaworks board read ops --limit 50 --human-only
```

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `channel` | positional | 必須 | チャネル名 |

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--limit` | int | 20 | 最大件数 |
| `--human-only` | flag | — | 人間のメッセージのみ |

---

### `board post` — チャネル投稿

```bash
animaworks board post alice general "本日の進捗報告です"
```

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `from_anima` | positional | 必須 | 送信Anima名 |
| `channel` | positional | 必須 | チャネル名 |
| `text` | positional | 必須 | 本文 |

---

### `board dm-history` — DM履歴取得

```bash
animaworks board dm-history alice bob --limit 30
```

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `from_anima` | positional | 必須 | 自分側Anima名 |
| `peer` | positional | 必須 | 相手Anima名 |

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--limit` | int | 20 | 最大件数 |

---

## 5. Anima 管理

### `anima list` — 一覧表示

```bash
animaworks anima list            # サーバーAPI経由
animaworks anima list --local    # ファイルシステム直接スキャン
```

---

### `anima create` — 新規作成

```bash
animaworks anima create --from-md character.md --role engineer
animaworks anima create --from-md character.md --supervisor alice --name bob
animaworks anima create --template default --name carol
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--name` | string | Anima名（blank時は必須） |
| `--template` | string | テンプレート名 |
| `--from-md` | path | キャラクターシートMD |
| `--supervisor` | string | 上司Anima名 |
| `--role` | choice | engineer, researcher, manager, writer, ops, general |

---

### `anima info` — 詳細情報

```bash
animaworks anima info alice
animaworks anima info alice --json
```

モデル名、実行モード、credential、コンテキストウィンドウ、max_turns 等を表示。

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--json` | flag | JSON形式で出力 |

---

### `anima status` — プロセス状態

```bash
animaworks anima status          # 全Anima
animaworks anima status alice    # 特定Anima
```

---

### `anima restart` — プロセス再起動

```bash
animaworks anima restart alice
```

---

### `anima enable` / `anima disable` — 有効化・無効化

```bash
animaworks anima enable alice
animaworks anima disable alice
```

---

### `anima delete` — 削除

```bash
animaworks anima delete alice              # ZIPアーカイブ後に削除
animaworks anima delete alice --no-archive  # アーカイブなしで削除
animaworks anima delete alice --force       # 確認プロンプトをスキップ
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--no-archive` | flag | ZIPアーカイブを作成しない |
| `--force` | flag | 確認プロンプトをスキップ |

---

### `anima set-model` — モデル変更

```bash
animaworks anima set-model alice claude-sonnet-4-6
animaworks anima set-model alice openai/gpt-4.1 --credential azure
animaworks anima set-model --all claude-sonnet-4-6
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `anima` | positional | Anima名（`--all` 時は省略可） |
| `model` | positional | モデル名 |

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--credential` | string | クレデンシャル名 |
| `--all` | flag | 全有効Animaに適用 |

---

### `anima set-background-model` — バックグラウンドモデル設定

heartbeat / inbox / cron で使用するモデルを設定する。

```bash
animaworks anima set-background-model alice claude-sonnet-4-6
animaworks anima set-background-model alice --clear   # メインモデルにフォールバック
animaworks anima set-background-model --all claude-sonnet-4-6
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--credential` | string | クレデンシャル名 |
| `--all` | flag | 全有効Animaに適用 |
| `--clear` | flag | バックグラウンドモデルを解除 |

---

### `anima reload` — 設定ホットリロード

プロセス再起動なしで status.json を再読み込みする。

```bash
animaworks anima reload alice
animaworks anima reload --all
```

---

### `anima set-role` — ロール変更

```bash
animaworks anima set-role alice engineer
animaworks anima set-role alice manager --status-only   # テンプレート再適用なし
animaworks anima set-role alice writer --no-restart      # 自動再起動なし
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `anima` | positional | Anima名 |
| `role` | positional | engineer, researcher, manager, writer, ops, general |

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--status-only` | flag | status.jsonのみ更新 |
| `--no-restart` | flag | 自動再起動をスキップ |

---

### `anima rename` — リネーム

```bash
animaworks anima rename alice alice-v2
animaworks anima rename alice alice-v2 --force
```

config.json、ディレクトリ、supervisor参照を一括更新する。

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--force` | flag | 確認プロンプトをスキップ |

---

### `anima audit` — 監査レポート

部下Animaの活動サマリー、タスク状況、エラー頻度、ツール使用統計をレポートする。

```bash
animaworks anima audit bob
animaworks anima audit bob --days 7
```

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--days` | int | 1 | 監査対象日数（最大30） |

---

## 6. 設定 (Config)

### `config` — 対話式セットアップ

```bash
animaworks config -i    # 対話式ウィザード
```

---

### `config get` — 設定値取得

```bash
animaworks config get system.log_level
animaworks config get credentials.anthropic.api_key --show-secrets
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `key` | positional | ドット記法キー |

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--show-secrets` | flag | APIキー値を表示 |

---

### `config set` — 設定値変更

```bash
animaworks config set system.log_level DEBUG
animaworks config set heartbeat.interval_minutes 15
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `key` | positional | ドット記法キー |
| `value` | positional | 設定値 |

---

### `config list` — 設定一覧

```bash
animaworks config list
animaworks config list --section credentials --show-secrets
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--section` | string | セクションフィルタ |
| `--show-secrets` | flag | APIキー値を表示 |

---

### `config export-sections` — セクションエクスポート

プロンプトDBからテンプレートファイルへエクスポートする。

```bash
animaworks config export-sections
animaworks config export-sections --dry-run
```

---

## 7. RAG インデックス

### `index` — ベクトルインデックス構築

```bash
animaworks index                        # 全Animaの増分インデックス
animaworks index --anima alice          # 特定Animaのみ
animaworks index --full                 # フル再構築
animaworks index --shared               # 共有コレクションをインデックス
animaworks index --dry-run              # 実行せずに対象表示
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--anima` | string | 対象Anima名 |
| `--full` | flag | 既存インデックス削除後にフル再構築 |
| `--shared` | flag | common_knowledge/common_skillsを各Animaにインデックス |
| `--dry-run` | flag | 実行せずに対象を表示 |

---

## 8. モデル情報

### `models list` — 対応モデル一覧

```bash
animaworks models list
animaworks models list --mode S          # Sモード対応のみ
animaworks models list --json
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--mode` | choice | S, A, B, C でフィルタ |
| `--json` | flag | JSON出力 |

---

### `models info` — モデル詳細

```bash
animaworks models info claude-sonnet-4-6
```

実行モード、コンテキストウィンドウ、閾値等を表示。

---

### `models show` — models.json 表示

```bash
animaworks models show
animaworks models show --json
```

---

## 9. タスク管理

Anima のタスクキュー管理。`animaworks-tool` 経由で実行する。

### `animaworks-tool task add` — タスク追加

```bash
animaworks-tool task add --source human --instruction "APIドキュメントを書いて" --assignee alice --deadline 1d
```

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--source` | choice | "anima" | human, anima |
| `--instruction` | string | 必須 | タスク指示文 |
| `--assignee` | string | 必須 | 担当Anima名 |
| `--summary` | string | instruction[:100] | 1行要約 |
| `--deadline` | string | null | ISO8601期限 or `1d`, `2h` 等 |

---

### `animaworks-tool task update` — タスク更新

```bash
animaworks-tool task update --task-id abc123 --status done --summary "完了しました"
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--task-id` | string | タスクID |
| `--status` | choice | pending, in_progress, done, cancelled, blocked |
| `--summary` | string | 更新後の要約 |

---

### `animaworks-tool task list` — タスク一覧

```bash
animaworks-tool task list
animaworks-tool task list --status pending
```

| オプション | 型 | 説明 |
|-----------|-----|------|
| `--status` | choice | pending, in_progress, done, cancelled, blocked |

---

## 10. ログ・コスト

### `logs` — ログ表示

```bash
animaworks logs alice                  # 特定Animaのログ
animaworks logs --all                  # サーバー+全Animaのログ
animaworks logs alice --lines 100      # 100行表示
animaworks logs alice --date 20260305  # 日付指定
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `anima` | positional | Anima名（省略時は `--all` 必要） |

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--all` | flag | — | 全ログ表示 |
| `--lines` | int | 50 | 表示行数 |
| `--date` | string | null | 日付（YYYYMMDD） |

---

### `cost` — トークン使用量・コスト

```bash
animaworks cost                     # 全Animaの30日分
animaworks cost alice               # 特定Animaのみ
animaworks cost --today             # 当日のみ
animaworks cost alice --days 7 --json
```

| 引数 | 型 | 説明 |
|------|-----|------|
| `anima` | positional | Anima名（省略時は全Anima） |

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--days` | int | 30 | 集計日数 |
| `--today` | flag | — | 当日のみ |
| `--json` | flag | — | JSON出力 |

---

## 11. アセット管理

### `optimize-assets` — 3Dアセット最適化

```bash
animaworks optimize-assets --anima alice              # 特定Anima
animaworks optimize-assets --all                       # 全最適化一括
animaworks optimize-assets --anima alice --simplify 0.3
animaworks optimize-assets --dry-run                   # プレビューのみ
```

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `-a`, `--anima` | string | null | 対象Anima |
| `--dry-run` | flag | — | 変更内容のみ表示 |
| `--simplify` | float | 0.27 | メッシュ簡略化比率 |
| `--texture-compress` | flag | — | テクスチャをWebPに変換 |
| `--texture-resize` | int | 1024 | テクスチャ解像度 |
| `--all` | flag | — | strip+simplify+texture+draco一括適用 |
| `--skip-backup` | flag | — | バックアップを作成しない |

---

### `remake-assets` — アセット再生成（スタイル転送）

```bash
animaworks remake-assets bob --style-from alice
animaworks remake-assets bob --style-from alice --image-style realistic
animaworks remake-assets bob --style-from alice --vibe-strength 0.8 --steps fullbody,bustup
animaworks remake-assets bob --style-from alice --dry-run
```

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `anima` | positional | 必須 | 再生成対象Anima名 |

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--style-from` | string | 必須 | スタイル参照Anima名 |
| `--steps` | string | 全ステップ | カンマ区切り（fullbody, bustup, chibi, 3d, rigging, animations） |
| `--prompt` | string | null | プロンプト上書き |
| `--vibe-strength` | float | 0.6 | Vibe Transfer強度（0.0–1.0） |
| `--vibe-info-extracted` | float | 0.8 | 情報抽出レベル（0.0–1.0） |
| `--seed` | int | null | 再現用シード |
| `--image-style` | choice | null | anime, realistic |
| `--no-backup` | flag | — | バックアップをスキップ |
| `--dry-run` | flag | — | API呼び出しなしで内容表示 |

---

### `migrate-cron` — cron.mdマイグレーション

日本語形式のcron.mdを標準cron式に変換する。

```bash
animaworks migrate-cron
```

---

## よく使う操作フロー

### 初回セットアップ

```bash
animaworks init                          # ランタイム初期化
animaworks start                         # サーバー起動
# ブラウザで http://localhost:18500 にアクセス
```

### Anima追加

```bash
animaworks anima create --from-md character.md --role engineer --supervisor alice
animaworks index --anima bob             # RAGインデックス構築
animaworks anima restart bob             # （サーバー起動中なら自動起動）
```

### モデル変更

```bash
animaworks anima set-model alice claude-sonnet-4-6
animaworks anima reload alice            # ホットリロード（再起動不要）
```

### バックグラウンドモデルでコスト最適化

```bash
animaworks anima set-background-model alice claude-sonnet-4-6
animaworks anima restart alice           # background-modelは再起動が必要
```

### トラブルシューティング

```bash
animaworks logs alice --lines 100        # ログ確認
animaworks anima status alice            # プロセス状態
animaworks anima info alice              # 設定確認
animaworks anima restart alice           # 再起動
animaworks stop --force && animaworks start  # 強制リスタート
```
