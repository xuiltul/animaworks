---
name: animaworks-guide
description: >-
  animaworksコマンド・CLIの完全リファレンス。
  サーバー操作(start/serve/stop/restart/status/reset)、チャット・メッセージング(chat/send/board)、
  Anima管理(list/info/create/enable/disable/delete/restart/set-model/set-background-model/
  set-outbound-limit/set-role/reload/rename/audit)、
  モデル情報(models list/info/show)、ハートビート・cron、ログ・コスト(logs/cost)、
  タスク管理(task add/update/list)、設定管理(config get/set/list/export-sections/-i)、
  プロファイル(profile list/add/remove/start/stop/start-all/stop-all)、RAGインデックス管理(index)、
  アセット操作(optimize-assets/remake-assets)、animaworks-toolの外部ツール実行、
  バックグラウンドタスク確認(check_background_task/list_background_tasks)を網羅。
  「コマンド」「使い方」「CLI」「animaworks」「起動」「停止」「再起動」「送信方法」
  「Anima作成」「ロール変更」「モデル変更」「ステータス確認」「インデックス」
  「モデル一覧」「モデル情報」「ログ」「コスト」「タスク」「設定」「プロファイル」
  「バックグラウンドタスク確認」「タスク状態」
---

# AnimaWorks CLI 完全リファレンス

AnimaWorks の操作は全て `animaworks` コマンドで行う。
このスキルは全サブコマンドの書式・引数・具体例をまとめたリファレンス。

運用の考え方やルールは `common_knowledge/` を参照:
- メッセージングのルール → `communication/messaging-guide.md`
- タスク管理 → `operations/task-management.md`
- ツール体系 → `operations/tool-usage-overview.md`
- 組織構造 → `reference/organization/structure.md`
- モデル選択・設定 → `reference/operations/model-guide.md`

---

## サーバー操作（基本的には使わないこと）

```bash
animaworks start                         # サーバー起動（デフォルト: 0.0.0.0:18500）
animaworks serve                         # start のエイリアス
animaworks start --port 8080             # ポート指定
animaworks start --foreground            # フォアグラウンド起動（デバッグ用）
animaworks stop                          # サーバー停止
animaworks stop --force                  # 強制停止（SIGTERM後SIGKILL、孤立プロセスも終了）
animaworks restart                       # サーバー再起動
animaworks restart --force               # 強制停止してから再起動
animaworks status                        # システム状態確認（プロセス・Anima一覧）
animaworks reset                         # ランタイムディレクトリ削除＋再初期化
animaworks reset --restart               # リセット後サーバー自動起動
```

---

## Anima管理（anima サブコマンド）

### 一覧・状態・詳細情報

```bash
animaworks anima list                    # 全Anima一覧（名前・有効/無効・モデル・supervisor）
animaworks anima list --local            # API不使用・ファイルシステム直接スキャン
animaworks anima status                  # 全Animaのプロセス状態（State・モデル・PID・Uptime）
animaworks anima status {名前}           # 特定Animaのプロセス状態
animaworks anima info {名前}             # 設定詳細（モデル・ロール・credential・voice等）
animaworks anima info {名前} --json      # JSON出力
```

`anima info` の出力項目:
- Anima名、Enabled、Role、Model、Execution Mode
- Credential、Fallback Model、Max Turns、Max Chains
- Context Threshold、Max Tokens、LLM Timeout
- Thinking設定、Supervisor、Mode S Auth
- Voice設定（tts_provider、voice_id、speed、pitch）

### 作成

```bash
# キャラクターシート（MD）から作成（推奨）
animaworks anima create --from-md {ファイル} [--role {role}] [--name {名前}]

# テンプレートから作成
animaworks anima create --template {テンプレート名} [--name {名前}]

# ブランク作成
animaworks anima create --name {名前}
```

### 有効化・無効化・削除

```bash
animaworks anima enable {名前}           # 有効化（休養から復帰）
animaworks anima disable {名前}          # 無効化（休養）
animaworks anima delete {名前}           # 削除（ZIPアーカイブ後）
animaworks anima delete {名前} --no-archive  # アーカイブなしで削除
animaworks anima delete {名前} --force   # 確認なしで削除
animaworks anima restart {名前}          # プロセス再起動
animaworks anima audit {名前}            # 部下の直近活動を包括監査（デフォルト: 1日）
animaworks anima audit {名前} --days 7   # 直近7日間の監査
```

### モデル変更

```bash
animaworks anima set-model {名前} {モデル名}
animaworks anima set-model {名前} {モデル名} --credential {credential名}
animaworks anima set-model --all {モデル名}   # 全Anima一括変更
```

変更後にサーバーが起動中の場合は `anima restart {名前}` が必要。

### バックグラウンドモデル（Heartbeat/Cron用）

```bash
animaworks anima set-background-model {名前} {モデル名}   # Heartbeat・Cron用モデルを設定
animaworks anima set-background-model {名前} {モデル名} --credential {credential名}
animaworks anima set-background-model {名前} --clear    # オーバーライド解除（メインモデルにフォールバック）
animaworks anima set-background-model --all {モデル名}   # 全Anima一括変更
```

### アウトバウンド制限

```bash
animaworks anima set-outbound-limit {名前} --per-hour 30 --per-day 100   # 送信レート制限
animaworks anima set-outbound-limit {名前} --per-run 5                  # 1 runあたりの宛先数
animaworks anima set-outbound-limit {名前} --clear                      # ロールデフォルトに戻す
```

### 名前変更

```bash
animaworks anima rename {旧名前} {新名前}
animaworks anima rename {旧名前} {新名前} --force   # 確認なしで実行
```

### ロール変更

```bash
# ロール変更（テンプレート再適用 + 自動restart）
animaworks anima set-role {名前} {role}

# status.jsonのroleフィールドのみ変更（テンプレートは触らない）
animaworks anima set-role {名前} {role} --status-only

# ファイル更新のみ・再起動しない
animaworks anima set-role {名前} {role} --no-restart
```

set-role で自動更新されるファイル:
- `status.json` — role・モデル・max_turns をロールテンプレートの標準値に更新
- `specialty_prompt.md` — ロール別専門ガイドラインに差し替え
- `permissions.json` — ロール別のツール・コマンド許可範囲に差し替え

有効なロール: `engineer`, `researcher`, `manager`, `writer`, `ops`, `general`

### ホットリロード

```bash
animaworks anima reload {名前}           # status.json からモデル設定を再読み込み（プロセス再起動なし）
animaworks anima reload --all            # 全Animaをリロード
```

---

## モデル情報（models サブコマンド）

```bash
animaworks models list                   # 既知モデル一覧（名前・実行モード・コンテキスト窓・説明）
animaworks models list --mode S          # 実行モードでフィルタ（S/C/D/G/A/B）
animaworks models list --json            # JSON出力
animaworks models info {モデル名}        # 特定モデルの解決情報（実行モード・コンテキスト窓・閾値・ソース）
animaworks models show                   # models.json の現在の内容を表示
animaworks models show --json            # models.json を生JSON出力
```

詳細 → `reference/operations/model-guide.md`

---

## チャット・メッセージング

```bash
# Animaとチャット（人間→Anima）
animaworks chat {名前} "メッセージ"
animaworks chat {名前} "メッセージ" --from {送信者名}
animaworks chat {名前} "メッセージ" --local  # API不使用・直接実行

# Anima間メッセージ送信
animaworks send {送信者} {受信者} "メッセージ"
animaworks send {送信者} {受信者} "メッセージ" --intent report   # または --intent question
animaworks send {送信者} {受信者} "メッセージ" --reply-to {メッセージID}
animaworks send {送信者} {受信者} "メッセージ" --thread-id {スレッドID}

# ハートビート手動起動
animaworks heartbeat {名前}
animaworks heartbeat {名前} --local      # API不使用・直接実行
```

---

## Board（共有チャネル）

```bash
animaworks board read {チャネル名}                      # チャネルメッセージ読み取り
animaworks board read {チャネル名} --limit 50           # 最大件数指定
animaworks board read {チャネル名} --human-only         # 人間の投稿のみ
animaworks board post {送信者} {チャネル名} "テキスト"  # チャネルへ投稿
animaworks board dm-history {自分} {相手}               # DM履歴取得
animaworks board dm-history {自分} {相手} --limit 50    # 件数指定
```

---

## 設定管理（config サブコマンド）

```bash
animaworks config -i                     # 対話式ウィザード（credential・Anima設定）
animaworks config list                   # 全設定値の一覧表示
animaworks config list --section system  # セクションでフィルタ（例: system, credentials）
animaworks config list --show-secrets    # API keyを表示
animaworks config get {キー}             # 特定の設定値取得（ドット記法: system.log_level）
animaworks config get {キー} --show-secrets
animaworks config set {キー} {値}        # 設定値を変更
animaworks config export-sections        # tool_prompts.sqlite3 から templates/prompts/ へエクスポート
animaworks config export-sections --dry-run
```

**注意**: Animaのモデル・credential等は `status.json` がSSoT。`animas.{名前}.model` 等の直接設定は非推奨。`animaworks anima set-model` を使用すること。

---

## プロファイル（profile サブコマンド・マルチテナント）

複数のAnimaWorksインスタンス（別データディレクトリ）を管理する。

```bash
animaworks profile list                  # 全プロファイル一覧（data_dir・port・状態）
animaworks profile add {名前}             # プロファイル登録（data_dir: ~/.animaworks/{名前}）
animaworks profile add {名前} --data-dir /path/to/data --port 18510
animaworks profile remove {名前}         # 登録解除（データは残る）
animaworks profile start {名前}          # そのプロファイルのサーバー起動
animaworks profile stop {名前}           # そのプロファイルのサーバー停止
animaworks profile stop {名前} --force   # 強制停止
animaworks profile start-all             # 全プロファイルを起動
animaworks profile stop-all              # 全プロファイルを停止
animaworks profile stop-all --force      # 強制停止で全停止
```

---

## ログ閲覧（logs）

```bash
animaworks logs {名前}                   # 特定Animaのログ表示
animaworks logs --all                    # サーバー＋全Animaのログ表示
animaworks logs {名前} --lines 100       # 表示行数指定（デフォルト: 50）
animaworks logs {名前} --date 20260301   # 特定日のログ表示
```

---

## コスト確認（cost）

```bash
animaworks cost                          # 全Animaのトークン使用量・コスト
animaworks cost {名前}                   # 特定Animaのコスト
animaworks cost --today                  # 本日のみ
animaworks cost --days 7                 # 直近7日（デフォルト: 30日）
animaworks cost --json                   # JSON出力
```

---

## タスク管理（task サブコマンド）

```bash
animaworks task list                     # タスク一覧
animaworks task list --status pending    # ステータスでフィルタ（pending/in_progress/done/cancelled/blocked）
animaworks task add --assignee {名前} --instruction "タスク内容"
animaworks task add --assignee {名前} --instruction "内容" --source human --deadline 2026-03-10T18:00:00
animaworks task update --task-id {ID} --status done
animaworks task update --task-id {ID} --status done --summary "完了サマリー"
```

---

## RAGインデックス管理

```bash
animaworks index                         # 全Animaのインデックス増分更新
animaworks index --anima {名前}          # 特定Animaのみ
animaworks index --full                  # 全データ再インデックス
animaworks index --dry-run               # 変更内容の確認のみ（実行しない）
```

---

## アセット操作

### アセット最適化

```bash
animaworks optimize-assets                              # 全Animaの3Dアセット最適化
animaworks optimize-assets --anima {名前}               # 特定Animaのみ
animaworks optimize-assets --dry-run                    # 確認のみ
animaworks optimize-assets --simplify                   # メッシュ簡素化
animaworks optimize-assets --texture-compress           # テクスチャ圧縮
animaworks optimize-assets --texture-resize 512         # テクスチャリサイズ
```

### アセット再生成

```bash
animaworks remake-assets {名前} --style-from {参照Anima}   # スタイル転写でアセット再生成
animaworks remake-assets {名前} --style-from {参照} --steps portrait,fullbody
animaworks remake-assets {名前} --style-from {参照} --dry-run
animaworks remake-assets {名前} --style-from {参照} --no-backup
```

---

## 外部ツール実行（animaworks-tool）

Anima が外部サービス（Slack, Gmail, GitHub 等）を使う場合のコマンド。

```bash
# ヘルプ表示
animaworks-tool {ツール名} --help

# 実行
animaworks-tool {ツール名} {サブコマンド} [引数...]

# バックグラウンド実行（長時間ツール向け）
animaworks-tool submit {ツール名} {サブコマンド} [引数...]
```

### 具体例

```bash
animaworks-tool web_search query "AnimaWorks framework"
animaworks-tool slack send --channel "#general" --text "おはようございます"
animaworks-tool github issues --repo owner/repo
animaworks-tool submit image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

submit の詳細 → `common_knowledge/operations/background-tasks.md`

### バックグラウンドタスクの確認（Anima内部ツール）

submit で投入したタスクの進捗確認には、以下の内部ツールを使用する:
- `list_background_tasks` — 実行中・完了済みタスクの一覧
- `check_background_task(task_id)` — 特定タスクのステータス・結果取得

これらはCLIコマンドではなく、Animaが会話中に使用するMCPツール。

---

## 初期化・マイグレーション

```bash
animaworks init                          # ランタイムディレクトリ初期化（~/.animaworks/）
animaworks init --force                  # 既存にテンプレート差分をマージ
animaworks init --skip-anima             # インフラのみ初期化（Anima作成スキップ）
animaworks init --template {名前}       # テンプレートからAnimaを非対話で作成
animaworks init --from-md {PATH}         # MDファイルからAnimaを非対話で作成
animaworks init --blank {名前}           # ブランクAnimaを非対話で作成
animaworks init --from-md {PATH} --name {名前}  # 作成時の名前を上書き
animaworks migrate-cron                  # cron.md を日本語形式→標準cron式に変換
```

---

## グローバルオプション

```bash
animaworks --gateway-url http://host:port {コマンド}   # サーバーURLを指定
animaworks --data-dir /path/to/data {コマンド}         # ランタイムディレクトリを指定
```
