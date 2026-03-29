---
name: animaworks-guide
description: >-
  animaworksコマンドの完全リファレンス。サーバー操作・Anima管理・モデル・タスク・設定・RAG・資産・外部ツールのCLI書式をまとめる。
  Use when: サブコマンドの書式確認、サーバー起動停止、Anima作成・モデル変更・タスク追加・ログ・設定・インデックス操作が必要なとき。
---

# AnimaWorks CLI 完全リファレンス

AnimaWorks の操作は全て `animaworks` コマンドで行う。
このスキルは全サブコマンドの書式・引数・具体例をまとめたリファレンス。

運用の考え方やルールは `common_knowledge/` および `reference/` を参照:
- メッセージングのルール → `communication/messaging-guide.md`
- タスク管理 → `operations/task-management.md`
- ツール体系 → `operations/tool-usage-overview.md`
- 組織構造 → `reference/organization/structure.md`
- モデル選択・設定 → `reference/operations/model-guide.md`

---

## 非推奨・互換エイリアス

```bash
# トップレベル（警告が出る）
animaworks list                          # → animaworks anima list を使用
animaworks create-anima ...              # → animaworks anima create を使用

# --local（deprecated: ProcessSupervisor をバイパスする直接実行。HTTP API 経由が推奨）
animaworks chat {名前} "..." --local
animaworks heartbeat {名前} --local
```

推奨: `animaworks start`（または `serve`）でサーバーを起動し、`--local` を付けずに `chat` / `heartbeat` を使う。

`gateway` / `worker` サブコマンドは非表示の互換用（分散アーキ廃止後は使用しない）。

---

## サーバー操作（基本的には使わないこと）

```bash
animaworks start                         # サーバー起動（デフォルト: 0.0.0.0:18500）
animaworks serve                         # start のエイリアス
animaworks start --host 127.0.0.1        # バインドアドレス指定
animaworks start --port 8080             # ポート指定
animaworks start --foreground            # フォアグラウンド起動（-f、デバッグ用）
animaworks stop                          # サーバー停止
animaworks stop --force                  # 強制停止（SIGTERM後SIGKILL、孤立プロセスも終了）
animaworks restart                       # サーバー再起動（--host / --port / -f / --force 可）
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
animaworks anima permissions {名前}      # ツール許可を表示（load_permissions: JSON 優先、レガシーは permissions.md）
```

`anima info` の出力項目:
- Anima名、Enabled、Role、Model、Execution Mode（組み込みラベルは主に S/C/A/B。D・G 等は解決値がそのまま表示されうる）
- Credential、Fallback Model、Max Turns、Max Chains
- Context Threshold、Max Tokens、LLM Timeout
- Thinking / Thinking Effort、Supervisor、Mode S Auth
- Voice設定（tts_provider、voice_id、speed、pitch 等、status.json の voice 辞書を列挙）

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
```

### 活動監査（audit）

対象 Anima の `activity_log` を読む（ヘルプ文言は subordinate だが、名前を指定すれば任意の Anima で可）。

```bash
animaworks anima audit {名前}            # 活動監査（デフォルト: report モード・直近1日）
animaworks anima audit {名前} --days 7   # 集計日数（最大30）
animaworks anima audit --all             # 全Animaを対象（タイムラインをマージ）
animaworks anima audit {名前} --since 09:00   # 当日 JST の時刻以降（指定時は --days より優先）
animaworks anima audit {名前} --mode summary  # 統計サマリー（省略時は report＝時系列）
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
animaworks anima set-background-model --all --clear     # 有効な全Animaの background を一括クリア
animaworks anima set-background-model --all {モデル名}   # 有効な全Anima一括（モデルは位置引数でも可: `{モデル} --all`）
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

set-role で自動更新されるファイル（`--status-only` 以外）:
- `status.json` — role およびロール `defaults.json` 由来の model / max_turns 等をマージ
- `specialty_prompt.md` — ロールテンプレートから上書き
- `permissions.json` — ロールテンプレートから上書き（従来の `permissions.md` は読み込み時に JSON へ移行されうる）

有効なロール: `engineer`, `researcher`, `manager`, `writer`, `ops`, `general`

### ホットリロード

```bash
animaworks anima reload {名前}           # status.json からモデル設定を再読み込み（プロセス再起動なし）
animaworks anima reload --all            # 全Animaをリロード
```

---

## モデル情報（models サブコマンド）

```bash
animaworks models                        # サブコマンド一覧（help）
animaworks models list                   # 組み込みカタログ（KNOWN_MODELS）: 名前・モード・コンテキスト・注記
animaworks models list --mode S          # モードでフィルタ（CLI の choices は S / A / B / C のみ・大文字小文字可）
animaworks models list --json            # JSON出力
animaworks models info {モデル名}        # 任意モデル名の解決結果（実行モード・コンテキスト窓・閾値・ソース）
animaworks models show                   # ~/.animaworks/models.json のパターン一覧
animaworks models show --json            # models.json を生JSON出力
```

`models list` に無いモデル（例: `cursor/*`, `gemini/*`）は `models info <名前>` と `models.json` で確認する。

詳細 → `reference/operations/model-guide.md`

---

## チャット・メッセージング

```bash
# Animaとチャット（人間→Anima）
animaworks chat {名前} "メッセージ"
animaworks chat {名前} "メッセージ" --from {送信者名}
animaworks chat {名前} "メッセージ" --local  # deprecated（ヘルプ参照）

# Anima間メッセージ送信
animaworks send {送信者} {受信者} "メッセージ"
animaworks send {送信者} {受信者} "メッセージ" --intent report   # report / delegation / question または省略（空）
animaworks send {送信者} {受信者} "メッセージ" --reply-to {メッセージID}
animaworks send {送信者} {受信者} "メッセージ" --thread-id {スレッドID}

# ハートビート手動起動
animaworks heartbeat {名前}
animaworks heartbeat {名前} --local      # deprecated（ヘルプ参照）
```

---

## Board（共有チャネル）

```bash
animaworks board read {チャネル名}                      # チャネル読み取り（デフォルト --limit 20）
animaworks board read {チャネル名} --limit 50           # 最大件数
animaworks board read {チャネル名} --human-only         # 人間の投稿のみ
animaworks board post {送信者} {チャネル名} "テキスト"  # チャネルへ投稿
animaworks board dm-history {自分} {相手}               # DM履歴（デフォルト --limit 20）
animaworks board dm-history {自分} {相手} --limit 50    # 件数指定
```

---

## 設定管理（config サブコマンド）

```bash
animaworks config                        # ヘルプ表示（子コマンドまたは -i が無い場合）
animaworks config -i                     # 対話式ウィザード（credential・Anima設定）
animaworks config list                   # 全設定値の一覧表示
animaworks config list --section system  # セクションでフィルタ（例: system, credentials）
animaworks config list --show-secrets    # API keyを表示
animaworks config get {キー}             # 特定の設定値取得（ドット記法: system.log_level）
animaworks config get {キー} --show-secrets
animaworks config set {キー} {値}        # 設定値を変更
animaworks config export-sections        # tool_prompts.sqlite3 → ロケール別 templates/{ja|en}/prompts/
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

**前提**: 実行前に `ANIMAWORKS_ANIMA_DIR` を対象 Anima のディレクトリ（例: `~/.animaworks/animas/{名前}`）に設定すること。未設定だとエラーになる（Anima 子プロセス内の `animaworks-tool task` や、シェルで変数を付与した `animaworks task` が想定用途）。

```bash
animaworks task list                     # タスク一覧（JSON）
animaworks task list --status pending    # ステータスでフィルタ（pending/in_progress/done/cancelled/blocked）
animaworks task add --assignee {名前} --instruction "タスク内容"   # 既定 --source anima
animaworks task add --assignee {名前} --instruction "内容" --source human --deadline 2026-03-10T18:00:00
animaworks task add ... --relay-chain alice,bob   # カンマ区切りリレー鎖（任意）
animaworks task add ... --summary "1行要約"       # 省略時は instruction の先頭100文字
animaworks task update --task-id {ID} --status done
animaworks task update --task-id {ID} --status done --summary "完了サマリー"
```

---

## RAGインデックス管理

依存: RAG スタック未インストール時は `pip install 'animaworks[rag]'` が必要（CLI がエラー表示して終了）。

```bash
animaworks index                         # 全Anima: knowledge/episodes/procedures/skills + state/conversation.json の要約
                                         # + common_knowledge/common_skills（有効Animaごと）+ shared/users
animaworks index --anima {名前}          # 当該Animaのみ（共有コレクション・shared/users はスキップ）
animaworks index --anima {名前} --shared # 当該Animaのメモリに加え common_knowledge/common_skills も更新
animaworks index --full                  # コレクション削除からの全再構築（埋め込みモデル変更・L2→cosine 移行時は必須）
animaworks index --dry-run               # 確認のみ
```

- **conversation_summary**: `state/conversation.json` の `compressed_summary` をインデックス対象に含める（該当ファイルがある場合）。
- **L2 距離の既存コレクション**: サーバー非稼働でローカル Chroma に直接触る場合、`--full` なしだと cosine 移行の警告が出ることがある（メッセージに従い `--full` を実行）。
- **サーバー起動中**: `server.pid` 検出時、CLI は `ANIMAWORKS_VECTOR_URL` / `ANIMAWORKS_EMBED_URL` を設定して HTTP 経由でインデックス・埋め込みを行い、Chroma の同時アクセス競合を避ける。
- **埋め込みモデル**: `index_meta.json` の記録と設定が異なる場合、`--full` なしではエラー終了する。

---

## ランタイム移行（migrate）

```bash
animaworks migrate                       # 未適用のマイグレーションを実行
animaworks migrate --dry-run             # 変更プレビュー
animaworks migrate --verbose             # ファイル単位の詳細
animaworks migrate --list                # ステップ一覧と適用済みフラグ
animaworks migrate --force               # 状態に関わらず再適用
animaworks migrate --resync-db          # SQLite プロンプト DB の再同期のみ
```

実行中サーバーがあると警告が出る。`~/.animaworks/config.json` が無い場合は失敗する。

---

## アセット操作

### アセット最適化

```bash
animaworks optimize-assets                              # assets/ を持つ全Anima（anim_*.glb ストリップ、avatar_chibi*.glb に Draco は既定で実行）
animaworks optimize-assets --anima {名前}               # 特定Animaのみ（短縮形: -a {名前}）
animaworks optimize-assets --dry-run                    # 確認のみ
animaworks optimize-assets --all                        # 簡素化・テクスチャ処理・Draco 等をまとめて適用
animaworks optimize-assets --simplify                   # メッシュ簡素化（既定比率 0.27 前後）
animaworks optimize-assets --simplify 0.2              # 簡素化比率を数値で指定
animaworks optimize-assets --texture-compress           # WebP 化（--texture-resize 省略時は 1024 相当の扱い）
animaworks optimize-assets --texture-resize 512         # テクスチャ最大辺
animaworks optimize-assets --skip-backup                # 実行前バックアップをスキップ
```

### アセット再生成（Vibe Transfer）

```bash
animaworks remake-assets {名前} --style-from {参照Anima}   # 参照の fullbody を基準にスタイル転写
# --steps: カンマ区切り。既定は全ステップ。候補:
#   fullbody, bustup, icon, chibi, 3d, rigging, animations
animaworks remake-assets {名前} --style-from {参照} --steps fullbody,icon
animaworks remake-assets {名前} --style-from {参照} --prompt "..."   # prompt.txt の上書き
animaworks remake-assets {名前} --style-from {参照} --vibe-strength 0.6
animaworks remake-assets {名前} --style-from {参照} --vibe-info-extracted 0.8
animaworks remake-assets {名前} --style-from {参照} --seed 42          # fullbody の再現性
animaworks remake-assets {名前} --style-from {参照} --image-style anime|realistic
animaworks remake-assets {名前} --style-from {参照} --dry-run
animaworks remake-assets {名前} --style-from {参照} --no-backup
```

---

## 外部ツール実行（animaworks-tool）

Anima が外部サービス（Slack, Gmail, GitHub 等）を使う場合のコマンド。

第1引数が登録済みツール名または `submit` のとき、`animaworks` は内部で `animaworks-tool` に振り替わる（例: `animaworks web_search query "..."`）。

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
animaworks slack send --channel "#general" --text "おはようございます"   # 上記ショートカット可
animaworks-tool github issues --repo owner/repo
animaworks-tool submit image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

submit の詳細 → `common_knowledge/operations/background-tasks.md`

**ディレクトリの整理**

- **`state/background_tasks/`** — `submit` 由来の非同期タスク記述子（`pending` / `done` 等）。`internal list-background-tasks` / `check-background-task` はここを読む。
- **`state/background_notifications/`** — ツール完了通知などを heartbeat が吸い上げる用途（MCP・スケジューラ等が書き込む経路あり）。CLI の `list-background-tasks` とは別。

### 子プロセス向けサブコマンド（internal / vault / supervisor）

いずれも **`ANIMAWORKS_ANIMA_DIR` 必須**（`task` と同じ）。トップレベル `animaworks` にも同名サブコマンドが登録されている。

**internal**

```bash
animaworks-tool internal archive-memory {相対パス}    # knowledge/ episodes/ procedures/ 以下のみ
animaworks-tool internal check-permissions {ツール名} [アクション]
# ※実装は permissions.md が存在する場合はそのパース結果を使用（permissions.json は未参照）
animaworks-tool internal create-skill {名前} [--content ...]   # 省略時は標準入力
animaworks-tool internal manage-channel create|archive {チャネル名}
animaworks-tool internal list-background-tasks
animaworks-tool internal check-background-task {task_id}
```

**vault**（Anima 名前空間付き KV）

```bash
animaworks-tool vault get {キー}
animaworks-tool vault store {キー} {値}
animaworks-tool vault list
```

**supervisor**（`ANIMAWORKS_ANIMA_DIR` の Anima 名を起点に、status.json の supervisor 関係で配下を解決）

```bash
animaworks-tool supervisor org-dashboard
animaworks-tool supervisor ping [--name {部下名}]    # 省略時は全配下
animaworks-tool supervisor read-state {部下名}      # 位置引数（祖先関係チェックあり）
animaworks-tool supervisor task-tracker [--status delegated]   # 既定 status=delegated（文字列でフィルタ）
```

### バックグラウンドタスクの確認

- 会話中: MCP ツール `list_background_tasks` / `check_background_task`
- CLI: `animaworks-tool internal list-background-tasks` / `check-background-task {task_id}`（`ANIMAWORKS_ANIMA_DIR` 設定下、`state/background_tasks/`）

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

スキーマ移行・DB再同期は `animaworks migrate`（上記）を参照。

---

## グローバルオプション

```bash
animaworks --gateway-url http://host:port {コマンド}   # API ベース URL（既定: http://localhost:18500）
animaworks --data-dir /path/to/data {コマンド}         # ランタイムディレクトリ（~/.animaworks 相当）
```

`--data-dir` はサブコマンド実行前に `ANIMAWORKS_DATA_DIR` へ反映される。
