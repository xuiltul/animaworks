---
name: animaworks-guide
description: >-
  Complete reference for the animaworks command. Summarizes CLI formats for server operations, Anima management, models, tasks, configuration, RAG, assets, and external tools.
  Use when: Use when: checking subcommand formats, server startup and shutdown, creating Anima, changing models, adding tasks, logs, configuration, or index operations are needed.
---


# AnimaWorks CLI Complete Reference

All AnimaWorks operations are performed using the `animaworks` command.
This skill is a reference summarizing the formats, arguments, and concrete examples of all subcommands.

For operational principles and rules, refer to `common_knowledge/` and `reference/`:
- Messaging rules → `communication/messaging-guide.md`
- Task management → `operations/task-management.md`
- Tool system → `operations/tool-usage-overview.md`
- Organization structure → `reference/organization/structure.md`
- Model selection and configuration → `reference/operations/model-guide.md`

---

## Deprecated and Compatibility Aliases

```bash
# --local（deprecated: ProcessSupervisor をバイパスする直接実行。HTTP API 経由が推奨）
animaworks chat {名前} "..." --local
animaworks heartbeat {名前} --local
```

Recommended: Start the server with `animaworks start` (or `serve`), and use `chat` / `heartbeat` without adding `--local`.

`gateway` / `worker` subcommands are hidden compatibility options (do not use after the distributed architecture is discontinued).

---

## Server Operations (generally avoid using)

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

## Anima Management (anima subcommand)

### List, Status, and Detailed Information

```bash
animaworks anima list                    # 全Anima一覧（名前・有効/無効・モデル・supervisor）
animaworks anima list --local            # API不使用・ファイルシステム直接スキャン
animaworks anima status                  # 全Animaのプロセス状態（State・モデル・PID・Uptime）
animaworks anima status {名前}           # 特定Animaのプロセス状態
animaworks anima info {名前}             # 設定詳細（モデル・ロール・credential・voice等）
animaworks anima info {名前} --json      # JSON出力
animaworks anima permissions {名前}      # ツール許可を表示（load_permissions: JSON 優先、レガシーは permissions.md）
```

Output items of `anima info`:
- Anima name, Enabled, Role, Model, Execution Mode (built-in labels are S/C/D/G/X/A）
- Credential, Fallback Model, Context Threshold, Max Tokens, Thinking / Thinking Effort, Supervisor, Mode S Auth
- Voice settings (tts_provider, voice_id, speed, pitch, etc., listing the voice dictionary of status.json)

### Creation

```bash
# キャラクターシート（MD）から作成（推奨）
animaworks anima create --from-md {ファイル} [--role {role}] [--name {名前}]

# テンプレートから作成
animaworks anima create --template {テンプレート名} [--name {名前}]

# ブランク作成
animaworks anima create --name {名前}
```

### Enable, Disable, and Delete

```bash
animaworks anima enable {名前}           # 有効化（休養から復帰）
animaworks anima disable {名前}          # 無効化（休養）
animaworks anima delete {名前}           # 削除（ZIPアーカイブ後）
animaworks anima delete {名前} --no-archive  # アーカイブなしで削除
animaworks anima delete {名前} --force   # 確認なしで削除
animaworks anima restart {名前}          # プロセス再起動
```

### Activity Audit (audit)

Reads the `activity_log` of the target Anima (the help text says subordinate, but specifying a name allows any Anima).

```bash
animaworks anima audit {名前}            # 活動監査（デフォルト: report モード・直近1日）
animaworks anima audit {名前} --days 7   # 集計日数（最大30）
animaworks anima audit --all             # 全Animaを対象（タイムラインをマージ）
animaworks anima audit {名前} --since 09:00   # 当日 JST の時刻以降（指定時は --days より優先）
animaworks anima audit {名前} --mode summary  # 統計サマリー（省略時は report＝時系列）
```

### Model Change

```bash
animaworks anima set-model {名前} {モデル名}
animaworks anima set-model {名前} {モデル名} --credential {credential名}
animaworks anima set-model --all {モデル名}   # 全Anima一括変更
```

If the server is running after the change, `anima restart {名前}` is required.

### Background Model (for Heartbeat/Cron)

```bash
animaworks anima set-background-model {名前} {モデル名}   # Heartbeat・Cron用モデルを設定
animaworks anima set-background-model {名前} {モデル名} --credential {credential名}
animaworks anima set-background-model {名前} --clear    # オーバーライド解除（メインモデルにフォールバック）
animaworks anima set-background-model --all --clear     # 有効な全Animaの background を一括クリア
animaworks anima set-background-model --all {モデル名}   # 有効な全Anima一括（モデルは位置引数でも可: `{モデル} --all`）
```

### Rename

```bash
animaworks anima rename {旧名前} {新名前}
animaworks anima rename {旧名前} {新名前} --force   # 確認なしで実行
```

### Role Change

```bash
# ロール変更（テンプレート再適用 + 自動restart）
animaworks anima set-role {名前} {role}

# status.jsonのroleフィールドのみ変更（テンプレートは触らない）
animaworks anima set-role {名前} {role} --status-only

# ファイル更新のみ・再起動しない
animaworks anima set-role {名前} {role} --no-restart
```

Files automatically updated by set-role (other than `--status-only`):
- `status.json` — merges role and model / context_threshold / conversation_history_threshold etc. derived from the role `defaults.json`
- `specialty_prompt.md` — overwritten from the role template
- `permissions.json` — overwritten from the role template (the conventional `permissions.md` may be migrated to JSON at load time)

Valid roles: `engineer`, `researcher`, `manager`, `writer`, `ops`, `general`

### Hot Reload

```bash
animaworks anima reload {名前}           # status.json からモデル設定を再読み込み（プロセス再起動なし）
animaworks anima reload --all            # 全Animaをリロード
```

---

## Model Information (models subcommand)

```bash
animaworks models                        # サブコマンド一覧（help）
animaworks models list                   # 組み込みカタログ（KNOWN_MODELS）: 名前・モード・コンテキスト・注記
animaworks models list --mode S          # モードでフィルタ（CLI の choices は S / A / B / C のみ・大文字小文字可）
animaworks models list --json            # JSON出力
animaworks models info {モデル名}        # 任意モデル名の解決結果（実行モード・コンテキスト窓・閾値・ソース）
animaworks models show                   # ~/.animaworks/models.json のパターン一覧
animaworks models show --json            # models.json を生JSON出力
```

Models not in `models list` (e.g., `cursor/*`, `gemini/*`) can be checked with `models info <名前>` and `models.json`.

Details → `reference/operations/model-guide.md`

---

## Chat and Messaging

```bash
# Animaとチャット（人間→Anima）
animaworks chat {名前} "メッセージ"
animaworks chat {名前} "メッセージ" --from {送信者名}
animaworks chat {名前} "メッセージ" --local  # deprecated（ヘルプ参照）

# Anima間メッセージ送信（Animaコンテキストでは送信者は自分のAnima名、intentは必須）
animaworks send {自分} {受信者} "メッセージ" --intent report   # report または question。委譲には delegate_task を使う
animaworks send {自分} {受信者} "メッセージ" --intent question
animaworks send {自分} {受信者} "メッセージ" --intent report --reply-to {メッセージID}
animaworks send {自分} {受信者} "メッセージ" --intent report --thread-id {スレッドID}

# ハートビート手動起動
animaworks heartbeat {名前}
animaworks heartbeat {名前} --local      # deprecated（ヘルプ参照）
```

---

## Board (Shared Channel)

```bash
animaworks board read {チャネル名}                      # 前回の続きから読む（デフォルト --limit 20）
animaworks board read {チャネル名} --latest             # 最新だけを見る（読んだ位置は進めない）
animaworks board read {チャネル名} --limit 50           # 最大件数
animaworks board read {チャネル名} --human-only         # 人間の投稿のみ
animaworks board post {送信者} {チャネル名} "テキスト"  # チャネルへ投稿
animaworks board dm-history {自分} {相手}               # DM履歴（デフォルト --limit 20、direction both）
animaworks board dm-history {自分} {相手} --limit 50    # 件数指定
animaworks board dm-history {自分} {相手} --direction sent --hours 24 --keyword deploy
```

---

## Configuration Management (config subcommand)

```bash
animaworks config                        # ヘルプ表示（子コマンドまたは -i が無い場合）
animaworks config -i                     # 対話式ウィザード（credential・Anima設定）
animaworks config list                   # 全設定値の一覧表示
animaworks config list --section system  # セクションでフィルタ（例: system, credentials）
animaworks config list --show-secrets    # API keyを表示
animaworks config get {キー}             # 特定の設定値取得（ドット記法: system.mode）
animaworks config get {キー} --show-secrets
animaworks config set {キー} {値}        # 設定値を変更
```

**Note**: `status.json` is the SSoT for Anima models, credentials, etc. Direct configuration of `animas.{名前}.model` etc. is deprecated. Use `animaworks anima set-model` instead.

---

## Profiles (profile subcommand, multi-tenant)

Manages multiple AnimaWorks instances (separate data directories).

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

## Log Viewing (logs)

```bash
animaworks logs {名前}                   # 特定Animaのログ表示
animaworks logs --all                    # サーバー＋全Animaのログ表示
animaworks logs {名前} --lines 100       # 表示行数指定（デフォルト: 50）
animaworks logs {名前} --date 20260301   # 特定日のログ表示
```

---

## Cost Check (cost)

```bash
animaworks cost                          # 全Animaのトークン使用量・コスト
animaworks cost {名前}                   # 特定Animaのコスト
animaworks cost --today                  # 本日のみ
animaworks cost --days 7                 # 直近7日（デフォルト: 30日）
animaworks cost --json                   # JSON出力
```

---

## Task Management (task subcommand)

**Prerequisite**: Before running, set `ANIMAWORKS_ANIMA_DIR` to the target Anima's directory (e.g., `~/.animaworks/animas/{名前}`). An error will occur if it is not set (intended uses include `animaworks-tool task` inside an Anima child process and `animaworks task` with the variable set in the shell).

```bash
animaworks task list                     # タスク一覧（JSON）
animaworks task list --status pending    # ステータスでフィルタ（pending/in_progress/done/cancelled）
animaworks task add --assignee {名前} --instruction "タスク内容"   # 既定 --source anima。実行入力を登録しTaskExecが実行
animaworks task add --assignee {名前} --instruction "内容" --source human
animaworks task add ... --relay-chain alice,bob   # カンマ区切りリレー鎖（任意）
animaworks task add ... --summary "1行要約"       # 省略時は instruction の先頭100文字
# task add は実行可能タスク。追跡専用の backlog_task ツールとは用途が異なる
animaworks task update --task-id {ID} --status done
animaworks task update --task-id {ID} --status done --summary "完了サマリー"
```

---

## RAG Index Management

Dependency: If the RAG stack is not installed, `pip install 'animaworks[rag]'` is required (the CLI displays an error and exits).

```bash
animaworks index                         # 全Anima: knowledge/episodes/procedures/skills + state/conversation.json の要約
                                         # + common_knowledge/common_skills（有効Animaごと）+ shared/users
animaworks index --anima {名前}          # 当該Animaのみ（共有コレクション・shared/users はスキップ）
animaworks index --anima {名前} --shared # 当該Animaのメモリに加え common_knowledge/common_skills も更新
animaworks index --full                  # コレクション削除からの全再構築（埋め込みモデル変更・L2→cosine 移行時は必須）
animaworks index --dry-run               # 確認のみ
```

- **conversation_summary**: Includes the `compressed_summary` of `state/conversation.json` in the indexing target (if the relevant file exists).
- **Existing collections with L2 distance**: When directly accessing local Chroma while the server is not running, a cosine migration warning may appear without `--full` (follow the message and execute `--full`).
- **While the server is running**: When `server.pid` is detected, the CLI sets `ANIMAWORKS_VECTOR_URL` / `ANIMAWORKS_EMBED_URL` and performs indexing and embedding via HTTP, avoiding concurrent access conflicts with Chroma.
- **Embedding model**: If the record in `index_meta.json` differs from the configuration, the command exits with an error unless `--full` is used.

---

## Runtime Migration (migrate)

```bash
animaworks migrate                       # 未適用のマイグレーションを実行
animaworks migrate --dry-run             # 変更プレビュー
animaworks migrate --verbose             # ファイル単位の詳細
animaworks migrate --list                # ステップ一覧と適用済みフラグ
animaworks migrate --force               # 状態に関わらず再適用
```

A warning appears if a server is currently running. Fails if `~/.animaworks/config.json` is absent.

---

## Asset Operations

### Asset Optimization

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

### Asset Regeneration (Vibe Transfer)

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

## External Tool Execution (animaworks-tool)

Command used when Anima uses external services (Slack, Gmail, GitHub, etc.).

When the first argument is a registered tool name or `submit`, `animaworks` is internally replaced with `animaworks-tool` (e.g., `animaworks web_search query "..."`).

```bash
# ヘルプ表示
animaworks-tool {ツール名} --help

# 実行
animaworks-tool {ツール名} {サブコマンド} [引数...]

# バックグラウンド実行（長時間ツール向け）
animaworks-tool submit {ツール名} {サブコマンド} [引数...]
```

### Concrete Examples

```bash
animaworks-tool web_search query "AnimaWorks framework"
animaworks-tool slack send "#general" "おはようございます"   # gated: 明示的な許可が必要
animaworks-tool github issues --repo owner/repo
animaworks-tool submit image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

For submit details → `common_knowledge/operations/background-tasks.md`

**Directory Organization**

- **`state/background_tasks/`** — Execution result JSON saved by BackgroundTaskManager (`running` / `completed` / `failed`). The submitted content and attempts of `animaworks-tool submit` are stored in TaskStore, and `internal list-background-tasks` / `check-background-task` read this result JSON.
- **`state/background_notifications/`** — Used by heartbeat to pick up tool completion notifications, etc. (there are paths written by MCP, schedulers, etc.). Separate from the CLI's `list-background-tasks`.

### Subcommands for Child Processes (internal / vault / supervisor)

All require **`ANIMAWORKS_ANIMA_DIR`** (same as `task`). Subcommands with the same names are also registered on the top-level `animaworks`.

**internal**

```bash
animaworks-tool internal archive-memory {相対パス} [--reason "理由"]  # knowledge/ procedures/ state/overflow_inbox/ のみ。episodes/ は対象外
# --reason の既定値は "archived via CLI"。パス権限・保護ファイル検査と索引更新は archive_memory_file ツールが行う
animaworks-tool internal check-permissions {ツール名} [アクション]
# ※ permissions.json（external_tools の allow/deny と gated action）で判定。判定できないときは不許可として返る
animaworks-tool internal create-skill {名前} [--content ...] [--location personal|common]  # 省略時は標準入力
animaworks-tool internal manage-channel create|archive|add_member|remove_member|info {チャネル名} [--member 名前] [--description 説明]
animaworks-tool internal list-background-tasks [--status running|completed|failed|pending]
animaworks-tool internal check-background-task {task_id}
```

**vault** (Anima-namespaced KV)

There are two storage locations: “your own Anima namespace” and the “`shared` section.” Anima can read its own namespace and `shared`, but can only write to and delete from its own namespace. Writing to and deleting from `shared` is restricted to operators who have not set `ANIMAWORKS_ANIMA_DIR`. If unspecified, `get` searches in the order of its own namespace → `shared`, while `list` displays only these two.

```bash
animaworks-tool vault get {キー}              # 自分の名前空間 → shared の順に探す
animaworks-tool vault get {キー} --shared     # shared のみ
animaworks-tool vault store {キー} {値}       # 自分の名前空間へ
printf '%s' "{値}" | animaworks-tool vault store {キー} --shared   # オペレーター専用（stdin 経由のみ）
animaworks-tool vault list                    # {"sections":{"自分の名前空間":[...], "shared":[...]}}
animaworks-tool vault list --shared           # shared のみ
animaworks-tool vault delete {キー}           # 自分の名前空間から削除
animaworks-tool vault delete {キー} --shared  # オペレーター専用
```

Anima cannot perform writes or deletions via `--shared`. Values for which “only one valid copy can exist at a time,” such as single-use tokens, should be stored in one location and not distributed.

**supervisor** (resolve descendants through the supervisor relationships in status.json, starting from the Anima name in `ANIMAWORKS_ANIMA_DIR`)

```bash
animaworks-tool supervisor org-dashboard
animaworks-tool supervisor ping [--name {部下名}]    # 省略時は全配下
animaworks-tool supervisor read-state {部下名}      # 位置引数（祖先関係チェックあり）
animaworks-tool supervisor task-tracker [--status delegated]   # 既定 status=delegated（文字列でフィルタ）
```

### Checking Background Tasks

- During conversation: MCP tools `list_background_tasks` / `check_background_task`
- CLI: `animaworks-tool internal list-background-tasks` / `check-background-task {task_id}` (under `ANIMAWORKS_ANIMA_DIR` configuration, `state/background_tasks/`)

---

## Initialization and Migration

```bash
animaworks init                          # ランタイムディレクトリ初期化（~/.animaworks/）
animaworks init --force                  # 既存にテンプレート差分をマージ
animaworks init --skip-anima             # インフラのみ初期化（Anima作成スキップ）
animaworks init --template {名前}       # テンプレートからAnimaを非対話で作成
animaworks init --from-md {PATH}         # MDファイルからAnimaを非対話で作成
animaworks init --blank {名前}           # ブランクAnimaを非対話で作成
animaworks init --from-md {PATH} --name {名前}  # 作成時の名前を上書き
旧形式の cron.md はサーバ起動時と `animaworks migrate` で自動変換される。
```

For schema migration, refer to `animaworks migrate` (above).

---

## Global Options

```bash
animaworks --gateway-url http://host:port {コマンド}   # API ベース URL（既定: http://localhost:18500）
animaworks --data-dir /path/to/data {コマンド}         # ランタイムディレクトリ（~/.animaworks 相当）
```

`--data-dir` is applied to `ANIMAWORKS_DATA_DIR` before subcommand execution.
