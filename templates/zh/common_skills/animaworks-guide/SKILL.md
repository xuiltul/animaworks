---
name: animaworks-guide
description: >-
  animaworks 命令的完整参考。汇总服务器操作、Anima 管理、模型、任务、配置、RAG、资产、外部工具的 CLI 格式。
  Use when: 需要确认子命令格式、启动停止服务器、创建 Anima、更改模型、添加任务、日志、配置、索引操作时。
---


# AnimaWorks CLI 完整参考

AnimaWorks 的所有操作均通过 `animaworks` 命令进行。
本技能汇总了所有子命令的格式、参数和具体示例的参考。

运用思路和规则请参阅 `common_knowledge/` 及 `reference/`：
- 消息传递规则 → `communication/messaging-guide.md`
- 任务管理 → `operations/task-management.md`
- 工具体系 → `operations/tool-usage-overview.md`
- 组织结构 → `reference/organization/structure.md`
- 模型选择・配置 → `reference/operations/model-guide.md`

---

## 已弃用・兼容别名

```bash
# --local（deprecated: ProcessSupervisor をバイパスする直接実行。HTTP API 経由が推奨）
animaworks chat {名前} "..." --local
animaworks heartbeat {名前} --local
```

推荐：使用 `animaworks start`（或 `serve`）启动服务器，不加 `--local` 而使用 `chat` / `heartbeat`。

`gateway` / `worker` 子命令是隐藏的兼容用途（分布式架构废止后不再使用）。

---

## 服务器操作（基本上不应使用）

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

## Anima 管理（anima 子命令）

### 列表・状态・详细信息

```bash
animaworks anima list                    # 全Anima一覧（名前・有効/無効・モデル・supervisor）
animaworks anima list --local            # API不使用・ファイルシステム直接スキャン
animaworks anima status                  # 全Animaのプロセス状態（State・モデル・PID・Uptime）
animaworks anima status {名前}           # 特定Animaのプロセス状態
animaworks anima info {名前}             # 設定詳細（モデル・ロール・credential・voice等）
animaworks anima info {名前} --json      # JSON出力
animaworks anima permissions {名前}      # ツール許可を表示（load_permissions: JSON 優先、レガシーは permissions.md）
```

`anima info` 的输出项：
- Anima 名、Enabled、Role、Model、Execution Mode（内置标签为 S/C/D/G/X/A）
- Credential、Fallback Model、Context Threshold、Max Tokens、Thinking / Thinking Effort、Supervisor、Mode S Auth
- Voice 配置（tts_provider、voice_id、speed、pitch 等，列出 status.json 的 voice 字典）

### 创建

```bash
# キャラクターシート（MD）から作成（推奨）
animaworks anima create --from-md {ファイル} [--role {role}] [--name {名前}]

# テンプレートから作成
animaworks anima create --template {テンプレート名} [--name {名前}]

# ブランク作成
animaworks anima create --name {名前}
```

### 启用・禁用・删除

```bash
animaworks anima enable {名前}           # 有効化（休養から復帰）
animaworks anima disable {名前}          # 無効化（休養）
animaworks anima delete {名前}           # 削除（ZIPアーカイブ後）
animaworks anima delete {名前} --no-archive  # アーカイブなしで削除
animaworks anima delete {名前} --force   # 確認なしで削除
animaworks anima restart {名前}          # プロセス再起動
```

### 活动审计（audit）

读取目标 Anima 的 `activity_log`（帮助文本写的是 subordinate，但指定名称后可用于任意 Anima）。

```bash
animaworks anima audit {名前}            # 活動監査（デフォルト: report モード・直近1日）
animaworks anima audit {名前} --days 7   # 集計日数（最大30）
animaworks anima audit --all             # 全Animaを対象（タイムラインをマージ）
animaworks anima audit {名前} --since 09:00   # 当日 JST の時刻以降（指定時は --days より優先）
animaworks anima audit {名前} --mode summary  # 統計サマリー（省略時は report＝時系列）
```

### 模型更改

```bash
animaworks anima set-model {名前} {モデル名}
animaworks anima set-model {名前} {モデル名} --credential {credential名}
animaworks anima set-model --all {モデル名}   # 全Anima一括変更
```

更改后若服务器正在运行，需要 `anima restart {名前}`。

### 后台模型（用于 Heartbeat/Cron）

```bash
animaworks anima set-background-model {名前} {モデル名}   # Heartbeat・Cron用モデルを設定
animaworks anima set-background-model {名前} {モデル名} --credential {credential名}
animaworks anima set-background-model {名前} --clear    # オーバーライド解除（メインモデルにフォールバック）
animaworks anima set-background-model --all --clear     # 有効な全Animaの background を一括クリア
animaworks anima set-background-model --all {モデル名}   # 有効な全Anima一括（モデルは位置引数でも可: `{モデル} --all`）
```

### 重命名

```bash
animaworks anima rename {旧名前} {新名前}
animaworks anima rename {旧名前} {新名前} --force   # 確認なしで実行
```

### 角色更改

```bash
# ロール変更（テンプレート再適用 + 自動restart）
animaworks anima set-role {名前} {role}

# status.jsonのroleフィールドのみ変更（テンプレートは触らない）
animaworks anima set-role {名前} {role} --status-only

# ファイル更新のみ・再起動しない
animaworks anima set-role {名前} {role} --no-restart
```

set-role 自动更新的文件（`--status-only` 以外）：
- `status.json` — 合并 role 及角色 `defaults.json` 来源的 model / context_threshold / conversation_history_threshold 等
- `specialty_prompt.md` — 从角色模板覆盖
- `permissions.json` — 从角色模板覆盖（传统的 `permissions.md` 可能在读取时迁移为 JSON）

有效角色：`engineer`, `researcher`, `manager`, `writer`, `ops`, `general`

### 热重载

```bash
animaworks anima reload {名前}           # status.json からモデル設定を再読み込み（プロセス再起動なし）
animaworks anima reload --all            # 全Animaをリロード
```

---

## 模型信息（models 子命令）

```bash
animaworks models                        # サブコマンド一覧（help）
animaworks models list                   # 組み込みカタログ（KNOWN_MODELS）: 名前・モード・コンテキスト・注記
animaworks models list --mode S          # モードでフィルタ（CLI の choices は S / A / B / C のみ・大文字小文字可）
animaworks models list --json            # JSON出力
animaworks models info {モデル名}        # 任意モデル名の解決結果（実行モード・コンテキスト窓・閾値・ソース）
animaworks models show                   # ~/.animaworks/models.json のパターン一覧
animaworks models show --json            # models.json を生JSON出力
```

`models list` 中不存在的模型（例：`cursor/*`, `gemini/*`）请通过 `models info <名前>` 和 `models.json` 确认。

详情 → `reference/operations/model-guide.md`

---

## 聊天・消息传递

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

## Board（共享频道）

```bash
animaworks board read {チャネル名}                      # チャネル読み取り（デフォルト --limit 20）
animaworks board read {チャネル名} --limit 50           # 最大件数
animaworks board read {チャネル名} --human-only         # 人間の投稿のみ
animaworks board post {送信者} {チャネル名} "テキスト"  # チャネルへ投稿
animaworks board dm-history {自分} {相手}               # DM履歴（デフォルト --limit 20、direction both）
animaworks board dm-history {自分} {相手} --limit 50    # 件数指定
animaworks board dm-history {自分} {相手} --direction sent --hours 24 --keyword deploy
```

---

## 配置管理（config 子命令）

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

**注意**：Anima 的模型・credential 等以 `status.json` 为 SSoT。直接配置 `animas.{名前}.model` 等不推荐。请使用 `animaworks anima set-model`。

---

## 配置文件（profile 子命令・多租户）

管理多个 AnimaWorks 实例（不同数据目录）。

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

## 日志查看（logs）

```bash
animaworks logs {名前}                   # 特定Animaのログ表示
animaworks logs --all                    # サーバー＋全Animaのログ表示
animaworks logs {名前} --lines 100       # 表示行数指定（デフォルト: 50）
animaworks logs {名前} --date 20260301   # 特定日のログ表示
```

---

## 成本确认（cost）

```bash
animaworks cost                          # 全Animaのトークン使用量・コスト
animaworks cost {名前}                   # 特定Animaのコスト
animaworks cost --today                  # 本日のみ
animaworks cost --days 7                 # 直近7日（デフォルト: 30日）
animaworks cost --json                   # JSON出力
```

---

## 任务管理（task 子命令）

**前提**：执行前需将 `ANIMAWORKS_ANIMA_DIR` 设置到目标 Anima 的目录（例：`~/.animaworks/animas/{名前}`）。未设置会报错（预期用途为 Anima 子进程内的 `animaworks-tool task`，或在 shell 中附加变量的 `animaworks task`）。

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

## RAG 索引管理

依赖：RAG 栈未安装时需要 `pip install 'animaworks[rag]'`（CLI 会显示错误并退出）。

```bash
animaworks index                         # 全Anima: knowledge/episodes/procedures/skills + state/conversation.json の要約
                                         # + common_knowledge/common_skills（有効Animaごと）+ shared/users
animaworks index --anima {名前}          # 当該Animaのみ（共有コレクション・shared/users はスキップ）
animaworks index --anima {名前} --shared # 当該Animaのメモリに加え common_knowledge/common_skills も更新
animaworks index --full                  # コレクション削除からの全再構築（埋め込みモデル変更・L2→cosine 移行時は必須）
animaworks index --dry-run               # 確認のみ
```

- **conversation_summary**：将 `state/conversation.json` 的 `compressed_summary` 纳入索引对象（若存在相应文件）。
- **L2 距离的现有集合**：服务器未运行而直接操作本地 Chroma 时，若无 `--full` 可能出现 cosine 迁移警告（按消息提示执行 `--full`）。
- **服务器运行中**：检测到 `server.pid` 时，CLI 会设置 `ANIMAWORKS_VECTOR_URL` / `ANIMAWORKS_EMBED_URL` 并通过 HTTP 进行索引・嵌入，以避免 Chroma 并发访问冲突。
- **嵌入模型**：若 `index_meta.json` 的记录与配置不一致，无 `--full` 时会报错退出。

---

## 运行时迁移（migrate）

```bash
animaworks migrate                       # 未適用のマイグレーションを実行
animaworks migrate --dry-run             # 変更プレビュー
animaworks migrate --verbose             # ファイル単位の詳細
animaworks migrate --list                # ステップ一覧と適用済みフラグ
animaworks migrate --force               # 状態に関わらず再適用
```

若有正在运行的服务器会发出警告。无 `~/.animaworks/config.json` 时会失败。

---

## 资产操作

### 资产优化

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

### 资产重新生成（Vibe Transfer）

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

## 外部工具执行（animaworks-tool）

Anima 使用外部服务（Slack、Gmail、GitHub 等）时的命令。

当第 1 个参数为已注册工具名或 `submit` 时，`animaworks` 会在内部转为 `animaworks-tool`（例：`animaworks web_search query "..."`）。

```bash
# ヘルプ表示
animaworks-tool {ツール名} --help

# 実行
animaworks-tool {ツール名} {サブコマンド} [引数...]

# バックグラウンド実行（長時間ツール向け）
animaworks-tool submit {ツール名} {サブコマンド} [引数...]
```

### 具体示例

```bash
animaworks-tool web_search query "AnimaWorks framework"
animaworks-tool slack send "#general" "おはようございます"   # gated: 明示的な許可が必要
animaworks-tool github issues --repo owner/repo
animaworks-tool submit image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

submit 的详情 → `common_knowledge/operations/background-tasks.md`

**目录整理**

- **`state/background_tasks/`** — BackgroundTaskManager 保存的执行结果 JSON（`running` / `completed` / `failed`）。`animaworks-tool submit` 的投入内容和尝试保存到 TaskStore，`internal list-background-tasks` / `check-background-task` 读取此结果 JSON。
- **`state/background_notifications/`** — 供 heartbeat 吸收工具完成通知等用途（存在 MCP・调度器等写入的路径）。与 CLI 的 `list-background-tasks` 不同。

### 面向子进程的子命令（internal / vault / supervisor）

均 **必须 `ANIMAWORKS_ANIMA_DIR`**（与 `task` 相同）。顶层 `animaworks` 也注册了同名子命令。

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

**vault**（带 Anima 命名空间的 KV）

保存位置有两个：「自己的 Anima 命名空间」和「`shared` 节」。Anima 可以读取自己的命名空间和 `shared`，但只能写入・删除自己的命名空间。对 `shared` 的写入・删除仅供未设置 `ANIMAWORKS_ANIMA_DIR` 的操作员使用。`get` 未指定时按自己的命名空间 → `shared` 的顺序搜索，`list` 仅显示这两者。

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

Anima 无法通过 `--shared` 执行写入・删除。像单次使用令牌这样「有效副本同时只能存在一个」的值，应确定唯一保管位置，不要分散。

**supervisor**（以 `ANIMAWORKS_ANIMA_DIR` 的 Anima 名为起点，通过 status.json 的 supervisor 关系解析下属）

```bash
animaworks-tool supervisor org-dashboard
animaworks-tool supervisor ping [--name {部下名}]    # 省略時は全配下
animaworks-tool supervisor read-state {部下名}      # 位置引数（祖先関係チェックあり）
animaworks-tool supervisor task-tracker [--status delegated]   # 既定 status=delegated（文字列でフィルタ）
```

### 后台任务确认

- 对话中：MCP 工具 `list_background_tasks` / `check_background_task`
- CLI：`animaworks-tool internal list-background-tasks` / `check-background-task {task_id}`（在 `ANIMAWORKS_ANIMA_DIR` 配置下，`state/background_tasks/`）

---

## 初始化・迁移

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

模式迁移请参阅 `animaworks migrate`（上述）。

---

## 全局选项

```bash
animaworks --gateway-url http://host:port {コマンド}   # API ベース URL（既定: http://localhost:18500）
animaworks --data-dir /path/to/data {コマンド}         # ランタイムディレクトリ（~/.animaworks 相当）
```

`--data-dir` 会在子命令执行前反映到 `ANIMAWORKS_DATA_DIR`。
