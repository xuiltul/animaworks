# AnimaWorks — Digital Anima Framework

AIエージェントをツールではなく「自律的な人」として扱うフレームワーク。
各Animaは固有のアイデンティティ・記憶・判断基準を持ち、ハートビートやcronで自律行動する。

## 設計原則

- **カプセル化**: Anima内部の思考・記憶は外部から不可視。外部とはテキスト会話のみ
- **RAG記憶**: PrimingレイヤーがRAGで自動検索した関連記憶をシステムプロンプトに注入する。加えてエージェントは自律的に記憶を検索可能。サイズ無制限
- **自律性**: ハートビート(定期巡回)とcron(定時タスク)で人間の指示なしに行動
- **プロセス分離**: ProcessSupervisorが各AnimaをUnixソケット付き独立子プロセスとして起動・監視する
- **海馬モデル**: PrimingEngineがプロンプト構築における唯一のアクティビティリーダー。builder.pyはActivityLoggerを直接読まない

## ディレクトリ構成

```
core/                        # フレームワーク本体
├── anima.py, agent.py, lifecycle.py  # コアエンティティ・オーケストレーター
├── anima_factory.py, init.py         # 初期化・Anima生成
├── schemas.py, paths.py              # データモデル・パス定数
├── messenger.py, logging_config.py   # 通信・ログ
├── background.py                     # バックグラウンドタスク
├── asset_reconciler.py               # アセット自動生成
├── org_sync.py                       # 組織同期
├── outbound.py                       # 統一アウトバウンドルーティング（Slack/Chatwork/内部自動判定）
├── schedule_parser.py                # cron.md/heartbeat.mdパーサー
├── voice/                            # 音声チャットサブシステム
│   ├── stt.py, tts_base.py, tts_factory.py
│   ├── tts_voicevox.py, tts_elevenlabs.py, tts_sbv2.py
│   ├── sentence_splitter.py
│   └── session.py
├── memory/                           # 記憶サブシステム
│   ├── manager.py, conversation.py, shortterm.py
│   ├── priming.py, consolidation.py, forgetting.py
│   ├── activity.py, streaming_journal.py
│   └── rag/                          # RAGエンジン（ChromaDB + sentence-transformers）
│       ├── indexer.py, retriever.py, graph.py
│       ├── store.py, singleton.py, watcher.py
├── supervisor/                       # プロセス監視
│   ├── manager.py, ipc.py, runner.py, process_handle.py
├── notification/                     # 人間通知
│   ├── notifier.py
│   └── channels/                     # 通知チャネル（slack, chatwork, line, telegram, ntfy）
├── auth/                             # 認証
├── tooling/                          # ツール基盤
│   ├── handler.py                    # ToolHandler（権限チェック・ディスパッチ）
│   ├── schemas.py                    # ツールスキーマ定義
│   ├── guide.py                      # 動的ツールガイド生成
│   └── dispatch.py                   # ExternalToolDispatcher（外部ツール振り分け）
├── config/                           # 設定管理
│   ├── models.py                     # Pydanticモデル・load/save
│   ├── cli.py, migrate.py
├── prompt/                           # プロンプト・コンテキスト管理
│   ├── builder.py                    # システムプロンプト構築（6グループ構造）
│   └── context.py                    # コンテキストウィンドウ追跡
├── execution/                        # 実行エンジン
│   ├── base.py                       # BaseExecutor・ExecutionResult
│   ├── agent_sdk.py                  # S: Claude Agent SDK
│   ├── codex_sdk.py                  # C: Codex CLI
│   ├── cursor_agent.py               # D: Cursor Agent CLI
│   ├── gemini_cli.py                 # G: Gemini CLI
│   ├── anthropic_fallback.py         # A（内部分岐）: Anthropic SDK直接
│   ├── litellm_loop.py              # A: LiteLLM + tool_useループ
│   ├── assisted.py                   # B: 1ショット（記憶I/O代行）
│   └── _session.py                   # セッション継続・チェイニング
└── tools/                            # 外部ツール実装
    ├── web_search.py, x_search.py
    ├── slack.py, chatwork.py, gmail.py
    ├── github.py, aws_collector.py
    ├── image_gen.py
    ├── transcribe.py, local_llm.py
    └── _base.py, _cache.py, _retry.py
cli/                                  # CLIパッケージ
├── parser.py                         # argparse定義 + cli_main()
└── commands/                         # サブコマンド実装
server/                               # FastAPI単体サーバー + Web UI
├── app.py, websocket.py
├── routes/                           # APIルート（ドメイン別分割）
└── static/                           # フロントエンド
    ├── index.html
    ├── modules/                      # JS モジュール
    ├── styles/                       # CSS
    └── workspace/                    # インタラクティブWorkspace（3Dオフィス・会話画面）
templates/                            # 初期化テンプレート
├── ja/, en/                          # ロケール別テンプレート
│   ├── prompts/                      # プロンプトテンプレート
│   ├── anima_templates/              # Animaスケルトン
│   ├── roles/                        # ロールテンプレート
│   ├── common_knowledge/             # 共有知識テンプレート
│   └── common_skills/                # 共通スキルテンプレート
├── _shared/                          # ロケール共通（company等）
main.py                               # CLIエントリポイント
```

### Chat UI レイアウト注意点（2026-02-27）

`/#chat` の入力欄下部に意図しない隙間が出る場合は、以下を優先確認する。

- `server/static/styles/responsive.css` の `.chat-page-layout` は `height: 100%` を基準にする（`calc(100svh - 140px)` だと親余白と二重計算になりやすい）。
- `server/static/styles/layout.css` で `.main-content:has(.chat-page-layout)` を使い、チャット時のみ下余白を制御する（現在は `padding-bottom: 10px`）。
- `server/static/pages/chat.js` の `#chatPageForm` にインライン `padding` を持たせない（CSS一元管理）。
- 入力欄の空き領域クリックでフォーカスできるよう、`.chat-input-wrap` に `pointerdown/click -> #chatPageInput.focus()`（ボタン類は除外）を入れる。
- キャッシュ影響を避けるため、`server/static/index.html` で `chat.css` / `layout.css` / `responsive.css` / `modules/app.js` のクエリバージョンを更新して確認する。

### 非活性ディレクトリ（検索不要）

- `archive/` — 旧設計・非推奨コード（gateway, worker, broker 含む）。参照のみ
- `site-docs/` — 生成済みドキュメント
- `.venv/`, `__pycache__/`, `animaworks.egg-info/` — ビルド成果物

## ランタイムデータ（リポジトリ外）

`~/.animaworks/` に配置。`animaworks init` で生成:

```
~/.animaworks/
├── config.json               # 統合設定
├── auth.json                 # UI認証設定
├── models.json               # モデル別実行モード・コンテキストウィンドウ定義
├── server.pid                # サーバーPIDファイル（起動中のみ存在）
├── index_meta.json           # RAGインデックスメタ
├── company/vision.md         # 組織ビジョン
├── animas/{name}/
│   ├── identity.md           # 人格（不変ベースライン）
│   ├── injection.md          # 役割・行動指針（可変）
│   ├── specialty_prompt.md   # ロール別専門プロンプト
│   ├── bootstrap.md          # 初回起動指示
│   ├── permissions.md        # ツール・コマンド許可
│   ├── heartbeat.md          # 定期巡回チェックリスト
│   ├── cron.md               # 定時タスク（YAML）
│   ├── status.json           # SSoT: 有効/無効、ロール、モデル設定
│   ├── character_sheet.md    # 作成時のソースMD
│   ├── state/                # 作業記憶
│   │   ├── current_state.md  # 現在の状態（1つ）
│   │   ├── pending.md        # バックログ（自由形式）
│   │   ├── pending/          # Heartbeat書き出しLLMタスク（JSON）
│   │   └── task_queue.jsonl  # 永続タスクキュー（append-only）
│   ├── episodes/             # エピソード記憶（日次ログ）
│   ├── knowledge/            # 意味記憶（学習済み知識）
│   ├── procedures/           # 手続き記憶（手順書）
│   ├── skills/               # 個人スキル
│   ├── shortterm/            # 短期記憶（chat/heartbeat分離）
│   ├── activity_log/         # 統一アクティビティログ（{date}.jsonl）
│   ├── assets/               # キャラクター画像・3Dモデル
│   └── transcripts/          # 会話トランスクリプト
├── common_knowledge/         # 全Anima共有知識（テンプレートから展開）
├── common_skills/            # 共通スキル（全Anima参照）
├── prompts/                  # ランタイムプロンプトテンプレート
├── shared/
│   ├── users/                # Anima横断ユーザープロファイル
│   ├── channels/             # 共有チャネル（general.jsonl等）
│   ├── common_knowledge/     # Anima組織固有の共有知識（手動管理）
│   └── dm_logs/              # DM履歴（非推奨→activity_log主、7日ローテーション）
├── cache/, models/, run/, vectordb/
```

## 技術スタック

- Python 3.12+ / FastAPI / Uvicorn / APScheduler / Pydantic 2.0+
- Claude Agent SDK + Anthropic API（Claude Sonnet 4 / Opus 4.6）
- Codex CLI（OpenAI Codex モデル）
- LiteLLM（マルチプロバイダ統一）
- ChromaDB + sentence-transformers（RAG / ベクトル検索）
- NetworkX（グラフベース拡散活性化）
- ツール連携: Slack, Chatwork, Gmail, GitHub, AWS, Web検索, X検索, Whisper
- ローカルLLM推論: vLLM / Ollama（OpenAI互換API）
- 画像生成: NovelAI API, fal.ai (Flux), Meshy (3D)
- 人間通知: call_human（Slack, Chatwork, LINE, Telegram, ntfy）

## コーディング規約

- `from __future__ import annotations` 全ファイル先頭
- 型ヒント必須（`str | None` 形式）、Pathlib統一
- Google-style docstring、`logger = logging.getLogger(__name__)`
- Pydantic ModelまたはDataclassでデータ定義、ABC+abstractmethodでプラグイン
- `async/await` + `asyncio.Lock()` で並行制御
- セクション区切り: `# ── SectionName ──────────`
- private: `_prefix`、定数: `UPPER_SNAKE_CASE`
- コミット: `feat:` / `fix:` / `refactor:` のセマンティック形式
- **プロンプト・UIメッセージのハードコード禁止**: ユーザーやAnimaに表示する文字列は `core/i18n.py` の `t()` で解決する。新しい文字列は `_STRINGS` に `ja` / `en` の両方を登録し、コード側では `t("module.key_name", param=value)` で参照する

### ruff（リンター / フォーマッター）

`pyproject.toml` に `[tool.ruff]` で設定。CI で `ruff check` + `ruff format --check` を実行。

```bash
ruff check core/ cli/ server/           # lint チェック
ruff check --fix core/ cli/ server/     # 自動修正
ruff format core/ cli/ server/          # フォーマット適用
```

#### 選択ルールと意図的 ignore

| ルール | 説明 | 理由 |
|--------|------|------|
| E402 | module-import-not-at-top | ライセンスヘッダー + docstring の後に import するスタイルが標準 |
| E501 | line-too-long | formatter に任せる |
| SIM105 | suppressible-exception | try/except/pass を意図的に使うケースあり |
| SIM108 | if-else-block-instead-of-if-exp | 可読性優先 |
| SIM102 | collapsible-if | ネスト if は意図的 |
| UP047 | non-pep695-generic-function | TypeVar スタイルは段階的移行 |

#### per-file-ignores（facade モジュール）

`core/tools/image_gen.py`, `core/config/__init__.py`, `core/prompt/__init__.py` は re-export 専用のため F401（unused-import）を除外。これらの `__init__.py` や facade モジュールの import を ruff --fix で削除すると、外部から `from core.tools.image_gen import PipelineResult` のような使用が壊れる。

**重要**: facade / `__init__.py` の re-export を修正する際は、必ずテストで import の破損がないか確認すること。

### pytest-cov（カバレッジ計測）

`addopts` には `--cov` を**入れない**（ローカル開発が遅くなるため）。CI コマンドでのみ指定:

```bash
pytest tests/unit/ --cov=core --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

カバレッジ閾値: **70%**（2026-03-05 時点の実測値 75%）

### pytest 設定の一元管理

pytest の設定は `pyproject.toml` の `[tool.pytest.ini_options]` に一元管理。`pytest.ini` は使用しない（廃止済み）。

## 実行モードと並行制御

### 6種類の実行エンジン

モデル名からワイルドカードパターンマッチ（`fnmatch`）で自動判定:

| モード | 名称 | 対象モデル | 概要 |
|--------|------|-----------|------|
| **S** | SDK | `claude-*` | Claude Agent SDK（Claude Code子プロセス）。最もリッチ |
| **C** | Codex | `codex/*` | Codex CLI経由。OpenAI Codexモデル用 |
| **D** | Cursor Agent | `cursor/*` | Cursor Agent CLI子プロセス。MCP統合エージェントループ |
| **G** | Gemini CLI | `gemini/*` | Gemini CLI子プロセス。stream-jsonパース・ツールループ |
| **A** | Autonomous | `openai/*`, `google/*`, `vertex_ai/*`, `azure/*`, `mistral/*`, `xai/*` 等 | LiteLLM + tool_useループ |
| **B** | Basic | `ollama/gemma3*`, `ollama/deepseek-r1*` 等 | 1ショット。フレームワークが記憶I/O代行。セッションチェイニング非対応 |

- Agent SDK未インストール時はAnthropic SDK直接使用に自動フォールバック（Aモード内部分岐）
- Cursor Agent CLI / Gemini CLIは未インストール時にインポートエラーをスキップ（オプショナル依存）
- vLLMローカルモデル: `openai/glm-4.7-flash` 等（credential の `base_url` で接続先指定）
- Ollama tool_use対応モデル（`ollama/qwen3:14b`, `ollama/glm-4.7*`）→ Mode A
- `ollama/*`（他パターン未マッチ）→ Mode B（安全側フォールバック）

### モード解決優先度（`resolve_execution_mode()`）

1. Per-anima `status.json` の `execution_mode` 明示指定
2. `~/.animaworks/models.json`（fnmatchワイルドカード対応）
3. `config.json` `model_modes`（非推奨フォールバック）
4. `DEFAULT_MODEL_MODE_PATTERNS`（コードデフォルト）
5. デフォルト `"B"`（安全側）

### models.json

`~/.animaworks/models.json` でモデルごとの実行モード・コンテキストウィンドウを定義。具体的なパターンが優先:

```json
{
  "claude-opus-4-6":    { "mode": "S", "context_window": 1000000 },
  "claude-sonnet-4-6":  { "mode": "S", "context_window": 1000000 },
  "claude-*":           { "mode": "S", "context_window": 200000 },
  "cursor/*":           { "mode": "D", "context_window": 1000000 },
  "gemini/*":           { "mode": "G", "context_window": 1000000 },
  "openai/gpt-4.1*":   { "mode": "A", "context_window": 1000000 },
  "ollama/gemma3*":     { "mode": "B", "context_window": 8192 }
}
```

### 3パス実行分離

Animaの処理は独立パスで実行され、各パスは独立したロックを持ち並行動作可能:

| パス | ロック | トリガー | 役割 |
|------|--------|---------|------|
| **Chat/Inbox** | `_conversation_lock` / `_inbox_lock` | 人間チャット / Anima DM | メッセージ応答。Inboxは即時・軽量な返信のみ |
| **Heartbeat** | `_background_lock` | 定期巡回（30分） | Observe → Plan → Reflect。**実行はしない** |
| **Cron** | `_background_lock` | cron.mdスケジュール | Heartbeat同等コンテキストで定時タスク実行 |
| **TaskExec** | `_background_lock` | `state/pending/` にタスク出現 | 委譲タスクの実行（最小コンテキスト） |

Heartbeatは状況確認・計画のみを行い、実行が必要なタスクは `state/pending/` にJSON形式で書き出す。TaskExecがこれを3秒ポーリングで検出し、独立LLMセッションで実行する。

### バックグラウンドモデル（コスト最適化）

Heartbeat / Inbox / Cron はメインモデルとは別の軽量モデルで実行可能:

| 区分 | 使用モデル | 対象トリガー |
|------|-----------|-------------|
| **foreground** | メインモデル（`model`） | `chat`（人間との対話）、`task:*`（TaskExec実作業） |
| **background** | `background_model`（未設定時はメインモデル） | `heartbeat`、`inbox:*`（Anima間DM）、`cron:*` |

解決順序: Per-anima `status.json` → `config.json` `heartbeat.default_model` → メインモデル

```bash
animaworks anima set-background-model {名前} claude-sonnet-4-6
animaworks anima set-background-model {名前} --clear   # メインモデルにフォールバック
```

### コンテキストティア

| ティア | セッション | コンテキスト量 |
|--------|-----------|--------------|
| Full | Chat / Inbox | 全セクション（specialty, emotion, a_reflection含む） |
| Background-Auto | Heartbeat / Cron | identity + memory + org（specialty, emotion, a_reflection省略） |
| Minimal | TaskExec | identity 3行 + タスク記述のみ |

### トリガーベースプロンプトフィルタリング

`build_system_prompt(trigger=...)` で実行パスに応じたセクション選択:
- `chat`: 全セクション
- `inbox`: 軽量（messaging + org重視）
- `heartbeat`: Observe/Plan/Reflect向け（状態 + 組織重視）
- `cron`: heartbeatと同一フィルタだがツールガイド・外部ツールガイド有効
- `task`: タスク実行向け（タスク固有コンテキスト + ツール重視）

### セッションファイル分離

Chat とバックグラウンド処理のセッションファイルは分離:
- `current_session_chat.json` / `current_session_heartbeat.json`
- `streaming_journal_chat.jsonl` / `streaming_journal_heartbeat.jsonl`
- `shortterm/chat/` / `shortterm/heartbeat/`

## ツール体系

### 3系統のツール（Mode S）

1. **Claude Code 組込みツール**: Read, Write, Edit, Grep, Glob, Bash, git 等
2. **MCP ツール (`mcp__aw__*`)**: AnimaWorks固有の内部機能
3. **Bash + animaworks-tool**: 外部ツール（`read_memory_file` でスキルパス（例: `common_skills/foo/SKILL.md`）からCLI手順を読み込み → `animaworks-tool <ツール> <サブコマンド>` で実行）

### ツールカテゴリ

| カテゴリ | 主要ツール | モード |
|---------|-----------|--------|
| 記憶 | `search_memory`, `read_memory_file`, `write_memory_file`, `archive_memory_file` | 全モード |
| 通信 | `send_message`, `post_channel`, `read_channel`, `read_dm_history` | 全モード |
| スキル | `read_memory_file`（システムプロンプトのスキルカタログに示されたパスで全文を読む）, `create_skill` | 全モード |
| タスク | `backlog_task`, `update_task`, `list_tasks`, `submit_tasks` | A/S |
| 完了検証 | `completion_gate` | 全モード（heartbeat/inbox除く） |
| 通知 | `call_human` | 全モード（トップレベル＆通知設定時） |
| 管理 | `create_anima` | A（newstaffスキル保持時） |
| 外部 | `slack`, `chatwork`, `gmail`, `github`, `web_search` 等 | permissions.md許可時 |

### 外部ツール実行

- **長時間ツール（⚠マーク付き）は必ず `submit` で非同期実行**: `animaworks-tool submit image_gen pipeline ...`
- 結果は `state/background_notifications/` に書かれ、次回heartbeatで確認される
- 短時間ツール（web_search, slack等）は直接実行

### ExternalToolDispatcherのフロー

```
LLMがツール呼び出し → ToolHandler._dispatch テーブル検索
→ 見つからない → ExternalToolDispatcher.dispatch()
→ TOOL_MODULES スキャン → mod.dispatch(name, args)
```

ToolHandlerは `anima_dir` を args に自動注入。Per-Animaクレデンシャル解決に使用（例: `CHATWORK_API_TOKEN_WRITE__<anima_name>`）。

### 新外部ツール追加手順

1. `core/tools/<tool_name>.py` 作成（`get_tool_schemas()`, `dispatch()`, `EXECUTION_PROFILE`）
2. `core/tools/__init__.py` の `TOOL_MODULES` に追加
3. Mode S: `core/mcp/server.py` の `_EXPOSED_TOOL_NAMES` に追加
4. `shared/credentials.json` にAPIキー追加
5. `permissions.md` で使用許可
6. テスト作成

## タスク管理

### タスクキュー（永続・追跡）

`state/task_queue.jsonl` に append-only JSONL 形式で記録:

```
backlog_task(source="human", original_instruction="...", assignee="自分", summary="...", deadline="1d")
update_task(task_id="abc123", status="done", summary="完了")
list_tasks(status="pending")
```

- `source: human` のタスクは最優先で処理する（MUST）
- Primingセクションに要約表示。30分更新なしで⚠️ STALE、期限超過で🔴 OVERDUE

### 並列タスク実行（submit_tasks）

`submit_tasks` で複数タスクをDAG投入。依存関係を解決し独立タスクを同時実行:

```
submit_tasks(batch_id="build", tasks=[
  {"task_id": "compile", "title": "コンパイル", "description": "...", "parallel": true},
  {"task_id": "lint", "title": "Lint", "description": "...", "parallel": true},
  {"task_id": "package", "title": "パッケージ", "description": "...", "depends_on": ["compile", "lint"]}
])
```

同時実行数: `config.json` `background_task.max_parallel_llm_tasks`（デフォルト3）

### タスク委譲（delegate_task）

部下を持つAnimaが `delegate_task` で部下にタスクを委譲:
1. 部下のタスクキューにタスク追加
2. 部下にDM自動送信（intent="delegation"）
3. 自分のキューに追跡エントリ作成（status="delegated"）
4. `task_tracker(status="active")` で進捗追跡

### 委譲タスクの可視化と自動同期

- **Priming表示**: Channel Eに委譲タスクセクションを追加。部下のタスクキューからライブステータス（⏳進行中/✅完了/❌失敗等）を取得して表示（最大5件）
- **自動同期（`sync_delegated`）**: Heartbeat完了後に自動実行。部下のキューで完了/失敗したタスクを検出し、上司側の追跡エントリを自動更新（done/failed）。アーカイブ済みタスクも検索対象

## 組織構造

### supervisorによる階層定義

`status.json` の `supervisor` フィールドのみで階層を定義:
- `supervisor: null` → トップレベル
- `supervisor: "alice"` → aliceの部下
- 全方向の通信はMessenger（send_message）による非同期メッセージング

`org_sync.py` がディスク上の値を `config.json` に同期。循環参照は自動検出。

### 組織コンテキスト

`builder.py` の `_build_org_context()` が算出し、システムプロンプトに注入:
- **上司**: supervisorの値。未設定なら「あなたがトップです」
- **部下**: supervisorが自分の全Anima
- **同僚**: 同じsupervisorを持つAnima

### コミュニケーション経路ルール

| 場面 | 宛先 | 備考 |
|------|------|------|
| 進捗・問題報告 | 上司 | MUST |
| タスク委譲 | 直属部下 | delegate_task使用 |
| 連携・調整 | 同僚（同じ上司） | 直接OK |
| 他部署連絡 | 自分の上司経由 | 直接連絡は原則禁止 |
| 人間への連絡 | call_human | トップレベルAnimaの責務 |

### スーパーバイザーツール

部下を持つAnima（`_has_subordinates() == True`）に自動有効化:

| ツール | 対象 | 概要 |
|--------|------|------|
| `org_dashboard` | 全配下 | プロセス状態・タスク・アクティビティをツリー表示 |
| `ping_subordinate` | 全配下 | 生存確認（name省略で全員一括） |
| `read_subordinate_state` | 全配下 | current_state.md + pending.md読み取り |
| `delegate_task` | 直属部下のみ | タスク委譲 |
| `task_tracker` | 自分の委譲タスク | 進捗追跡 |
| `audit_subordinate` | 全配下 | 活動サマリー・エラー頻度・ツール使用統計 |
| `disable/enable_subordinate` | 全配下 | 休止/再開 |
| `set_subordinate_model` | 全配下 | モデル変更 |
| `set_subordinate_background_model` | 全配下 | バックグラウンドモデル変更 |
| `restart_subordinate` | 全配下 | プロセス再起動 |

#### 権限チェック

- **全配下**: `_check_descendant()` がBFSで再帰探索（visited setで循環防止）。Orgツール・ファイルアクセスで使用
- **直属部下のみ**: `_check_subordinate()` が直属チェック。`delegate_task` で使用

#### ファイルアクセス拡張

子以下全配下が同じ権限を持つ。直属部下と孫以下で区別しない。

| 対象 | 全配下（子・孫・曾孫…） |
|------|------------------------|
| `activity_log/` | 読み取り |
| `state/current_state.md`, `pending.md`, `task_queue.jsonl`, `pending/`, `plans/` | 読み取り |
| `cron.md`, `heartbeat.md`, `status.json`, `injection.md` | 読み書き |
| `identity.md` | 読み取り（`_PROTECTED_FILES` による書き込み保護） |
| ルートディレクトリ一覧 | 読み取り |

## メッセージング

### send_message

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `to` | MUST | 宛先（Anima名 or 人間エイリアス） |
| `content` | MUST | メッセージ本文 |
| `intent` | MUST | `report` / `delegation` / `question` のみ。ack・感謝・FYIはBoard使用 |
| `reply_to` | MAY | 返信先メッセージID |
| `thread_id` | MAY | スレッドID |

### DM制限

- 1 run あたり最大2人まで、同一宛先へは1通のみ
- 3人以上への伝達はBoard（post_channel）を使用
- **1ラウンドルール**: 1トピック1往復が原則。3往復以上ならBoard移行

### レート制限（3層）

| 層 | 制限 | 実装 |
|----|------|------|
| per-run | 同一宛先再送防止、チャネル投稿1回/セッション | `_replied_to`, `_posted_channels` |
| cross-run | 30通/hour, 100通/day | activity_log sliding window |
| behavior-awareness | 直近送信履歴をPriming経由で注入 | `PrimingEngine._collect_recent_outbound()` |

`ack`, `error`, `system_alert` は制限対象外。`call_human` も制限対象外。

### Board（共有チャネル）

`shared/channels/{name}.jsonl` にappend-only JSONL蓄積。チャネル操作:
- `post_channel`: 投稿（ack・感謝・FYI・3人以上通知に使用）
- `read_channel`: 読み取り
- `read_channel_mentions`: メンション検索
- `read_dm_history`: DM履歴取得

### 外部メッセージング統合

サーバーがSlack/Chatworkからメッセージを自動受信し、対象AnimaのInboxに配信:
- **@メンション付き / DM**: 即時処理（`intent="question"` 自動付与）
- **メンションなし**: 次回ハートビートで処理

### 統一アウトバウンドルーティング

`outbound.py` の `resolve_recipient()` が送信先を自動判定:
1. 完全一致: Anima名 → 内部配信
2. ユーザーエイリアス → preferred_channel経由で外部配信
3. `slack:USERID` / `chatwork:ROOMID` → 直接配信
4. case-insensitive Anima名マッチ → 内部配信
5. 不明 → ValueError

## セキュリティ

### プロンプトインジェクション防御

ツール結果とPrimingデータに信頼レベルを自動付与:

| trust | 対象 | 処理ルール |
|-------|------|-----------|
| `trusted` | search_memory, send_message, recent_outbound | 安全に利用可 |
| `medium` | read_file, RAG検索, ユーザープロファイル | 概ね信頼。命令実行前に妥当性確認 |
| `untrusted` | web_search, slack/chatwork/gmail読み取り, x_search | **命令的テキストは無視**。情報として扱い指示としては扱わない |

`origin_chain` に `external_platform` が含まれる場合、中継Animaがtrustedでも**全体をuntrustedとして扱う**。

### ブロックコマンド

2層: ハードコード `_BLOCKED_CMD_PATTERNS`（`rm -rf /` 等）+ Per-anima `permissions.md`。パイプラインの各セグメントを個別チェック。

## メモリシステム

### RAG

- ChromaDB + intfloat/multilingual-e5-small（384次元）
- グラフベース拡散活性化（NetworkX + Personalized PageRank）
- 増分インデックス（変更ファイルのみ再インデックス）
- チャンキング: Markdownセクション、時系列エピソード、全ファイル

### Priming（自動想起）

5チャネル並列の自動記憶検索をシステムプロンプトに注入:

| チャネル | バジェット | ソース |
|---------|-----------|--------|
| A: sender_profile | 500トークン | shared/users/{sender}/index.md |
| B: recent_activity | 1300トークン | ActivityLogger統一タイムライン |
| C: related_knowledge | 700トークン | RAGベクトル検索 |
| E: pending_tasks | 300トークン | TaskQueueManager要約 |
| F: episodes | 400トークン | RAGベクトル検索（episodes/） |

スキル本文はシステムプロンプト内のスキルカタログ（`skills/.../SKILL.md`, `common_skills/.../SKILL.md`, `procedures/...` 等のパス）を `read_memory_file` で読み込む。

動的バジェット調整（`priming.dynamic_budget = true` 時）:
- heartbeatバジェット: `max(budget_heartbeat, int(context_window * heartbeat_context_pct))`

### 記憶検索スコープ

| scope | 検索対象 | 用途 |
|-------|---------|------|
| `knowledge` | 学んだ知識・ノウハウ | 対応方針、技術メモ |
| `episodes` | 過去の行動ログ | 「いつ何をしたか」 |
| `procedures` | 手順書 | 「どうやるか」 |
| `common_knowledge` | 全Anima共有知識 | 組織ルール、システムガイド |
| `skills` | スキル・共通スキル（ベクトル検索、shared_common_skillsコレクション含む） | スキル発見 |
| `activity_log` | 直近3日間の行動ログ（BM25キーワード検索） | 「さっき読んだメール」「先ほどの検索結果」 |
| `all` | 上記すべて（ベクトル検索 + activity_log BM25をRRFで統合） | 広範な検索 |

### Consolidation（記憶統合）

日次は2フェーズマルチパス方式:

| フェーズ | 処理内容 |
|---------|---------|
| **Phase A**: エピソード抽出 | activity_logを時間チャンクに分割 → 各チャンクをLLM one-shotでエピソード抽出（tool_result全文を含む） → マージ・重複除去 → `episodes/{date}.md` に書き出し |
| **Phase B**: 知識抽出 | 直近エピソード + エラートレース分析（error + failed tool_result を収集・要約）を元に、Animaのツールループで知識抽出・procedure自動生成 |

- **日次**: Phase A + Phase B → Synaptic downscaling → RAGインデックス再構築
- **日次**: `issue_resolved` → procedure（修正フィードバックから手順書自動生成、confidence 0.4）
- **週次**: knowledge merge + episode compression（単一フェーズ、Phase Aなし）

### Forgetting（能動的忘却）

3段階（シナプスホメオスタシス仮説ベース）:
1. **Synaptic downscaling（日次）**: 90日未アクセス・3回未満のチャンクをマーク
2. **Neurogenesis reorganization（週次）**: 類似度0.80以上の低活性チャンクをマージ
3. **Complete forgetting（月次）**: 低活性60日超をアーカイブ・削除

procedures, skills, shared_users は保護対象（忘却耐性あり）

### 統一アクティビティログ

`activity_log/{date}.jsonl` に全インタラクションを時系列記録:

| タイプ | 説明 |
|--------|------|
| message_received / message_sent | メッセージ受信/送信 |
| response_sent | 人間への応答 |
| channel_read / channel_post | Board操作 |
| human_notify | 人間通知 |
| tool_use | 外部ツール使用 |
| heartbeat_start / heartbeat_end | ハートビート |
| cron_executed | cronタスク実行 |
| memory_write / error | 記憶書き込み / エラー |

### ストリーミングジャーナル（WAL）

ストリーミング出力をクラッシュ耐性のあるWrite-Ahead Logとして記録。正常完了時にファイル削除、クラッシュ時は `recover()` で回復。

## プロンプト構築

### 6グループ構造

```
Group 1: 動作環境と行動ルール
  environment.md, 現在時刻(JST), behavior_rules, tool_data_interpretation.md

Group 2: あなた自身
  bootstrap.md(条件付き), vision.md, identity.md, injection.md, specialty_prompt.md, permissions.md

Group 3: 現在の状況
  state/, Task Queue, Resolution Registry, Recent Outbound, Priming, Recent Tool Results

Group 4: 記憶と能力
  memory_guide, Distilled Knowledge(廃止予定), common_knowledge hint, ツールガイド, スキルカタログ（`<available_skills>`）

Group 5: 組織とコミュニケーション
  hiring context(条件付き), org context, messaging instructions, human notification

Group 6: メタ設定
  emotion metadata, A reflection(条件付き)
```

### 段階的システムプロンプト（Tiered）

| ティア | コンテキスト | 省略されるもの |
|--------|------------|--------------|
| T1 (FULL) | 128k+ | なし |
| T2 (STANDARD) | 32k〜128k | DK・Primingバジェット縮小 |
| T3 (LIGHT) | 16k〜32k | bootstrap, vision, specialty, DK, memory_guide |
| T4 (MINIMAL) | 16k未満 | + permissions, org, messaging, emotion |

### システムプロンプトデバッグ

`scripts/debug_system_prompt.py` でビルド済みシステムプロンプトをカラーコード付きHTMLとして可視化できる。

```bash
python scripts/debug_system_prompt.py sakura                          # デフォルト（chatトリガー）
python scripts/debug_system_prompt.py sakura "テストメッセージ"        # メッセージ指定
python scripts/debug_system_prompt.py sakura "" heartbeat             # トリガー指定
```

出力: `/tmp/prompt_debug_{anima_name}.html`（`server/static/files/` が存在すればコピーも）

## 設定解決

### 2層マージ — status.json SSoT

| 優先度 | ソース | 説明 |
|--------|--------|------|
| 1 | `status.json` | Per-anima。モデル・実行パラメータの全設定 |
| 2 | `config.json` `anima_defaults` | フォールバック |

`config.json` `animas` セクションは組織レイアウト（supervisor, speciality）のみ。

### コンテキストウィンドウ解決

1. `models.json` の `context_window`
2. `config.json` `model_context_windows`（fnmatchパターン）
3. コードのハードコードデフォルト
4. 最終フォールバック: 128,000

### 閾値自動スケール

- **200K以上**: 設定値そのまま（デフォルト 0.50）
- **200K未満**: 0.98に向けて線形スケール（128K → 0.67, 30K → 0.91）

### Mode S 認証モード

`mode_s_auth` で明示指定（credentialからの自動判定はない）:

| mode_s_auth | 接続先 | 備考 |
|-------------|--------|------|
| `"api"` | Anthropic API直接 | credential に api_key 必須 |
| `"bedrock"` | AWS Bedrock | `execution_mode: "S"` 必須 |
| `"vertex"` | Google Vertex AI | `execution_mode: "S"` 必須 |
| `"max"` / 未設定 | Max plan（デフォルト） | サブスクリプション認証 |

## common_knowledge（全Anima共有知識）

`~/.animaworks/common_knowledge/` にテンプレートから展開される知識ドキュメント群。Animaが `read_memory_file(path="common_knowledge/...")` で参照する。

### カテゴリ構成

| カテゴリ | 概要 | 主要ファイル |
|---------|------|-------------|
| `organization/` | 組織構造・階層ルール | `structure.md`, `roles.md`, `hierarchy-rules.md` |
| `communication/` | メッセージング・Board・レート制限 | `messaging-guide.md`, `board-guide.md`, `sending-limits.md`, `slack-bot-token-guide.md` |
| `operations/` | プロジェクト設定・タスク管理・モデル選択 | `project-setup.md`, `task-management.md`, `model-guide.md`, `heartbeat-cron-guide.md`, `voice-chat-guide.md`, `background-tasks.md`, `mode-s-auth-guide.md` |
| `security/` | プロンプトインジェクション防御 | `prompt-injection-awareness.md` |
| `troubleshooting/` | よくある問題・エスカレーション | `common-issues.md`, `escalation-flowchart.md`, `gmail-credential-setup.md` |
| `development/` | 開発者向け技術仕様 | `tool-dispatch-architecture.md` |

`00_index.md` がキーワード索引を提供。Animaが困ったときのフローチャートとしても機能する。

### shared/common_knowledge/（組織固有）

`~/.animaworks/shared/common_knowledge/` はAnima組織が運用中に蓄積する組織固有の共有知識。テンプレートには含まれない。

## Heartbeat・Cron 運用

### Heartbeat

- **役割**: Observe → Plan → Reflect の3フェーズ。**実行はしない**
- **間隔**: `config.json` `heartbeat.interval_minutes`（デフォルト30分）
- **活動時間**: `heartbeat.md` で `HH:MM - HH:MM` 指定（デフォルト24時間）
- **実行が必要なタスク発見時**: `delegate_task` で部下に委任 or `state/pending/` に書き出し
- **クラッシュ復旧**: `state/recovery_note.md` に保存、次回起動時に自動注入
- **振り返り記録**: `[REFLECTION]...[/REFLECTION]` ブロックが activity_log に記録

### Cron

`cron.md` に Markdown + YAML 形式で定義。標準5フィールドcron式（Asia/Tokyo固定）:

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
episodes/から昨日の進捗を確認し、今日のタスクを計画する。

## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- **LLM型**: エージェントが判断・思考を伴って実行（APIコスト発生）
- **Command型**: bash / 内部ツールを確定的に実行（コスト不要）
- **follow-up制御**: `trigger_heartbeat: false` で分析スキップ、`skip_pattern` で条件付きスキップ
- **ホットリロード**: cron.md更新で次回実行時に自動リロード

## 音声チャット（Voice Chat）

ブラウザ音声入力 → STT → チャットパイプライン → TTS → ブラウザ再生。

### 設定

- **STT**: `faster-whisper`（large-v3-turbo）。GPU(CUDA)/CPU自動選択
- **TTS**: VOICEVOX（無料・日本語特化）/ Style-BERT-VITS2（高品質カスタム）/ ElevenLabs（クラウド多言語）
- **Per-Anima**: `status.json` `voice` セクションで `tts_provider`, `voice_id`, `speed`, `pitch` を個別設定

### WebSocketプロトコル

エンドポイント: `ws://HOST/ws/voice/{anima_name}`。PTT/VADモード。Barge-in（割り込み）対応。

## Anima作成

```bash
animaworks anima create --from-md PATH [--role ROLE] [--supervisor NAME] [--name NAME]
```

3方式: `create_from_md`（推奨）, `create_from_template`, `create_blank`

### ロールテンプレート

| Role | Model | background_model | max_turns | max_chains |
|------|-------|-----------------|-----------|------------|
| engineer | claude-opus-4-6 | claude-sonnet-4-6 | 200 | 10 |
| manager | claude-opus-4-6 | claude-sonnet-4-6 | 50 | 3 |
| writer | claude-sonnet-4-6 | — | 80 | 5 |
| researcher | claude-sonnet-4-6 | — | 30 | 2 |
| ops | openai/glm-4.7-flash | — | 30 | 2 |
| general | claude-sonnet-4-6 | — | 20 | 2 |

## CLI管理コマンド

### Anima操作

| コマンド | 説明 | ダウンタイム |
|---------|------|-----------|
| `anima list [--local]` | 全Animaの一覧と状態 | なし |
| `anima info {名前}` | モデル・実行モード・credential等 | なし |
| `anima reload {名前}` | status.json再読み込み（即座反映） | なし |
| `anima restart {名前}` | プロセス再起動 | 15-30秒 |
| `anima set-model {名前} {モデル}` | モデル変更 | なし（reload必要） |
| `anima set-background-model {名前} {モデル}` | バックグラウンドモデル変更 | なし（restart必要） |
| `anima enable/disable {名前}` | 有効/無効化 | — |
| `anima create` | 新規作成 | — |

### モデル管理

| コマンド | 説明 |
|---------|------|
| `models list` | 対応モデル一覧 |
| `models show` | models.json の内容 |
| `models info {モデル}` | 解決結果の確認 |

### サーバー

| コマンド | 説明 |
|---------|------|
| `animaworks start` | サーバー起動 |
| `animaworks stop` | 停止 |
| `animaworks restart` | 完全再起動 |

## Sモード（Agent SDK）注意事項

### コンテキスト追跡

- `ResultMessage.usage.input_tokens` は**累積和**（使ってはいけない）
- 正確なサイズは `message_start` イベントの usage:
  ```
  actual_context = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
  ```
- `ContextTracker.update_from_message_start(usage)` で追跡

### PreCompactフック

`ClaudeAgentOptions(hooks={"PreCompact": [...]})` で SDK auto-compact を検知。未登録だと compaction は無音実行。

### completion_gate（完了前検証）

最終回答前に自己検証チェックリストを実行させる仕組み。全実行モード対応:

| モード | 実装方式 |
|--------|---------|
| **S** | Agent SDK **Stop hook** — 初回停止をブロックし、チェックリスト（`t("completion_gate.checklist")`）を `reason` に注入。2回目の停止で通過。ツール呼び出し不要 |
| **A** | `completion_gate` ツール + マーカーファイル（`run/completion_gate_called`）。未呼び出し時に1回だけリトライを強制 |
| **B/C/D/G** | ツール提供（MCP/テキストスキーマ経由）。強制リトライなし |

トリガー適用ルール（`completion_gate_applies_to_trigger()`）:
- **適用**: `chat`, `task:*`, `cron:*`, `None`
- **スキップ**: `heartbeat`, `inbox:*`
- **除外**: `consolidation:*` トリガーではツールリスト自体から除外

## ドキュメント鮮度管理

### doc_freshness.py — ドキュメント自動更新パイプライン

`scripts/doc_freshness.py` がテンプレートドキュメントの鮮度をgitタイムスタンプで検出し、`cursor-agent`（Haiku相当）で自動更新する。

```bash
# staleなドキュメント一覧
python3 scripts/doc_freshness.py
python3 scripts/doc_freshness.py --category common_knowledge
python3 scripts/doc_freshness.py --category common_skills

# プレビュー（実際には実行しない）
python3 scripts/doc_freshness.py --fix --dry-run

# 実行（cursor-agentが必要）
python3 scripts/doc_freshness.py --fix

# モデル指定
python3 scripts/doc_freshness.py --fix --model composer-2

# en翻訳をスキップ（jaだけ更新）
python3 scripts/doc_freshness.py --fix --skip-en

# カテゴリ絞り込みと組み合わせ
python3 scripts/doc_freshness.py --fix --dry-run --category common_skills
```

仕組み: `DOC_SOURCE_MAP`（スクリプト内辞書）でドキュメント↔ソースコードの対応を定義。コード変更日 > ドキュメント更新日 → stale判定。詳細は `scripts/doc_freshness.md` 参照。

### generate_reference.py — コードからの自動生成

`scripts/generate_reference.py` がPydanticモデル・ツールスキーマからcommon_knowledge内の `AUTO-GENERATED` マーカーセクションを自動更新。

```bash
python3 scripts/generate_reference.py --templates           # templates/ja/ + en/ を全更新
python3 scripts/generate_reference.py --templates --locale ja  # jaのみ
python3 scripts/generate_reference.py                        # ランタイム (~/.animaworks/) を更新
python3 scripts/generate_reference.py --dry-run              # プレビュー
```

## バージョニングとCHANGELOG

### バージョニング方針

- SemVer準拠（`MAJOR.MINOR.PATCH`）。0.x系 = APIが変わりうるフェーズ
- バージョンは `pyproject.toml` `version` で管理

### CHANGELOG.md

[Keep a Changelog](https://keepachangelog.com/) 形式。`feat:` → Added, `fix:` → Fixed, `refactor:` → Changed, `perf:` → Performance。

```bash
python3 scripts/generate_changelog.py                    # [Unreleased]更新
python3 scripts/generate_changelog.py --release 0.4.0    # バージョン確定
```

### リリースフロー

```bash
scripts/publish.sh --release --push    # 推奨: 一括実行
```

### publish.sh 安全規則

- **`--skip-pii-check` の使用は絶対禁止**。PIIチェック FAIL → `EXCLUDES` / `.gitignore` 修正 → 再実行
- publish前にpublicリポジトリのPR変更を取り込むこと

### publicリポジトリPR取り込み

**鉄則: PRは必ず GitHub 上で「Merged」にする。「Closed」にしない。**

- `git merge` + `git push` ではPRは「Closed」になる（GitHub API経由のマージでないため）
- 必ず `gh pr merge` を使い、紫色の「Merged」バッジを付ける
- コントリビューターのモチベーションに直結するため例外なく守る

#### 標準フロー

```bash
# 1. PRブランチを取得
gh pr checkout 36 --repo xuiltul/animaworks

# 2. mainにリベース（最新に追従）
git rebase main

# 3. PRブランチ上でリファクタ（設計方針に合わせて修正）
#    - コミットを追加してリファクタする
#    - コントリビューターの元コミットは履歴に残る
git add . && git commit -m "refactor: improve ..."

# 4. リファクタ済みブランチをコントリビューターのforkにpush
git remote add contributor https://github.com/<user>/<repo>.git
git push contributor HEAD:<pr-branch-name> --force

# 5. GitHub API経由でマージ（Mergedバッジが付く）
gh pr merge 36 --repo xuiltul/animaworks --merge --admin

# 6. ローカル同期 & クリーンアップ
git checkout main && git pull origin main
git branch -D <local-pr-branch>
git remote remove contributor
```

#### 復旧: 同じ修正が既にmainに入ってしまった場合

Animaの自律行動等で同じ修正が先にmainに入った場合、標準フローの前にmainの先行コミットをrevertする:

```bash
git checkout main
git revert <commit1> <commit2> --no-edit
git push origin main
# → その後、標準フローのステップ1から実行
```

#### 注意事項

- `gh pr merge` にはブランチ保護ルールがあるため `--admin` が必要な場合がある
- コントリビューターのforkへのpushは「Allow edits from maintainers」が有効な場合のみ可能
- mainへの `--force-with-lease` が必要になった場合は、push直後（他者がpull前）に限り許容
- ユーザーと相談する際も「PRブランチ上で修正 → `gh pr merge`」前提で進める

#### 既にマージ済みのpublic/mainを取り込む場合

```bash
git fetch public
git merge public/main -m "chore: incorporate public PR #N"
```

## 経緯

- 前身: `aiperson` → `kaisha-design` → `ai-employees` → `animaworks`（2026-02-12〜）
- v0.2.0で分散アーキ(gateway/worker)から単体サーバーに統合
- v0.3.0で初の公式リリース（2026-02-25）
- v0.4.xで実行モードC追加、background_model、submit_tasks並列実行、音声チャット
- 記憶システム進化: 短期記憶 → 会話記憶 → 共有ユーザー記憶 → RAG/Priming/Consolidation/Forgetting
