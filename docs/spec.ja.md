# Digital Anima 要件定義書 v1.4

## 1. 概要

Digital Animaは、AIエージェントを「1人の人間」としてカプセル化する最小単位。

**核心の設計原則:**

- 内部状態は外部から不可視。外部との接点はテキスト会話のみ
- 記憶は「書庫型」。必要な時に必要な記憶だけを自分で検索して取り出す
- 全コンテキストは共有しない。自分の言葉で圧縮・解釈して伝える
- ハートビートにより指示待ちではなくプロアクティブに行動する
- 役割・理念は後から注入する。Digital Anima自体は「空の器」

**技術方針:**

- エージェント実行は **6モード**（モデル名パターンで自動判定、`resolve_execution_mode()`）: **S** Claude Agent SDK、**C** Codex CLI、**D** Cursor Agent CLI、**G** Gemini CLI、**A** LiteLLM + tool_use（Anthropic 直フォールバック含む）、**B** 1ショット補助（フレームワークが記憶I/O代行）
- 設定は統合 **config.json**（Pydanticバリデーション）、記憶は **Markdown** で記述する
- ユーザー向け UI 文言は **`core/i18n`** の `t()` で解決（ハードコード禁止）
- 複数Animaが **階層構造** で協調動作する（`supervisor` フィールドで階層定義、同期委任）

-----

## 2. アーキテクチャ

```
┌──────────────────────────────────────────────────────┐
│                   Digital Anima                      │
│                                                       │
│  Identity ──── 自分が誰か（常駐）                      │
│  Agent Core ── 6実行モード（モデル名で自動解決）          │
│    ├ S: Claude Agent SDK                               │
│    ├ C: Codex CLI                                        │
│    ├ D: Cursor Agent CLI                                 │
│    ├ G: Gemini CLI                                       │
│    ├ A: LiteLLM + tool_use（クラウドAPI・一部Ollama等）   │
│    └ B: 1ショット補助（弱モデル・FW代行）                 │
│  Memory ───── 書庫型長期記憶（自律検索で想起）          │
│    ├ 会話記憶（state/conversation.json、ローリング圧縮）│
│    ├ 短期記憶（shortterm/chat/・heartbeat/ 分離）      │
│    └ 統一活動ログ（activity_log/、JSONL時系列）        │
│  Boards ───── Slack型共有チャネル                       │
│  Permissions ─ ツール/ファイル/コマンド制限             │
│  Communication ─ テキスト＋ファイル参照                 │
│  Lifecycle ── メッセージ受信/ハートビート/cron          │
│  Injection ── 役割/理念/行動規範（後から注入）          │
│                                                       │
└──────────────────────────────────────────────────────┘
        ▲                       │
   テキスト(受信)          テキスト(送信)
```

-----

## 3. ファイル構成

```
animaworks/
├── core/
│   ├── anima.py               # DigitalAnimaクラス
│   ├── agent.py               # AgentCore（実行モード選択・サイクル管理）
│   ├── anima_factory.py       # Anima生成（テンプレート/空白/MD）
│   ├── init.py                # ランタイム初期化
│   ├── schemas.py             # データモデル（Message, CycleResult等）
│   ├── paths.py               # パス解決
│   ├── messenger.py           # Anima間メッセージ送受信
│   ├── lifecycle/             # ハートビート・cron・Inbox（パッケージ、APScheduler）
│   │   ├── __init__.py        #   LifecycleManager（Scheduler/Inbox/レート制限等ミックスイン統合）
│   │   ├── scheduler.py       #   スケジュール登録
│   │   ├── inbox_watcher.py   #   Inbox 監視
│   │   ├── rate_limiter.py    #   メッセージ連鎖・クールダウン
│   │   ├── system_crons.py    #   サーバ単位の定期ジョブ
│   │   └── system_consolidation.py # 組織横断 consolidation トリガー
│   ├── outbound.py            # 統一アウトバウンドルーティング（Slack/Chatwork/内部自動判定）
│   ├── background.py          # バックグラウンドタスク管理
│   ├── asset_reconciler.py    # アセット自動生成
│   ├── org_sync.py            # 組織構造同期（status.json → config.json）
│   ├── schedule_parser.py     # cron.md/heartbeat.mdパーサー
│   ├── logging_config.py      # ログ設定
│   ├── memory/                # 記憶サブシステム（詳細は memory.md 参照）
│   │   ├── manager.py         #   書庫型記憶の検索・書き込み
│   │   ├── conversation.py    #   会話記憶（エントリ）
│   │   ├── conversation_*.py  #   圧縮・確定・モデル・プロンプト等（分割モジュール）
│   │   ├── shortterm.py       #   短期記憶（chat/heartbeat分離）
│   │   ├── activity.py        #   統一アクティビティログ（JSONL時系列）
│   │   ├── streaming_journal.py #  ストリーミングジャーナル（WAL）
│   │   ├── priming/           #   自動想起レイヤー（6チャネル、engine + channel_a〜f + budget）
│   │   ├── consolidation.py   #   記憶統合（日次/週次）
│   │   ├── forgetting.py      #   能動的忘却（3段階）
│   │   ├── reconsolidation.py #   記憶再統合
│   │   ├── task_queue.py      #   永続タスクキュー
│   │   ├── resolution_tracker.py # 解決レジストリ
│   │   ├── rag_search.py      #   検索オーケストレーション
│   │   └── rag/               #   RAGエンジン（ChromaDB + sentence-transformers）
│   │       ├── indexer.py, retriever.py, graph.py, store.py, http_store.py
│   │       └── watcher.py     #   ファイル変更監視
│   ├── supervisor/            # プロセス監視
│   │   ├── manager.py         #   ProcessSupervisor（子プロセス起動・監視）
│   │   ├── ipc.py             #   Unix Domain Socket IPC
│   │   ├── runner.py          #   Animaプロセスランナー
│   │   ├── process_handle.py  #   プロセスハンドル管理
│   │   ├── pending_executor.py #   TaskExec（state/pending/ タスク実行）
│   │   ├── scheduler_manager.py #  子プロセス側スケジューラ
│   │   ├── inbox_rate_limiter.py, streaming_handler.py, transport.py 等
│   │   └── _mgr_*.py          #   ヘルス・調整など内部補助
│   ├── notification/          # 人間通知
│   │   ├── notifier.py        #   HumanNotifier（call_human統合）
│   │   ├── reply_routing.py   #   返信ルーティング
│   │   └── channels/          #   Slack, Chatwork, LINE, Telegram, ntfy
│   ├── voice/                 # 音声チャットサブシステム
│   │   ├── stt.py             #   VoiceSTT（faster-whisper）
│   │   ├── tts_*.py           #   TTSプロバイダ（VOICEVOX, ElevenLabs, SBV2）
│   │   └── session.py         #   VoiceSession（STT→Chat IPC→TTS）
│   ├── mcp/                   # stdio MCP（Mode S: ツール名 `mcp__aw__*`）
│   │   └── server.py
│   ├── config/                # 設定管理
│   │   ├── models.py          #   公開ファサード（load_config / load_permissions 等の再エクスポート）
│   │   ├── schemas.py         #   Pydantic モデル定義（AnimaWorksConfig 本体）
│   │   ├── io.py              #   config.json 読み書き・キャッシュ
│   │   ├── model_mode.py      #   resolve_execution_mode / DEFAULT_MODEL_MODE_PATTERNS
│   │   ├── resolver.py        #   status.json マージ解決
│   │   ├── vault.py           #   VaultManager（~/.animaworks/vault.json + vault.key）
│   │   └── cli.py ほか        #   migrate, anima_registry, model_config, global_permissions 等
│   ├── prompt/                # プロンプト・コンテキスト管理
│   │   ├── builder.py         #   システムプロンプト構築（6グループ構造）
│   │   ├── assembler.py, sections.py, org_context.py, messaging.py
│   │   └── context.py         #   コンテキストウィンドウ追跡
│   ├── tooling/               # ツール基盤
│   │   ├── handler.py         #   ToolHandler 本体（ディスパッチ集約）
│   │   ├── handler_base.py, handler_memory.py, handler_comms.py, handler_skills.py
│   │   ├── handler_perms.py, handler_org.py, handler_org_dashboard.py
│   │   ├── handler_delegation.py, handler_subordinate_control.py, handler_create_anima.py
│   │   ├── schemas/           #   ツールスキーマ（ドメイン別 Python モジュール）
│   │   ├── guide.py, dispatch.py, permissions.py, skill_tool.py, skill_creator.py
│   │   └── prompt_db.py, org_helpers.py
│   ├── execution/             # 実行エンジン
│   │   ├── base.py            #   BaseExecutor ABC
│   │   ├── agent_sdk.py       #   Mode S: Claude Agent SDK
│   │   ├── codex_sdk.py       #   Mode C: Codex CLI
│   │   ├── cursor_agent.py    #   Mode D: Cursor Agent CLI
│   │   ├── gemini_cli.py      #   Mode G: Gemini CLI
│   │   ├── anthropic_fallback.py # Mode A 内: Anthropic SDK 直接
│   │   ├── litellm_loop.py    #   Mode A: LiteLLM + tool_use
│   │   ├── assisted.py        #   Mode B: フレームワーク補助
│   │   └── _session.py ほか   #   セッション・SDK ストリーム・サニタイズ等
│   ├── i18n/                  #   ユーザー向け文言（t() / _STRINGS）
│   └── tools/                 # 外部ツール実装
│       ├── web_search.py, x_search.py, slack.py, chatwork.py
│       ├── gmail.py, github.py, google_calendar.py, google_tasks.py
│       ├── discord.py, notion.py, machine.py
│       ├── call_human.py, transcribe.py, aws_collector.py, local_llm.py
│       ├── image_gen.py       #   画像・3D（サブパッケージ image/）
│       └── …
├── cli/                       # CLIパッケージ
│   ├── parser.py              #   argparse定義 + cli_main()
│   └── commands/              #   サブコマンド実装
├── server/
│   ├── app.py                 # FastAPI（lifespan・ミドルウェア・静的配信・ルータ登録）
│   ├── slack_socket.py        # Slack Socket Mode クライアント
│   ├── websocket.py           # WebSocketManager（ダッシュボード用 `/ws`）
│   ├── stream_registry.py     # チャット/SSE ストリームのプロデューサー登録・掃除
│   ├── reload_manager.py      # ConfigReloadManager（設定ホットリロード）
│   ├── room_manager.py        # 会議室（MeetingRoom / RoomManager、`shared/meetings`）
│   ├── localhost.py           # ローカルホスト信頼判定（認証バイパス用）
│   ├── events.py              # ダッシュボード向けイベント emit（WebSocket）
│   ├── dependencies.py        # 互換スタブ（プロセス分離後は IPC 利用）
│   ├── routes/                # API ルート（`/api` プレフィックスで include）
│   │   ├── animas.py, chat.py, sessions.py
│   │   ├── chat_*.py          # チャット処理分割（chunk_handler, emotion, images, producer, resume 等）
│   │   ├── memory_routes.py, logs_routes.py
│   │   ├── channels.py        # Board / 共有チャネル・DM 履歴
│   │   ├── voice.py           # 音声 WebSocket `ws/voice/{name}`（ルータ直下）
│   │   ├── websocket_route.py # ダッシュボード WebSocket `/ws`
│   │   ├── room.py            # 会議室 REST + SSE
│   │   ├── internal.py        # 内部 API（埋め込み/ベクトル、CLI 連携通知、メッセージ取得）
│   │   ├── auth.py            # UI セッション認証
│   │   ├── setup.py           # 初回セットアップ API（`/api/setup/*`）
│   │   ├── activity_report.py, brainstorm.py
│   │   ├── external_tasks.py, team_presets.py
│   │   ├── chat_ui_state.py, tool_prompts.py, users.py
│   │   ├── system.py, assets.py, config_routes.py
│   │   ├── webhooks.py        # 外部メッセージング Webhook
│   │   └── media_proxy.py     # 外部画像プロキシ（assets ルートから利用）
│   └── static/                # Web UI（`/` にマウント、HTML は no-cache）
│       ├── index.html         # SPA シェル（`#/chat`, `#/board` 等）
│       ├── setup/             # セットアップウィザード（`/setup`）
│       ├── modules/, pages/, shared/, styles/
│       └── workspace/         # 3D オフィス Workspace（`/workspace`）
├── templates/
│   ├── ja/, en/               # ロケール別テンプレート
│   │   ├── prompts/           #   プロンプトテンプレート
│   │   ├── anima_templates/   #   Anima雛形（_blank）
│   │   ├── roles/             #   ロールテンプレート（engineer, researcher, manager, writer, ops, general）
│   │   ├── common_knowledge/  #   共有知識テンプレート
│   │   └── common_skills/     #   共通スキルテンプレート
│   └── _shared/               # ロケール共通（組織ビジョン等）
├── main.py                    # CLIエントリポイント
└── tests/                     # テストスイート
```

### 3.1 Animaディレクトリ（`~/.animaworks/animas/{name}/`）

各Animaは以下のファイル・ディレクトリで構成される:

|ファイル / ディレクトリ      |説明                              |
|----------------------|--------------------------------|
|`identity.md`         |性格・得意分野（不変ベースライン）          |
|`injection.md`        |役割・理念・行動規範（差替可能）            |
|`permissions.json` / `permissions.md`|ツール/ファイル/コマンド権限。`permissions.json` を優先し、未移行時は `permissions.md` から自動移行（`load_permissions` は `core/config/schemas.py` で定義、`core/config/models.py` からインポート）|
|`heartbeat.md`        |定期チェック間隔・活動時間               |
|`cron.md`             |定時タスク（YAML）                   |
|`bootstrap.md`        |初回起動時の自己構築指示                |
|`status.json`         |有効/無効、ロール、モデル設定             |
|`specialty_prompt.md` |ロール別専門プロンプト                  |
|`assets/`             |キャラクター画像・3Dモデル              |
|`transcripts/`        |会話トランスクリプト                   |
|`skills/`             |個人スキル（YAML frontmatter + Markdown本文）|
|`activity_log/`       |統一活動ログ（日付別JSONL）            |
|`state/`              |作業記憶（current_state.md, pending.md, pending/, task_queue.jsonl）|
|`episodes/`           |エピソード記憶（日次ログ）               |
|`knowledge/`          |意味記憶（学習済み知識）                |
|`procedures/`         |手続き記憶（手順書）                   |
|`shortterm/`          |短期記憶（chat/・heartbeat/ 分離、セッション継続）|

### 3.2 config.json（統合設定）

全設定を `~/.animaworks/config.json` に統合。Pydantic `AnimaWorksConfig`（`core/config/schemas.py`）でバリデーションする。暗号化シークレットストア **Vault** は `config.json` には含めず、`~/.animaworks/vault.json` + `vault.key` を `core/config/vault.py` の `VaultManager` が管理する。

**トップレベル構造:**

|セクション             |説明                              |
|--------------------|--------------------------------|
|`version`           |スキーマバージョン（整数）                    |
|`setup_complete`    |初回セットアップ完了フラグ                   |
|`locale`            |既定ロケール（例: `ja`）                  |
|`system`            |動作モード、ログレベル                   |
|`credentials`       |プロバイダ別APIキー・エンドポイント（名前付きマップ）   |
|`model_modes`       |※非推奨。`~/.animaworks/models.json` に置換済み。フォールバックとして参照    |
|`model_context_windows`|※後方互換。優先は `models.json` の `context_window`    |
|`model_max_tokens`  |モデル名パターン（fnmatch）→ 既定 `max_tokens` オーバーライド|
|`anima_defaults`    |全Animaに適用されるデフォルト値             |
|`animas`            |組織レイアウト（supervisor, speciality）のみ。モデル設定は status.json SSoT|
|`consolidation`     |記憶統合設定（日次/週次の実行時刻・閾値、日次インデックス等）|
|`rag`               |RAG設定（埋め込みモデル、グラフ拡散活性化、検索スコア閾値等）|
|`prompt`            |システムプロンプト構築（例: 注入サイズ警告しきい値文字数）   |
|`priming`           |自動想起設定（メッセージタイプ別トークンバジェット）     |
|`image_gen`         |画像生成設定（スタイル一貫性、Vibe Transfer）    |
|`human_notification`|人間通知設定（チャネル: Slack/LINE/Telegram/Chatwork/ntfy）|
|`server`            |サーバーランタイム設定（IPC、keep-alive、ストリーミング）|
|`external_messaging`|外部メッセージング統合（Slack Socket Mode、Chatwork Webhook）|
|`background_task`   |バックグラウンドツール実行設定（対象ツール・閾値・並列LLM数）|
|`activity_log`      |ログローテーション設定（rotation_mode, max_size_mb, max_age_days）|
|`heartbeat`         |ハートビート間隔・タイムアウト・カスケード防止・idle compaction 等|
|`voice`             |音声チャット設定（STT/TTSプロバイダ）                 |
|`housekeeping`      |定期ディスククリーンアップ設定                       |
|`machine`           |`machine` 外部エージェントツールのエンジン優先順位        |
|`workspaces`        |エイリアス → 絶対パス（ワークスペース登録）          |
|`activity_level`    |グローバル活動レベル（10–400%。HB 間隔・max_turns に反映）|
|`activity_schedule` |時間帯別 `activity_level`（空なら固定 `activity_level`）|
|`ui`                |UI テーマ・デモモード等                      |

**設定解決（2層マージ — status.json SSoT）:**

Anima起動時のモデル設定は `status.json` を Single Source of Truth（SSoT）として2層で解決する:

1. **Layer 1: status.json**（最優先）— `animas/{name}/status.json` のモデル・実行パラメータ
2. **Layer 2: config.json anima_defaults**（フォールバック）— `config.anima_defaults` の全体デフォルト

`config.json` の `animas` セクションは組織レイアウト（`supervisor`, `speciality`）のみを保持する。

**AnimaModelConfig フィールド（config.json animas）:**

|フィールド       |型              |説明                        |
|---------------|---------------|--------------------------|
|`supervisor`   |`str \| null`  |上位Animaの名前                  |
|`speciality`   |`str \| null`  |自由記述の専門分野                   |
|`model`        |`str \| null`  |オーバーライド用（status.json が優先）   |

**status.json のモデル関連フィールド（SSoT）:**

|フィールド                           |型              |デフォルト                   |説明                        |
|---------------------------------|---------------|------------------------|--------------------------|
|`model`                          |`str`          |`claude-sonnet-4-6`       |使用するモデル名（プロバイダprefix付き可）|
|`fallback_model`                 |`str \| null`  |`null`                  |フォールバックモデル                  |
|`max_tokens`                     |`int`          |`8192`                  |1回のレスポンスの最大トークン数             |
|`max_turns`                      |`int`          |`20`                    |1サイクルの最大ターン数                |
|`credential`                     |`str`          |`"anthropic"`           |使用するcredentials名             |
|`context_threshold`              |`float`        |`0.50`                  |短期記憶外部化の閾値（コンテキスト使用率）       |
|`max_chains`                     |`int`          |`2`                     |自動セッション継続の最大回数              |
|`conversation_history_threshold` |`float`        |`0.30`                  |会話記憶の圧縮トリガー（コンテキスト使用率）      |
|`background_model`               |`str \| null`  |`null`                  |heartbeat/inbox/cron用の軽量モデル。未設定時はメインモデルを使用|
|`execution_mode`                 |`str \| null`  |`null`（自動検出）             |`"S"` / `"A"` / `"B"` / `"C"` / `"D"` / `"G"`。未設定時は models.json または DEFAULT_MODEL_MODE_PATTERNS で解決|
|`supervisor`                     |`str \| null`  |`null`                  |上位Animaの名前                  |
|`speciality`                     |`str \| null`  |`null`                  |自由記述の専門分野                   |

**config.json 例:**

```json
{
  "version": 1,
  "system": { "mode": "server", "log_level": "INFO" },
  "credentials": {
    "anthropic": { "api_key": "", "base_url": null },
    "ollama": { "api_key": "dummy", "base_url": "http://localhost:11434/v1" }
  },
  "anima_defaults": {
    "model": "claude-sonnet-4-6",
    "max_tokens": 4096,
    "max_turns": 20,
    "credential": "anthropic",
    "context_threshold": 0.50,
    "conversation_history_threshold": 0.30
  },
  "animas": {
    "alice": {},
    "bob": { "model": "gpt-4o", "credential": "openai", "supervisor": "alice" }
  }
}
```

**セキュリティ:** config.jsonのパーミッションは `0o600`（owner read/write only）で保存される。APIキーは環境変数での管理も引き続きサポート。

**コンテキストウィンドウ解決**（`resolve_context_window()` — `core/prompt/context.py`）:

1. `~/.animaworks/models.json` の `context_window`（最優先）
2. config.json `model_context_windows`（fnmatch ワイルドカードパターン）
3. コードのハードコードデフォルト辞書
4. `_DEFAULT_CONTEXT_WINDOW` = 128,000（最終フォールバック）

コンパクション閾値は自動スケール: 200K 以上のウィンドウでは設定値（デフォルト 0.50）をそのまま使用、200K 未満は 0.98 に向けて線形スケール。

### 3.3 モデル・認証設定（credentials）

#### プロバイダ別 credentials 設定

`config.json` の `credentials` セクションに、プロバイダごとの認証情報を名前付きマップで定義する。Animaの `status.json` からキー名で参照する。

```json
{
  "credentials": {
    "anthropic": {
      "type": "api_key",
      "api_key": "sk-ant-api03-xxxxx",
      "keys": {},
      "base_url": null
    },
    "bedrock": {
      "type": "api_key",
      "api_key": "",
      "keys": {
        "aws_access_key_id": "AKIA...",
        "aws_secret_access_key": "...",
        "aws_region_name": "ap-northeast-1"
      },
      "base_url": null
    },
    "azure": {
      "type": "api_key",
      "api_key": "BKQ5t...",
      "keys": { "api_version": "2025-01-01-preview" },
      "base_url": "https://your-resource.openai.azure.com"
    },
    "vertex": {
      "type": "api_key",
      "api_key": "",
      "keys": {
        "vertex_project": "my-gcp-project",
        "vertex_location": "asia-northeast1",
        "vertex_credentials": "/path/to/service-account.json"
      },
      "base_url": null
    },
    "vllm-gpu": {
      "api_key": "dummy",
      "base_url": "http://localhost:8000/v1"
    }
  }
}
```

vLLM は OpenAI 互換 API を提供するため、`openai/` プレフィックスで接続。`api_key` は認証不要でもダミー値が必要。Anima設定: `model: "openai/glm-4.7-flash"`, `credential: "vllm-gpu"`。

#### モデル名の命名規則

モデル名にはプロバイダプレフィックスを含める（LiteLLMの命名規則に準拠）:

| プロバイダ | 形式 | 例 |
|-----------|------|-----|
| Anthropic直接 | `claude-{tier}-{version}` | `claude-opus-4-6`, `claude-sonnet-4-6` |
| AWS Bedrock | `bedrock/{region}.anthropic.claude-{tier}-{version}` | `bedrock/jp.anthropic.claude-sonnet-4-6` |
| Azure OpenAI | `azure/{deployment-name}` | `azure/gpt-4.1-mini` |
| Google Vertex AI | `vertex_ai/{model-name}` | `vertex_ai/gemini-2.5-flash` |
| OpenAI直接 | `openai/{model-name}` | `openai/gpt-4.1` |
| Codex | `codex/{model-name}` | `codex/gpt-5.3-codex` |
| Cursor Agent CLI | `cursor/{model-name}` | `cursor/claude-sonnet-4-6` |
| Gemini CLI | `gemini/{model-name}` | （CLI 側のモデル表記に準拠） |
| Ollama | `ollama/{model-name}` | `ollama/qwen3:8b` |
| vLLM（ローカル） | `openai/{model-name}` + credential の base_url | `openai/glm-4.7-flash` |

#### status.json のモデル関連フィールド

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `model` | Yes | モデル名（上記プレフィックス付き） |
| `credential` | Yes | config.jsonの`credentials`キー名 |
| `execution_mode` | No | 実行モード。未設定時は`DEFAULT_MODEL_MODE_PATTERNS`で自動解決 |
| `mode_s_auth` | No | Mode S（Agent SDK）使用時の認証方式（`"api"` / `"bedrock"` / `"vertex"`） |

#### 実行モード（execution_mode）の自動解決

`status.json` に `execution_mode` を設定しない場合、`resolve_execution_mode()` が以下の優先度で解決する:

1. Per-anima 明示オーバーライド（status.json の `execution_mode`）
2. `models.json`（`~/.animaworks/models.json`、ユーザー編集可）
3. `config.json` `model_modes`（非推奨フォールバック）
4. `DEFAULT_MODEL_MODE_PATTERNS`（コードデフォルト）
5. デフォルト `"B"`（安全側）

**DEFAULT_MODEL_MODE_PATTERNS の主なマッピング**（`core/config/model_mode.py`。より具体的なパターンが先にマッチするよう解決時に特異度ソートされる）:

| パターン | モード | 説明 |
|---------|-------|------|
| `claude-*` | S | Claude直接 → Agent SDK |
| `codex/*` | C | Codex → CLI wrapper |
| `cursor/*` | D | Cursor Agent CLI |
| `gemini/*` | G | Gemini CLI |
| `openai/*`, `azure/*`, `bedrock/*`, `vertex_ai/*`, `google/*`, `mistral/*`, `xai/*`, `cohere/*`, `zai/*`, `minimax/*`, `moonshot/*`, `deepseek/deepseek-chat` 等 | A | クラウド API 等 → LiteLLM + tool_use |
| `ollama/qwen3.5*`, `ollama/qwen3:*`（一部サイズ）, `ollama/qwen3-coder:*`, `ollama/llama4:*`, `ollama/mistral-small3.2:*`, `ollama/devstral*`, `ollama/glm-4.7*`, `ollama/glm-5*`, `ollama/minimax*`, `ollama/kimi-k2*`, `ollama/gpt-oss*` 等 | A | tool_use 実績のある Ollama 系 |
| `ollama/qwen3:0.6b` 〜 `8b`, `ollama/gemma3*`, `ollama/deepseek-r1*`, `ollama/deepseek-v3*`, `ollama/phi4*` 等 | B | 弱い/推論特化で tool が不安定な系 → Basic |
| `ollama/*` | B | 上記いずれにも当たらない Ollama → Basic（安全側） |

**注意:** `bedrock/*` はデフォルトMode A。Mode Sで使いたい場合は `"execution_mode": "S"` と `"mode_s_auth": "bedrock"` の両方を明示設定すること。

#### 構成パターン例

**Claude Opus（Anthropic Max Plan）:**
```json
{ "model": "claude-opus-4-6", "credential": "anthropic" }
```

**Claude Sonnet（AWS Bedrock経由 + Mode S）:**
```json
{
  "model": "bedrock/jp.anthropic.claude-sonnet-4-6",
  "credential": "bedrock",
  "execution_mode": "S",
  "mode_s_auth": "bedrock"
}
```

**Azure OpenAI:**
```json
{ "model": "azure/gpt-4.1-mini", "credential": "azure" }
```

**RAGConfig フィールド:**

|フィールド                      |型        |デフォルト                         |説明                          |
|----------------------------|---------|-------------------------------|----------------------------|
|`enabled`                   |`bool`   |`true`                         |RAG機能の有効/無効                  |
|`embedding_model`           |`str`    |`intfloat/multilingual-e5-small`|使用する埋め込みモデル                  |
|`use_gpu`                   |`bool`   |`false`                        |GPU使用の有無                     |
|`enable_spreading_activation`|`bool`  |`true`                         |グラフベース拡散活性化の有無               |
|`max_graph_hops`            |`int`    |`2`                            |グラフ探索の最大ホップ数                 |
|`enable_file_watcher`       |`bool`   |`true`                         |ファイル変更監視の有無                  |
|`graph_cache_enabled`       |`bool`   |`true`                         |グラフキャッシュの有効/無効               |
|`implicit_link_threshold`   |`float`  |`0.75`                         |暗黙リンク生成の類似度閾値                |
|`spreading_memory_types`    |`list`   |`["knowledge", "episodes"]`    |拡散活性化の対象記憶タイプ                |
|`min_retrieval_score`       |`float`  |`0.30`                         |検索結果の最低スコア（下回るヒットを落とす）      |
|`skill_match_min_score`     |`float`  |`0.75`                         |Priming スキルマッチの類似度下限            |

**PromptConfig フィールド（抜粋）:**

|フィールド                      |型      |デフォルト |説明                          |
|---------------------------|-------|-------|----------------------------|
|`injection_size_warning_chars`|`int`|`5000` |injection 系ファイルが長すぎる場合の警告しきい値（文字）|

**PrimingConfig フィールド:**

|フィールド              |型      |デフォルト |説明                          |
|--------------------|-------|-------|----------------------------|
|`dynamic_budget`    |`bool` |`true` |動的バジェット配分の有無                 |
|`budget_greeting`   |`int`  |`500`  |挨拶メッセージ時のトークンバジェット           |
|`budget_question`   |`int`  |`2000` |質問メッセージ時のトークンバジェット           |
|`budget_request`    |`int`  |`3000` |リクエストメッセージ時のトークンバジェット        |
|`budget_heartbeat`  |`int`  |`200`  |ハートビート時のトークンバジェット（フォールバック）|
|`heartbeat_context_pct`|`float`|`0.05`|動的バジェット時のHB用コンテキスト割合（5%）   |

**ServerConfig フィールド:**

|フィールド                      |型       |デフォルト |説明                          |
|----------------------------|--------|-------|----------------------------|
|`session_ttl_days`          |`int \| null`|`7`（`null` で無期限）|UI セッション Cookie の TTL |
|`ipc_stream_timeout`        |`int`   |`60`   |IPCストリーミングのチャンク単位タイムアウト（秒）    |
|`keepalive_interval`        |`int`   |`30`   |keep-alive送信間隔（秒）             |
|`max_streaming_duration`    |`int`   |`1800` |ストリーミング最大持続時間（秒）             |
|`busy_hang_threshold`       |`int`   |`900`  |子プロセス「busy」無応答とみなす秒数（`HealthConfig` へ反映）|
|`stream_checkpoint_enabled` |`bool`  |`true` |ストリーミング中のツール結果保存             |
|`stream_retry_max`          |`int`   |`3`    |ストリーム切断時の自動リトライ最大回数          |
|`stream_retry_delay_s`      |`float` |`5.0`  |リトライ間の待機時間（秒）                |
|`llm_num_retries`           |`int`   |`3`    |LLM API 呼び出しの再試行回数（429/5xx/ネットワーク）|
|`media_proxy`               |`object`|（既定あり）|外部画像プロキシのモード・許可ドメイン・レート制限等（`MediaProxyConfig`）|

**ExternalMessagingConfig フィールド:**

|フィールド              |型           |デフォルト    |説明                          |
|--------------------|------------|---------|----------------------------|
|`preferred_channel` |`str`       |`"slack"`|優先送信チャネル（`"slack"` or `"chatwork"`）|
|`user_aliases`      |`dict`      |`{}`     |ユーザーエイリアス → 連絡先マッピング         |
|`slack`             |`object`    |         |Slack設定（enabled, mode, anima_mapping）|
|`chatwork`          |`object`    |         |Chatwork設定（enabled, mode, anima_mapping）|

**BackgroundTaskConfig フィールド:**

|フィールド                  |型      |デフォルト |説明                          |
|----------------------|-------|-------|----------------------------|
|`enabled`             |`bool` |`true` |バックグラウンド実行の有効/無効             |
|`eligible_tools`      |`dict` |       |対象ツール名 → `{ "threshold_s": int }`（`BackgroundToolConfig`）|
|`result_retention_hours`|`int` |`24`   |結果保持時間（時間）                   |
|`max_parallel_llm_tasks`|`int` |`3`    |`submit_tasks` 等の同時 LLM 実行上限（1–10）|

**ActivityLogConfig フィールド（補足）:**

|フィールド           |型      |デフォルト |説明                          |
|----------------|-------|-------|----------------------------|
|`rotation_enabled`|`bool`|`true` |ローテーション処理の有効/無効              |

**HeartbeatConfig フィールド（補足）:**

|フィールド                    |型        |デフォルト |説明                          |
|-------------------------|---------|-------|----------------------------|
|`interval_minutes`       |`int`    |`30`   |ハートビート間隔（分、`heartbeat.md` ではなく設定駆動）|
|`soft_timeout_seconds`   |`int`    |`300`  |HB セッションに折り返しリマインダを入れるまでの秒数   |
|`hard_timeout_seconds`   |`int`    |`600`  |HB セッションを強制終了するまでの秒数          |
|`max_turns`              |`int\|null`|`null`|HB 専用 `max_turns`（未設定時は Anima 側設定）|
|`default_model`          |`str\|null`|`null`|heartbeat/cron のグローバル背景モデル（未設定時は各 Anima の `background_model` / メイン）|
|`msg_heartbeat_cooldown_s`|`int`   |`300`  |メッセージ起因 HB のクールダウン（秒）        |
|`cascade_window_s`       |`int`    |`1800` |カスケード検出のスライディングウィンドウ（秒）    |
|`cascade_threshold`    |`int`    |`3`    |ウィンドウ内の往復上限（ペアあたり）          |
|`depth_window_s`         |`int`    |`600`  |双方向深さ制限のウィンドウ（秒）             |
|`max_depth`              |`int`    |`6`    |双方向やり取りの深さ上限                  |
|`actionable_intents`     |`list`   |`["report","question"]`|メッセージ起因 HB の対象 intent        |
|`idle_compaction_minutes`|`float`|`10`   |最終ストリーム終了から何分後にアイドル時 auto-compact するか|
|`enable_read_ack`        |`bool`   |`false`|既読 ACK 送信（既定オフ、感謝ループ抑制）      |
|`channel_post_cooldown_s`|`int`   |`300`  |同一 Anima の Board 投稿間隔の下限（秒、0 で無制限）|
|`max_messages_per_hour`, `max_messages_per_day`|`int`|`30` / `100`|※レートは `ROLE_OUTBOUND_DEFAULTS` + status.json を優先。項目は後方互換で残存|

### 3.4 HTTP サーバー（`server/`）の構成と挙動

`server/app.py` が FastAPI アプリを組み立てる。主な要素は次のとおり。

**ルーティング**

- `create_router()`（`server/routes/__init__.py`）が **`/api` 配下**に REST をまとめる（animas, chat, channels, memory, sessions, system, config, logs, assets, internal, auth, users, room, webhooks など）。
- **`/ws`** — ダッシュボード用 WebSocket（`websocket_route.py` → `WebSocketManager`）。
- **`/ws/voice/{name}`** — 音声チャット（`voice.py`、`/api` プレフィックス外でマウント）。
- **`create_setup_router()`** — 初回セットアップ **`/api/setup/*`**。完了後はミドルウェアが 403 でブロックし、通常 UI からは `/setup` を `/` にリダイレクトする。

**静的ファイル**

- **`/setup`** — `static/setup/`（ウィザード専用）。
- **`/`** — `static/` 全体（`index.html` ほか）。`.js` / `.css` / `.html` および `/`・`/workspace` には **Cache-Control: no-cache** を付与し、更新反映を容易にする。

**ミドルウェア（適用順のイメージ）**

1. **RequestLoggingMiddleware** — 純 ASGI。`X-Request-ID`（または生成 ID）を structlog コンテキストに束縛。`/api/system/health` 等はログ抑制。
2. **static_cache_control** — 上記キャッシュヘッダ。
3. **setup_guard** — `config.setup_complete` が false の間は `/api/setup` と `/setup` 以外をブロックし `/setup/` へ誘導。
4. **auth_guard** — `core.auth` の `auth_mode`。`local_trust` は無認証。それ以外は `/api/*`・`/ws`・`/ws/*` でセッション Cookie を検証（例外: ログイン、セットアップ、**公開アイコン** `GET /api/animas/{name}/assets/icon*.png`、**検証済み localhost** かつ `trust_localhost` 時）。

**ライフサイクル（`setup_complete` 時）**

- 起動時: **`permissions.global.json`** を読み込み（欠落時は致命的エラーで終了）。`WebSocketManager` ハートビート、`StreamRegistry` のクリーンアップループ開始。
- **APScheduler（サーバプロセス内）**: 孤児 Anima 検出、定期アセット調整、Claude CLI/SDK 自動更新チェック、グローバル権限ファイル整合性チェックなど。
- **バックグラウンドタスク**（サーバ起動直後から UI 応答優先）: 全 Anima 子プロセス起動、frontmatter 移行、組織同期、Slack Socket Mode、`ConfigReloadManager` 登録、アセットリコンシル。
- 子プロセス向け環境変数 **`ANIMAWORKS_EMBED_URL` / `ANIMAWORKS_VECTOR_URL`** — RAG の埋め込み・ベクトル操作を **HTTP でサーバに集約**（子プロセス側のモデル常駐を抑える）。

**内部 API（`routes/internal.py`、プレフィックス `/api` 経由）**

- **`POST /internal/embed`** — 埋め込み推論（子プロセスから HTTP 呼び出し）。
- **`POST /internal/vector/*`** — Chroma 互換の query/upsert/delete 等（コレクション単位）。
- **`POST /internal/message-sent`** — CLI 等からの送信通知（WebSocket ブロードキャスト等）。
- **`GET /messages/{message_id}`** — 共有 inbox ストアからメッセージ JSON を検索。

**会議室**

- `RoomManager` が `shared/meetings` を永続化。`room.py` が作成・参加者管理・**SSE ストリーミング**付きミーティングチャットを提供（最大参加者数などはリクエストバリデーションで制限）。

-----

## 4. 記憶システム（書庫型）

### 4.1 設計理念

従来のAIエージェントは記憶を機械的に切り詰めてプロンプトに詰め込む（切り詰め型）。これは「直近の記憶しかない前向性健忘」に等しい。

書庫型は異なる。人間が書庫から必要な資料を引き出すように、Digital Animaは **必要な時に必要な記憶だけを自分で検索して取り出す。** 記憶の量に上限はない。コンテキストに入るのは「今必要なもの」だけ。

### 4.2 脳科学モデルとの対応

```
┌─────────────────────────────────────────────────┐
│  ワーキングメモリ（前頭前皮質）                    │
│  = コンテキストウィンドウ                          │
│  容量制限あり。「今考えていること」の一時保持       │
│  → SDKに委譲。追加実装は不要                       │
└──────────────────┬──────────────────────────────┘
                    │ 想起（検索）/ 記銘（書き込み）
┌──────────────────┴──────────────────────────────┐
│  長期記憶（大脳皮質・海馬系）                      │
│                                                    │
│  episodes/   エピソード記憶 — いつ何があったか     │
│  knowledge/  意味記憶 — 学んだ教訓・知識           │
│  procedures/ 手続き記憶 — 作業の手順書             │
└────────────────────────────────────────────────┘
```

### 4.3 記憶ディレクトリの役割

|ディレクトリ                   |脳の対応              |内容              |更新方法                 |
|--------------------------|-------------------|----------------|---------------------|
|`state/`                  |ワーキングメモリの永続部分    |今の状態・未完了タスク      |毎サイクル上書き             |
|`state/conversation.json` |会話記憶              |ローリング会話履歴        |閾値超過時にLLM要約で圧縮      |
|`shortterm/chat/`         |チャット用短期記憶      |コンテキスト引き継ぎ       |セッション切替時に自動外部化     |
|`shortterm/heartbeat/`   |ハートビート用短期記憶   |コンテキスト引き継ぎ       |chat/heartbeat で分離管理       |
|`episodes/`               |エピソード記憶（海馬）      |日別の行動ログ          |日付ファイルに追記           |
|`knowledge/`              |意味記憶（側頭葉皮質）      |教訓・ルール・相手の特性     |トピック別に作成・更新        |
|`procedures/`             |手続き記憶（基底核）       |作業手順書            |必要に応じて改訂           |

### 4.4 記憶操作の概要

- **想起（思い出す）**: 判断の前に必ず書庫を検索する（`knowledge/` → `episodes/` → `procedures/`）
- **記銘（書き込む）**: 行動の後に `episodes/` にログ追記、新知識は `knowledge/` に記録
- **統合（振り返り）**: エピソード記憶から意味記憶への転送（日次/週次で自動実行）
- **忘却**: 低活性チャンクの段階的アーカイブ（3段階: downscaling → reorganization → forgetting）
- **Priming（自動想起）**: 6チャネル並列の自動記憶検索をシステムプロンプトに注入

成功の鍵は「記憶を検索せずに判断するのは禁止」というシステムプロンプトの強い指示。

> 記憶システムの技術的詳細（会話記憶、短期記憶、アクティビティログ、ストリーミングジャーナル、Priming、RAG、統合・忘却の各サブシステム）については [memory.md](memory.md) を参照。

-----

## 5. Identity（自己定義）

Digital Animaが「自分は何者か」を認識する情報。ワーキングメモリに常駐する。

```markdown
# Identity: Tanaka

## 性格特性
- 慎重で、リスクを先に考える
- 詳細志向で、曖昧さを嫌う

## 視点
技術的実現可能性を重視する。「それは本当に動くのか」が常に判断の起点。

## 得意なこと
- バックエンド設計、パフォーマンス最適化

## 苦手なこと
- UI/UXデザインの判断、ユーザーの感情的ニーズの把握
```

-----

## 6. Permissions（権限）

Digital Animaが「何ができるか」の制限。権限の制限は「視野の狭さ」を生み、他者への依存 = 組織の価値を生む。

```markdown
# Permissions: Tanaka

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 使えないツール
WebSearch, WebFetch

## 読める場所
- /project/src/backend/ 配下
- /project/docs/ 配下
- /shared/reports/ 配下

## 書ける場所
- /project/src/backend/ 配下
- /workspace/Tanaka/ 配下

## 見えない場所
- /project/.env
- /project/src/frontend/ 配下（Suzukiの管轄）

## 実行できるコマンド
npm test, npm run build, git diff, git log

## 実行できないコマンド
git push（承認必要）, rm -rf, docker
```

フロントのコードが読めないから「フロント側の制約を教えて」と同僚に聞く必要がある。この「知らないから聞く」が組織の水平コミュニケーション。

-----

## 7. Communication（通信）

### 原則

- テキスト＋ファイル参照のみ。内部状態の直接共有は禁止
- 自分の言葉で圧縮・解釈して伝える。全コンテキストは送らない
- 長い内容はファイルとして置き「ここに置いたから見て」と伝える

### メッセージ構造

```json
{
  "id": "20260213_100000_abc",
  "thread_id": "",
  "reply_to": "",
  "from_person": "Tanaka",
  "to_person": "Suzuki",
  "content": "認証APIの設計を見直した。auth-api-design.md に置いたので確認してほしい。",
  "intent": "report",
  "source": "internal",
  "timestamp": "2026-02-13T10:00:00Z"
}
```

### Intent（メッセージ意図）

`send_message` の `intent` フィールドでメッセージの意図を明示する。DM は意味のある通信にのみ使用し、ack や感謝は Board（共有チャネル）に投稿する。

| intent | 説明 | 用途 |
|--------|------|------|
| `report` | 上位への報告 | 進捗・問題報告（MUST） |
| `delegation` | 部下への委任 | delegate_task と連動 |
| `question` | 質問・相談 | 同僚への連携・調整 |

### Board（共有チャネル）

`shared/channels/{name}.jsonl` に append-only JSONL 蓄積する Slack 型共有チャネル。ack・感謝・FYI・3人以上への通知にはBoardを使用する。

### レート制限（3層）

DM の過剰送信を防ぐ 3 層の制限:

| 層 | 制限 | 実装 |
|----|------|------|
| per-run | 同一宛先再送防止、1 run あたり最大 2 人 | `_replied_to`, `_posted_channels` |
| cross-run | 30通/hour, 100通/day | activity_log sliding window |
| behavior-awareness | 直近送信履歴をPriming経由で注入 | `PrimingEngine._collect_recent_outbound()` |

`ack`, `error`, `system_alert`, `call_human` はレート制限対象外。

### コミュニケーション経路ルール

| 場面 | 宛先 | 備考 |
|------|------|------|
| 進捗・問題報告 | 上司 | MUST |
| タスク委譲 | 直属部下 | delegate_task 使用 |
| 連携・調整 | 同僚（同じ上司） | 直接OK |
| 他部署連絡 | 自分の上司経由 | 直接連絡は原則禁止 |
| 人間への連絡 | call_human | トップレベルAnimaの責務 |

Suzukiは設計書だけを見る。Tanakaの思考過程や破棄した案は見えない。この情報の非対称性が、異なるバックグラウンドからの新しい視点を可能にする。

-----

## 8. Lifecycle（生命サイクル）

### 8.1 起動トリガーと実行パス

Digital Animaは自分の内部時計を持つ。4つの実行パスは独立したロックを持ち、並行動作可能。

|パス   |ロック |トリガー |役割 |
|-------|------|---------|-----|
|**Chat/Inbox** | `_conversation_lock` / `_inbox_lock` | 人間チャット / Anima DM | メッセージ応答。Inboxは即時・軽量な返信のみ |
|**Heartbeat** | `_background_lock` | 定期巡回（30分） | Observe → Plan → Reflect。実行はしない |
|**Cron** | `_background_lock` | cron.mdスケジュール | Heartbeat同等コンテキストで定時タスク実行 |
|**TaskExec** | `_background_lock` | state/pending/ にタスク出現 | 委譲タスクの実行（最小コンテキスト） |

Heartbeatは状況確認・計画のみを行い、実行が必要なタスクは `state/pending/` にJSON形式で書き出す。TaskExecがこれを3秒ポーリングで検出し、独立LLMセッションで実行する。

### 8.2 ハートビート

一定間隔で「顔を上げて周囲を見渡す」行為。メインのコンテキストを保持したまま実行し、何もなければ何もしない。

```markdown
# Heartbeat: Tanaka

## 実行間隔
30分ごと

## 活動時間
9:00 - 22:00（JST）

## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 自分の作業領域に新しいファイルが置かれていないか
- 何もなければ何もしない（HEARTBEAT_OK）

## 通知ルール
- 緊急と判断した場合のみ関係者に通知
- 同じ内容の通知は24時間以内に繰り返さない
```

### 8.3 cron

自分の時計で決まった時間に決まったことを行う。ハートビートと違い、必ず何かを実行して結果を出す。

cronは外部のスケジューラーや組織構造に依存しない。**各Digital Animaが自分のcronを持つ。** 人間が自分の習慣として毎朝日記を書くのと同じ。

`cron.md` に Markdown + YAML 形式で定義。標準5フィールドcron式（Asia/Tokyo固定）:

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
episodes/から昨日の進捗を確認し、今日のタスクを計画する。

## 週次振り返り
schedule: 0 17 * * 5
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。

## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- **LLM型** (`type: llm`): エージェントが判断・思考を伴って実行（APIコスト発生）
- **Command型** (`type: command`): bash / 内部ツールを確定的に実行（コスト不要）
- **ホットリロード**: cron.md更新で次回実行時に自動リロード

**ハートビートとcronの違い:**

|項目    |ハートビート        |cron          |
|------|--------------|--------------|
|人間での例 |仕事中に時々メールを確認  |毎朝の日課、週次の振り返り |
|コンテキスト|保持する          |保持しない（新規セッション）|
|判断    |「気にすべきことがあるか？」|盲目的に実行する      |
|何もない時 |何もしない         |必ず何かを出力する     |
|所属    |個人の内部         |個人の内部         |

### 8.4 1サイクルの流れ

```
起動（メッセージ or ハートビート or cron）
  ↓
想起: 関連する記憶を書庫から検索
  ↓
思考・行動: Agent Core（S/C/D/G/A/B モード）が処理
  ↓
通信: 結果を要約してテキスト送信 or ファイル作成
  ↓
記銘: 行動ログ・教訓・知識を書き込み
  ↓
状態更新: state/ を更新
  ↓
休止
```

-----

## 9. Injectable Slot（後から注入）

Digital Animaは「空の器」。役割・理念はMarkdownで注入する。

```markdown
# Injection: Tanaka

## 役割
テックリード。技術的意思決定とコードレビューを担当。
担当範囲はバックエンドアーキテクチャ。

## 理念
高品質なソフトウェアを通じてユーザーの課題を解決する。

## 行動規範
- 品質は妥協しない
- シンプルさを追求する
- 迷ったときは「ユーザーにとって何が最善か」に立ち返る

## やらないこと
- 本番DBへの直接アクセス
- フロントエンドの実装（Suzukiに任せる）
- 承認なしのmainブランチへのpush
```

-----

## 10. システムプロンプトの構築

各Markdown・テンプレートを結合して1つのシステムプロンプトを構築する。`core/prompt/builder.py` の `build_system_prompt()` が6グループ構造の中心となり、セクション組み立ては `assembler.py` / `sections.py` 等と連携する。`trigger` パラメータ（`chat` / `inbox` / `heartbeat` / `cron` / `task`）で実行パスに応じたセクション選択が行われる。

```
Group 1: 動作環境と行動ルール
  - environment.md（ガードレール・フォルダ構造）
  - 現在時刻（JST）
  - behavior_rules（検索してから判断）
  - tool_data_interpretation.md（ツール結果・Primingの信頼レベル解釈ルール）

Group 2: あなた自身
  - bootstrap.md（初回起動指示、条件付き）
  - company/vision.md（組織ビジョン）
  - identity.md（人格）
  - injection.md（役割・行動指針）
  - specialty_prompt.md（ロール別専門プロンプト）
  - permissions.json（優先）／未移行時は permissions.md からの自動移行結果（ツール・ファイル・コマンド許可）

Group 3: 現在の状況
  - state/current_state.md + pending.md（進行中タスク）
  - Task Queue（永続タスクキュー、条件付き）
  - Resolution Registry（解決済み問題、直近7日、条件付き）
  - Recent Outbound（直近2時間の送信履歴、最大3件）
  - Priming（RAG自動想起結果、条件付き）
  - Recent Tool Results（直近ツール結果、条件付き）

Group 4: 記憶と能力
  - memory_guide（記憶ディレクトリ案内）
  - common_knowledge hint（共有リファレンスヒント、条件付き）
  - 雇用ルール（newstaffスキル保持時、条件付き）
  - ツールガイド（実行モード別）
  - 外部ツールガイド（権限あり時、条件付き）

Group 5: 組織とコミュニケーション
  - hiring context（ソロトップレベル時、条件付き）
  - org context（組織構成ツリー）
  - messaging instructions（メッセージング指示）
  - human notification guidance（トップレベル＆通知有効時、条件付き）

Group 6: メタ設定
  - emotion metadata（表情メタデータ指示）
  - A reflection（Aモード時の自己修正プロンプト、条件付き）
```

**段階的システムプロンプト（Tiered System Prompt）:** コンテキストウィンドウに応じて4段階（T1 FULL 128k+ / T2 STANDARD 32k〜128k / T3 LIGHT 16k〜32k / T4 MINIMAL 16k未満）でプロンプト内容を調整する。

**スキル注入（段階開示）:** チャネル D（skill_match）はマッチしたスキル/手続きの**名前のみ**を返却。全文は `skill` ツールでオンデマンド読み込み。バジェットはメッセージタイプで決定: greeting=500, question=2000, request=3000, heartbeat=200（`PrimingConfig` の既定）。

「記憶を検索せずに判断するのは禁止」を `behavior_rules` に含めることが書庫型記憶の成功の鍵（実験で検証済み）。

-----

## 11. 実装済み機能

- **Digital Animaクラス** — カプセル化・自律動作。1 Anima = 1ディレクトリ
- **6実行モード** — S: Agent SDK / C: Codex CLI / D: Cursor Agent CLI / G: Gemini CLI / A: LiteLLM + tool_use（Anthropic 直フォールバック含む）/ B: Assisted（1ショット補助）
- **バックグラウンドモデル** — heartbeat/inbox/cronはメインモデルとは別の軽量モデルで実行可能（コスト最適化）
- **ProcessSupervisor** — 各AnimaをUnixソケット付き独立子プロセスとして起動・監視
- **書庫型記憶** — episodes / knowledge / procedures / state。詳細は [memory.md](memory.md) 参照
- **Priming（6チャネル）** — sender_profile, recent_activity, related_knowledge, skill_match, pending_tasks, episodes
- **記憶統合・忘却** — 日次/週次のconsolidation、3段階のforgetting（シナプスホメオスタシス仮説ベース）
- **Board/共有チャネル** — Slack型共有チャネル。チャネル投稿・メンション・DM履歴のREST API
- **統一アウトバウンドルーティング** — 宛先名から内部Anima/外部プラットフォーム（Slack/Chatwork）を自動解決して配送
- **ハートビート・cron・TaskExec** — APSchedulerによるスケジュール管理。state/pending/ のタスクはTaskExecが3秒ポーリングで実行
- **Anima間メッセージ** — Messenger経由のテキスト通信。intent制御（report/delegation/question）、3層レート制限
- **スーパーバイザーツール** — 部下を持つAnimaに自動有効化（下記ツール一覧参照）
- **統合設定** — config.json + Pydanticバリデーション。status.json SSoT、models.json で実行モードオーバーライド
- **クレデンシャルVault** — `vault.json` + `vault.key`（PyNaCl SealedBox、`core/config/vault.py`）。ツール: `vault_get` / `vault_store` / `vault_list`
- **共通ツールディレクトリ** — `~/.animaworks/common_tools/*.py` をスキャンし、コアツールと同名でなければ読み込み（`core/tools/__init__.py`）
- **FastAPIサーバー** — REST（`/api`）+ ダッシュボード WebSocket（`/ws`）+ 音声（`/ws/voice/{name}`）+ 初回セットアップウィザード（`/setup`）+ SPA（`#/chat` 等）+ Workspace（`/workspace`）。内部 embed/vector API で子プロセス RAG を集約、会議室 API+SSE、`StreamRegistry` / `ConfigReloadManager`、Slack Socket Mode 統合
- **音声チャット** — WebSocket /ws/voice/{name}。STT（faster-whisper）→ Chat IPC → TTS（VOICEVOX/ElevenLabs/SBV2）
- **Anima生成** — テンプレート / 空白（_blank）/ MDファイル（create --from-md）からの生成
- **スキル段階開示** — マッチしたスキル名のみ注入。全文は `skill` ツールでオンデマンド読み込み
- **外部メッセージング統合** — Slack Socket Mode（リアルタイム双方向）, Chatwork Webhook（受信）
- **永続タスクキュー** — task_queue.jsonl。滞留検知・DAG並列実行（submit_tasks）・委任プロンプト注入
- **解決レジストリ** — shared/resolutions.jsonlによるAnima横断の課題解決追跡
- **人間通知** — call_human 統合。Slack, Chatwork, LINE, Telegram, ntfy チャネル
- **外部ツール** — web_search, x_search, slack, chatwork, gmail, github, google_calendar, google_tasks, discord, notion, machine, transcribe, aws_collector, local_llm, image_gen, call_human 等（`permissions` と `ExternalToolDispatcher` 経由）

### 11.1 内部ツール一覧

フレームワークが提供する内部ツール。Claude Code 互換ツール + AnimaWorks 固有ツールを組み合わせる。Mode S は MCP 経由、Mode C/D/G は各 CLI のツール経路、Mode A/B はネイティブ tool_use 等として提供される。外部連携は `Bash` + `animaworks-tool` CLI 経由が主。

**記憶:**

| ツール | 説明 |
|--------|------|
| `search_memory` | 書庫型記憶の検索（scope: knowledge/episodes/procedures/common_knowledge/all） |
| `read_memory_file` | 記憶ファイルの読み取り |
| `write_memory_file` | 記憶ファイルの書き込み |
| `archive_memory_file` | 記憶ファイルのアーカイブ |
| `report_procedure_outcome` | 手続き記憶の実行結果フィードバック |
| `report_knowledge_outcome` | 知識記憶の有用性フィードバック |

**通信:**

| ツール | 説明 |
|--------|------|
| `send_message` | Anima間DM送信（intent必須: report/delegation/question） |
| `post_channel` | Board（共有チャネル）への投稿 |
| `read_channel` | Board読み取り |
| `manage_channel` | チャネル管理（作成・ACL設定） |
| `read_dm_history` | DM履歴の取得 |
| `call_human` | 人間への通知（Slack/LINE/Telegram/Chatwork/ntfy） |

**タスク:**

| ツール | 説明 |
|--------|------|
| `backlog_task` | タスクキューへの追加（人間由来は最優先） |
| `submit_tasks` | 複数タスクの DAG 投入（依存解決・並列実行） |
| `update_task` | タスク状態の更新 |
| `list_tasks` | タスク一覧（モードによりツールまたは CLI） |

**スキル:**

| ツール | 説明 |
|--------|------|
| `skill` | スキル参照（段階開示: 名前一覧→全文オンデマンド） |
| `create_skill` | 新規スキルの作成 |

**Vault:**

| ツール | 説明 |
|--------|------|
| `vault_get` | シークレットの取得 |
| `vault_store` | シークレットの保存 |
| `vault_list` | シークレット一覧 |

**バックグラウンドタスク:**

| ツール | 説明 |
|--------|------|
| `check_background_task` | バックグラウンドタスクの状態確認 |
| `list_background_tasks` | バックグラウンドタスク一覧 |

**スーパーバイザー**（部下を持つAnimaに自動有効化）:

| ツール | 対象 | 説明 |
|--------|------|------|
| `org_dashboard` | 全配下 | プロセス状態・タスク・アクティビティをツリー表示 |
| `ping_subordinate` | 全配下 | 生存確認（name省略で全員一括） |
| `read_subordinate_state` | 全配下 | current_state.md + pending.md 読み取り |
| `delegate_task` | 直属部下 | タスク委譲（部下キューに追加＋DM送信） |
| `task_tracker` | 自分の委譲タスク | 進捗追跡 |
| `audit_subordinate` | 全配下 | 活動サマリー・エラー頻度・ツール使用統計 |
| `disable_subordinate` | 全配下（子・孫…） | 休止（`_check_descendant`） |
| `enable_subordinate` | 全配下 | 再開 |
| `set_subordinate_model` | 全配下 | モデル変更 |
| `set_subordinate_background_model` | 全配下 | バックグラウンドモデル変更 |
| `restart_subordinate` | 全配下 | プロセス再起動 |

**その他:**

| ツール | 説明 |
|--------|------|
| `check_permissions` | 権限チェック |
| `create_anima` | 新規Anima作成（newstaffスキル保持時） |

-----

## 12. 設計判断の記録

|判断                             |理由                                                         |
|-------------------------------|-----------------------------------------------------------|
|記憶はJSON → Markdownファイル         |実験でMarkdownの方がAIが自然に読み書きでき、Grep検索との相性が良いと判明                |
|記憶の忘却はスコアベース → [IMPORTANT]タグ＋統合|シンプルなタグ方式の方が実用的。統合（consolidation）で自然に重要度が整理される             |
|config.md → config.json          |per-anima MDから統合JSONへ。Pydanticバリデーション + per-anima overrides|
|エージェントループは自作しない                |Claude Agent SDKに委譲。車輪の再発明はしない                             |
|実行モード6分岐（S/C/D/G/A/B）        |Claude SDK、Codex/Cursor/Gemini CLI、LiteLLM 汎用、Assisted 弱モデル。全て Anima カプセル内（モデル名パターンで自動解決）|
|agent.pyリファクタリング                |execution/, tooling/, memory/ に分離。ProcessSupervisor で子プロセス起動         |
|lifecycle をパッケージ化                |スケジューラ・Inbox・レート制限・系統 cron/consolidation を `core/lifecycle/` にモジュール分割|
|設定スキーマ拡張                        |`prompt` / `machine` / `workspaces` / `activity_level` / `activity_schedule` / `ui` 等。Vault は config 外の `vault.json`|
|i18n 一元化                            |ユーザー向け文字列は `core/i18n` の `t()` に集約（コード内ハードコード禁止）|
|権限は「視野の制限」                     |知らないことがあるから他者に聞く。全知は組織を無意味にする                              |
|書庫型記憶を採用                       |切り詰め型（直近N件をプロンプトに詰める）では記憶がスケールしない。書庫型なら記憶量に上限がない           |
|cronは「個」の内部時計                  |cronは組織のスケジューラーではなく、各Digital Animaが自分で持つ習慣。人間が自分の日課を持つのと同じ|
