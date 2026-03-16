# Digital Anima 要件定義書 v1.2

## 1. 概要

Digital Animaは、AIエージェントを「1人の人間」としてカプセル化する最小単位。

**核心の設計原則:**

- 内部状態は外部から不可視。外部との接点はテキスト会話のみ
- 記憶は「書庫型」。必要な時に必要な記憶だけを自分で検索して取り出す
- 全コンテキストは共有しない。自分の言葉で圧縮・解釈して伝える
- ハートビートにより指示待ちではなくプロアクティブに行動する
- 役割・理念は後から注入する。Digital Anima自体は「空の器」

**技術方針:**

- エージェント実行は **4モード** で動作する: **S** (Claude Agent SDK — セッション管理をSDKに委任), **A** (Anthropic SDK / LiteLLM + tool_use), **B** (1ショット補助), **C** (Codex CLI wrapper)
- 設定は統合 **config.json**（Pydanticバリデーション）、記憶は **Markdown** で記述する
- 複数Animaが **階層構造** で協調動作する（`supervisor` フィールドで階層定義、同期委任）

-----

## 2. アーキテクチャ

```
┌──────────────────────────────────────────────────────┐
│                   Digital Anima                      │
│                                                       │
│  Identity ──── 自分が誰か（常駐）                      │
│  Agent Core ── 4実行モード                             │
│    ├ S: Claude Agent SDK（セッション管理をSDKに委任）   │
│    ├ A: Anthropic SDK / LiteLLM + tool_use             │
│    │  （Claude, GPT-4o, Gemini等・自律）               │
│    ├ B: 1ショット補助（Ollama等・FW代行）              │
│    └ C: Codex CLI wrapper（Codex経由実行）              │
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
│   ├── lifecycle.py           # ハートビート・cron管理（APScheduler）
│   ├── outbound.py            # 統一アウトバウンドルーティング（Slack/Chatwork/内部自動判定）
│   ├── background.py          # バックグラウンドタスク管理
│   ├── asset_reconciler.py    # アセット自動生成
│   ├── org_sync.py            # 組織構造同期（status.json → config.json）
│   ├── schedule_parser.py     # cron.md/heartbeat.mdパーサー
│   ├── logging_config.py      # ログ設定
│   ├── memory/                # 記憶サブシステム（詳細は memory.md 参照）
│   │   ├── manager.py         #   書庫型記憶の検索・書き込み
│   │   ├── conversation.py    #   会話記憶（ローリング圧縮）
│   │   ├── shortterm.py       #   短期記憶（chat/heartbeat分離）
│   │   ├── activity.py        #   統一アクティビティログ（JSONL時系列）
│   │   ├── streaming_journal.py #  ストリーミングジャーナル（WAL）
│   │   ├── priming.py         #   自動想起レイヤー（6チャネル）
│   │   ├── consolidation.py   #   記憶統合（日次/週次）
│   │   ├── forgetting.py      #   能動的忘却（3段階）
│   │   ├── reconsolidation.py #   記憶再統合
│   │   ├── task_queue.py      #   永続タスクキュー
│   │   ├── resolution_tracker.py # 解決レジストリ
│   │   └── rag/               #   RAGエンジン（ChromaDB + sentence-transformers）
│   │       ├── indexer.py, retriever.py, graph.py, store.py
│   │       └── watcher.py     #   ファイル変更監視
│   ├── supervisor/            # プロセス監視
│   │   ├── manager.py         #   ProcessSupervisor（子プロセス起動・監視）
│   │   ├── ipc.py             #   Unix Domain Socket IPC
│   │   ├── runner.py          #   Animaプロセスランナー
│   │   ├── process_handle.py  #   プロセスハンドル管理
│   │   ├── pending_executor.py #   TaskExec（state/pending/ タスク実行）
│   │   └── scheduler_manager.py #  APScheduler統合管理
│   ├── notification/          # 人間通知
│   │   ├── notifier.py        #   HumanNotifier（call_human統合）
│   │   ├── reply_routing.py   #   返信ルーティング
│   │   └── channels/          #   Slack, Chatwork, LINE, Telegram, ntfy
│   ├── voice/                 # 音声チャットサブシステム
│   │   ├── stt.py             #   VoiceSTT（faster-whisper）
│   │   ├── tts_*.py           #   TTSプロバイダ（VOICEVOX, ElevenLabs, SBV2）
│   │   └── session.py         #   VoiceSession（STT→Chat IPC→TTS）
│   ├── mcp/                   # MCPサーバー（Mode S/C用）
│   │   └── server.py
│   ├── config/                # 設定管理
│   │   ├── models.py          #   Pydantic統合設定モデル
│   │   ├── vault.py           #   クレデンシャルVault（暗号化シークレット管理）
│   │   ├── cli.py             #   configサブコマンド
│   │   └── migrate.py         #   レガシー設定マイグレーション
│   ├── prompt/                # プロンプト・コンテキスト管理
│   │   ├── builder.py         #   システムプロンプト構築（6グループ構造）
│   │   └── context.py         #   コンテキストウィンドウ追跡
│   ├── tooling/               # ツール基盤
│   │   ├── handler*.py        #   ツール実行ディスパッチ・権限チェック（ドメイン別分割）
│   │   ├── schemas.py         #   ツールスキーマ定義
│   │   ├── guide.py           #   動的ツールガイド生成
│   │   ├── dispatch.py        #   外部ツール振り分け
│   │   └── permissions.py     #   権限評価エンジン
│   ├── execution/             # 実行エンジン
│   │   ├── base.py            #   BaseExecutor ABC
│   │   ├── agent_sdk.py       #   Mode S: Claude Agent SDK
│   │   ├── anthropic_fallback.py # Mode A: Anthropic SDK直接
│   │   ├── litellm_loop.py    #   Mode A: LiteLLM + tool_use
│   │   ├── assisted.py        #   Mode B: フレームワーク補助
│   │   ├── codex_sdk.py       #   Mode C: Codex CLI wrapper
│   │   └── _session.py        #   セッション継続・チェイニング
│   └── tools/                 # 外部ツール実装
│       ├── web_search.py, x_search.py, slack.py
│       ├── chatwork.py, gmail.py, github.py
│       ├── google_calendar.py, google_tasks.py, call_human.py
│       ├── transcribe.py, aws_collector.py
│       ├── image_gen.py       #   画像・3Dモデル生成
│       └── local_llm.py
├── cli/                       # CLIパッケージ
│   ├── parser.py              #   argparse定義 + cli_main()
│   └── commands/              #   サブコマンド実装
├── server/
│   ├── app.py                 # FastAPIアプリケーション
│   ├── slack_socket.py        # Slack Socket Modeクライアント
│   ├── routes/                # APIルート（ドメイン別分割）
│   │   ├── animas.py, chat.py, sessions.py
│   │   ├── memory_routes.py, logs_routes.py
│   │   ├── channels.py        #   Board/共有チャネル・DM履歴API
│   │   ├── voice.py           #   音声チャットWebSocket（/ws/voice/{name}）
│   │   ├── system.py, assets.py, config_routes.py
│   │   ├── webhooks.py        #   外部メッセージングWebhook
│   │   └── websocket_route.py
│   ├── websocket.py           # WebSocket管理
│   └── static/                # Web UI
│       ├── index.html         # ダッシュボード
│       ├── modules/           # JSモジュール
│       └── workspace/         # インタラクティブWorkspace（3Dオフィス）
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
|`permissions.json`    |ツール/ファイル/コマンド権限（Pydantic検証JSON。従来のpermissions.mdを置換）|
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

全設定を `~/.animaworks/config.json` に統合。Pydantic `AnimaWorksConfig` モデルでバリデーションし、person単位のオーバーライドをサポートする。

**トップレベル構造:**

|セクション             |説明                              |
|--------------------|--------------------------------|
|`system`            |動作モード、ログレベル                   |
|`credentials`       |プロバイダ別APIキー・エンドポイント（名前付きマップ）   |
|`model_modes`       |※非推奨。`~/.animaworks/models.json` に置換済み。フォールバックとして参照    |
|`model_context_windows`|モデル名パターン→コンテキストウィンドウサイズのオーバーライド（fnmatch）|
|`anima_defaults`    |全Animaに適用されるデフォルト値             |
|`animas`            |組織レイアウト（supervisor, speciality）のみ。モデル設定は status.json SSoT|
|`consolidation`     |記憶統合設定（日次/週次の実行時刻・閾値）         |
|`rag`               |RAG設定（埋め込みモデル、グラフ拡散活性化等）       |
|`priming`           |自動想起設定（メッセージタイプ別トークンバジェット）     |
|`image_gen`         |画像生成設定（スタイル一貫性、Vibe Transfer）    |
|`human_notification`|人間通知設定（チャネル: Slack/LINE/Telegram/Chatwork/ntfy）|
|`server`            |サーバーランタイム設定（IPC、keep-alive、ストリーミング）|
|`external_messaging`|外部メッセージング統合（Slack Socket Mode、Chatwork Webhook）|
|`background_task`   |バックグラウンドツール実行設定（対象ツール・閾値）     |
|`activity_log`      |ログローテーション設定（rotation_mode, max_size_mb, max_age_days）|
|`heartbeat`         |ハートビートスケジュール・カスケード防止設定         |
|`vault`             |クレデンシャルVault設定（暗号化シークレットストア）     |
|`voice`             |音声チャット設定（STT/TTSプロバイダ）                 |
|`housekeeping`      |定期ディスククリーンアップ設定                       |

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
|`execution_mode`                 |`str \| null`  |`null`（自動検出）             |`"S"` / `"A"` / `"B"` / `"C"`。未設定時は models.json または DEFAULT_MODEL_MODE_PATTERNS で解決|
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

**DEFAULT_MODEL_MODE_PATTERNS の主なマッピング:**

| パターン | モード | 説明 |
|---------|-------|------|
| `claude-*` | S | Claude直接 → Agent SDK |
| `codex/*` | C | Codex → CLI wrapper |
| `openai/*`, `azure/*`, `bedrock/*`, `vertex_ai/*`, `google/*` 等 | A | クラウドAPI → LiteLLM + tool_use |
| `ollama/qwen3.5*`, `ollama/glm-4.7*` 等 | A | tool_use 対応 Ollama |
| `ollama/*` | B | その他 Ollama → Basic（1ショット補助） |

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

**PrimingConfig フィールド:**

|フィールド              |型      |デフォルト |説明                          |
|--------------------|-------|-------|----------------------------|
|`dynamic_budget`    |`bool` |`true` |動的バジェット配分の有無                 |
|`budget_greeting`   |`int`  |`500`  |挨拶メッセージ時のトークンバジェット           |
|`budget_question`   |`int`  |`1500` |質問メッセージ時のトークンバジェット           |
|`budget_request`    |`int`  |`3000` |リクエストメッセージ時のトークンバジェット        |
|`budget_heartbeat`  |`int`  |`200`  |ハートビート時のトークンバジェット（フォールバック）|
|`heartbeat_context_pct`|`float`|`0.05`|動的バジェット時のHB用コンテキスト割合（5%）   |

**ServerConfig フィールド:**

|フィールド                      |型       |デフォルト |説明                          |
|----------------------------|--------|-------|----------------------------|
|`ipc_stream_timeout`        |`int`   |`60`   |IPCストリーミングのチャンク単位タイムアウト（秒）    |
|`keepalive_interval`        |`int`   |`30`   |keep-alive送信間隔（秒）             |
|`max_streaming_duration`    |`int`   |`1800` |ストリーミング最大持続時間（秒）             |
|`stream_checkpoint_enabled` |`bool`  |`true` |ストリーミング中のツール結果保存             |
|`stream_retry_max`          |`int`   |`3`    |ストリーム切断時の自動リトライ最大回数          |
|`stream_retry_delay_s`      |`float` |`5.0`  |リトライ間の待機時間（秒）                |

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
|`eligible_tools`      |`dict` |       |対象ツール名 → 閾値（threshold_s）のマップ  |
|`result_retention_hours`|`int` |`24`   |結果保持時間（時間）                   |

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
思考・行動: Agent Core（S/A/B/Cモード）が処理
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

各Markdown・テンプレートを結合して1つのシステムプロンプトを構築する。`core/prompt/builder.py` の `build_system_prompt()` が6グループ構造で組み立てる。`trigger` パラメータ（`chat` / `inbox` / `heartbeat` / `cron` / `task`）で実行パスに応じたセクション選択が行われる。

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
  - permissions.json（ツール・ファイル・コマンド許可）

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
  - ツールガイド（S/A/B/Cモード別）
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

**スキル注入（段階開示）:** チャネル D はマッチしたスキル/手続きの**名前のみ**を返却。全文は `skill` ツールでオンデマンド読み込み。バジェットはメッセージタイプで決定: greeting=500, question=1500, request=3000, heartbeat=200。

「記憶を検索せずに判断するのは禁止」を `behavior_rules` に含めることが書庫型記憶の成功の鍵（実験で検証済み）。

-----

## 11. 実装済み機能

- **Digital Animaクラス** — カプセル化・自律動作。1 Anima = 1ディレクトリ
- **4実行モード** — S: Claude Agent SDK / A: Anthropic SDK or LiteLLM + tool_use / B: Assisted（1ショット補助）/ C: Codex CLI wrapper
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
- **クレデンシャルVault** — 暗号化シークレット管理（vault_get/vault_store/vault_list）
- **FastAPIサーバー** — REST + WebSocket + Web UI（3Dオフィス・会話画面）
- **音声チャット** — WebSocket /ws/voice/{name}。STT（faster-whisper）→ Chat IPC → TTS（VOICEVOX/ElevenLabs/SBV2）
- **Anima生成** — テンプレート / 空白（_blank）/ MDファイル（create --from-md）からの生成
- **スキル段階開示** — マッチしたスキル名のみ注入。全文は `skill` ツールでオンデマンド読み込み
- **外部メッセージング統合** — Slack Socket Mode（リアルタイム双方向）, Chatwork Webhook（受信）
- **永続タスクキュー** — task_queue.jsonl。滞留検知・DAG並列実行（submit_tasks）・委任プロンプト注入
- **解決レジストリ** — shared/resolutions.jsonlによるAnima横断の課題解決追跡
- **人間通知** — call_human 統合。Slack, Chatwork, LINE, Telegram, ntfy チャネル
- **外部ツール** — web_search, x_search, slack, chatwork, gmail, github, google_calendar, google_tasks, transcribe, aws_collector, local_llm, image_gen, call_human

### 11.1 内部ツール一覧

フレームワークが提供する内部ツール。全モード共通の18ツール構成（Claude Code互換8 + AW必須10）。Mode S/C では MCP 経由、Mode A/B ではネイティブ tool_use として提供される。それ以外の機能は `Bash` + `animaworks-tool` CLI 経由でアクセスする。

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
| `submit_tasks` | タスクキューへの追加・複数タスクのDAG投入（依存関係解決・並列実行） |
| `update_task` | タスク状態の更新 |
| タスク一覧 | CLI: `animaworks-tool task list` で取得 |

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
| `disable_subordinate` | 直属部下 | 休止 |
| `enable_subordinate` | 直属部下 | 再開 |
| `set_subordinate_model` | 直属部下 | モデル変更 |
| `set_subordinate_background_model` | 直属部下 | バックグラウンドモデル変更 |
| `restart_subordinate` | 直属部下 | プロセス再起動 |

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
|実行モード4分岐（S/A/B/C）             |Claude SDK最優先、Anthropic SDKフォールバック、LiteLLM汎用、Assisted弱モデル対応、Codex CLI wrapper。全てAnimaカプセル内|
|agent.pyリファクタリング                |execution/, tooling/, memory/ に分離。ProcessSupervisor で子プロセス起動         |
|権限は「視野の制限」                     |知らないことがあるから他者に聞く。全知は組織を無意味にする                              |
|書庫型記憶を採用                       |切り詰め型（直近N件をプロンプトに詰める）では記憶がスケールしない。書庫型なら記憶量に上限がない           |
|cronは「個」の内部時計                  |cronは組織のスケジューラーではなく、各Digital Animaが自分で持つ習慣。人間が自分の日課を持つのと同じ|
