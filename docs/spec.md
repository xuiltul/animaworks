# Digital Anima 要件定義書 v1.0

## 1. 概要

Digital Animaは、AIエージェントを「1人の人間」としてカプセル化する最小単位。

**核心の設計原則:**

- 内部状態は外部から不可視。外部との接点はテキスト会話のみ
- 記憶は「書庫型」。必要な時に必要な記憶だけを自分で検索して取り出す
- 全コンテキストは共有しない。自分の言葉で圧縮・解釈して伝える
- ハートビートにより指示待ちではなくプロアクティブに行動する
- 役割・理念は後から注入する。Digital Anima自体は「空の器」

**技術方針:**

- エージェント実行は **4モード** で動作する: **A1** (Claude Agent SDK), **A1 Fallback** (Anthropic SDK直接), **A2** (LiteLLM + tool_use), **B** (1ショット補助)
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
│    ├ A1: Claude Agent SDK（Claude専用・自律行動）       │
│    ├ A1 Fallback: Anthropic SDK直接（SDK未インストール時）│
│    ├ A2: LiteLLM + tool_use（GPT-4o, Gemini等・自律） │
│    └ B:  1ショット補助（Ollama等・FW代行）             │
│  Memory ───── 書庫型長期記憶（自律検索で想起）          │
│    ├ 会話記憶（ローリング圧縮）                        │
│    ├ 短期記憶（セッション継続）                        │
│    └ 統一活動ログ（JSONL時系列）                       │
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
│   ├── outbound.py            # アウトバウンドメッセージルーティング
│   ├── schedule_parser.py     # スケジュール記法パーサー
│   ├── logging_config.py      # ログ設定
│   ├── memory/                # 記憶サブシステム
│   │   ├── manager.py         #   書庫型記憶の検索・書き込み
│   │   ├── conversation.py    #   会話記憶（ローリング圧縮）
│   │   ├── shortterm.py       #   短期記憶（セッション継続）
│   │   ├── activity.py        #   統一アクティビティログ（JSONL時系列）
│   │   ├── streaming_journal.py #  ストリーミングジャーナル（WAL）
│   │   ├── priming.py         #   自動想起レイヤー
│   │   ├── consolidation.py   #   記憶統合（日次/週次）
│   │   └── forgetting.py      #   能動的忘却（3段階）
│   ├── config/                # 設定管理
│   │   ├── models.py          #   Pydantic統合設定モデル
│   │   ├── cli.py             #   configサブコマンド
│   │   └── migrate.py         #   レガシー設定マイグレーション
│   ├── prompt/                # プロンプト・コンテキスト管理
│   │   ├── builder.py         #   システムプロンプト構築（24セクション）
│   │   └── context.py         #   コンテキストウィンドウ追跡
│   ├── tooling/               # ツール基盤
│   │   ├── handler.py         #   ツール実行ディスパッチ・権限チェック
│   │   ├── schemas.py         #   ツールスキーマ定義
│   │   ├── guide.py           #   動的ツールガイド生成
│   │   └── dispatch.py        #   外部ツール振り分け
│   ├── execution/             # 実行エンジン
│   │   ├── base.py            #   BaseExecutor ABC
│   │   ├── agent_sdk.py       #   Mode A1: Claude Agent SDK
│   │   ├── anthropic_fallback.py # Mode A1 Fallback: Anthropic SDK直接
│   │   ├── litellm_loop.py    #   Mode A2: LiteLLM + tool_use
│   │   ├── assisted.py        #   Mode B: フレームワーク補助
│   │   └── _session.py        #   セッション継続・チェイニング
│   └── tools/                 # 外部ツール実装
│       ├── web_search.py, x_search.py, slack.py
│       ├── chatwork.py, gmail.py, github.py
│       ├── transcribe.py, aws_collector.py
│       ├── image_gen.py       #   画像・3Dモデル生成
│       └── local_llm.py
├── cli/                       # CLIパッケージ
│   ├── parser.py              #   argparse定義 + cli_main()
│   └── commands/              #   サブコマンド実装
├── server/
│   ├── app.py                 # FastAPIアプリケーション
│   ├── routes/                # APIルート（ドメイン別分割）
│   │   ├── animas.py, chat.py, sessions.py
│   │   ├── memory_routes.py, logs_routes.py
│   │   ├── channels.py       #   Board/共有チャネル・DM履歴API
│   │   ├── system.py, assets.py, config_routes.py
│   │   ├── webhooks.py        #   外部メッセージングWebhook
│   │   └── websocket_route.py
│   ├── websocket.py           # WebSocket管理
│   └── static/                # Web UI
│       ├── index.html         # ダッシュボード
│       ├── modules/           # JSモジュール（activity, animas, api, app,
│       │                      #   chat, history, login, memory, router,
│       │                      #   state, status, touch, websocket）
│       └── workspace/         # インタラクティブWorkspace
├── templates/
│   ├── prompts/               # プロンプトテンプレート
│   ├── anima_templates/       # Anima雛形（_blank）
│   ├── roles/                 # ロールテンプレート（engineer, researcher, manager, writer, ops, general）
│   └── company/               # 組織ビジョンテンプレート
├── main.py                    # CLIエントリポイント
└── tests/                     # テストスイート
```

### 3.1 Animaディレクトリ（`~/.animaworks/animas/{name}/`）

各Animaは以下のファイル・ディレクトリで構成される:

|ファイル / ディレクトリ      |説明                              |
|----------------------|--------------------------------|
|`identity.md`         |性格・得意分野（不変ベースライン）          |
|`injection.md`        |役割・理念・行動規範（差替可能）            |
|`permissions.md`      |ツール/ファイル/コマンド権限              |
|`heartbeat.md`        |定期チェック間隔・活動時間               |
|`cron.md`             |定時タスク（YAML）                   |
|`bootstrap.md`        |初回起動時の自己構築指示                |
|`status.json`         |有効/無効、ロール、モデル設定             |
|`specialty_prompt.md` |ロール別専門プロンプト                  |
|`assets/`             |キャラクター画像・3Dモデル              |
|`transcripts/`        |会話トランスクリプト                   |
|`skills/`             |個人スキル（YAML frontmatter + Markdown本文）|
|`activity_log/`       |統一活動ログ（日付別JSONL）            |
|`state/`              |作業記憶（current_task, pending）    |
|`episodes/`           |エピソード記憶（日次ログ）               |
|`knowledge/`          |意味記憶（学習済み知識）                |
|`procedures/`         |手続き記憶（手順書）                   |
|`shortterm/`          |短期記憶（セッション継続）               |

### 3.2 config.json（統合設定）

全設定を `~/.animaworks/config.json` に統合。Pydantic `AnimaWorksConfig` モデルでバリデーションし、person単位のオーバーライドをサポートする。

**トップレベル構造:**

|セクション             |説明                              |
|--------------------|--------------------------------|
|`system`            |動作モード、ログレベル                   |
|`credentials`       |プロバイダ別APIキー・エンドポイント（名前付きマップ）   |
|`model_modes`       |モデル名→実行モード（A1/A2/B）のカスタムマッピング  |
|`anima_defaults`    |全Animaに適用されるデフォルト値             |
|`animas`            |Anima単位のオーバーライド（未指定フィールドはdefaults適用）|
|`consolidation`     |記憶統合設定（日次/週次の実行時刻・閾値）         |
|`rag`               |RAG設定（埋め込みモデル、グラフ拡散活性化等）       |
|`priming`           |自動想起設定（メッセージタイプ別トークンバジェット）     |
|`image_gen`         |画像生成設定（スタイル一貫性、Vibe Transfer）    |
|`human_notification`|人間通知設定（チャネル: Slack/LINE/Telegram/Chatwork/ntfy）|
|`server`            |サーバーランタイム設定（IPC、keep-alive、ストリーミング）|
|`external_messaging`|外部メッセージング統合（Slack Socket Mode、Chatwork Webhook）|
|`background_task`   |バックグラウンドツール実行設定（対象ツール・閾値）     |

**AnimaModelConfig フィールド:**

|フィールド                           |型              |デフォルト                   |説明                        |
|---------------------------------|---------------|------------------------|--------------------------|
|`model`                          |`str`          |`claude-sonnet-4-20250514`|使用するモデル名（bare name、プロバイダprefixなし）|
|`fallback_model`                 |`str \| null`  |`null`                  |フォールバックモデル                  |
|`max_tokens`                     |`int`          |`4096`                  |1回のレスポンスの最大トークン数             |
|`max_turns`                      |`int`          |`20`                    |1サイクルの最大ターン数                |
|`credential`                     |`str`          |`"anthropic"`           |使用するcredentials名             |
|`context_threshold`              |`float`        |`0.50`                  |短期記憶外部化の閾値（コンテキスト使用率）       |
|`max_chains`                     |`int`          |`2`                     |自動セッション継続の最大回数              |
|`conversation_history_threshold` |`float`        |`0.30`                  |会話記憶の圧縮トリガー（コンテキスト使用率）      |
|`execution_mode`                 |`str \| null`  |`null`（自動検出）             |`"autonomous"` or `"assisted"`|
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
    "model": "claude-sonnet-4-20250514",
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
|`budget_heartbeat`  |`int`  |`200`  |ハートビート時のトークンバジェット            |

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
|`shortterm/`              |短期記憶（セッション継続）    |コンテキスト引き継ぎ       |セッション切替時に自動外部化     |
|`episodes/`               |エピソード記憶（海馬）      |日別の行動ログ          |日付ファイルに追記           |
|`knowledge/`              |意味記憶（側頭葉皮質）      |教訓・ルール・相手の特性     |トピック別に作成・更新        |
|`procedures/`             |手続き記憶（基底核）       |作業手順書            |必要に応じて改訂           |

### 4.4 記憶操作

**想起（思い出す）** — 判断の前に必ず書庫を検索する。

1. `knowledge/` をキーワード検索（相手名、トピック等）
1. 必要に応じて `episodes/` も検索（過去に何があったか）
1. 手順が不明なら `procedures/` を確認
1. 関連する記憶を読み込んでから判断する

**記銘（書き込む）** — 行動の後に記憶を更新する。

1. `episodes/YYYY-MM-DD.md` に行動ログを追記
1. 新しく学んだことがあれば `knowledge/` に書き込み
1. 重要な教訓は `[IMPORTANT]` タグを付けて保護
1. `state/current_task.md` を更新

**統合（振り返り）** — エピソード記憶から意味記憶への転送。脳科学でいう睡眠中の記憶固定化に相当する。

- `episodes/` のログからパターンを抽出し `knowledge/` に一般化して書き出す
- ハートビートまたはcronで定期実行する

### 4.5 エピソード記憶のフォーマット

```markdown
# 2026-02-12 行動ログ

## 09:15 Chatwork未返信チェック
- トリガー: ハートビート
- 判断: 2件の未返信を発見。社内は自分で対応、社外はエスカレーション
- 結果: 返信案を作成し承認を取得
- 教訓: なし

## 14:30 田中さんへの返信
- トリガー: メッセージ受信
- 判断: 過去にカジュアル文面でリジェクトされた記憶あり → フォーマルで作成
- 結果: 承認済み
- 教訓: 建設業界への対応方針が正しいことを再確認
```

### 4.6 知識のフォーマット

```markdown
# 対応方針

## コミュニケーションルール
- [IMPORTANT] 必ずフォーマルなビジネス文面で対応すること
- カジュアルな文面はNG（2026-02-11にリジェクトされた）
- 建設業界はフォーマルなコミュニケーションを重視する

## 連絡先
- 主な担当者: 田中さん
```

### 4.7 実験による検証結果

書庫型記憶は手動テスト（2026-02-12実施）で全5項目S判定を取得済み。

- **想起**: プロンプトに含まれていない過去の記憶を自発的に検索して活用できた
- **記銘**: 行動ログ・教訓・新規知識を適切にファイルに書き込めた
- **Reflexion**: リジェクト（失敗）から教訓を抽出し、次回の判断を変えられた
- **統合**: 個別エピソードからメタパターンを抽出し知識として一般化できた
- **復元**: コンテキストクリア後も `state/` と `episodes/` から状態を復元できた

成功の鍵は「記憶を検索せずに判断するのは禁止」というシステムプロンプトの強い指示。

### 4.8 会話記憶（ConversationMemory）

ローリングチャット履歴。蓄積量が `conversation_history_threshold`（デフォルト30%）を超えると、古いターンをLLM要約で圧縮し、直近ターンは原文保持。`state/conversation.json` に保存。

### 4.9 短期記憶（ShortTermMemory）

セッション継続のための外部化記憶。A2モードでコンテキスト閾値を超えた際、`session_state.json`（機械用）と `session_state.md`（次回プロンプト注入用）を生成。`shortterm/archive/` に自動退避（最大100件）。

### 4.10 統一アクティビティログ（ActivityLogger）

全インタラクションを単一タイムラインとして記録する append-only JSONL ログ。

- **ファイル配置**: `{anima_dir}/activity_log/{date}.jsonl`
- **イベントタイプ**: `message_received`, `response_sent`, `channel_read`, `channel_post`, `dm_received`, `dm_sent`, `human_notify`, `tool_use`, `heartbeat_start`, `heartbeat_end`, `cron_executed`, `memory_write`, `error`
- **エントリフィールド**: `ts`（ISO 8601）, `type`, `content`, `summary`, `from`/`to`, `channel`, `tool`, `via`, `meta`
- **Priming統合**: 直近の活動をシステムプロンプトに注入し、セッション間の文脈継続を実現
- **用途**: Priming層の「直近の活動」チャネルの単一データソース。従来の散在したtranscript, dm_log, heartbeat_historyファイルを統一

### 4.11 ストリーミングジャーナル（StreamingJournal）

クラッシュ耐性のある Write-Ahead Log（WAL）。ストリーミング出力をインクリメンタルにディスクへ永続化する。

- **ファイル配置**: `{anima_dir}/shortterm/streaming_journal.jsonl`
- **ライフサイクル**: `open()` → `write_text()` / `write_tool_*()` → `finalize()`（正常時はファイル削除）
- **クラッシュリカバリ**: 次回起動時に `recover()` で孤立ジャーナルを読み取り、`JournalRecovery` として復元
- **バッファリング**: 1秒間隔 or 500文字でフラッシュ、fsync による永続化保証
- **イベント種別**: `start`（トリガー・セッション情報）, `text`（テキストチャンク）, `tool_start` / `tool_end`, `done`

### 4.12 RAG設定

埋め込みモデルは `config.json` の `rag.embedding_model` で選択可能。デフォルトは `intfloat/multilingual-e5-small`（384次元）。GPU使用やグラフベース拡散活性化もconfig.jsonで設定する。

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
  "type": "message",
  "content": "認証APIの設計を見直した。auth-api-design.md に置いたので確認してほしい。",
  "attachments": [],
  "timestamp": "2026-02-13T10:00:00Z"
}
```

### メッセージタイプ

現在の実装では `type` フィールドのデフォルトは `"message"` の単一型。以下のタイプ分類は将来拡張として設計に残す。

|type（将来拡張） |説明        |
|------------|----------|
|request     |上位からの依頼・指示|
|report      |上位への報告    |
|consultation|同僚への相談    |
|broadcast   |全体通知      |

Suzukiは設計書だけを見る。Tanakaの思考過程や破棄した案は見えない。この情報の非対称性が、異なるバックグラウンドからの新しい視点を可能にする。

-----

## 8. Lifecycle（生命サイクル）

### 8.1 起動トリガー

Digital Animaは自分の内部時計を持つ。3つのトリガーはすべて「個」に属する。

|トリガー   |内容                     |
|-------|-----------------------|
|メッセージ受信|他者からメッセージが届いたら起動       |
|ハートビート |定期的に状況を確認。何もなければ何もしない  |
|cron   |自分の時計で、決まった時間に決まったことを実行|

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

```markdown
# Cron: Tanaka

## 毎朝の業務計画（毎日 9:00 JST）
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は /workspace/Tanaka/daily-plan.md に書き出す。

## 週次振り返り（毎週金曜 17:00 JST）
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
（記憶の統合 = 脳科学でいう睡眠中の記憶固定化）
```

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
思考・行動: Agent Core（A1/A2/Bモード）が処理
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

各Markdown・テンプレートを結合して1つのシステムプロンプトを構築する。`core/prompt/builder.py` の `build_system_prompt()` が24セクションを順に組み立てる。

```
システムプロンプト =
   1. environment（ガードレール・フォルダ構造）
   2. bootstrap（初回起動指示 — 条件付き）
   3. company/vision.md（組織ビジョン）
   4. identity.md（あなたは誰か）
   5. injection.md（役割・理念）
   6. specialty_prompt.md（ロール別専門プロンプト — 条件付き）
   7. permissions.md（何ができるか）
   8. state/current_task.md + pending.md（進行中タスク・未完了タスク）
   9. 直近の活動サマリー（ActivityLoggerから取得）
  10. priming（RAG自動想起 — メッセージタイプ別バジェット）
  11. memory_guide（記憶ディレクトリガイド + ファイル一覧）
  12. common_knowledge（共有リファレンスヒント — 条件付き）
  13. マッチしたスキル全文注入（説明ベースマッチング + バジェット内）
  14. 非マッチ個人スキル（テーブル形式）
  15. 非マッチ共通スキル（テーブル形式）
  16. 雇用ルール（newstaffスキル保持時のみ — 条件付き）
  17. 外部ツールガイド（許可時のみ — 条件付き）
  18. A2 reflection（A2モード時の自己修正プロンプト）
  19. emotion metadata（表情メタデータ指示）
  20. hiring context（他のAnimaが0名の時のみ — 条件付き）
  21. behavior_rules（検索してから判断せよ）
  22. org context（組織構成 — supervisor/subordinates/peers）
  23. messaging instructions（メッセージ送受信 + 同僚Anima一覧）
  24. human notification guidance（トップレベルAnima + 通知有効時のみ — 条件付き）
```

各セクションは `---` で区切られ、条件付きセクション（bootstrap, tools_guide等）は該当時のみ注入される。

スキル注入はメッセージ内容から説明ベースマッチングを行い、マッチしたスキルの全文をバジェット内で注入する（トリガーベーススキル注入）。バジェットはメッセージタイプで決定: greeting=1000, question=3000, request=5000, heartbeat=2000。

「記憶を検索せずに判断するのは禁止」を `behavior_rules` に含めることが書庫型記憶の成功の鍵（実験で検証済み）。

-----

## 11. 実装済み機能

- **Digital Animaクラス** — カプセル化・自律動作。1 Anima = 1ディレクトリ
- **4実行モード** — A1: Claude Agent SDK / A1 Fallback: Anthropic SDK直接 / A2: LiteLLM + tool_use / B: Assisted（1ショット補助）
- **書庫型記憶** — episodes（日別ログ）/ knowledge（教訓・知識）/ procedures（手順書）/ state（作業記憶）
- **会話記憶** — ローリング圧縮。閾値超過時にLLM要約で古いターンを圧縮
- **短期記憶** — セッション継続。コンテキスト閾値超過時にJSON+MDで外部化
- **統一アクティビティログ** — 全インタラクションをJSONL時系列で記録。Priming統合で文脈継続
- **ストリーミングジャーナル（WAL）** — クラッシュ耐性。テキスト・ツール結果のインクリメンタル永続化
- **Board/共有チャネル** — Slack型共有チャネル。チャネル投稿・メンション・DM履歴のREST API
- **統一アウトバウンドルーティング** — 宛先名から内部Anima/外部プラットフォーム（Slack/Chatwork）を自動解決して配送
- **ハートビート・cron** — APSchedulerによるスケジュール管理。日本語スケジュール記法対応
- **Anima間メッセージ** — Messenger経由のテキスト通信。階層委任（supervisor → 配下 同期委任）
- **統合設定** — config.json + Pydanticバリデーション。person単位のオーバーライド
- **FastAPIサーバー** — REST + WebSocket + Web UI（3Dオフィス・会話画面）
- **外部ツール10種** — web_search, slack, chatwork, gmail, github, x_search, transcribe, aws_collector, local_llm, image_gen
- **Anima生成** — テンプレート / 空白（_blank）/ MDファイルからの生成
- **トリガーベーススキル注入** — メッセージ内容から説明ベースマッチング。マッチしたスキル全文をバジェット内で注入
- **外部メッセージング統合** — Slack Socket Mode（リアルタイム双方向）, Chatwork Webhook（受信）
- **埋め込みモデル設定** — config.jsonでRAG用埋め込みモデルを選択可能
- **A1 Fallbackエグゼキューター** — Claude Agent SDK未インストール時にAnthropic SDK直接利用でtool_useループ実行

-----

## 12. 設計判断の記録

|判断                             |理由                                                         |
|-------------------------------|-----------------------------------------------------------|
|記憶はJSON → Markdownファイル         |実験でMarkdownの方がAIが自然に読み書きでき、Grep検索との相性が良いと判明                |
|記憶の忘却はスコアベース → [IMPORTANT]タグ＋統合|シンプルなタグ方式の方が実用的。統合（consolidation）で自然に重要度が整理される             |
|config.md → config.json          |per-anima MDから統合JSONへ。Pydanticバリデーション + per-anima overrides|
|エージェントループは自作しない                |Claude Agent SDKに委譲。車輪の再発明はしない                             |
|実行モード4分岐                       |Claude SDK最優先、Anthropic SDKフォールバック、LiteLLM汎用、Assisted弱モデル対応。全てAnimaカプセル内|
|agent.pyリファクタリング                |1848行→465行。execution/, tool_handler, tool_schemas に分離         |
|権限は「視野の制限」                     |知らないことがあるから他者に聞く。全知は組織を無意味にする                              |
|書庫型記憶を採用                       |切り詰め型（直近N件をプロンプトに詰める）では記憶がスケールしない。書庫型なら記憶量に上限がない           |
|cronは「個」の内部時計                  |cronは組織のスケジューラーではなく、各Digital Animaが自分で持つ習慣。人間が自分の日課を持つのと同じ|
