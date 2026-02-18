# AnimaWorks

**Organization-as-Code for LLM Agents**

AnimaWorksは、AIエージェントをツールではなく、組織の自律的なメンバーとして扱うフレームワーク。各エージェント（Digital Anima）は固有のアイデンティティ・記憶・通信チャネルを持ち、自分のスケジュールで自律的に動作する。人間の組織と同じ原理でメッセージパッシングによって協働する。

> *不完全な個の協働が、単一の全能者より堅牢な組織を作る。*

**[English README](README.md)**

## 3つの核心

- **カプセル化** — 内部の思考・記憶は外から見えない。他者とはテキスト会話だけでつながる
- **書庫型記憶** — 記憶をプロンプトに詰め込むのではなく、必要な時に自分で書庫を検索して思い出す
- **自律性** — 指示を待つのではなく、自分の時計（ハートビート・cron）で動き、自分の理念で判断する

## アーキテクチャ

```
┌──────────────────────────────────────────────────────┐
│            Digital Anima: (Alice)                     │
├──────────────────────────────────────────────────────┤
│  Identity ────── 自分が誰か（常駐）                     │
│  Agent Core ──── 4つの実行モード                        │
│    ├ A1: Claude Agent SDK（Claude モデル専用）           │
│    ├ A1 Fallback: Anthropic SDK直接（SDK未インストール時）│
│    ├ A2: LiteLLM + tool_use（GPT-4o, Gemini 等）       │
│    └ B:  LiteLLM テキストベースツールループ（Ollama 等）   │
│  Memory ──────── 書庫型。自律検索で想起                  │
│  Boards ──────── Slack型共有チャネル                     │
│  Permissions ─── ツール/ファイル/コマンド制限             │
│  Communication ─ テキスト＋ファイル参照                  │
│  Lifecycle ───── メッセージ/ハートビート/cron             │
│  Injection ───── 役割/理念/行動規範（注入式）             │
└──────────────────────────────────────────────────────┘
```

## 脳科学にインスパイアされた記憶システム

従来のAIエージェントは記憶を切り詰めてプロンプトに詰め込む（＝直近の記憶しかない健忘）。AnimaWorksの書庫型記憶は、人間が書庫から資料を引き出すように **必要な時に必要な記憶だけを自分で検索して取り出す。**

| ディレクトリ | 脳科学モデル | 内容 |
|---|---|---|
| `episodes/` | エピソード記憶 | 日別の行動ログ |
| `knowledge/` | 意味記憶 | 教訓・ルール・学んだ知識 |
| `procedures/` | 手続き記憶 | 作業手順書 |
| `state/` | ワーキングメモリ | 今の状態・未完了タスク |
| `shortterm/` | 短期記憶 | セッション継続（コンテキスト引き継ぎ） |
| `activity_log/` | 統一活動記録 | 全インタラクションのJSONL時系列ログ |

### 記憶ライフサイクル

- **Priming（自動想起）** — 4チャネル並列の記憶検索をシステムプロンプトに自動注入（送信者プロファイル、直近の活動、関連知識、スキルマッチング）
- **Consolidation（記憶統合）** — 日次（エピソード → 意味記憶、NREM睡眠アナログ）および週次（知識マージ + エピソード圧縮）
- **Forgetting（能動的忘却）** — シナプスホメオスタシス仮説に基づく3段階の忘却:
  1. シナプスダウンスケーリング（日次）: 低アクセスチャンクをマーク
  2. ニューロジェネシス再編（週次）: 類似する低活性チャンクをマージ
  3. 完全忘却（月次）: 非活性チャンクをアーカイブ・削除

## クイックスタート

### 必要環境

- Python 3.12+
- Anthropic API キー

### インストール

```bash
git clone https://github.com/<your-org>/animaworks.git
cd animaworks
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 環境変数

```bash
# 必須
export ANTHROPIC_API_KEY=sk-ant-...

# オプション
export ANIMAWORKS_DATA_DIR=~/.animaworks  # ランタイムデータ（デフォルト: ~/.animaworks）
```

### 初期化と実行

```bash
# ランタイムディレクトリを初期化（初回のみ）
animaworks init

# サーバー起動
animaworks start

# Web UIを開く
# ダッシュボード: http://localhost:18500/
# ワークスペース: http://localhost:18500/workspace/
```

## CLIコマンドリファレンス

### サーバー管理

| コマンド | 説明 |
|---|---|
| `animaworks start [--host HOST] [--port PORT]` | サーバー起動（デフォルト: `0.0.0.0:18500`） |
| `animaworks stop` | サーバー停止（graceful shutdown） |
| `animaworks restart [--host HOST] [--port PORT]` | サーバー再起動 |

### 初期化

| コマンド | 説明 |
|---|---|
| `animaworks init` | ランタイムディレクトリを初期化（対話式セットアップ） |
| `animaworks init --force` | テンプレートの差分マージ（既存データを保持） |
| `animaworks init --from-md PATH [--name NAME]` | MDファイルからAnima作成 |
| `animaworks init --blank NAME` | 空のAnimaスケルトンを作成 |
| `animaworks reset [--restart]` | ランタイムディレクトリをリセット |

### Anima管理

| コマンド | 説明 |
|---|---|
| `animaworks create-anima [--from-md PATH] [--role ROLE] [--name NAME]` | Animaを新規作成 |
| `animaworks anima status [ANIMA]` | Animaプロセスの状態表示 |
| `animaworks anima restart ANIMA` | Animaプロセスを再起動 |
| `animaworks list` | 全Animaを一覧表示 |

### コミュニケーション

| コマンド | 説明 |
|---|---|
| `animaworks chat ANIMA "メッセージ" [--from NAME]` | Animaにメッセージを送信 |
| `animaworks send FROM TO "メッセージ"` | Anima間メッセージ |
| `animaworks heartbeat ANIMA` | ハートビートを手動トリガー |

### 設定・診断

| コマンド | 説明 |
|---|---|
| `animaworks config list [--section SECTION]` | 設定値を一覧表示 |
| `animaworks config get KEY` | 設定値を取得（ドット記法: `system.mode`） |
| `animaworks config set KEY VALUE` | 設定値を変更 |
| `animaworks status` | システムステータス表示 |
| `animaworks logs [ANIMA] [--lines N]` | ログ表示 |

## 実行モード

Animaごとにモデルと実行モードを選択可能。config.jsonで個別に設定する。

| モード | エンジン | 対象モデル | ツール |
|--------|----------|-----------|--------|
| A1 | Claude Agent SDK | Claudeモデル | Read/Write/Edit/Bash/Grep/Glob |
| A1 Fallback | Anthropic SDK直接 | Claude（Agent SDK未インストール時） | search_memory, read/write_file 等 |
| A2 | LiteLLM + tool_use | GPT-4o, Gemini 他 | search_memory, read/write_file, execute_command 等 |
| B | LiteLLM テキストベースツールループ | Ollama 等 | 疑似ツールコール（テキスト解析JSON） |

## 階層とロール

- `supervisor` フィールドのみで階層を定義。supervisor未設定 = トップレベル
- ロールテンプレート（`--role`）で役職別の専門プロンプト・権限・デフォルトパラメータを適用:

| ロール | デフォルトモデル | 用途 |
|--------|----------------|------|
| `engineer` | Opus | 複雑な推論、コード生成 |
| `manager` | Opus | 調整、意思決定 |
| `writer` | Sonnet | コンテンツ作成 |
| `researcher` | Haiku | 情報収集 |
| `ops` | ローカルモデル | ログ監視、定型業務 |
| `general` | Sonnet | 汎用 |

- 全方向の通信（指示・報告・連携）はMessengerによる非同期メッセージング
- 各AnimaはProcessSupervisorが独立子プロセスとして起動し、Unix Domain Socket経由で通信

## Web UI

- `http://localhost:18500/` — ダッシュボード（Anima状態、活動タイムライン、設定）
- `http://localhost:18500/workspace/` — インタラクティブ Workspace（3Dオフィス、会話画面）

## 人格の追加

1人 = 1ディレクトリ。`~/.animaworks/animas/{name}/` にMarkdownファイルを配置する:

```
animas/alice/
├── identity.md          # 性格・得意分野（不変）
├── injection.md         # 役割・理念・行動規範（差替可能）
├── permissions.md       # ツール/ファイル権限
├── heartbeat.md         # 定期チェック間隔
├── cron.md              # 定時タスク（YAML）
├── bootstrap.md         # 初回起動時の自己構築指示
├── status.json          # 有効/無効、ロール、モデル設定
├── specialty_prompt.md  # ロール別専門プロンプト
├── assets/              # キャラクター画像・3Dモデル
├── activity_log/        # 統一活動ログ（日付別JSONL）
└── skills/              # 拡張スキル（YAML frontmatter + Markdown本文）
```

またはMarkdownキャラクターシートから作成:

```bash
animaworks create-anima --from-md character_sheet.md --role engineer --name alice
```

## 技術スタック

| コンポーネント | 技術 |
|---|---|
| エージェント実行 | Claude Agent SDK / Anthropic SDK / LiteLLM |
| LLMプロバイダ | Anthropic, OpenAI, Google, Ollama (via LiteLLM) |
| Webフレームワーク | FastAPI + Uvicorn |
| タスクスケジュール | APScheduler |
| 設定管理 | Pydantic + JSON + Markdown |
| 記憶基盤 | ChromaDB + sentence-transformers（RAG / ベクトル検索） |
| グラフ活性化 | NetworkX（拡散活性化 + PageRank） |
| 人間通知 | Slack, Chatwork, LINE, Telegram, ntfy |
| 外部メッセージング | Slack Socket Mode, Chatwork Webhook |
| 画像生成 | NovelAI, fal.ai (Flux), Meshy (3D) |

## プロジェクト構成

```
animaworks/
├── main.py              # CLIエントリポイント
├── core/                # Digital Animaコアエンジン
│   ├── anima.py         #   カプセル化された人格クラス
│   ├── agent.py         #   実行モード選択・サイクル管理
│   ├── anima_factory.py #   Anima生成（テンプレート/空白/MD）
│   ├── memory/          #   記憶サブシステム
│   │   ├── manager.py   #     書庫型記憶の検索・書き込み
│   │   ├── priming.py   #     自動想起レイヤー（4チャネル並列）
│   │   ├── consolidation.py #  記憶統合（日次/週次）
│   │   ├── forgetting.py #    能動的忘却（3段階）
│   │   └── rag/         #     RAGエンジン（ChromaDB + embeddings）
│   ├── execution/       #   実行エンジン（A1/A1F/A2/B）
│   ├── tooling/         #   ツールディスパッチ・権限チェック
│   ├── prompt/          #   システムプロンプト構築（24セクション）
│   ├── supervisor/      #   プロセス隔離（Unixソケット）
│   └── tools/           #   外部ツール実装
├── cli/                 # CLIパッケージ（argparse + サブコマンド）
├── server/              # FastAPIサーバー + Web UI
│   ├── routes/          #   APIルート（ドメイン別分割）
│   └── static/          #   ダッシュボード + Workspace UI
└── templates/           # デフォルト設定・プロンプトテンプレート
    ├── roles/           #   ロールテンプレート（6種）
    └── anima_templates/ #   Animaスケルトン
```

## 著者について

AnimaWorksは、精神科医として人間の不完全さを診てきた経験と、経営者として複数の組織を動かしてきた経験から生まれた。

「不完全な個の協働が、単一の全能者より堅牢な組織を作る」— これがOrganization-as-Codeの根底にある思想。

## 設計思想の詳細

詳しい設計理念は [vision.md](docs/vision.md)、技術仕様は [spec.md](docs/spec.md) を参照。

## ライセンス

Apache License 2.0。詳細は [LICENSE](LICENSE) を参照。
