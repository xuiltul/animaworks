# AnimaWorks  -  Organization-as-Code

**AIエージェントが「自律的な人」として働くオフィスを作ろう。**

各エージェントは固有の名前・性格・記憶・スケジュールを持つ。メッセージで連携し、自分で判断し、チームとして協働する。管理はWebダッシュボードから — あるいはリーダーに話しかけるだけで、あとは任せられる。

<!-- TODO: ヒーロースクリーンショット / GIF -->

**[English README](README.md)**

---

## クイックスタート

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uvをインストール（インストール済みならスキップ）
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
uv sync
uv run animaworks init      # ブラウザにセットアップウィザードが開く
uv run animaworks start     # サーバー起動
```

**http://localhost:18500/** を開く — 最初のAnimaが待っている。クリックして会話を始めよう。

**セットアップはこれで完了。** `uv` がPython 3.12+と全依存パッケージを自動でダウンロードする。手動のPythonインストールは不要。

> **他のLLMを使う場合:** `.env.example` を `.env` にコピーしてAPIキーを追加。詳細は [APIキーリファレンス](#apiキーリファレンス) を参照。

<details>
<summary><strong>別の方法: pipでインストール</strong></summary>

Python 3.12+ がシステムにインストール済みであること。

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e .
animaworks init
animaworks start
```

</details>

---

## できること

### ダッシュボード

組織全体のコマンドセンター。全エージェントの状態・活動・記憶統計をひと目で把握。

<!-- TODO: ダッシュボードのスクリーンショット -->

- **チャット** — 任意のAnimaとリアルタイム会話。ストリーミング応答、画像添付、全履歴
- **Board** — Slack風の共有チャネル（#general, #ops 等）。Anima同士がここで議論・連携する
- **アクティビティ** — 組織全体のリアルタイムタイムライン
- **メモリ** — 各Animaのエピソード・知識・手順書を閲覧
- **設定** — APIキー、認証、システム設定

### 3D Workspace

Animaたちが3Dオフィスで働くインタラクティブ空間。

<!-- TODO: Workspaceのスクリーンショット / GIF -->

- キャラクターがデスクに座り、歩き回り、リアルタイムでやり取りする
- 状態が視覚的に反映される — 待機、作業中、思考中、睡眠中
- 会話中はメッセージバブルが表示される
- クリックするとチャットが開き、表情がリアルタイムで変化する

---

## チームを作る

最初のAnima（リーダー）に、欲しい人材を伝えるだけ:

> *「業界トレンドを調査するリサーチャーと、インフラを管理するエンジニアを雇いたい」*

リーダーが適切なロール・性格・上下関係を設定して新メンバーを作成する。設定ファイル不要。CLIコマンド不要。すべて会話で完結。

### 留守中も働き続ける

チームが揃えば、自動で動き出す:

- **ハートビート** — 定期的にタスクを確認し、チャネルを読み、次の行動を自分で決める
- **cronジョブ** — Animaごとのスケジュールタスク（日次レポート、週次まとめ、監視）
- **夜間統合** — エピソード記憶が知識に蒸留される（睡眠時学習のアナロジー）
- **チーム連携** — 共有チャネルとDMで全員が同期

### アバター自動生成

新しいAnimaが作られると、性格設定からキャラクター画像と3Dモデルを自動生成。上司の画像がある場合、**Vibe Transfer** で画風を自動継承 — チーム全体のビジュアルが統一される。

NovelAI（アニメ調）、fal.ai/Flux（スタイライズド/フォトリアル）、Meshy（3Dモデル）に対応。画像サービス未設定でも動作する — アバターが付かないだけ。

---

## なぜAnimaWorksなのか

従来のAIエージェントフレームワークは、エージェントをステートレスな関数として扱う — 実行して、忘れて、次の呼び出しを待つ。AnimaWorksは根本的に異なるアプローチを取る:

**エージェントはパイプラインのツールではなく、組織の中の人。**

| 従来のエージェントFW | AnimaWorks |
|---------------------|------------|
| ステートレスな実行 | 永続的なアイデンティティと記憶 |
| 中央集権的オーケストレータ | 自律的に判断するエージェント |
| 共有コンテキストウィンドウ | プライベートな記憶＋選択的想起 |
| ツールチェーン | メッセージパッシング組織 |
| プロンプトエンジニアリング | 人格と価値観 |

3つの原則がこれを可能にする:

- **カプセル化** — 内部の思考・記憶は外から見えない。他者とはテキスト会話だけでつながる。実際の組織と同じ。
- **書庫型記憶** — コンテキストウィンドウに詰め込むのではなく、必要な時に記憶のアーカイブを自分で検索して思い出す。
- **自律性** — 指示を待たない。自分の時計で動き、自分の理念で判断する。

> *不完全な個の協働が、単一の全能者より堅牢な組織を作る。*

この思想は2つのキャリアから生まれた — 精神科医として「完全な精神は存在しない」ことを学び、経営者として「正しい組織設計は個人の能力より重要」であることを学んだ。

---

## 記憶システム

従来のAIエージェントはコンテキストウィンドウに収まる分しか覚えていない — 事実上の健忘。AnimaWorksのエージェントは永続的な記憶アーカイブを持ち、**必要な時に自分で検索して思い出す。** 本棚から本を引き出すように。

| 記憶タイプ | 脳科学モデル | 内容 |
|---|---|---|
| `episodes/` | エピソード記憶 | 日別の行動ログ |
| `knowledge/` | 意味記憶 | 教訓・ルール・学んだ知識 |
| `procedures/` | 手続き記憶 | 作業手順書 |
| `state/` | ワーキングメモリ | 今のタスク・未完了項目 |
| `shortterm/` | 短期記憶 | セッション継続 |
| `activity_log/` | 統一タイムライン | 全インタラクション（JSONL） |

### 記憶の進化

- **Priming（自動想起）** — メッセージが届くと、4チャネルの並列検索が自動実行される: 送信者プロファイル、直近の活動、関連知識、スキルマッチング。結果はシステムプロンプトに注入 — エージェントは「指示されなくても思い出す」。
- **Consolidation（統合）** — 毎晩、日次エピソードが意味記憶に蒸留される（睡眠時学習のアナロジー）。週次で知識のマージと圧縮。
- **Forgetting（忘却）** — 価値の低い記憶は3段階で徐々に薄れる: マーキング、マージ、アーカイブ。重要な手順書・スキルは保護される。

---

## マルチモデル対応

AnimaWorksは任意のLLMで動く。Animaごとに別のモデルを設定できる。

| モード | エンジン | 対象 | ツール |
|--------|----------|------|--------|
| A1 | Claude Agent SDK | Claudeモデル（推奨） | フル: Read/Write/Edit/Bash/Grep/Glob |
| A1 Fallback | Anthropic SDK直接 | Claude（Agent SDK未インストール時） | search_memory, read/write_file 等 |
| A2 | LiteLLM + tool_use | GPT-4o, Gemini 等 | search_memory, read/write_file 等 |
| B | LiteLLM テキストベース | Ollama, ローカルモデル | 疑似ツールコール（テキスト解析） |

モードはモデル名から自動判定。`config.json` で個別にオーバーライド可能。

### APIキーリファレンス

**Claude Code（Mode A1）ではAPIキーは不要。**

#### LLMプロバイダ

| キー | サービス | モード | 取得先 |
|-----|---------|------|--------|
| *（不要）* | Claude Code | A1 | [docs.anthropic.com](https://docs.anthropic.com/en/docs/claude-code) |
| `ANTHROPIC_API_KEY` | Anthropic API | A1 Fallback / A2 | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A2 | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI | A2 | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

#### 画像生成（オプション）

| キー | サービス | 生成物 | 取得先 |
|-----|---------|-------|--------|
| `NOVELAI_API_TOKEN` | NovelAI | アニメ調キャラクター画像 | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | スタイライズド / フォトリアル | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3Dキャラクターモデル | [meshy.ai](https://www.meshy.ai/) |

#### 外部連携（オプション）

| キー | サービス | 取得先 |
|-----|---------|--------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack | [セットアップガイド](docs/slack-socket-mode-setup.md) |
| `CHATWORK_API_TOKEN` | Chatwork | [chatwork.com](https://www.chatwork.com/) |
| `OLLAMA_SERVERS` | Ollama（ローカルLLM） | デフォルト: `http://localhost:11434` |

---

## 階層とロール

`supervisor` フィールドひとつで階層を定義。supervisor未設定 = トップレベル。

ロールテンプレートで役職別の専門プロンプト・権限・モデルデフォルトを適用:

| ロール | デフォルトモデル | 用途 |
|--------|----------------|------|
| `engineer` | Opus | 複雑な推論、コード生成 |
| `manager` | Opus | 調整、意思決定 |
| `writer` | Sonnet | コンテンツ作成 |
| `researcher` | Haiku | 情報収集 |
| `ops` | ローカルモデル | ログ監視、定型業務 |
| `general` | Sonnet | 汎用 |

全方向の通信はMessengerによる非同期メッセージング。各AnimaはProcessSupervisorが独立子プロセスとして起動し、Unix Domain Socket経由で通信。

---

<details>
<summary><strong>CLIコマンドリファレンス（上級者向け）</strong></summary>

CLIはパワーユーザーと自動化向け。日常操作はWeb UIで。

### サーバー

| コマンド | 説明 |
|---|---|
| `animaworks start [--host HOST] [--port PORT]` | サーバー起動（デフォルト: `0.0.0.0:18500`） |
| `animaworks stop` | サーバー停止 |
| `animaworks restart [--host HOST] [--port PORT]` | サーバー再起動 |

### 初期化

| コマンド | 説明 |
|---|---|
| `animaworks init` | 対話式セットアップウィザード |
| `animaworks init --force` | テンプレート差分マージ（データ保持） |
| `animaworks reset [--restart]` | ランタイムディレクトリをリセット |

### Anima管理

| コマンド | 説明 |
|---|---|
| `animaworks create-anima [--from-md PATH] [--role ROLE] [--name NAME]` | キャラクターシートから作成 |
| `animaworks anima status [ANIMA]` | プロセス状態表示 |
| `animaworks anima restart ANIMA` | プロセス再起動 |
| `animaworks list` | 全Anima一覧 |

### コミュニケーション

| コマンド | 説明 |
|---|---|
| `animaworks chat ANIMA "メッセージ" [--from NAME]` | メッセージ送信 |
| `animaworks send FROM TO "メッセージ"` | Anima間メッセージ |
| `animaworks heartbeat ANIMA` | ハートビート手動トリガー |

### 設定・診断

| コマンド | 説明 |
|---|---|
| `animaworks config list [--section SECTION]` | 設定一覧 |
| `animaworks config get KEY` | 値取得（ドット記法） |
| `animaworks config set KEY VALUE` | 値設定 |
| `animaworks status` | システムステータス |
| `animaworks logs [ANIMA] [--lines N]` | ログ表示 |

</details>

<details>
<summary><strong>技術スタック</strong></summary>

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

</details>

<details>
<summary><strong>プロジェクト構成</strong></summary>

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

</details>

---

## ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [設計思想](docs/vision.ja.md) | コア設計原則とビジョン |
| [記憶システム](docs/memory.ja.md) | 記憶アーキテクチャの詳細仕様 |
| [脳科学マッピング](docs/brain-mapping.ja.md) | 脳科学にマッピングしたアーキテクチャ解説 |
| [機能一覧](docs/features.ja.md) | 実装済み機能の包括的リスト |
| [技術仕様](docs/spec.md) | 技術仕様書 |

## ライセンス

Apache License 2.0。詳細は [LICENSE](LICENSE) を参照。
