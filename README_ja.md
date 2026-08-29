# AnimaWorks — Organization-as-Code

### ソフトウェアを出荷するAI組織。

AnimaWorksは、永続的なAIエージェントを「動く組織」にするフレームワークです。ゴールを渡すと、エージェントたちが仕事を分解し、並列のworktreeで実装し、互いにテスト・レビューし、Pull Requestを作り、CIの失敗を修復し、コンフリクトを解消し、デプロイ結果まで見届けます。人間に確認を求めるのは、本当に人間が必要な場面だけです。

```text
タスク → エージェントチーム → 並列worktree → 実装 → テスト → レビュー
     → Pull Request → CI修復 → デプロイ → 観測 → 修復
```

AnimaWorksはこのパイプラインをハードコードしていません。フレームワークが担うのは、GitHubイベントのタスク化、PR単位の直列実行、マルチモデルレビューの編成。残りはエージェントたちが人間のエンジニアと同じやり方で回します — git、テスト、CI、そしてロールごとの作業規約で。だからこそ同じ組織が、メール対応も議事録もSlack投稿もこなせます。ビルドスクリプトではなく、組織だからです。

## 本番での実績

過去6ヶ月、8体のAnimaからなるAnimaWorks組織が、本番SaaSプロダクトの日常的な開発運用を担ってきました:

| 指標（2026年3月〜8月） | 値 |
|---|---|
| エージェントが作成したPull Request | **302件**（267件マージ） |
| エージェントが運用したPull Request — レビュー・CI修復・コンフリクト解消 | **752件**（721件マージ） |
| 組織が自発的に起こしたタスクの割合 | **99.7%**（31,215タスク中、人間起点は92件） |
| GitHubイベントから自動でタスク化された件数（8月のみ） | **2,508件** |

これらの数字は、コミットのauthor情報ではなく一次実行記録（エージェントごとのactivity log・タスクキュー・作業メモ）から集計したものです。共有クレデンシャルの下では人間とエージェントのpushが混ざるためです。証跡が確認できないPRは除外しています。対象リポジトリは非公開のため、公開するのは集計値のみです。

AnimaWorks自身も同じ方法で開発されています。このリポジトリに定義されたエージェントたちが、このリポジトリのPRをレビューし、CIを直し、リリースを出しています。人間の仕事は主に方向づけと例外対応です。

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — リアルタイム組織ツリーとアクティビティフィード" width="720">
  <br><em>Workspaceダッシュボード: 各Animaのロール・ステータス・直近のアクションがリアルタイムで見えます。</em>
</p>

<p align="center">
  <img src="docs/images/pixel-workspace.gif" alt="AnimaWorks ドット絵オフィス — 稼働中の組織のライブビュー" width="720">
  <br><em>ドット絵オフィスはシミュレーションではありません。稼働中の組織のライブビューです — ステータスラベルの一つひとつが、実際に動いているタスクです。</em>
</p>

**[English README](README.md)** | **[简体中文 README](README_zh.md)** | **[한국어 README](README_ko.md)**

---

## :rocket: 今すぐ試す

**Claude Code CLIが入っているか、Codexにログイン済みならAPIキーは不要**です。

まず、ワンライナーでクローンとインストールを実行:

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
```

次に、デモチームを起動:

```bash
uv run animaworks demo
```

**http://localhost:18501** を開けば準備完了。3人のチーム（マネージャー＋エンジニア＋アシスタント）が3日分のアクティビティ履歴付きで動き出します。初回インストールはPython 3.12+とML系依存パッケージをダウンロードするため数分かかりますが、2回目以降のデモ起動は数秒です。[デモの詳細はこちら →](demo/README.ja.md)

> プリセット: `en-business`（既定）/ `en-anime` / `ja-business` / `ja-anime` — 例: `uv run animaworks demo --preset ja-anime`。既存デモのプリセット切替には `--reset` が必要です。デモはクローンしたリポジトリが必要です（pipパッケージには同梱されません）。

自分の組織を作りたいときは `uv run animaworks start` を実行 — 下のセットアップウィザードが最初のエージェント作成を案内します。

---

## クイックスタート

macOS / Linux / WSL:

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv sync --all-extras        # codex/claude実行系のextraを追加
uv run animaworks start     # サーバー起動 — 初回はセットアップウィザードが開きます
```

Windows (PowerShell):

```powershell
git clone https://github.com/xuiltul/animaworks.git
cd animaworks
uv sync --all-extras
uv run animaworks start
```

OpenAI の Codex を APIキーなしで使う場合は、初回起動前に `codex login` を実行してください。

**http://localhost:18500/** を開くと、セットアップウィザードが5ステップで案内します:

1. **言語** — UIの表示言語を選択
2. **ユーザー情報** — オーナーアカウントを作成
3. **プロバイダ認証** — APIキー入力（OpenAIはCodex Loginも可）とアバター画風の選択
4. **最初のAnima** — 最初のエージェントに名前をつける
5. **確認** — 内容を確認して完了

`.env` を手で書く必要はありません。ウィザードが `config.json` に自動保存します。

セットアップスクリプトが [uv](https://docs.astral.sh/uv/) のインストール、リポジトリのクローン、依存パッケージの導入までやってくれます。**macOS、Linux、WSL** では Python の事前インストールなしに動きます。**Windows** は上の PowerShell 手順を使ってください。なお Mode S（Claude Agent SDK）は Windows では利用できません — Codex / Gemini / API系モードを使ってください。

> **`uv sync` には必ず `--all-extras` を付けてください。** `setup.sh` が実行する素の `uv sync` でも本体は動きますが、Mode C（Codex）には `codex` extra が必要です。また後から extras 無しの sync を実行すると、venv から `codex` / `claude` 実行パッケージが消えて該当モードのAnimaが一斉に壊れます。

> **他のLLMを使いたい場合:** Claude、GPT、Gemini、ローカルモデル等に対応しています。セットアップウィザードでAPIキーを入力するか、OpenAI/Codex では **Codex Login** も使えます。後からダッシュボードの **Settings** で変更できます。詳細は [APIキーリファレンス](#apiキーリファレンス) を参照してください。

<details>
<summary><strong>別の方法: スクリプトを確認してから実行</strong></summary>

`curl | bash` を直接実行したくない場合、先にスクリプトの中身を確認してみてください:

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh -o setup.sh
cat setup.sh            # スクリプトの中身を確認
bash setup.sh           # 確認後に実行
```

</details>

<details>
<summary><strong>別の方法: uvでステップごとに手動インストール</strong></summary>

```bash
# uvをインストール（インストール済みならスキップ）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# クローンとインストール
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
uv sync --all-extras    # Python 3.12+と全依存パッケージ（codex/claude extras含む）を自動ダウンロード

# 起動
uv run animaworks start
```

</details>

<details>
<summary><strong>別の方法: pipで手動インストール</strong></summary>

> **macOS ユーザーへ:** macOS Sonoma以前のシステムPython (`/usr/bin/python3`) はバージョン3.9のため、AnimaWorksの要件（3.12+）を満たしません。[Homebrew](https://brew.sh/) で `brew install python@3.13` をインストールするか、上のuvによる方法を使ってください（uvはPythonを自動管理します）。

Python 3.12+（3.12/3.13推奨）がシステムにインストール済みであること。

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
python3 --version       # 3.12+ であることを確認
pip install --upgrade pip && pip install -e .
animaworks start
```

注: 素の `pip install -e .` には Codex extra が含まれません。Mode C を使う場合は `.[codex]` を追加してください。

</details>

---

## ループの仕組み

典型的な変更は、組織の中をこう流れます:

1. **タスクが届く** — 人間から、他のエージェントから、スケジュール（heartbeat / cron）から、あるいはGitHubイベントから。Webhookゲートウェイが、CIの失敗・レビューコメント・`@bot` コマンド・マージコンフリクトを自動でエージェントのタスクに変換します（`gh-ci-*` / `gh-review-*` / `gh-comment-*`）。PR単位の重複排除と再試行上限付きです。
2. **マネージャーが分解する** — `delegate_task` で受け入れ条件・作業場所・排他キーを付けてエンジニアに委譲します。同じPRに触れるタスクは排他キーで直列化され、エージェント同士が同じブランチで衝突することはありません。
3. **エンジニアが隔離worktreeで実装・テストする** — 共有知識として同梱されるロール別の作業規約（PdM / エンジニア / レビュアー / テスター）に従います。この隔離は固定のパイプライン段ではなく、エージェントがgitで実行する運用規約です — だからこそ厄介なケースにも対応できます。
4. **レビューはマルチモデル** — PRごとに、設定された各モデルで1本ずつレビューパスを発行し、全パスの指摘を統合タスクが総合判定します（承認 / 修正要求）。新しいpushが来ると古いレビュータスクは自動キャンセルされ、やり直します。
5. **CIの失敗は仕事として戻ってくる** — 失敗したworkflow runは、PR番号とコミットに紐づく修復タスクとして実装エージェントに届きます（二重発行なし）。実験的なスタンドアロンループ（`python3 -m swe.ci_autofix`）は、修正→lint/テストゲート→レビュー→コミットを回し、3回失敗で人間にエスカレーションします。
6. **デプロイと実行時確認もエージェントの仕事** — エージェントはブランチを隔離環境にデプロイし、ログ・エラー・UIの状態を読んで、テストで捕まらなかった問題を見つけます。本番の組織の活動記録には、数百件のデプロイ・実行時観測アクションが残っています。
7. **人間は例外に介入する** — 組織は、詰まったときや権限を超える判断のときにエスカレーションします（`call_human`）。スーパーバイザプロセスは別途、エージェントの死活を監視し、ハングしたプロセスを再起動し、自身の記憶インデックスを修復します。

人間の役割は「エージェントの操作」から「組織のオーナー」へ移ります。意図を伝え、重要なものをレビューし、例外を判断する。

---

## 他のフレームワークとの違い

|  | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents |
|--|-----------|--------|-----------|----------|---------------|
| **設計思想** | 自律エージェントの組織 | ロールベースのチーム | グラフワークフロー | 個人アシスタント | 軽量SDK |
| **記憶** | 脳科学ベース: ハイブリッドRAG（ベクトル＋BM25＋グラフ）・atomic facts・統合・能動的忘却・自動想起 | Cognitive Memory（手動forget） | チェックポイント＋cross-threadストア | SuperMemory知識グラフ | セッション内のみ |
| **自律性** | Heartbeat（観察→計画→振返り）+ Cron + TaskExec + GitHubイベントゲートウェイ — 24/7稼働 | 人間がキック | 人間がキック | Cron + heartbeat | 人間がキック |
| **組織構造** | 上司→部下の階層・委譲・監査・ダッシュボード | Crew内フラットロール | — | 単一エージェント | Handoffのみ |
| **プロセス** | エージェント毎に独立OSプロセス・IPC・自動再起動 | 共有プロセス | 共有プロセス | 単一プロセス | 共有プロセス |
| **マルチモデル** | 7エンジン: Claude SDK / Codex / Cursor Agent / Gemini CLI / Grok Build / LiteLLM / Assisted — エンジン別のフォールバック連鎖付き | LiteLLM | LangChainモデル | OpenAI互換 | OpenAI中心 |

> AnimaWorksはタスクランナーではありません。考えて、覚えて、忘れて、少しずつ育つ組織です。僕は実際の事業運営の中で、AIのチームとして使いながら開発しています。

---

## できること

### ダッシュボード

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks ダッシュボード — リアルタイム組織図" width="720">
  <br><em>ダッシュボード: 全Animaのリアルタイムステータス付き組織図。</em>
</p>

Web UIは6つの画面（ハッシュルーター `#/…`）とWorkspaceアプリで構成されます:

- **ホーム** — ライブステータス付き組織図、要対応を示す注目チップ、LLM使用量パネル（Claude / OpenAI / nanoGPT）、システムステータスバー、最近の活動、外部タスクウィジェット。各Animaの詳細ページ（overview / process / schedule / memory / assets）はここから開きます
- **チャット** — 好きなAnimaとリアルタイム会話: ストリーミング応答（SSE）、画像添付、マルチスレッド履歴、右側タブ（state / activity / heartbeat / cron）、記憶ブラウザ（episodes / knowledge / procedures）。**ミーティングモード**は最大5名のAnimaを司会者付きで同じ部屋に集めます。チャットタブ長押しで**音声ポップアップ**（しゃべるアニメーションアバター付き）
- **Board** — Slack風の共有チャネルとDM。Anima同士が議論・連携します。ブリッジされたDiscordチャネルもここに表示
- **タスク** — タスクボード: キュー・処理中・保留・抑制・バックグラウンド実行・結果。プライミングにも連動し、今見るべきタスクだけを会話に出します
- **アクティビティ** — 組織全体のSVGスイムレーンタイムライン、ライブtoolティッカー付きNowボード、セッション再生、ログ
- **設定** — 4タブ（general / activity / API・認証 / users）。初回は `/setup/` のウィザード
- **Workspace** — 別タブで開く独立アプリ: **3Dオフィス**（`/workspace/`、組織図ビュー切替・しゃべるバストアップ付き）と**ドット絵オフィス**（`/workspace/pixel/`、ステータスラベルの全てが実タスクのライブ2Dビュー）
- **テーマと言語** — UIテーマ11種＋アニメ/リアル表示モード。セットアップウィザードは17言語、ダッシュボード本体は `ja` / `en` / `ko`

### 組織を作って、任せる

リーダーに「こういう人が欲しい」と伝えると、ロール、性格、上下関係を判断して新しいメンバーを作れます。設定ファイルやCLIを直接触らなくても、会話を起点に組織を育てられます。

チームが揃うと、Animaは自分のスケジュールと記憶を使って継続的に動きます:

- **ハートビート** — 定期的に状況を確認して、次に何をするか自分で判断します
- **cronジョブ** — 日次レポート、週次まとめ、監視。Animaごとに設定でき、LLMタスクとコマンド実行の両方に対応
- **タスク委譲** — マネージャーが受け入れ条件付きでタスクを振り、進捗を追い、報告を受けます
- **並列タスク実行** — 複数タスクを同時投入。独立タスクは並列、排他キーを共有するタスクは順番に実行されます
- **GitHubイベントゲートウェイ** — 監視対象リポジトリのCI失敗・レビューコメント・コンフリクトが自動でタスクになります
- **夜間統合** — 日中のエピソード記憶が、寝ている間に知識へ昇華されます
- **チーム連携** — 共有チャネルとDMで、必要な相手へ状況を共有します

### 記憶システム

従来のAIエージェントは、コンテキストウィンドウに入る分しか覚えていません。AnimaWorksのAnimaはファイルベースの長期記憶を持ち、必要な時に検索して思い出します。すべてを毎回詰め込むのではなく、今の会話や行動に関係する記憶だけを取り出します。

- **自動想起（Priming）** — メッセージが届くと6チャンネルが並列で動きます: 送信者プロファイル、直近活動、重要知識、関連知識、保留タスク、エピソード（Neo4jバックエンドではグラフ文脈も）。取得した記憶は決定論的なゲートが、本文・ポインタ・根拠・抑制のどれで出すかを決めます
- **意図的想起** — 自動想起で足りない時は、Anima自身が `search_memory` や `read_memory_file` で記憶を探します。検索はハイブリッド（ベクトル＋BM25＋atomic facts＋エンティティレジストリ）で、確信度ゲート付きです
- **行動前のアクションルール照合** — 外部送信など副作用のある操作の前に、関連するアクションルールを照合して提示します。必要な記憶を読むまで実行を保留する設定も可能です
- **統合（Consolidation）** — 毎晩、Anima自身が2相のパスを回します（エピソード抽出→自分のツールループでの知識抽出）。フレームワークは後処理としてインデックス再構築と活性度調整を行います。週次では重複・矛盾する知識のマージ候補を提示し、検索インデックスを再構築します
- **忘却（Forgetting）** — 数ヶ月使われない記憶は低活性化マークを経て月次でアーカイブされます。重要な知識と成熟した手順は保護されます。失敗をきっかけに、機能しなくなった手順を改訂する再統合もあります
- **プラグイン式バックエンド** — 安定既定は `legacy`（隔離vector worker経由のChromaDB。破損時は自動隔離・再構築）。Neo4jグラフバックエンド（エンティティ抽出・コミュニティ検出・グラフ想起）は実験的なオプトインです

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks チャット — 複数Animaとのマルチスレッド会話" width="720">
  <br><em>チャット: マネージャーがコード修正をレビューしながら、エンジニアが進捗報告している。</em>
</p>

### マルチモデル対応

どのLLMでも動きます。Animaごとに別のモデルを使い分けられます。

| モード | エンジン | 対象 | ツール |
|--------|----------|------|--------|
| S (SDK) | Claude Agent SDK | Claudeモデル（推奨） | Claude Code 組込み（Read/Write/Edit/Bash/Grep/Glob 等）＋ **stdio MCP**（`mcp__aw__*`）で AnimaWorks 内部ツール。Agent SDK が使えない環境では専用の Anthropic SDK エグゼキュータにフォールバック |
| C (Codex) | Codex CLI（SDK ラッパ） | OpenAI Codex CLIモデル | Codex サンドボックス＋ **AnimaWorks MCP**（`core/mcp/server.py`）で内部ツール |
| D (Cursor) | Cursor Agent CLI | `cursor/*` モデル | MCP統合のエージェントループ |
| G (Gemini CLI) | Gemini CLI | `gemini/*` モデル | stream-json パース・ツールループ |
| X (Grok Build) | Grok Build CLI ラッパー（ACP stdio） | `grok/*` モデル | ACP stdio 経由の Grok Build エージェントループ |
| A (Autonomous) | LiteLLM + tool_use | GPT, Gemini, Mistral, Bedrock, Vertex, xAI, DeepSeek 等 | CC 互換（Read/Write/Edit/Bash/Grep/Glob、**WebSearch/WebFetch**）＋記憶・メッセージ・タスク・**todo_write**・スキル作成など |
| B (Basic) | LiteLLM 1ショット | tool_use が不安定なローカル系（例: 小型 Ollama） | プロンプト内の擬似ツール呼び出しでループ。フレームワークが記憶I/O を代行 |

モード解決は `status.json` の `execution_mode` が最優先、次に `models.json` のテーブル、最後に組み込みのモデル名パターン（`fnmatch`）。tool_use対応のOllamaモデル（例: `ollama/qwen3:14b`, `ollama/glm-4.7*`）はA、それ以外の `ollama/*` はBに割り当てられます。各CLIエンジンにはLiteLLMまで落ちるフォールバック連鎖があります（Codex/Grokはレートガード連動）。Heartbeat・Cron・Inbox はメインとは別の **background_model** で回せます（コスト最適化）。拡張思考（Extended thinking）にも対応しています。

### 音声チャット

ブラウザだけでAnimaと声で会話できます（押して話す or ハンズフリー、WebSocket経由）。

- **STT**: faster-whisper（ストリーミング・LocalAgreement-2の逐次確定）
- **TTS**: VOICEVOX / Style-BERT-VITS2（AivisSpeech） / ElevenLabs / Irodori。Animaごとに声・話速・ピッチを設定可能
- **低遅延フロントレーン** — 小型ローカルモデルが即座に応答し、必要に応じて本体エージェントへの委譲（`ask_anima`）や記憶の読み出しを行います
- **自発発語** — フロントレーン有効時、沈黙が続くとAnimaが自分から話し始めます
- **アニメーションアバター** — 音声ポップアップは疑似Live2Dのバストアップを駆動します（静止画5フレームの瞬き・口パク。リギング・Live2D SDK不使用）

### アバター自動生成

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks アセット管理 — リアリスティックなアバターと表情バリアント" width="720">
  <br><em>性格設定から全身・バストアップ・表情バリアントを自動生成します。上司の画風を自動継承するVibe Transfer付き。</em>
</p>

7ステップのパイプラインが、全身画・7種の表情付きバストアップ・アイコン・ちびキャラ、さらに（アニメ調では）idle/sitting/waving/talkingアニメーション付きのリグ済み3Dモデルまで生成します。バックエンドは NovelAI（アニメ調）、fal.ai/Flux（スタイライズド/フォトリアル）、Meshy（3D）に加え、Codex画像生成とローカルDiffusersに対応。Vibe Transfer（NovelAI）で新しいAnimaが上司の画風を継承できます。画像サービスを設定しなくても本体は動きます。

---

## なぜAnimaWorksなのか

**一人では何もできない。だから、組織を作りました。**

このプロジェクトは、3つのキャリアの交差点から生まれました。

**経営者として** — 僕は「一人では何もできない」ことを知っています。優秀なエンジニアも必要だし、コミュニケーションが得意なスタッフもいます。黙々と働くワーカーもいれば、時折鋭いアイデアを出してくれる人もいます。天才だけでは組織は回りません。多様な力を合わせたとき、一人では成し遂げられなかったことが成し遂げられます。

**精神科医として** — LLMの内部構造を観察したとき、人間の脳と驚くほど似た構造があることに気づきました。想起、学習、忘却、固定化——脳が記憶を処理するメカニズムを、LLMの記憶システムとしてそのまま実装したら、人間の脳を再現できるかもしれない。だったら、LLMを「擬似的な人間」として扱うことができれば、人間と同じように組織を作れるはずです。

**エンジニアとして** — 30年間コードを書いてきました。ロジックを組む楽しさ、自動化の快感を知っています。理想をすべてコードに詰め込めば、僕の理想の組織を作れます。

優れた「単独AI秘書」のフレームワークはすでにたくさんあります。でも、コードで人間に近い単位を作り、それを組織として機能させるプロジェクトはまだ少ないと感じていました。AnimaWorksは、僕自身が事業に組み込み、日々使いながら育てているAI組織です。

> *不完全な個の協働が、単一の全能者より堅牢な組織を作る。*

3つの原則がこれを支えています:

- **カプセル化** — 内部の思考・記憶は外から見えません。他者とはテキスト会話だけでつながります。現実の組織と同じです。
- **RAG記憶（書庫型）** — ウィンドウにすべてを詰め込みません。Priming が RAG で関連チャンクを拾い、エージェントは `search_memory` 等で自分から思い出します。
- **自律性** — 指示を待ちません。自分の時計で動いて、自分の価値観で判断します。

---

<details>
<summary><strong>APIキーリファレンス</strong></summary>

#### LLMプロバイダ

| キー | サービス | モード | 取得先 |
|-----|---------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A / C（Codex Login 時は省略可） | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

**OpenAI Codex（Mode C）** は `OPENAI_API_KEY` を使う方法に加えて、ローカルの **Codex Login**（`codex login`）も利用できます。セットアップウィザードや Settings で選択してください。

**Grok Build（Mode X）** は Grok Build CLI ラッパー（ACP stdio）経由で `grok/*` モデルを利用します。事前に `grok` CLI をインストールし、`grok login` を実行してください。

**Azure OpenAI**、**Vertex AI (Gemini)**、**AWS Bedrock**、**vLLM** は `config.json` の `credentials` セクションで設定します。詳細は[技術仕様](docs/spec.ja.md)を参照してください。

**Ollama** 等のローカルモデルはAPIキー不要です。`OLLAMA_SERVERS`（デフォルト: `http://localhost:11434`）で接続先を指定します。

認証情報は `config.json` の `credentials` → vault → 共有credentialsファイル → 環境変数の順で解決されるため、多くのキーは暗号化vault（`animaworks vault`）にも置けます。

#### 画像生成（オプション）

| キー | サービス | 生成物 | 取得先 |
|-----|---------|-------|--------|
| `NOVELAI_TOKEN` | NovelAI | アニメ調キャラクター画像 | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | スタイライズド / フォトリアル | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3Dキャラクターモデル | [meshy.ai](https://www.meshy.ai/) |

#### 音声チャット（オプション）

| 要件 | サービス | 備考 |
|------|---------|------|
| `pip install animaworks[transcribe]` | STT（faster-whisper） | 初回使用時にモデル自動DL。GPU推奨 |
| VOICEVOX Engineを起動 | TTS（VOICEVOX） | デフォルト: `http://localhost:50021` |
| AivisSpeech/SBV2を起動 | TTS（Style-BERT-VITS2） | デフォルト: `http://localhost:5000` |
| Irodoriサーバーを起動 | TTS（Irodori） | デフォルト: `http://localhost:7861` |
| `ELEVENLABS_API_KEY` | TTS（ElevenLabs） | クラウドAPI（環境変数） |

#### 外部連携（オプション）

| キー | サービス | 取得先 |
|-----|---------|--------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack（ツール＋Socket Mode受信） | [セットアップガイド](docs/slack-socket-mode-setup.ja.md) |
| `CHATWORK_API_TOKEN` | Chatwork（ツール＋Webhook受信） | [chatwork.com](https://www.chatwork.com/) |
| `DISCORD_BOT_TOKEN`（または Anima 単位 `DISCORD_BOT_TOKEN__<名前>`） | Discord（ツール＋Gateway受信＋通知） | [Discord Developer Portal](https://discord.com/developers/applications) |
| `NOTION_API_TOKEN`（または `NOTION_API_TOKEN__<名前>`） | Notion | [Notion integrations](https://www.notion.so/my-integrations) |
| `GITHUB_WEBHOOK_SECRET` ＋ `gh auth login` | GitHub Webhookゲートウェイ（CI/レビュー/コンフリクト→タスク化） | リポジトリ設定 |

Gmail / Google Calendar / Google Sheets / Google Tasks / X検索 / AWSコレクタ / Zoom会議取り込み（RTMS）/ ローカルLLMツールは `config.json` の `credentials`（OAuth またはサービスアカウント）で設定します。人間への通知チャネル: Slack, Chatwork, Discord, LINE, Telegram, ntfy。詳細は [技術仕様](docs/spec.ja.md) を参照してください。

</details>

<details>
<summary><strong>階層とロール</strong></summary>

`supervisor` フィールドひとつで上下関係を定義します。未設定ならトップレベルです。

ロールテンプレートで、役職に応じた専門プロンプト・権限・モデルが自動適用されます:

| ロール | デフォルトモデル | 用途 |
|--------|----------------|------|
| `engineer` | Claude Opus 4.6 | 複雑な推論、コード生成 |
| `manager` | Claude Opus 4.6 | 調整、意思決定 |
| `writer` | Claude Sonnet 4.6 | コンテンツ作成 |
| `researcher` | Claude Sonnet 4.6 | 情報収集 |
| `ops` | Ollama (GLM-4.7) | ログ監視、定型業務 |
| `general` | Claude Sonnet 4.6 | 汎用 |

マネージャーには**スーパーバイザーツール**が自動で付きます。タスク委譲、進捗追跡、部下の再起動/無効化、組織ダッシュボード、部下の状態読み取り——現実の管理職がやることと同じです。

各AnimaはProcessSupervisorが独立プロセスとして起動し、ローカルIPCで通信します（Unix系では Unix socket、Windows では loopback TCP）。

</details>

<details>
<summary><strong>セキュリティ</strong></summary>

自律的に動くエージェントにツールを渡す以上、セキュリティは本気でやる必要があります。実際に仕事で使うので、妥協はできません。AnimaWorksは防御を多層に重ねています:

| レイヤー | 内容 |
|---------|------|
| **信頼境界ラベリング** | 外部データ（Web検索、Slack、メール）は出所でタグ付けされ、セッション中に見た最小信頼度が伝播。untrustedソースからの指示には従わないようモデルに明示 |
| **記憶の出所追跡** | 外部コンテンツ由来の記憶は出所がRAGメタデータまで保持され、想起時もAnima自身の知識と区別されます |
| **コマンドセキュリティ** | シェルインジェクション検出（既定は記録・enforce可） → グローバル禁止リスト（強制。`permissions.global.json` 無しではサーバーが起動しない） → 個別エージェント禁止コマンド → 個別エージェント許可リスト → パストラバーサル検出 |
| **ファイルサンドボックス** | 各エージェントは `permissions.json` で自ディレクトリに閉じ込め。identityと権限ファイル自体は書き込み保護 |
| **プロセス隔離** | エージェントごとに独立OSプロセス。ローカルIPCで通信（Unix socket、Windowsは loopback TCP） |
| **レート制限** | セッション内の宛先重複排除とロール別上限 → 時間・日単位の横断上限（ログが読めない場合はfail-closed） → 直近送信履歴のプロンプト注入による自己認識 |
| **カスケード防止** | 会話深度制限＋カスケード検出。5分クールダウンと遅延処理 |
| **認証・セッション管理** | Argon2idハッシュ、48バイトランダムトークン、最大10セッション、TTLは設定可能 |
| **Webhook検証** | Slack・Chatwork・Zoom・GitHub のHMAC署名検証（リプレイ防止付き） |
| **SSRF緩和** | メディアプロキシがプライベートIPとDNSリバインディングをブロック、HTTPS強制、Content-Type・マジックバイト検証 |
| **アウトバウンドルーティング** | 未知の宛先はfail-closed。明示的な設定なしに任意の外部送信は不可 |
| **エージェント間メッセージの完全性** | 送信者名の名簿照合と、中継メッセージ全件のorigin chain追跡 |

詳細: **[セキュリティアーキテクチャ](docs/security.ja.md)**

</details>

<details>
<summary><strong>CLIコマンドリファレンス（上級者向け）</strong></summary>

CLIはパワーユーザーと自動化向けです。日常操作はWeb UIで十分です。

### サーバー・デモ

| コマンド | 説明 |
|---|---|
| `animaworks start [--host HOST] [--port PORT] [-f]` | サーバー起動（`-f` でフォアグラウンド。既定ポート18500） |
| `animaworks stop [--force]` / `restart` | サーバー停止 / 再起動 |
| `animaworks demo [--preset NAME] [--port PORT] [--reset]` | デモ組織を起動（既定ポート18501・専用データディレクトリ） |

### 初期化

| コマンド | 説明 |
|---|---|
| `animaworks init [--force] [--template NAME] [--from-md PATH] [--blank]` | ランタイムディレクトリを初期化 |
| `animaworks migrate [--dry-run] [--list] [--force] [--resync-db]` | ランタイムデータのマイグレーション（起動時にも自動実行） |
| `animaworks reset [--restart]` | ランタイムディレクトリをリセット |
| `animaworks import hermes\|openclaw --path P [--apply]` | 他フレームワークからのエージェント移行 |

### Anima管理

| コマンド | 説明 |
|---|---|
| `animaworks anima create [--from-md PATH] [--template NAME] [--role ROLE] [--supervisor NAME] [--name NAME]` | 新規作成 |
| `animaworks anima list / info / status / restart / disable / enable` | 確認・制御 |
| `animaworks anima set-model / set-background-model / set-memory-backend / set-role / set-outbound-limit` | Anima単位の設定 |
| `animaworks anima reload [--all]` | status.jsonからホットリロード |
| `animaworks anima delete / rename / merge / merge-finalize` | ライフサイクル操作 |
| `animaworks anima audit [--days N]` / `permissions` / `repair-bootstrap` | 診断 |

### コミュニケーション

| コマンド | 説明 |
|---|---|
| `animaworks chat ANIMA "メッセージ" [--from NAME]` | メッセージ送信 |
| `animaworks send FROM TO "メッセージ"` | Anima間メッセージ |
| `animaworks board read/post/dm-history …` | 共有チャネルの読み書き |
| `animaworks heartbeat ANIMA` | ハートビート手動トリガー |

### 設定・メンテナンス

| コマンド | 説明 |
|---|---|
| `animaworks config list / get KEY / set KEY VALUE` | 設定 |
| `animaworks status` / `logs [ANIMA]` | システムステータス・ログ |
| `animaworks index [--anima NAME] [--full]` | RAGインデックス管理 |
| `animaworks repair-rag --anima NAME --full` / `rag-repair-status` | RAGの隔離・再構築 |
| `animaworks memory status / migrate / backup / rollback / cleanup` | memory backendと記憶データ |
| `animaworks skills install / list / inspect / remove / quarantine` | Skill Hub の操作 |
| `animaworks task add / update / list` | タスクキュー操作 |
| `animaworks vault status / init / get / store / list` | 暗号化credentialボールト |
| `animaworks company create / list / assign / adopt / split / export` | 複数会社の組織管理 |
| `animaworks cost` / `profile` / `models list` / `tmp list/clean` | コスト・プロファイル・モデル・一時ファイル整理 |
| `animaworks mcp --anima NAME` | 外部クライアント向けstdio MCPサーバーを起動 |

### 自動化ヘルパー

`python3 -m swe.ci_autofix` は失敗したCIランを修復する実験的なv0ループです。`gh` で最新の失敗ログを読み、
設定されたArchitectに修正させ、ローカルゲート（ruff / pytest）を通し、Reviewerに判定させてコミットし、
3回失敗したら `call_human` でエスカレーションします。詳細は
[`swe/README.md`](swe/README.md#4-ci-auto-fix-loop-v0)。

</details>

<details>
<summary><strong>技術スタック</strong></summary>

| コンポーネント | 技術 |
|---|---|
| エージェント実行 | Claude Agent SDK / Codex CLI / Cursor Agent CLI / Gemini CLI / Grok Build CLI / Anthropic SDK（フォールバック） / LiteLLM |
| Mode S 連携 | stdio **MCP**（`python -m core.mcp.server`、ツール名 `mcp__aw__*`） |
| LLMプロバイダ | Anthropic, OpenAI, Google, Azure, Vertex AI, AWS Bedrock, Ollama, vLLM ほか（LiteLLM 経由） |
| Webフレームワーク | FastAPI + Uvicorn |
| GitHub連携 | Webhookゲートウェイ（HMAC検証）→タスクディスパッチ、マルチパスレビュー編成、Anima別identity付き `gh` CLIツール |
| リアルタイム | WebSocket（ダッシュボード・音声）、SSE（チャット・ミーティング）、`StreamRegistry` でストリーム寿命管理 |
| タスクスケジュール | APScheduler（ハートビート・cron・統合・死活監視・RAG修復） |
| タスク管理 | タスクキュー（JSONL）＋PR単位排他キー付きpendingタスク実行器＋TaskBoard（SQLite） |
| 記憶基盤 | ChromaDB（隔離vector worker経由）＋BM25＋sentence-transformers＋NetworkX＋atomic facts＋エンティティレジストリ。オプションでNeo4jグラフバックエンド |
| 設定・マイグレーション | Pydantic 2.0+ / JSON / Markdown、`core/migrations/`（起動時マイグレーション） |
| 国際化 | `core/i18n` の `t()`。ウィザード17言語・ダッシュボード ja/en/ko |
| スキル基盤 | Skill Hub、明示的skill activation、router、curator、procedure-to-skill promotion |
| 拡張ツール | `core/tools/*.py` の自動登録に加え、`~/.animaworks/common_tools/` と `animas/<名>/tools/` をスキャン |
| 音声チャット | faster-whisper (STT) + VOICEVOX / SBV2 / ElevenLabs / Irodori (TTS) + ローカルフロントレーンモデル |
| メッセージング | 受信: Slack Socket Mode, Chatwork Webhook, Discord Gateway, Zoom RTMS ／ 人間通知: Slack, Chatwork, Discord, LINE, Telegram, ntfy |
| 画像生成 | NovelAI, fal.ai (Flux), Meshy (3D), Codex画像生成, ローカルDiffusers |
| Workspaceアプリ | Three.js 3Dオフィス＋2Dドット絵オフィス（同じライブイベントストリームで駆動） |

</details>

<details>
<summary><strong>プロジェクト構成</strong></summary>

```
animaworks/
├── main.py              # CLIエントリポイント
├── core/                # Digital Animaコアエンジン
│   ├── anima.py, agent.py  # コアエンティティ・オーケストレーション
│   ├── lifecycle/       # スケジューラ・統合ジョブ・inboxウォッチ等
│   ├── memory/          # 記憶（priming, consolidation, forgetting, RAG, facts, retrieval）
│   ├── skills/          # Skill Hub・activation・router・curator・promotion
│   ├── taskboard/       # TaskBoard ストア・状態・クリーンアップ
│   ├── execution/       # 実行エンジン（S/C/D/G/X/A/B）＋サニタイズ
│   ├── mcp/             # Mode S・外部クライアント向け stdio MCP サーバー
│   ├── platform/        # 子プロセス・ロック・Codex/Cursor/Gemini/Grok 周辺
│   ├── tooling/         # ToolHandler・スキーマ・権限・外部ディスパッチ
│   ├── prompt/          # システムプロンプト構築
│   ├── supervisor/      # ProcessSupervisor・IPC・TaskExec・死活監視・ストリーミング
│   ├── voice/           # 音声チャット（STT + TTS + フロントレーン）
│   ├── config/          # 設定（Pydantic・models.json・グローバル権限）
│   ├── auth/            # UI 認証まわり
│   ├── notification/    # 人間通知チャネル
│   ├── migrations/      # ランタイムデータマイグレーション
│   ├── i18n/            # 翻訳文字列（`t()`）
│   ├── tools/           # 外部ツール実装（slack, discord, gmail, github, …）
│   ├── tasks_dispatch.py, review_multipass.py  # GitHubイベント→タスク配線、マルチモデルレビュー
│   └── …
├── cli/                 # CLIパッケージ（demo含む）
├── server/              # FastAPI + 静的Web UI + Workspaceアプリ
│   ├── app.py           # アプリ生成・lifespan・認証/セットアップガード・静的マウント
│   ├── github_gateway.py, slack_socket.py, discord_gateway.py, zoom_gateway.py
│   ├── routes/          # REST/WebSocketルート（chat, room, voice, webhooks, …）
│   └── static/          # ダッシュボード、setupウィザード、workspace/（3D）、workspace/pixel/
├── swe/                 # 実験的CI自動修復ループ・SWEハーネス
├── demo/                # デモプリセットと同梱履歴
└── templates/           # 初期化テンプレート（ja / en / ko）ロール別作業規約を含む
```

</details>

---

## ドキュメント

**[ドキュメント総合インデックス](docs/README.ja.md)** — 読む順序の案内、アーキテクチャ詳説、設計仕様の一覧。

| ドキュメント | 説明 |
|-------------|------|
| [設計理念](docs/vision.ja.md) | 「不完全な個の協働」という根本思想 |
| [機能一覧](docs/features.ja.md) | AnimaWorksで何ができるかの全体像 |
| [記憶システム](docs/memory.ja.md) | エピソード記憶・意味記憶・手続き記憶・プライミング・能動的忘却 |
| [セキュリティ](docs/security.ja.md) | 多層防御モデル、データ出自追跡、敵対的脅威分析 |
| [脳科学マッピング](docs/brain-mapping.ja.md) | 各モジュールと人間の脳の対応関係 |
| [技術仕様](docs/spec.ja.md) | 実行モード、プロンプト構築、設定解決 |

## ライセンス

Apache License 2.0。詳細は [LICENSE](LICENSE) を参照してください。
