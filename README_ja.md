# AnimaWorks — Organization-as-Code

**一人では何もできない。だから、組織を作りました。**

AIエージェントを「ツール」ではなく「自律的に働く人」として扱うフレームワークです。各Animaは名前も性格も記憶もスケジュールも持っていて、メッセージで連携して、自分で判断して、チームとして動きます。リーダーに話しかけるだけで、あとは勝手に回ります。

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — リアルタイム組織ツリーとアクティビティフィード" width="720">
  <br><em>Workspaceダッシュボード: 各Animaのロール・ステータス・直近のアクションがリアルタイムで見えます。</em>
</p>

<p align="center">
  <img src="docs/images/workspace-demo.gif" alt="AnimaWorks 3Dワークスペース — エージェントが自律的に協働" width="720">
  <br><em>3Dオフィス: Animaたちがデスクに座ったり、歩き回ったり、勝手にメッセージをやり取りしています。</em>
</p>

**[English README](README.md)** | **[简体中文 README](README_zh.md)** | **[한국어 README](README_ko.md)**

---

## 他のフレームワークとの違い

|  | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents |
|--|-----------|--------|-----------|----------|---------------|
| **設計思想** | 自律エージェントの組織 | ロールベースのチーム | グラフワークフロー | 個人アシスタント | 軽量SDK |
| **記憶** | 脳科学ベース: 統合・3段階忘却・6ch自動想起（信頼タグ付き） | Cognitive Memory（手動forget） | チェックポイント＋cross-threadストア | SuperMemory知識グラフ | セッション内のみ |
| **自律性** | Heartbeat（観察→計画→振返り）+ Cron + TaskExec — 24/7稼働 | 人間がキック | 人間がキック | Cron + heartbeat | 人間がキック |
| **組織構造** | 上司→部下の階層・委譲・監査・ダッシュボード | Crew内フラットロール | — | 単一エージェント | Handoffのみ |
| **プロセス** | エージェント毎に独立OSプロセス・IPC・自動再起動 | 共有プロセス | 共有プロセス | 単一プロセス | 共有プロセス |
| **マルチモデル** | 6エンジン: Claude SDK / Codex / Cursor Agent / Gemini CLI / LiteLLM / Assisted | LiteLLM | LangChainモデル | OpenAI互換 | OpenAI中心 |

> AnimaWorksはタスクランナーではありません。考えて、覚えて、忘れて、成長する組織です。チームで業務を支え、会社として運用することが出来ます。私は実際のAI法人として運用しています。

---

## :rocket: 今すぐ試す — Dockerデモ

60秒で動きます。APIキーとDockerだけあれば大丈夫です。

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
cp .env.example .env          # ANTHROPIC_API_KEY を貼り付け
docker compose up              # http://localhost:18500 を開く
```

3人のチーム（マネージャー＋エンジニア＋コーディネーター）がすぐに動き出します。3日分のアクティビティ履歴付きです。[デモの詳細はこちら →](demo/README.ja.md)

> 言語・スタイルの切替: `PRESET=ja-anime docker compose up` — [全プリセット一覧](demo/README.ja.md#プリセット)

---

## クイックスタート

macOS / Linux / WSL:

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start     # サーバー起動 — 初回はセットアップウィザードが開きます
```

Windows (PowerShell):

```powershell
git clone https://github.com/xuiltul/animaworks.git
cd animaworks
uv sync
uv run animaworks start
```

OpenAI の Codex を APIキーなしで使う場合は、初回起動前に `codex login` を実行してください。

**http://localhost:18500/** を開くと、セットアップウィザードが順番に聞いてきます:

1. **言語** — UIの表示言語を選択
2. **ユーザー情報** — オーナーアカウントを作成
3. **プロバイダ認証** — APIキーを入力するか、OpenAI では Codex Login を選択
4. **最初のAnima** — 最初のエージェントに名前をつける

`.env` を手で書く必要はありません。ウィザードが `config.json` に自動保存します。

セットアップスクリプトが [uv](https://docs.astral.sh/uv/) のインストール、リポジトリのクローン、Python 3.12+と全依存パッケージのダウンロードまで全部やってくれます。**macOS、Linux、WSL** では Python の事前インストールなしに動きます。**Windows** は上の PowerShell 手順を使ってください。

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
uv sync                 # Python 3.12+と全依存パッケージを自動ダウンロード

# 起動
uv run animaworks start
```

</details>

<details>
<summary><strong>別の方法: pipで手動インストール</strong></summary>

> **macOS ユーザーへ:** macOS Sonoma以前のシステムPython (`/usr/bin/python3`) はバージョン3.9のため、AnimaWorksの要件（3.12+）を満たしません。[Homebrew](https://brew.sh/) で `brew install python@3.13` をインストールするか、上のuvによる方法を使ってください（uvはPythonを自動管理します）。

Python 3.12+ がシステムにインストール済みであること。

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
python3 --version       # 3.12+ であることを確認
pip install --upgrade pip && pip install -e .
animaworks start
```

</details>

---

## できること

### ダッシュボード

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks ダッシュボード — 19体のAnima組織図" width="720">
  <br><em>ダッシュボード: 4階層・19体のAnimaが稼働中。リアルタイムステータス表示。</em>
</p>

- **チャット** — 好きなAnimaとリアルタイムで会話。ストリーミング応答、画像添付、マルチスレッド、全履歴
- **音声チャット** — ブラウザだけで声で会話（押して話す or ハンズフリー）。VOICEVOX / SBV2 / ElevenLabs対応
- **Board** — Slack風の共有チャネル。Anima同士が勝手に議論・連携します
- **アクティビティ** — 組織全体で何が起きているか、リアルタイムのフィード
- **メモリ** — 各Animaが何を覚えているか覗けます。エピソード・知識・手順書
- **3D Workspace** — Animaたちが3Dオフィスで働いている様子を眺められます
- **多言語対応** — UI 17言語。テンプレートは日本語＋英語で自動フォールバック

### 組織を作って、任せる

リーダーに「こういう人が欲しい」と言うだけで、ロール・性格・上下関係を判断して新メンバーを作ってくれます。設定ファイルもCLIも不要です。会話だけで組織が育ちます。

チームが揃えば、人間がいなくても勝手に動きます:

- **ハートビート** — 定期的に状況を確認して、次に何をするか自分で判断します
- **cronジョブ** — 日次レポート、週次まとめ、監視。Animaごとにスケジュール設定できます
- **タスク委譲** — マネージャーが部下にタスクを振って、進捗を追って、報告を受けます
- **並列タスク実行** — 複数タスクを同時投入。依存関係を解決して独立タスクを並列実行します
- **夜間統合** — 日中のエピソード記憶が、寝ている間に知識へ昇華されます
- **チーム連携** — 共有チャネルとDMで全員が勝手に同期します

### 記憶システム

従来のAIエージェントは、コンテキストウィンドウに入る分しか覚えていません。AnimaWorksのAnimaは永続的な記憶を持っていて、必要な時に自分で検索して思い出します。本棚から本を引くようなものです。

- **自動想起（Priming）** — メッセージが届くと6チャネルの並列検索が走ります。送信者プロファイル、直近の活動、関連知識、スキル、未完了タスク、過去のエピソード。指示しなくても勝手に思い出します
- **統合（Consolidation）** — 毎晩、日中のエピソードが知識に昇華されます。脳科学の睡眠時記憶固定化と同じ仕組みです。解決した問題は自動で手順書になります
- **忘却（Forgetting）** — 使われない記憶は3段階で薄れていきます。マーキング→マージ→アーカイブ。大事な手順書やスキルは保護されます。人間の脳と同じで、忘れることも大事です

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks チャット — 複数Animaとのマルチスレッド会話" width="720">
  <br><em>チャット: マネージャーがコード修正をレビューしながら、エンジニアが進捗報告している。</em>
</p>

### マルチモデル対応

どのLLMでも動きます。Animaごとに別のモデルを使い分けられます。

| モード | エンジン | 対象 | ツール |
|--------|----------|------|--------|
| S (SDK) | Claude Agent SDK | Claudeモデル（推奨） | フル: Read/Write/Edit/Bash/Grep/Glob |
| C (Codex) | Codex SDK | OpenAI Codex CLIモデル | フル: Mode S同等 |
| D (Cursor) | Cursor Agent CLI | `cursor/*` モデル | MCP統合のエージェントループ |
| G (Gemini CLI) | Gemini CLI | `gemini/*` モデル | stream-json パース・ツールループ |
| A (Autonomous) | LiteLLM + tool_use | GPT, Gemini, Mistral, vLLM 等 | Read, Write, search_memory, send_message 等（18ツール統一） |
| B (Basic) | LiteLLM 1ショット | Ollama, 小型ローカルモデル | フレームワークが記憶I/Oを代行 |

モードはモデル名から自動判定されます。Heartbeat・Cron・Inboxはメインとは別の軽量モデルで回せます（コスト最適化）。拡張思考（Extended thinking）にも対応しています。

### アバター自動生成

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks アセット管理 — リアリスティックなアバターと表情バリアント" width="720">
  <br><em>性格設定から全身・バストアップ・表情バリアントを自動生成します。上司の画風を自動継承するVibe Transfer付き。</em>
</p>

NovelAI（アニメ調）、fal.ai/Flux（スタイライズド/フォトリアル）、Meshy（3Dモデル）に対応しています。画像サービスを設定しなくても動きます。アバターが付かないだけです。アバターがつくと、愛着が湧きます（笑）

---

## なぜAnimaWorksなのか

このプロジェクトは、3つのキャリアの交差点から生まれました。

**経営者として** — 僕は「一人では何もできない」ことを知っています。優秀なエンジニアも必要だし、コミュニケーションが得意なスタッフもいます。黙々と働くワーカーもいれば、時折鋭いアイデアを出してくれる人もいます。天才だけでは組織は回りません。多様な力を合わせたとき、一人では成し遂げられなかったことが成し遂げられます。

**精神科医として** — LLMの内部構造を観察したとき、人間の脳と驚くほど似た構造があることに気づきました。想起、学習、忘却、固定化——脳が記憶を処理するメカニズムを、LLMの記憶システムとしてそのまま実装したら、人間の脳を再現できるかもしれない。だったら、LLMを「擬似的な人間」として扱うことができれば、人間と同じように組織を作れるはずです。

**エンジニアとして** — 30年間コードを書いてきました。ロジックを組む楽しさ、自動化の快感を知っています。理想をすべてコードに詰め込めば、僕の理想の組織を作れます。

優れた「単独AI秘書」のフレームワークはすでにたくさんあります。でも、コードで人間を再現して、組織として機能させたプロジェクトはまだありませんでした。AnimaWorksは、僕が実際に自分の事業に組み込みながら育てている、本物の組織そのものです。実際に自分の事業で使いながら、育てています。

> *不完全な個の協働が、単一の全能者より堅牢な組織を作る。*

3つの原則がこれを支えています:

- **カプセル化** — 内部の思考・記憶は外から見えません。他者とはテキスト会話だけでつながります。現実の組織と同じです。
- **書庫型記憶** — コンテキストウィンドウに詰め込みません。必要な時に、自分の記憶を自分で検索して思い出します。
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

**Azure OpenAI**、**Vertex AI (Gemini)**、**AWS Bedrock**、**vLLM** は `config.json` の `credentials` セクションで設定します。詳細は[技術仕様](docs/spec.md)を参照してください。

**Ollama** 等のローカルモデルはAPIキー不要です。`OLLAMA_SERVERS`（デフォルト: `http://localhost:11434`）で接続先を指定します。

#### 画像生成（オプション）

| キー | サービス | 生成物 | 取得先 |
|-----|---------|-------|--------|
| `NOVELAI_API_TOKEN` | NovelAI | アニメ調キャラクター画像 | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | スタイライズド / フォトリアル | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3Dキャラクターモデル | [meshy.ai](https://www.meshy.ai/) |

#### 音声チャット（オプション）

| 要件 | サービス | 備考 |
|------|---------|------|
| `pip install faster-whisper` | STT（Whisper） | 初回使用時にモデル自動DL。GPU推奨 |
| VOICEVOX Engineを起動 | TTS（VOICEVOX） | デフォルト: `http://localhost:50021` |
| AivisSpeech/SBV2を起動 | TTS（Style-BERT-VITS2） | デフォルト: `http://localhost:5000` |
| `ELEVENLABS_API_KEY` | TTS（ElevenLabs） | クラウドAPI |

#### 外部連携（オプション）

| キー | サービス | 取得先 |
|-----|---------|--------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack | [セットアップガイド](docs/slack-socket-mode-setup.ja.md) |
| `CHATWORK_API_TOKEN` | Chatwork | [chatwork.com](https://www.chatwork.com/) |

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
| `ops` | vLLM (GLM-4.7-flash) | ログ監視、定型業務 |
| `general` | Claude Sonnet 4.6 | 汎用 |

マネージャーには**スーパーバイザーツール**が自動で付きます。タスク委譲、進捗追跡、部下の再起動/無効化、組織ダッシュボード、部下の状態読み取り——現実の管理職がやることと同じです。

各AnimaはProcessSupervisorが独立プロセスとして起動し、ローカルIPCで通信します（Unix系では Unix socket、Windows では loopback TCP）。

</details>

<details>
<summary><strong>セキュリティ</strong></summary>

自律的に動くエージェントにツールを渡す以上、セキュリティは本気でやる必要があります。実際に仕事で使うので、妥協はできません。AnimaWorksは10層の多層防御を実装しています:

| レイヤー | 内容 |
|---------|------|
| **信頼境界ラベリング** | 外部データ（Web検索、Slack、メール）はすべて `untrusted` タグ付き — untrustedソースからの指示には従わないようモデルに明示 |
| **5層コマンドセキュリティ** | シェルインジェクション検出 → ハードコードブロックリスト → 個別エージェント禁止コマンド → 個別エージェント許可リスト → パストラバーサル検出 |
| **ファイルサンドボックス** | 各エージェントは自ディレクトリに閉じ込め。`permissions.json` や `identity.md` はエージェント自身が書き換え不可 |
| **プロセス隔離** | エージェントごとに独立OSプロセス。ローカルIPCで通信（Unix socket または loopback TCP） |
| **3層レート制限** | セッション内重複排除 → ロール別送信上限 → 直近送信履歴のプロンプト注入による自己認識 |
| **カスケード防止** | 深度制限＋カスケード検出。5分クールダウンと遅延処理 |
| **認証・セッション管理** | Argon2idハッシュ、48バイトランダムトークン、最大10セッション |
| **Webhook検証** | Slack（リプレイ防止付きHMAC-SHA256）とChatworkの署名検証 |
| **SSRF緩和** | メディアプロキシがプライベートIPをブロック、HTTPS強制、Content-Type検証 |
| **アウトバウンドルーティング** | 未知の宛先はfail-closed。明示的な設定なしに任意の外部送信は不可 |

詳細: **[セキュリティアーキテクチャ](docs/security.ja.md)**

</details>

<details>
<summary><strong>CLIコマンドリファレンス（上級者向け）</strong></summary>

CLIはパワーユーザーと自動化向けです。日常操作はWeb UIで十分です。

### サーバー

| コマンド | 説明 |
|---|---|
| `animaworks start [--host HOST] [--port PORT] [-f]` | サーバー起動（`-f` でフォアグラウンド実行） |
| `animaworks stop [--force]` | サーバー停止 |
| `animaworks restart [--host HOST] [--port PORT]` | サーバー再起動 |

### 初期化

| コマンド | 説明 |
|---|---|
| `animaworks init` | ランタイムディレクトリを初期化（非対話） |
| `animaworks init --force` | テンプレート差分マージ（データ保持） |
| `animaworks migrate [--dry-run] [--list] [--force]` | ランタイムデータのマイグレーション（起動時に自動実行） |
| `animaworks reset [--restart]` | ランタイムディレクトリをリセット |

### Anima管理

| コマンド | 説明 |
|---|---|
| `animaworks anima create [--from-md PATH] [--template NAME] [--role ROLE] [--supervisor NAME] [--name NAME]` | 新規作成 |
| `animaworks anima list [--local]` | 全Anima一覧 |
| `animaworks anima info ANIMA [--json]` | 詳細設定 |
| `animaworks anima status [ANIMA]` | プロセス状態表示 |
| `animaworks anima restart ANIMA` | プロセス再起動 |
| `animaworks anima disable ANIMA` / `enable ANIMA` | 無効化 / 有効化 |
| `animaworks anima set-model ANIMA MODEL` | モデル変更 |
| `animaworks anima set-background-model ANIMA MODEL` | バックグラウンドモデル設定 |
| `animaworks anima reload ANIMA [--all]` | status.jsonからホットリロード |

### コミュニケーション

| コマンド | 説明 |
|---|---|
| `animaworks chat ANIMA "メッセージ" [--from NAME]` | メッセージ送信 |
| `animaworks send FROM TO "メッセージ"` | Anima間メッセージ |
| `animaworks heartbeat ANIMA` | ハートビート手動トリガー |

### 設定・メンテナンス

| コマンド | 説明 |
|---|---|
| `animaworks config list [--section SECTION]` | 設定一覧 |
| `animaworks config get KEY` / `set KEY VALUE` | 値の取得 / 設定 |
| `animaworks status` | システムステータス |
| `animaworks logs [ANIMA] [--lines N] [--all]` | ログ表示 |
| `animaworks index [--reindex] [--anima NAME]` | RAGインデックス管理 |
| `animaworks models list` / `models info MODEL` | モデル一覧・詳細 |

</details>

<details>
<summary><strong>技術スタック</strong></summary>

| コンポーネント | 技術 |
|---|---|
| エージェント実行 | Claude Agent SDK / Codex SDK / Cursor Agent CLI / Gemini CLI / Anthropic SDK / LiteLLM |
| LLMプロバイダ | Anthropic, OpenAI, Google, Azure, Vertex AI, AWS Bedrock, Ollama, vLLM |
| Webフレームワーク | FastAPI + Uvicorn |
| タスクスケジュール | APScheduler |
| 設定管理 | Pydantic 2.0+ / JSON / Markdown |
| 記憶基盤 | ChromaDB + sentence-transformers + NetworkX |
| 音声チャット | faster-whisper (STT) + VOICEVOX / SBV2 / ElevenLabs (TTS) |
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
│   ├── anima.py, agent.py, lifecycle.py  # コアエンティティ・オーケストレーター
│   ├── memory/          # 記憶サブシステム（priming, consolidation, forgetting, RAG）
│   ├── execution/       # 実行エンジン（S/C/D/G/A/B）
│   ├── tooling/         # ツールディスパッチ・権限チェック
│   ├── prompt/          # システムプロンプト構築（6グループ構造）
│   ├── supervisor/      # プロセス監視
│   ├── voice/           # 音声チャット（STT + TTS）
│   ├── config/          # 設定管理（Pydanticモデル）
│   ├── notification/    # 人間通知チャネル
│   └── tools/           # 外部ツール実装
├── cli/                 # CLIパッケージ
├── server/              # FastAPIサーバー + Web UI
└── templates/           # 初期化テンプレート（ja / en）
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
