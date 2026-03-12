# AnimaWorks — Organization-as-Code

**一人では何もできない。だから、組織を作った。**

経営者として知っている——天才一人より、不完全な個が協働するチームのほうが強い。精神科医としてLLMを診たとき、人間の脳と驚くほど似た構造に気づいた。30年のエンジニアリングで培った自動化への執念を注ぎ込んで、LLMで動く本物の組織を作った。それがAnimaWorksだ。

各Animaは固有の名前・性格・記憶・スケジュールを持ち、メッセージで連携し、自分で判断し、チームとして協働する。リーダーに話しかけるだけで、あとは任せられる。

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — リアルタイム組織ツリーとアクティビティフィード" width="720">
  <br><em>Workspaceダッシュボード: 各Animaのロール・ステータス・直近アクションをリアルタイムで一覧表示。</em>
</p>

<p align="center">
  <img src="docs/images/workspace-demo.gif" alt="AnimaWorks 3Dワークスペース — エージェントが自律的に協働" width="720">
  <br><em>3Dオフィス: Animaたちがデスクに座り、歩き回り、メッセージをやり取り——すべて自律的に。</em>
</p>

**[English README](README.md)** | **[简体中文 README](README_zh.md)**

---

## :rocket: 今すぐ試す — Dockerデモ

60秒で体験できる。APIキーとDockerだけでいい。

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
cp .env.example .env          # ANTHROPIC_API_KEY を貼り付け
docker compose up              # http://localhost:18500 を開く
```

3人のチーム（マネージャー＋エンジニア＋コーディネーター）が即座に動き出す。3日分のアクティビティ履歴付き。[デモの詳細はこちら →](demo/README.ja.md)

> 言語・スタイルの切替: `PRESET=ja-anime docker compose up` — [全プリセット一覧](demo/README.ja.md#プリセット)

---

## クイックスタート

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start     # サーバー起動 — 初回はセットアップウィザードが開く
```

**http://localhost:18500/** を開く。セットアップウィザードが順番に聞いてくる:

1. **言語** — UIの表示言語を選択
2. **ユーザー情報** — オーナーアカウントを作成
3. **APIキー** — LLMのAPIキーを入力（リアルタイムで接続検証）
4. **最初のAnima** — 最初のエージェントに名前をつける

`.env`を手で書く必要はない。ウィザードが `config.json` に自動保存する。

**これだけで終わりだ。** セットアップスクリプトが [uv](https://docs.astral.sh/uv/) をインストールし、リポジトリをクローンし、Python 3.12+と全依存パッケージを自動でダウンロードする。**macOS、Linux、WSL**でPythonの事前インストールなしに動く。

> **他のLLMを使いたい場合:** Claude、GPT、Gemini、ローカルモデル等に対応している。セットアップウィザードでAPIキーを入力するか、ダッシュボードの **Settings** から後で追加できる。詳細は [APIキーリファレンス](#apiキーリファレンス) を参照。

<details>
<summary><strong>別の方法: スクリプトを確認してから実行</strong></summary>

`curl | bash` を直接実行したくない場合、先にスクリプトの中身を確認できます:

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

組織全体を俯瞰するコマンドセンター。全Animaの状態・活動・記憶統計がひと目でわかる。

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks ダッシュボード — 19体のAnima組織図" width="720">
  <br><em>ダッシュボード: 4階層・19体のAnimaが稼働中。リアルタイムステータス表示。</em>
</p>

- **チャット** — 好きなAnimaとリアルタイムで話せる。ストリーミング応答、画像添付、マルチスレッド、全履歴
- **音声チャット** — 声で直接会話できる（押して話す or ハンズフリー）
- **Board** — Slack風の共有チャネル。Anima同士がここで勝手に議論・連携する
- **アクティビティ** — 組織全体で何が起きているか、リアルタイムで流れてくる
- **メモリ** — 各Animaが何を覚えているか、エピソード・知識・手順書を覗ける
- **設定** — APIキー、認証、システム設定
- **多言語対応** — 17言語に対応。UI全体がローカライズされる

### 3D Workspace

Animaたちが3Dオフィスで働いている様子を眺められる。

- デスクに座って作業したり、歩き回ったり、勝手にやり取りしている
- 待機、作業中、思考中、睡眠中——状態がそのまま見える
- 会話中は吹き出しが表示される
- クリックすればチャットが開く。表情もリアルタイムで変わる

---

## チームを作る

現実の組織と同じだ。リーダーに「こういう人が欲しい」と言えばいい:

> *「業界トレンドを調査するリサーチャーと、インフラを管理するエンジニアを雇いたい」*

リーダーが適切なロール・性格・上下関係を判断して、新メンバーを作る。設定ファイルを書く必要はない。CLIを叩く必要もない。会話だけで組織が育つ。

### 留守中も、組織は回り続ける

人間の組織と同じように、チームが揃えば勝手に動き出す:

- **ハートビート** — 定期的にタスクを確認し、チャネルを読み、次に何をすべきか自分で判断する
- **cronジョブ** — Animaごとのスケジュールタスク。日次レポート、週次まとめ、監視
- **タスク委譲** — マネージャーが部下にタスクを振り、進捗を追い、報告を受ける
- **並列タスク実行** — 複数タスクを同時投入。依存関係を解決して独立タスクを並列実行
- **夜間統合** — 日中のエピソード記憶が、寝ている間に知識へ昇華される
- **チーム連携** — 共有チャネルとDMで全員が勝手に同期する

### アバター自動生成

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks アセット管理 — リアリスティックなアバターと表情バリアント" width="720">
  <br><em>アセット管理: リアリスティックな全身・バストアップ・表情バリアント — すべて人格設定から自動生成。</em>
</p>

新しいAnimaが作られると、性格設定からキャラクター画像と3Dモデルを自動生成する。上司の画像がある場合は **Vibe Transfer** で画風を自動継承——チーム全体のビジュアルが統一される。

NovelAI（アニメ調）、fal.ai/Flux（スタイライズド/フォトリアル）、Meshy（3Dモデル）に対応。画像サービスを設定しなくても動く——アバターが付かないだけだ。

---

## なぜAnimaWorksなのか

このプロジェクトは、3つのキャリアの交差点から生まれた。

**経営者として**——僕は「一人では何もできない」ことを知っている。自分一人では何もできない。優秀なエンジニアも必要だし、コミュニケーションが得意なやつもいる。黙々と働くワーカーもいれば、時折鋭いアイデアを出してくれる人もいる。天才だけでは組織は回らない。多様な力を合わせたとき、一人では成し遂げられなかったことが成し遂げられる。

**精神科医として**——LLMの内部構造を観察したとき、人間の脳と驚くほど似た構造があることに気づいた。想起、学習、忘却、固定化——脳が記憶を処理するメカニズムを、LLMの記憶システムとしてそのまま実装できる。だったら、LLMを「擬似的な人間」として扱い、人間と同じように組織を作れるはずだ。

**エンジニアとして**——30年間コードを書いてきた。ロジックを組む楽しさ、自動化の快感を知っている。理想をすべてコードに詰め込めば、僕の理想の組織を作れる。

優れた「単独AI秘書」のフレームワークはすでにある。でも、コードで人間を再現し、組織として機能させたプロジェクトはまだなかった。AnimaWorksは、僕が実際に自分の事業に組み込みながら育てている、本物の法人組織そのものだ。

> *不完全な個の協働が、単一の全能者より堅牢な組織を作る。*

| 従来のエージェントFW | AnimaWorks |
|---------------------|------------|
| 実行したら忘れる | 記憶を持って積み重ねる |
| 一つの司令塔が全部指示する | 各自が判断して動く |
| 全員が同じ情報を見る | 必要な時に自分で思い出す |
| ツールを順番に呼ぶ | メッセージで連携する組織 |
| プロンプトを調整する | 人格と価値観で判断する |

3つの原則がこれを支える:

- **カプセル化** — 内部の思考・記憶は外から見えない。他者とはテキスト会話だけでつながる。現実の組織と同じだ。
- **書庫型記憶** — コンテキストウィンドウに詰め込まない。必要な時に、自分の記憶を自分で検索して思い出す。
- **自律性** — 指示を待たない。自分の時計で動き、自分の理念で判断する。

---

## 記憶システム

精神科医の目で見ると、従来のAIエージェントは事実上の健忘だ。コンテキストウィンドウに収まる分しか覚えていない。AnimaWorksのAnimaは永続的な記憶アーカイブを持ち、**必要な時に自分で検索して思い出す。** 本棚から本を引き出すように。脳科学で言う「想起」そのものだ。

| 記憶タイプ | 脳科学モデル | 内容 |
|---|---|---|
| `episodes/` | エピソード記憶 | 日別の行動ログ |
| `knowledge/` | 意味記憶 | 教訓・ルール・学んだ知識 |
| `procedures/` | 手続き記憶 | 作業手順書 |
| `skills/` | スキル記憶 | 再利用可能なタスク別指示書 |
| `state/` | ワーキングメモリ | 今のタスク・未完了項目・タスクキュー |
| `shortterm/` | 短期記憶 | セッション継続（chat/heartbeat分離） |
| `activity_log/` | 統一タイムライン | 全インタラクション（JSONL） |

### 記憶は進化する

- **Priming（自動想起）** — メッセージが届くと、6チャネルの並列検索が走る。送信者プロファイル、直近の活動、関連知識、スキル、未完了タスク、過去のエピソード。結果がシステムプロンプトに注入され、Animaは「指示されなくても思い出す」。
- **Consolidation（統合）** — 毎晩、日中のエピソードが意味記憶に昇華される。脳科学で言う睡眠時の記憶固定化と同じメカニズムだ。解決済みの問題は自動で手順書になる。週次で知識のマージと圧縮。
- **Forgetting（忘却）** — 使われない記憶は3段階で徐々に薄れる。マーキング、マージ、アーカイブ。大事な手順書やスキルは保護される。人間の脳と同じで、忘れることも大事だ。

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks チャット — 複数Animaとのマルチスレッド会話" width="720">
  <br><em>チャット: マルチスレッド会話 — マネージャーがコード修正をレビューし、エンジニアが進捗報告している。</em>
</p>

---

## 音声チャット

ブラウザだけでAnimaと音声会話。アプリ不要。

- **押して話す（PTT）** — マイクボタン長押しで録音、離すと送信
- **VADモード** — ハンズフリー: 発話を自動検出して録音開始/送信
- **割り込み（Barge-in）** — Animaの発話中に話し始めると自動で中断
- **複数TTSプロバイダ** — VOICEVOX、Style-BERT-VITS2/AivisSpeech、ElevenLabs
- **Animaごとの声** — 各Animaに異なる声・話し方を設定可能

仕組みはシンプルで、テキストチャットと同じパイプラインを通る: 音声 → STT（faster-whisper）→ Anima推論 → 応答テキスト → TTS → 音声再生。Anima自身は音声会話だと知らない——テキストに応答しているだけだ。

---

## マルチモデル対応

どのLLMでも動く。Animaごとに別のモデルを使い分けられる——これが「適材適所」の実装だ。

| モード | エンジン | 対象 | ツール |
|--------|----------|------|--------|
| S (SDK) | Claude Agent SDK | Claudeモデル（推奨） | フル: Read/Write/Edit/Bash/Grep/Glob（サブプロセス経由） |
| C (Codex) | Codex SDK | OpenAI Codex CLIモデル | フル: Mode S同等（Codexサブプロセス経由） |
| A (Autonomous) | LiteLLM + tool_use | GPT-4o, Gemini, Mistral, vLLM 等 | search_memory, read/write_file, send_message 等 |
| A (Fallback) | Anthropic SDK直接 | Claude（Agent SDK未インストール時） | Mode Aと同等 |
| B (Basic) | LiteLLM 1ショット | Ollama, 小型ローカルモデル | フレームワークがモデルに代わって記憶I/Oを代行 |

モードはモデル名からワイルドカードパターンマッチで自動判定。`status.json`で個別にオーバーライド可能。`~/.animaworks/models.json`でモデルごとの実行モード・コンテキストウィンドウを定義できる。

**バックグラウンドモデル** — Heartbeat・Cron・Inboxはメインモデルとは別の軽量モデルで実行可能（コスト最適化）。`animaworks anima set-background-model {名前} {モデル}` で設定。

**Extended thinking（拡張思考）** はClaude、Gemini等の対応モデルで利用可能 — Animaの思考過程をUIで表示できる。

### APIキーリファレンス

#### LLMプロバイダ

| キー | サービス | モード | 取得先 |
|-----|---------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A / C | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

**Azure OpenAI**、**Vertex AI (Gemini)**、**AWS Bedrock**、**vLLM** は `config.json` の `credentials` セクションで設定。詳細は[技術仕様](docs/spec.md)を参照。

**Ollama** 等のローカルモデルはAPIキー不要。`OLLAMA_SERVERS`（デフォルト: `http://localhost:11434`）で接続先を指定。

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

---

## 階層とロール

現実の組織と同じように、`supervisor` フィールドひとつで上下関係を定義する。未設定ならトップレベル。

ロールテンプレートで、役職に応じた専門プロンプト・権限・モデルが自動で適用される:

| ロール | デフォルトモデル | 用途 |
|--------|----------------|------|
| `engineer` | Claude Opus 4.6 | 複雑な推論、コード生成 |
| `manager` | Claude Opus 4.6 | 調整、意思決定 |
| `writer` | Claude Sonnet 4.6 | コンテンツ作成 |
| `researcher` | Claude Sonnet 4.6 | 情報収集 |
| `ops` | vLLM (GLM-4.7-flash) | ログ監視、定型業務 |
| `general` | Claude Sonnet 4.6 | 汎用 |

マネージャーには**スーパーバイザーツール**が自動で付く。タスク委譲、進捗追跡、部下の再起動/無効化、組織ダッシュボード、部下の状態読み取り——現実の管理職がやることと同じだ。

通信はすべてMessengerによる非同期メッセージング。各AnimaはProcessSupervisorが独立子プロセスとして起動し、Unix Domain Socket経由で通信する。

---

## セキュリティ

自律的に動くエージェントにツールを渡す以上、セキュリティは本気でやらないといけない。AnimaWorksは10層の多層防御を実装している:

| レイヤー | 内容 |
|---------|------|
| **信頼境界ラベリング** | 外部データ（Web検索、Slack、メール）はすべて `untrusted` タグ付き — untrustedソースからの指示には従わないようモデルに明示 |
| **5層コマンドセキュリティ** | シェルインジェクション検出 → ハードコードブロックリスト → 個別エージェント禁止コマンド → 個別エージェント許可リスト → パストラバーサル検出 |
| **ファイルサンドボックス** | 各エージェントは自ディレクトリに閉じ込め。`permissions.md` や `identity.md` はエージェント自身が書き換え不可 |
| **プロセス隔離** | エージェントごとに独立OSプロセス。Unix Domain Socket通信（TCP不使用） |
| **3層レート制限** | セッション内重複排除 → ロール別送信上限（manager 60通/時・300通/日 〜 general 15通/時・50通/日、status.jsonで個別上書き可） → 直近送信履歴のプロンプト注入による自己認識 |
| **カスケード防止** | 二層制御: 深度制限（10分間で同一ペア最大6ターン）＋カスケード検出（30分間で同一ペア最大3往復）。5分クールダウンと遅延処理 |
| **認証・セッション管理** | Argon2idハッシュ、48バイトランダムトークン、最大10セッション、0600ファイル権限 |
| **Webhook検証** | Slack（リプレイ防止付きHMAC-SHA256）とChatworkの署名検証 |
| **SSRF緩和** | メディアプロキシがプライベートIPをブロック、HTTPS強制、Content-Type検証、DNS解決チェック |
| **アウトバウンドルーティング** | 未知の宛先はfail-closed。明示的な設定なしに任意の外部送信は不可 |

詳細: **[セキュリティアーキテクチャ](docs/security.ja.md)**

---

<details>
<summary><strong>CLIコマンドリファレンス（上級者向け）</strong></summary>

CLIはパワーユーザーと自動化向け。日常操作はWeb UIで。

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
| `animaworks reset [--restart]` | ランタイムディレクトリをリセット |

### Anima管理

| コマンド | 説明 |
|---|---|
| `animaworks anima create [--from-md PATH] [--template NAME] [--role ROLE] [--supervisor NAME] [--name NAME]` | 新規作成（キャラクターシート/テンプレート/空白） |
| `animaworks anima list [--local]` | 全Anima一覧（名前・有効/無効・モデル・supervisor） |
| `animaworks anima info ANIMA [--json]` | 詳細設定（モデル・ロール・credential・voice等） |
| `animaworks anima status [ANIMA]` | プロセス状態表示 |
| `animaworks anima restart ANIMA` | プロセス再起動 |
| `animaworks anima disable ANIMA` | Animaを無効化（停止） |
| `animaworks anima enable ANIMA` | Animaを有効化（起動） |
| `animaworks anima set-model ANIMA MODEL [--credential CRED]` | モデル変更 |
| `animaworks anima set-background-model ANIMA MODEL [--credential CRED]` | Heartbeat・Cron用モデル設定 |
| `animaworks anima set-background-model ANIMA --clear` | バックグラウンドモデル解除 |
| `animaworks anima reload ANIMA [--all]` | status.json から設定をホットリロード（再起動不要） |

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
| `animaworks config get KEY` | 値取得（ドット記法） |
| `animaworks config set KEY VALUE` | 値設定 |
| `animaworks status` | システムステータス |
| `animaworks logs [ANIMA] [--lines N] [--all]` | ログ表示 |
| `animaworks index [--reindex] [--anima NAME]` | RAGインデックス管理 |
| `animaworks optimize-assets [--anima NAME]` | アセット画像最適化 |
| `animaworks remake-assets ANIMA --style-from REF` | アセット再生成（Vibe Transfer） |
| `animaworks models list` / `animaworks models info MODEL` | モデル一覧・詳細 |

</details>

<details>
<summary><strong>技術スタック</strong></summary>

| コンポーネント | 技術 |
|---|---|
| エージェント実行 | Claude Agent SDK / Codex SDK / Anthropic SDK / LiteLLM |
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
│   ├── anima_factory.py, init.py         # 初期化・Anima生成
│   ├── schemas.py, paths.py              # データモデル・パス定数
│   ├── messenger.py, outbound.py         # 通信・アウトバウンドルーティング
│   ├── background.py, asset_reconciler.py, org_sync.py
│   ├── schedule_parser.py                # cron.md / heartbeat.md パーサー
│   ├── memory/          # 記憶サブシステム
│   │   ├── manager.py, conversation.py, shortterm.py
│   │   ├── priming.py   # 自動想起（6チャネル並列）
│   │   ├── consolidation.py, forgetting.py
│   │   ├── activity.py, streaming_journal.py, task_queue.py
│   │   └── rag/         # RAGエンジン（ChromaDB + グラフ拡散活性化）
│   ├── execution/       # 実行エンジン（S/C/A/B）
│   │   ├── agent_sdk.py, codex_sdk.py, litellm_loop.py, assisted.py
│   │   └── anthropic_fallback.py, _session.py
│   ├── tooling/         # ツールディスパッチ・権限チェック・ガイド生成
│   ├── prompt/          # システムプロンプト構築（6グループ構造）
│   ├── supervisor/      # プロセス監視（pending_executor, scheduler_manager含む）
│   ├── voice/           # 音声チャット（STT + TTS + セッション管理）
│   ├── config/          # 設定管理（Pydanticモデル・vault）
│   ├── notification/    # 人間通知（slack, chatwork, line, telegram, ntfy）
│   ├── auth/            # 認証（Argon2id + セッション）
│   └── tools/           # 外部ツール実装
├── cli/                 # CLIパッケージ（argparse + サブコマンド）
├── server/              # FastAPIサーバー + Web UI
│   ├── routes/          # APIルート（ドメイン別分割）
│   └── static/          # ダッシュボード + Workspace UI
└── templates/           # 初期化テンプレート
    ├── ja/, en/         # ロケール別（prompts, roles, common_knowledge, common_skills）
    └── _shared/         # ロケール共通（company等）
```

</details>

---

## ドキュメント

**[ドキュメント総合インデックス](docs/README.ja.md)** — 読む順序の案内、アーキテクチャ詳説、研究論文、設計仕様の一覧。

| ドキュメント | 説明 |
|-------------|------|
| [設計理念](docs/vision.ja.md) | 「不完全な個の協働」という根本思想 |
| [機能一覧](docs/features.ja.md) | AnimaWorksで何ができるかの全体像 |
| [記憶システム](docs/memory.ja.md) | エピソード記憶・意味記憶・手続き記憶・プライミング・能動的忘却 |
| [セキュリティ](docs/security.ja.md) | 多層防御モデル、データ出自追跡、敵対的脅威分析 |
| [脳科学マッピング](docs/brain-mapping.ja.md) | 各モジュールと人間の脳の対応関係 |
| [技術仕様](docs/spec.ja.md) | 実行モード、プロンプト構築、設定解決 |

## ライセンス

Apache License 2.0。詳細は [LICENSE](LICENSE) を参照。
