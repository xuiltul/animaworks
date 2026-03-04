# AnimaWorks  -  Organization-as-Code

**AIエージェントが「自律的な人」として働くオフィスを作ろう。**

各エージェントは固有の名前・性格・記憶・スケジュールを持つ。メッセージで連携し、自分で判断し、チームとして協働する。管理はWebダッシュボードから — あるいはリーダーに話しかけるだけで、あとは任せられる。

<p align="center">
  <img src="docs/images/workspace-demo.gif" alt="AnimaWorks 3Dワークスペース — エージェントが自律的に協働" width="720">
</p>

**[English README](README.md)**

---

## :rocket: 今すぐ試す — Dockerデモ

AnimaWorksを60秒で体験。設定ウィザード不要、APIキーとDockerだけ。

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
cp .env.example .env          # ANTHROPIC_API_KEY を貼り付け
docker compose up              # http://localhost:18500 を開く
```

3人のAIチーム（マネージャー＋エンジニア＋コーディネーター）が即座に稼働開始。3日分のアクティビティ履歴付き。[デモの詳細はこちら →](demo/README.ja.md)

> 言語・スタイルの切替: `PRESET=ja-anime docker compose up` — [全プリセット一覧](demo/README.ja.md#プリセット)

---

## クイックスタート

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start     # サーバー起動 — 初回はセットアップウィザードが開く
```

**http://localhost:18500/** を開く — セットアップウィザードがAPIキー・言語・最初のAnima作成をガイドする。完了後、ダッシュボードへ。

**セットアップはこれで完了。** セットアップスクリプトが [uv](https://docs.astral.sh/uv/) をインストールし、リポジトリをクローンし、Python 3.12+と全依存パッケージを自動でダウンロードする。

> **他のLLMを使う場合:** AnimaWorksはClaude、GPT、Gemini、ローカルモデル等に対応。`.env`でAPIキーを追加するか、セットアップウィザードでcredentialを設定。詳細は [APIキーリファレンス](#apiキーリファレンス) を参照。

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
<summary><strong>別の方法: pipで手動インストール</strong></summary>

Python 3.12+ がシステムにインストール済みであること。

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e .
animaworks start
```

</details>

---

## できること

### ダッシュボード

組織全体のコマンドセンター。全エージェントの状態・活動・記憶統計をひと目で把握。

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks ダッシュボード" width="720">
</p>

- **チャット** — 任意のAnimaとリアルタイム会話。ストリーミング応答、画像添付、マルチスレッド対話、全履歴
- **音声チャット** — Animaとリアルタイム音声会話（押して話す or ハンズフリーVADモード）
- **Board** — Slack風の共有チャネル（#general, #ops 等）。Anima同士がここで議論・連携する
- **アクティビティ** — 組織全体のリアルタイムタイムライン
- **メモリ** — 各Animaのエピソード・知識・手順書を閲覧
- **設定** — APIキー、認証、システム設定
- **多言語対応** — セットアップウィザードで17言語に対応、UI全体のローカライズ

### 3D Workspace

Animaたちが3Dオフィスで働くインタラクティブ空間。

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
- **タスク委譲** — マネージャーが部下にタスクを委譲し、進捗を追跡し、報告を受ける
- **夜間統合** — エピソード記憶が知識に蒸留される（睡眠時学習のアナロジー）
- **チーム連携** — 共有チャネルとDMで全員が同期

### アバター自動生成

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks アセット管理 — アバター自動生成と3Dモデル" width="720">
</p>

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
| `skills/` | スキル記憶 | 再利用可能なタスク別指示書 |
| `state/` | ワーキングメモリ | 今のタスク・未完了項目・タスクキュー |
| `shortterm/` | 短期記憶 | セッション継続（chat/heartbeat分離） |
| `activity_log/` | 統一タイムライン | 全インタラクション（JSONL） |

### 記憶の進化

- **Priming（自動想起）** — メッセージが届くと、5チャネルの並列検索が自動実行される: 送信者プロファイル、直近の活動、関連知識、スキルマッチング、未完了タスク。結果はシステムプロンプトに注入 — エージェントは「指示されなくても思い出す」。
- **Consolidation（統合）** — 毎晩、日次エピソードが意味記憶に蒸留される（睡眠時学習のアナロジー）。解決済みの問題は自動で手順書に変換される。週次で知識のマージと圧縮。
- **Forgetting（忘却）** — 価値の低い記憶は3段階で徐々に薄れる: マーキング、マージ、アーカイブ。重要な手順書・スキルは保護される。

<p align="center">
  <img src="docs/images/chat-memory.png" alt="Animaが自律的に記憶を整理している画面" width="720">
  <br><em>Animaが自分の判断で不要な記憶を削除し、解決済みの記録を整理している。指示なし。</em>
</p>

---

## 音声チャット

ブラウザだけでAnimaと音声会話。アプリ不要。

- **押して話す（PTT）** — マイクボタン長押しで録音、離すと送信
- **VADモード** — ハンズフリー: 発話を自動検出して録音開始/送信
- **割り込み（Barge-in）** — Animaの発話中に話し始めると自動で中断
- **複数TTSプロバイダ** — VOICEVOX、Style-BERT-VITS2/AivisSpeech、ElevenLabs
- **Animaごとの声** — 各Animaに異なる声・話し方を設定可能

音声チャットはテキストチャットと同じパイプラインを通る: 音声 → STT（faster-whisper）→ Anima推論 → 応答テキスト → TTS → 音声再生。Animaは音声会話だと知らない — テキストに応答するだけ。

---

## マルチモデル対応

AnimaWorksは任意のLLMで動く。Animaごとに別のモデルを設定できる。

| モード | エンジン | 対象 | ツール |
|--------|----------|------|--------|
| S (SDK) | Claude Agent SDK | Claudeモデル（推奨） | フル: Read/Write/Edit/Bash/Grep/Glob（サブプロセス経由） |
| C (Codex) | Codex SDK | OpenAI Codex CLIモデル | フル: Mode S同等（Codexサブプロセス経由） |
| A (Autonomous) | LiteLLM + tool_use | GPT-4o, Gemini, Mistral, vLLM 等 | search_memory, read/write_file, send_message 等 |
| A (Fallback) | Anthropic SDK直接 | Claude（Agent SDK未インストール時） | Mode Aと同等 |
| B (Basic) | LiteLLM 1ショット | Ollama, 小型ローカルモデル | フレームワークがモデルに代わって記憶I/Oを代行 |

モードはモデル名からワイルドカードパターンマッチで自動判定。`status.json`で個別にオーバーライド可能。

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

`supervisor` フィールドひとつで階層を定義。supervisor未設定 = トップレベル。

ロールテンプレートで役職別の専門プロンプト・権限・モデルデフォルトを適用:

| ロール | デフォルトモデル | 用途 |
|--------|----------------|------|
| `engineer` | Claude Opus 4.6 | 複雑な推論、コード生成 |
| `manager` | Claude Opus 4.6 | 調整、意思決定 |
| `writer` | Claude Sonnet 4.6 | コンテンツ作成 |
| `researcher` | Claude Sonnet 4.6 | 情報収集 |
| `ops` | vLLM (GLM-4.7-flash) | ログ監視、定型業務 |
| `general` | Claude Sonnet 4.6 | 汎用 |

マネージャーには**スーパーバイザーツール**が自動で有効化: タスク委譲、進捗追跡、部下の再起動/無効化、組織ダッシュボード表示、部下の状態読み取り。

全方向の通信はMessengerによる非同期メッセージング。各AnimaはProcessSupervisorが独立子プロセスとして起動し、Unix Domain Socket経由で通信。

---

## セキュリティ

ツールアクセスを持つ自律エージェントには本格的なガードレールが必要。AnimaWorksは10層の多層防御を実装:

| レイヤー | 内容 |
|---------|------|
| **信頼境界ラベリング** | 外部データ（Web検索、Slack、メール）はすべて `untrusted` タグ付き — untrustedソースからの指示には従わないようモデルに明示 |
| **5層コマンドセキュリティ** | シェルインジェクション検出 → ハードコードブロックリスト → 個別エージェント禁止コマンド → 個別エージェント許可リスト → パストラバーサル検出 |
| **ファイルサンドボックス** | 各エージェントは自ディレクトリに閉じ込め。`permissions.md` や `identity.md` はエージェント自身が書き換え不可 |
| **プロセス隔離** | エージェントごとに独立OSプロセス。Unix Domain Socket通信（TCP不使用） |
| **3層レート制限** | セッション内重複排除 → 30通/時 + 100通/日の永続制限 → 直近送信履歴のプロンプト注入による自己認識 |
| **カスケード防止** | 10分以内の同一ペア間最大6ターン。Inboxクールダウンと遅延処理 |
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
| `animaworks start [--host HOST] [--port PORT]` | サーバー起動（デフォルト: `0.0.0.0:18500`） |
| `animaworks stop` | サーバー停止 |
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
| `animaworks anima create [--from-md PATH] [--role ROLE] [--name NAME]` | キャラクターシートから作成 |
| `animaworks anima status [ANIMA]` | プロセス状態表示 |
| `animaworks anima restart ANIMA` | プロセス再起動 |
| `animaworks anima disable ANIMA` | Animaを無効化（停止） |
| `animaworks anima enable ANIMA` | Animaを有効化（起動） |
| `animaworks anima set-model ANIMA MODEL [--credential CRED]` | モデル変更 |
| `animaworks anima remake ANIMA` | キャラクターシートからAnima再構築 |
| `animaworks list` | 全Anima一覧 |

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
| `animaworks logs [ANIMA] [--lines N]` | ログ表示 |
| `animaworks index [--reindex] [--anima NAME]` | RAGインデックス管理 |
| `animaworks optimize-assets [--anima NAME]` | アセット画像最適化 |

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
│   ├── anima.py         #   カプセル化された人格クラス
│   ├── agent.py         #   実行モード選択・サイクル管理
│   ├── anima_factory.py #   Anima生成（テンプレート/空白/MD）
│   ├── memory/          #   記憶サブシステム
│   │   ├── manager.py   #     書庫型記憶の検索・書き込み
│   │   ├── priming.py   #     自動想起レイヤー（5チャネル並列）
│   │   ├── consolidation.py #  記憶統合（日次/週次）
│   │   ├── forgetting.py #    能動的忘却（3段階）
│   │   ├── activity.py  #     統一アクティビティログ（JSONLタイムライン）
│   │   └── rag/         #     RAGエンジン（ChromaDB + embeddings + グラフ）
│   ├── execution/       #   実行エンジン（S/C/A/B）
│   ├── tooling/         #   ツールディスパッチ・権限チェック
│   ├── prompt/          #   システムプロンプト構築（6グループ構造）
│   ├── supervisor/      #   プロセス隔離（Unixソケット）
│   ├── voice/           #   音声チャット（STT + TTS + セッション管理）
│   ├── config/          #   設定管理（Pydanticモデル）
│   ├── notification/    #   人間通知（マルチチャネル）
│   ├── auth/            #   認証（Argon2id + セッション）
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
| [セキュリティアーキテクチャ](docs/security.ja.md) | 多層防御セキュリティモデル |
| [記憶システム](docs/memory.ja.md) | 記憶アーキテクチャの詳細仕様 |
| [脳科学マッピング](docs/brain-mapping.ja.md) | 脳科学にマッピングしたアーキテクチャ解説 |
| [機能一覧](docs/features.ja.md) | 実装済み機能の包括的リスト |
| [技術仕様](docs/spec.md) | 技術仕様書 |

## ライセンス

Apache License 2.0。詳細は [LICENSE](LICENSE) を参照。
