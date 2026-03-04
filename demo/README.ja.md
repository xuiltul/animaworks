# AnimaWorks デモ — 60秒で体験

**[English version](README.md)**

3体の自律AIエージェントが働くオフィスを、設定なし・ウィザードなしで起動。APIキーとDockerだけで始められます。

3日分のアクティビティ履歴が同梱されているため、ダッシュボードを開いた瞬間から「生きた組織」を体験できます。

## クイックスタート

### 1. クローン

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
```

### 2. APIキーを設定

```bash
cp .env.example .env
# .env を編集して Anthropic APIキーを貼り付け
```

APIキーがまだない場合は [console.anthropic.com](https://console.anthropic.com/) で取得できます。

### 3. 起動

```bash
docker compose up
```

**http://localhost:18500** を開けば準備完了。

---

## 何が見えるか

ダッシュボードを開くと、3人のチームがすでに稼働しています:

| エージェント | 役割 | やること |
|-------------|------|---------|
| **Kaito** | プロダクトマネージャー（リーダー） | 優先度設定、タスク委譲、進捗レビュー |
| **Sora** | リードエンジニア | 機能実装、技術的課題の調査 |
| **Hina** | チームアシスタント | スケジュール管理、コミュニケーションのハブ役 |

Kaitoがリーダーで、SoraとHinaはKaitoに報告します。この階層は完全に機能します — Kaitoはタスクを委譲し、進捗を確認でき、部下は自律的に報告します。

> **Note:** デフォルトプリセットは `en-anime`（英語）です。日本語プリセットを使うには:
> ```bash
> PRESET=ja-anime docker compose up
> ```

### 試してみること

- **Kaitoとチャット** — チームの進捗を聞いたり、新しい指示を出す
- **Activityフィードを見る** — エージェント同士のリアルタイムのやり取り
- **Boardを確認** — #generalチャネルでチームの議論が進行中
- **3D Workspaceを開く** — キャラクターがデスクに座り、動き回る様子
- **Soraに直接話しかける** — 技術的な質問をしてみる
- **5分待つ** — ハートビートが発火し、エージェントが自律的に動き始める

### 事前ロード済みの履歴

デモには3日分のシミュレートされたアクティビティが含まれています（日付は今日基準に自動調整）:

- 過去の会話や意思決定を含むアクティビティログ
- 進行中のタスク
- 共有 #general チャネルのメッセージ

ダッシュボードが空の状態にならず、コンテキストと履歴を持つチームを最初から体験できます。

---

## プリセット

言語とキャラクタースタイルの組み合わせで4つのプリセットがあります:

| プリセット | 言語 | スタイル | キャラクター |
|-----------|------|---------|-------------|
| `en-anime`（デフォルト） | 英語 | アニメ風カジュアル | Alex, Kai, Nova |
| `en-business` | 英語 | ビジネスプロフェッショナル | Alex, Kai, Nova |
| `ja-anime` | 日本語 | アニメ風カジュアル | Kaito, Sora, Hina |
| `ja-business` | 日本語 | ビジネスプロフェッショナル | Kaito, Sora, Hina |

プリセットの切り替えは `PRESET` 環境変数で:

```bash
PRESET=ja-anime docker compose up
```

> **注意:** プリセットは初回起動時のみ適用されます。切り替えるにはDockerボリュームを先に削除してください:
>
> ```bash
> docker compose down -v
> PRESET=ja-business docker compose up
> ```

---

## 設定

### 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `ANTHROPIC_API_KEY` | （必須） | Anthropic APIキー |
| `PRESET` | `en-anime` | 使用するプリセット |
| `TZ` | `Asia/Tokyo` | コンテナのタイムゾーン |

すべての変数は `.env` ファイルで設定するか、直接渡せます:

```bash
ANTHROPIC_API_KEY=sk-ant-... PRESET=ja-anime docker compose up
```

### ポート

デモサーバーはポート **18500** で動作します。別のポートを使う場合:

```bash
# docker-compose.yml のポートマッピングを変更:
ports:
  - "9000:18500"   # http://localhost:9000 でアクセス
```

---

## デモの仕組み

初回起動時、エントリポイントスクリプトが以下を実行します:

1. AnimaWorksランタイムを初期化
2. 選択されたプリセットのキャラクターシートから3体のエージェントを作成
3. プリセット固有の設定を適用（ハートビート間隔など）
4. キャラクターアセット（アバター）をコピー
5. 3日分のサンプルアクティビティデータをタイムスタンプ自動調整付きでロード
6. サーバーを起動

2回目以降の起動では初期化をスキップし、Dockerボリューム内の既存データを使用します。

### 自律動作

起動後、エージェントは自律的に動作します:

- **ハートビート** — 5分間隔（デモ設定）で、各エージェントが状況を確認し次の行動を判断
- **Cronタスク** — エージェントごとに定義されたスケジュールタスク（日次まとめ、監視など）
- **委譲チェーン** — KaitoがSora/Hinaにタスクを委譲、実行後に報告
- **Board活動** — エージェントが共有チャネルに進捗を投稿

何もしなくても動きます。見ているだけでもいいし、新しい指示を出すこともできます。

---

## データの永続化

エージェントのデータはDockerボリューム（`animaworks-demo-data`）に保存されます。会話、記憶、アクティビティログはコンテナ再起動後も保持されます。

完全にリセットする場合:

```bash
docker compose down -v    # ボリュームを削除
docker compose up         # 初期化からやり直し
```

---

## トラブルシューティング

### 「Animas will not be able to respond」と表示される

`ANTHROPIC_API_KEY` が未設定または無効です。`.env` ファイルを確認:

```bash
cat .env   # ANTHROPIC_API_KEY=sk-ant-api03-... と表示されるはず
```

### ポート18500が使用中

別のサービスがポートを使っています。停止するかマッピングを変更:

```bash
# ポートの使用状況を確認
lsof -i :18500

# または docker-compose.yml でポートを変更
ports:
  - "9000:18500"
```

### エージェントが応答しない

- APIキーが有効か確認（[console.anthropic.com](https://console.anthropic.com/) でテスト）
- コンテナログを確認: `docker compose logs -f`
- APIクレジットの残高を確認

### コンテナのビルドに失敗する

```bash
# キャッシュなしで再ビルド
docker compose build --no-cache
docker compose up
```

### すべてリセットしたい

```bash
docker compose down -v
docker compose up
```

---

## 次のステップ

自分のAI組織を作る準備ができましたか？

- **フルインストール** — ネイティブインストールは[メインREADME](../README_ja.md)を参照
- **自分のエージェントを作る** — Markdownでキャラクターシートを書くだけで、フレームワークが残りを処理
- **他のLLMを追加** — AnimaWorksはClaude、GPT、Gemini、ローカルモデル等に対応
- **ドキュメント** — [設計思想](../docs/vision.ja.md) · [記憶システム](../docs/memory.ja.md) · [セキュリティ](../docs/security.ja.md)

---

*このデモは [AnimaWorks](https://github.com/xuiltul/animaworks) の一部です — 自律型AI組織を構築するためのオープンソースフレームワーク。*
