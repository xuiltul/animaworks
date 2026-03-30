# 記憶システムガイド

Anima の記憶の仕組み・種類・使い分けのリファレンス。
記憶の検索・書き込み・整理の方法を確認するために参照すること。

## 記憶の全体像

あなたの記憶は、人間の脳の記憶モデルに対応した複数の種類で構成される:

| 記憶の種類 | ディレクトリ | 人間でいうと | 内容 |
|-----------|------------|------------|------|
| **短期記憶** | `shortterm/` | ワーキングメモリ | 直近の会話の文脈 |
| **エピソード記憶** | `episodes/` | 体験の記憶 | いつ何をしたか |
| **意味記憶** | `knowledge/` | 知識 | 学んだこと・ノウハウ |
| **手続き記憶** | `procedures/` | 身体が覚えた手順 | どうやるかのステップ |
| **スキル** | `skills/` | 特技・専門技能 | 実行可能な手順書 |

さらに、全 Anima で共有される記憶もある:

| 共有記憶 | パス | 内容 |
|---------|------|------|
| **共有知識** | `common_knowledge/` | フレームワークのリファレンス（このファイル自体も含む） |
| **共通スキル** | `common_skills/` | 全 Anima が使えるスキル |
| **組織共有知識** | `shared/common_knowledge/` | 組織が運用中に蓄積した知識 |
| **ユーザープロファイル** | `shared/users/` | Anima 横断のユーザー情報 |

---

## 短期記憶（shortterm/）

**直近の会話やセッションの文脈**を保持する。人間のワーキングメモリに相当する。

- `shortterm/chat/` と `shortterm/heartbeat/` でセッション種別ごとに分離（必要に応じて `thread_id` ごとのサブディレクトリ）
- 各セッション種別ディレクトリに `session_state.json` / `session_state.md` と `archive/` がある（完了・置き換え済みの状態は `archive/` へ）
- コンテキストウィンドウの使用率が閾値を超えると、古い部分が自動的に外部化される
- ストリーミング実行向けに、ツール完了位置などを記録するチェックポイント（再接続・再試行用）も同階層で管理される
- セッション間の文脈継続に使われる
- 日次ハウスキーピングで、短期記憶まわりのアーカイブには保持日数の上限がある（設定で調整可能）

短期記憶は自分で直接操作する必要はない。フレームワークが自動管理する。

---

## エピソード記憶（episodes/）

**「いつ何をしたか」の日次ログ**。人間の体験記憶に相当する。

- 日付ごとのファイル（例: `2026-03-09.md`）に自動記録される。同一日内の分割用に `2026-03-09_topic.md` のような **日付プレフィックス＋サフィックス** も扱う
- 統合作業（Consolidation）のエピソード収集は、直近24時間窓で上記パターンのファイルを読み、`## HH:MM — タイトル` 形式の見出しでエントリ分割する。見出しが無いファイルは更新時刻（mtime）で1エントリとして扱う
- 「先週何をしていたか」「この問題に以前対応したか」を思い出すために使う
- 日次・週次の Consolidation（記憶統合）では、Anima 自身のツールループで要約・知識抽出などが行われる（後述）

### 記憶の書き込み

```
write_memory_file(path="episodes/2026-03-09.md", content="...")
```

### 記憶の検索

```
search_memory(query="Slack API接続テスト", scope="episodes")
```

---

## 意味記憶（knowledge/）

**学んだ知識・ノウハウ・パターン**。人間の「知っていること」に相当する。

- エピソードから抽出された教訓やパターン
- 技術メモ、対応方針、判断基準
- Consolidation で自動蓄積されるほか、自分で能動的に書き込める
- レガシー形式のファイルは初回に YAML フロントマタ付きへ移行される（`knowledge/.migrated` マーカー）
- **再巩固**: フロントマターで `failure_count >= 2` かつ `confidence < 0.6` の knowledge は、手続きと同様に LLM による改訂対象になりうる（`ReconsolidationEngine` の knowledge パス）

例:
- 「Slack API のレート制限は Tier 1 で 1req/sec」
- 「このクライアントは月曜に連絡が多い」
- 「デプロイ前の確認項目リスト」

### 記憶の書き込み

```
write_memory_file(path="knowledge/slack-api-notes.md", content="...")
```

### 記憶の検索

```
search_memory(query="Slack API レート制限", scope="knowledge")
```

---

## 手続き記憶（procedures/）

**「どうやるか」のステップバイステップ手順書**。人間の「身体が覚えた手順」に相当する。

- 問題解決の手順、定型作業のフロー
- `issue_resolved` などのイベントから自動生成されることもある（confidence 0.4 などメタデータ付き）
- **スキルほどの全面保護はない**: メタデータに基づき忘却パイプラインの対象になりうる（後述の手続き専用ルール）
- **再巩固（reconsolidation）**: フロントマターで **`failure_count >= 2` かつ `confidence < 0.6`** のとき、LLM による手順書の改訂が走る。改訂後はカウンタリセット・バージョン番号更新・旧版を `archive/` に退避する（実装: `ReconsolidationEngine`）
- バージョン履歴は `archive/` に残り、古い版は一定数を超えると整理される（忘却エンジン側の手続きアーカイブ保持本数とも連動）

例:
- 「SSL証明書の更新手順」
- 「新規Animaのオンボーディング手順」
- 「本番障害時のエスカレーション手順」

### 記憶の書き込み

```
write_memory_file(path="procedures/ssl-renewal.md", content="...")
```

### 記憶の検索

```
search_memory(query="SSL証明書 更新", scope="procedures")
```

---

## スキル（skills/）

**実行可能な手順書・ツール使用ガイド**。「特技」に相当する。

- 個人スキル（`skills/`）と共通スキル（`common_skills/`）がある
- システムプロンプトのスキルカタログに、個人 `skills/`・`common_skills/`・`procedures/` 等のパスが一覧表示される（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`）
- 詳細が必要なときは `read_memory_file(path="...")` で全文を取得する
- **ベクトルストア上は常に忘却対象外**（`skills` / `shared_users` 型は保護）

### スキルの確認

```
read_memory_file(path="skills/newstaff/SKILL.md")  # スキルの全文を取得
```

### スキルの作成

```
create_skill(skill_name="deploy-procedure", description="本番デプロイ手順", body="...")
```

---

## 記憶の自動プロセス

### Priming（自動想起）

あなたが会話を始めるたびに、Priming エンジンが **5 チャネル（A, B, C/C0, E, F）** で関連記憶を並列検索し、システムプロンプトに注入する。実装では **C0（重要知識）** が先に取り、注入時には **Channel C の本文と連結**される（見た目は 1 つの「関連知識」ブロックになることが多い）。

| チャネル | 何を検索するか | 既定のチャネル別バジェット（トークン目安）※ |
|---------|--------------|--------------------------------|
| A: 送信者プロファイル | 相手のユーザー情報 | 500 |
| B: 直近の活動 | 統一アクティビティログ（タイムライン）＋共有チャネル等の補助 | 1300 |
| C0: 重要知識 | `[IMPORTANT]` タグ付き knowledge の**概要ポインタのみ**（ベクトルストア上の専用一覧取得） | 500 |
| C: 関連知識 | RAG（dense）で個人 knowledge + 共有 common_knowledge を検索 | 1000 |
| E: 保留タスク | タスクキュー要約 + 実行中の並列タスク + 委譲タスク状況 +（あれば）overflow inbox のファイル名一覧 | 500 |
| F: エピソード | RAG で episodes を検索 | 800 |

スキル・手続きのパス一覧はシステムプロンプトのスキルカタログ（`<available_skills>`、Group 4に注入）に載り、本文は `read_memory_file` で読み込む。スキルの意味的検索は `search_memory(scope="skills")` も利用可能。

※ `config.json` の `priming` で挨拶・質問・依頼・heartbeat 用の上限や `heartbeat_context_pct` を変えられる。`dynamic_budget` が有効なときは、メッセージ種別・heartbeat では `max(budget_heartbeat, context_window × heartbeat_context_pct)` などで全体トークン上限が変わり、各チャネルはその比率で縮む。**dynamic_budget が無効なとき**の Priming 全体の既定上限はコード上 **約2000トークン相当**（チャネル別バジェットの合計を基準に縮小割当）。

**複数クエリとマージ（Channel C / F）**: 現在メッセージ先頭（短く切り出し）、キーワード列、オプションで **直近の人間発話から組み立てたコンテキストクエリ**を最大3本まで使い、それぞれでベクトル検索したあと **チャンク単位で最高スコアを採用してマージ**する。キーワードは Unicode 対応の抽出に加え、**`knowledge/*.md` のファイル名（stem）** と一致する語を優先する。

**メッセージが空に近いとき**: heartbeat 等で本文が乏しい場合、`state/current_state.md` の先頭付近がキーワード抽出のフォールバックに使われる。

**Overflow 時の Channel C**: 長文コンテキストがファイルに逃がされている場合、指定されたファイル名（stem）に **RAG ヒットを制限**してノイズを減らす経路がある。

追加で以下も注入される:
- **直近の送信履歴**: 過去2時間の `channel_post` / `message_sent`（最大3件）
- **保留中の人間通知**: 直近24時間の `human_notify`（最大500トークン相当）

**Channel C と信頼度**: 検索結果はチャンクの `origin` 等から **medium** と **untrusted** に分かれる。untrusted は残りバジェットで別枠に切り詰めて注入され、プロンプトインジェクション対策の文脈で扱われる（詳細は `common_knowledge/security/` を参照）。

**`[IMPORTANT]` と C0**: `[IMPORTANT]` 付き knowledge は C0 で「タイトル＋（あれば）1行要約＋ `read_memory_file` への誘導」として載る。フロントマターに `summary` があればタイトル表示に優先して使われる。通常の RAG（C）の結果とは別経路で拾うため、クエリに引っかからなくても重要ルールを見失いにくい。業務上の必須ルールを knowledge に移すときは先頭に `[IMPORTANT]` を付ける。

Priming は自動で動くため、明示的な操作は不要。

### Consolidation（記憶統合）

**本番の統合作業（エピソード要約・知識への抽出・矛盾チェック等）は、Anima 自身がツールループで実行する**（`run_consolidation`）。`ConsolidationEngine` は主に前処理（エピソード・解決イベントの収集）と後処理（RAG 再構築・忘却の呼び出し）を担う。

日次統合は **2フェーズマルチパス方式**:

| フェーズ | 処理内容 |
|---------|---------|
| **Phase A: エピソード抽出** | activity_logの直近24時間分を時間チャンクに分割（モデルのコンテキストウィンドウに応じたサイズ）→ 各チャンクをLLM one-shotで処理（**tool_result全文**を含むため、旧方式のメタのみと比べ高精度）→ タイムラインヘッダの重複除去後マージ → `episodes/{date}.md` に書き出し |
| **Phase B: 知識抽出** | Phase Aの結果 + **エラートレース分析**（直近24時間のerror/failed tool_resultを収集、最大50件・3000文字）をAnimaのツールループで処理 → knowledge抽出、procedure自動生成（エラーパターンから `source: "error_trace_analysis"` 付き） |

| 頻度 | 処理の流れ（概要） |
|------|-------------------|
| **日次** | 直近24時間のエピソード数が閾値以上なら **Phase A → Phase B** → 続けて **Synaptic downscaling**（メタデータのみ）→ **RAG インデックス再構築** |
| **週次** | `run_consolidation(weekly)`（単一フェーズ、Phase Aなし） → **Neurogenesis reorganization**（低活性チャンク同士の LLM マージ）→ **RAG 再構築** |
| **月次** | **Complete forgetting**（低活性が長期続いたチャンクの削除・アーカイブ）と手続きアーカイブ整理 → **RAG 再構築**（Anima ループは走らない） |

日次は設定で無効化・エピソード閾値・`max_turns` を変えられる。実行が極端に短い場合はリトライがスケジュールされる。

### Forgetting（能動的忘却）

無限に記憶を溜め続けると検索精度が落ちるため、3段階で能動的に忘却する:

| 段階 | 頻度 | 条件 | 処理 |
|------|------|------|------|
| Synaptic downscaling | 日次 | knowledge/episodes: 90日間アクセスなし **かつ** 参照3回未満 → `low` マーク。procedures は別閾値（180日非使用かつ利用計3回未満、または失敗多発で utility が低い等） | 活性 `low` と `low_activation_since` を記録 |
| Neurogenesis reorganization | 週次 | 活性 `low` かつ保護外で、ベクトル類似度 **0.80 以上**のペア | LLM でマージ → 元チャンク削除・ディスク上のソースも統合（`archive/merged/` に退避） |
| Complete forgetting | 月次 | `low` のまま **90日超** かつ `access_count <= 2` | ベクトルから削除し、ソースを `archive/forgotten/` へ |

**保護ルール**（忘却されにくい条件）:

| 対象 | 保護条件 |
|------|---------|
| `skills/`, `shared/users/`（型として） | 常時保護（忘却対象外） |
| `[IMPORTANT]`（`importance == important`） | 原則保護。**ただし**最終アクセスまたは更新から **365日** アクセスがない場合はセーフティネットで保護解除され、通常の忘却フローに乗る（週次統合で内容が知識に取り込まれている想定） |
| Knowledge: `success_count >= 2` | 保護 |
| Procedures: `importance == "important"` / `protected == True` / `version >= 3` | 保護（手続き専用チェック） |

**Procedures の特別ルール**: 180日非活性かつ使用3回未満、または `failure_count >= 3` かつ utility が 0.3 未満の場合はダウンスケール対象になりうる。

---

## 記憶ツールの使い分け

| やりたいこと | ツール | 例 |
|------------|--------|-----|
| キーワードで記憶を探す | `search_memory` | `search_memory(query="API設定", scope="all")` |
| 特定ファイルを読む | `read_memory_file` | `read_memory_file(path="knowledge/api-notes.md")` |
| 記憶を書き込む | `write_memory_file` | `write_memory_file(path="knowledge/new-insight.md", content="...")` |
| 不要な記憶を整理する | `archive_memory_file` | `archive_memory_file(path="knowledge/outdated.md")` |

### scope（検索範囲）の選び方

| scope | 検索対象 | いつ使うか |
|-------|---------|----------|
| `knowledge` | 知識・ノウハウ | 「これについて何か知ってるかな？」 |
| `episodes` | 過去の行動ログ | 「前にこれやったことあるかな？」 |
| `procedures` | 手順書 | 「この作業の手順は？」 |
| `common_knowledge` | 共有リファレンス | 「フレームワークの仕様は？」 |
| `skills` | スキル・共通スキル（ベクトル検索） | 「この作業に使えるスキルは？」 |
| `activity_log` | 直近の行動ログ（ツール実行結果・メッセージ等） | 「さっき読んだメールの内容」「先ほどの検索結果」 |
| `all` | 上記すべて（ベクトル検索 + activity_log BM25をRRFで統合） | 幅広く検索したい場合 |

---

## RAG（ベクトル検索）の仕組み

記憶の検索には RAG（Retrieval-Augmented Generation）が使われる:

1. **インデックス**: `knowledge/`・`episodes/`・`procedures/`・共有 `common_knowledge/` などがチャンク化され、embedding でベクトルストア（既定では Chroma、Anima ごとの永続ディレクトリ）に格納される。ファイルハッシュを `index_meta.json` に保持し、**変更のあったファイルだけ**を差分更新する。
2. **会話要約の別コレクション**: `state/conversation.json` の **`compressed_summary`** を読み、`### ` 見出し単位でチャンク化し、**専用コレクション**（`memory_type: conversation_summary` / メタデータ `source: conversation_gist`）に載せる。通常の knowledge インデックスとは別枠で、長期チャットの圧縮メモを検索対象に含められる。
3. **`.ragignore`**: データディレクトリ（`~/.animaworks/`）直下の `.ragignore` に glob 風パターンを書くと、該当パスはインデックス対象から除外される（コメント行 `#` 可）。
4. **Embedding モデル**: `config.json` の `rag.embedding_model`（未設定時は `intfloat/multilingual-e5-small`）。`ANIMAWORKS_VECTOR_URL` / `ANIMAWORKS_EMBED_URL` が設定されていれば、子プロセスからサーバー経由でベクトル操作・埋め込み生成を委譲でき、埋め込み URL 設定時はローカルへのモデルロードを省略する。
5. **検索**: クエリをベクトル化し、類似度と**時間減衰**・参照頻度などを組み合わせてランキングする。`config.json` の `rag.min_retrieval_score` で結果の下限を切れる。Priming やツール経由の検索でも同じ下限が解決される。
6. **グラフ拡散**: `config.json` の `rag.enable_spreading_activation`（既定 true）と `rag.spreading_memory_types` で、知識グラフによる **spreading activation** を制御できる。
7. **増分更新と再構築**: ファイル変更に応じた再インデックスに加え、日次・週次・月次のライフサイクル後に **インデックス再構築** が走り整合を取る。

RAG は `search_memory` を呼ぶと自動的に使われる。仕組みを意識する必要はないが、
**検索精度を上げるコツ**:
- 具体的なキーワードを含むクエリを使う
- 記憶を書くときはタイトルと内容を明確にする（ファイル名が Priming のキーワード優先度に効く）
- 関連する情報は同じファイルにまとめる
