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
- **再固定化**: フロントマターで `failure_count >= 2` かつ `confidence < 0.6` の knowledge は、手続きと同様に LLM による改訂対象になりうる（`ReconsolidationEngine` の knowledge パス）

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
- **再固定化（reconsolidation）**: フロントマターで **`failure_count >= 2` かつ `confidence < 0.6`** のとき、LLM による手順書の改訂が走る。改訂後はカウンタリセット・バージョン番号更新・旧版を `archive/` に退避する（実装: `ReconsolidationEngine`）
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
- 必要なスキルは active skill context、Skill Router、Skill Hub、または `read_memory_file(path="...")` で読む
- スキル本文を常に全部読む必要はない。まず名前・説明・ポインタを見て、必要になった時だけ本文を読む
- 実績のある `procedures/` は probation skill や quarantine skill として昇格することがある
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

既定の `compact` は送信者情報、未完了タスク、明示された常駐ポインタ、直近の送信履歴、人間向け未通知事項を想起します。会話・タスクの依頼や質問では関連知識を検索しますが、通常の heartbeat・cron・報告イベントでは検索しません。広い活動履歴・エピソード・グラフの展開は任意の `full` または明示検索で利用します。`priming.max_tokens` は既定2,000で、通知と必須常駐ルールは別に保持します。Anima ごとの `status.json: priming_profile` でモデルルーティングを変えずに上書きできます。

過去の指示・顧客情報・継続作業が必要な場合に検索してください。全回答で儀式的に検索したり、使用のたびに成功を報告する必要はありません。スキル・手順の本文は必要時に読みます。自動常駐するのは明示された記憶だけで、`[IMPORTANT]` だけでは常駐指定になりません。

副作用のある行動には `[ACTION-RULE]`、権限、承認、重複実行防止が引き続き適用されます。停止された場合は指定された規則を読んでください。信頼できない検索結果は信頼された文脈と分離されます。

日次統合は未処理の活動チャンクからエピソードを生成し、成功した入力を記録します。活動原記録と記憶原本は保持します。知識の書き換えは別段階で既定無効です（`consolidation.knowledge_mutation_enabled`）。週次・月次の変更、蒸留、低活性化、自己修正、スキル自動学習、facts の自動生成も既定無効です。索引・修復・既存 facts の読み取りは維持します。任意の保守でも顧客別詳細・出典・安全規則を保持します。スキップ・変更なしは正常で、再試行の理由にはなりません。

Curator の昇格・退役は既定で提案だけです。安全上のブロックは直ちに隔離でき、運用者の明示操作も可能です。利用結果の件数は診断材料であり、仕事の品質を証明しません。

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
4. **Embedding モデル**: `config.json` の `rag.embedding_model`（未設定時は `intfloat/multilingual-e5-small`）。ChromaDB と埋め込み処理は通常 vector worker 経由で隔離され、必要に応じてサーバーや一時ワーカーに委譲される。
5. **検索**: クエリをベクトル化し、類似度と**時間減衰**・参照頻度などを組み合わせてランキングする。`config.json` の `rag.min_retrieval_score` で結果の下限を切れる。Priming やツール経由の検索でも同じ下限が解決される。
6. **グラフ拡散**: `config.json` の `rag.enable_spreading_activation`（既定 true）と `rag.spreading_memory_types` で、知識グラフによる **spreading activation** を制御できる。
7. **増分更新と再構築**: ファイル変更に応じた再インデックスに加え、日次・週次・月次のライフサイクル後に **インデックス再構築** が走り整合を取る。RAG不整合が検出された場合は、repair が `vectordb` を隔離して再構築できる。

RAG は `search_memory` を呼ぶと自動的に使われる。仕組みを意識する必要はないが、
**検索精度を上げるコツ**:
- 具体的なキーワードを含むクエリを使う
- 記憶を書くときはタイトルと内容を明確にする（ファイル名が Priming のキーワード優先度に効く）
- 関連する情報は同じファイルにまとめる
