# 記憶統合タスク（日次）

あなた（{anima_name}）の記憶を整理する時間です。以下の手順に従ってください。

## 今日のエピソード

{episodes_summary}

## 解決済みイベント

{resolved_events_summary}

## 今日のアクティビティログ（行動記録）
{activity_log_summary}

※ アクティビティログは行動の記録であり、推論過程は含まれません。
ここから知識を抽出する場合は以下に注意してください:
- 確実に事実と判断できるもののみ knowledge/ に記録
- 推測や解釈が必要なものは confidence: 0.5 で記録
- frontmatter に `source: "activity_log"` を付与

{reflections_summary}

## 既存の知識ファイル一覧

{knowledge_files_list}

## マージ候補（類似ファイルペア）

{merge_candidates}

## エラーパターン（過去24時間）

{error_patterns_summary}

---

## 作業手順

### Step 1: 重複ファイルの統合（MUST — 最優先）

**マージ候補が提示されている場合、すべてのペアについて統合を実施すること。**
加えて、上記のファイル一覧を確認し、同じトピックを扱う重複ファイルを自分で見つけること。

統合手順:
1. `read_memory_file` で両方の内容を確認
2. 情報を併合し `write_memory_file` で一方に書き込む
3. 不要になった方を `archive_memory_file` でアーカイブ
4. `[IMPORTANT]` タグがあれば統合先にも残す

- 「後で統合する」「複雑なので保留」は禁止。この場で完了させること
- 新規ファイル作成より既存ファイルへの統合を常に優先すること

### Step 2: エピソードからの知識抽出

今日のエピソードを確認し、実質的な情報があれば:
1. `search_memory` で関連する既存の knowledge/ / procedures/ を検索
2. 関連ファイルがあれば `read_memory_file` で確認し、`write_memory_file` で追記・更新
3. 該当する既存ファイルがない場合のみ、新規ファイルを作成

### Step 2.5: エラーパターン分析

上記「エラーパターン」セクションを確認し、繰り返し発生しているパターンがあれば:
1. `search_memory` で関連する既存の procedures/ を検索
2. 既存の手順書があれば `read_memory_file` で確認し、`write_memory_file` で追記・更新
3. 該当する既存ファイルがない場合のみ、`procedures/` に新規作成
4. 1回限りのエラーは記録不要（ノイズ）

新規作成時の frontmatter:
```
---
created_at: "YYYY-MM-DDTHH:MM:SS"
confidence: 0.4
auto_consolidated: true
source: "error_trace_analysis"
version: 1
---
```

### Step 3: 品質チェック
- 更新・作成した内容がエピソードの事実と矛盾していないか確認
- ファイル名はトピックを表すわかりやすい名前にすること

## 抽出すべき情報
- 具体的な設定値・認証情報の格納場所
- ユーザーやシステムの識別情報
- 手順・ワークフロー・プロセスの記録
- チーム構成・役割分担・指揮系統
- 技術的な判断とその理由
- 解決済みイベントから得られた教訓と手順

## 重要な制約
- **この作業はあなた自身が直接実行すること（MUST）**。`delegate_task`、`submit_tasks`、`send_message` は使用禁止。記憶操作ツールのみで作業を完結させること
- **Step 1 の統合を省略してはならない**。重複ファイルが存在するのに統合しなかった場合、それは失敗である

## 注意事項
- 挨拶のみの会話や実質的な情報を含まないやり取りは知識化不要
- [REFLECTION] タグ付きエントリは優先的に知識化を検討
- `[IMPORTANT]` タグ付きエントリは**必ず** knowledge/ に抽出すること（MUST）。既存ファイルと重複する場合は追記で統合。**本文にも `[IMPORTANT]` タグを残すこと**
- knowledge/ を新規作成する場合は YAML フロントマターを付与:
  ```
  ---
  created_at: "YYYY-MM-DDTHH:MM:SS"
  confidence: 0.7
  auto_consolidated: true
  success_count: 0
  failure_count: 0
  version: 1
  last_used: ""
  ---
  ```
- 完了後、実施内容のサマリーを出力（統合したペア数・アーカイブ数を含む）
