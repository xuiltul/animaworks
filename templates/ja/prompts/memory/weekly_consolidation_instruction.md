# 記憶統合タスク（週次）

{anima_name}、1週間分の記憶を整理する時間です。

## 作業手順

### 1. 知識ファイルの棚卸し
`Glob` で knowledge/ 内のファイル一覧を確認してください。
ファイル数が多い場合は `search_memory` で類似トピックを検索し、重複候補を特定してください。

### 2. 重複・類似ファイルの統合
重複や類似の knowledge/ ファイルが見つかった場合:
1. `read_memory_file` で両方の内容を確認
2. 統合した内容で新しいファイルを `write_memory_file` で作成（または良い方に追記）
3. 古い方を `archive_memory_file` でアーカイブ

**注意**: 統合時、元ファイルに `[IMPORTANT]` タグがある場合は統合先にも必ず残すこと（忘却防止に使用される）

### 3. [IMPORTANT] 知識の概念昇華
作成から30日以上経過した `[IMPORTANT]` 付き knowledge/ ファイルを概念統合する。具体的な出来事の記録を、普遍的な原則・ルールに昇華させる作業。

1. `search_memory` で `[IMPORTANT]` を含む knowledge/ を検索し、30日以上前のものを `read_memory_file` で確認
2. 関連テーマでグループ化し、各グループから抽象的な原則を抽出
3. `concept-{テーマ}.md` として `write_memory_file` で作成（本文先頭に `[IMPORTANT]` 付与）
4. 元ファイルから `[IMPORTANT]` タグを除去（ファイル自体は残す。自然に忘却される）

孤立した `[IMPORTANT]`（関連グループなし）や30日未満のものはスキップ。既に概念レベルのものは再統合不要。

### 4. 手続き知識の整理
`Glob` で procedures/ 内のファイルを確認し:
- 古くなった手順 → 現状に合わせて更新するか、`archive_memory_file` でアーカイブ
- 使われていない手順 → アーカイブを検討
- 類似の手順 → 統合

### 5. 古いエピソードの圧縮
`Glob` で episodes/ を確認し、30日以上前のファイルがあれば:
- `read_memory_file` で内容を確認
- [IMPORTANT] タグがないものは要点のみに圧縮して `write_memory_file` で上書き

### 6. 知識の矛盾解消
矛盾する知識ファイルがないか確認し:
- 最新の情報に基づいて正確な方を残す
- 古い方は `archive_memory_file` でアーカイブ

### 7. injection.md の整理
injection.md の文字数を確認してください。5000文字を超えている場合:
1. `read_memory_file(path="injection.md")` で内容を確認
2. 「役割定義」と「絶対遵守ルール」以外の内容を特定する
3. 業務ルール → knowledge/ に `[IMPORTANT]` 付きで移動
4. 手順的な内容 → procedures/ に移動
5. 一時的な指示（期限切れ・すでに完了）→ 削除
6. injection.md を上書きして簡潔にする

完了後、実施した内容のサマリーを出力してください。
