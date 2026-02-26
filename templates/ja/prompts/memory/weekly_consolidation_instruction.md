# 記憶統合タスク（週次）

{anima_name}、1週間分の記憶を整理する時間です。

## 作業手順

### 1. 知識ファイルの棚卸し
`list_directory` で knowledge/ 内のファイル一覧を確認してください。
ファイル数が多い場合は `search_memory` で類似トピックを検索し、重複候補を特定してください。

### 2. 重複・類似ファイルの統合
重複や類似の knowledge/ ファイルが見つかった場合:
1. `read_memory_file` で両方の内容を確認
2. 統合した内容で新しいファイルを `write_memory_file` で作成（または良い方に追記）
3. 古い方を `archive_memory_file` でアーカイブ

### 3. 手続き知識の整理
`list_directory` で procedures/ 内のファイルを確認し:
- 古くなった手順 → 現状に合わせて更新するか、`archive_memory_file` でアーカイブ
- 使われていない手順 → アーカイブを検討
- 類似の手順 → 統合

### 4. 古いエピソードの圧縮
`list_directory` で episodes/ を確認し、30日以上前のファイルがあれば:
- `read_memory_file` で内容を確認
- [IMPORTANT] タグがないものは要点のみに圧縮して `write_memory_file` で上書き

### 5. 知識の矛盾解消
矛盾する知識ファイルがないか確認し:
- 最新の情報に基づいて正確な方を残す
- 古い方は `archive_memory_file` でアーカイブ

完了後、実施した内容のサマリーを出力してください。
