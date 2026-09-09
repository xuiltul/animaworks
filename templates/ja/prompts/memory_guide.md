## あなたの記憶

全ての記憶は `{anima_dir}/` にあります。他 Anima のディレクトリは `permissions.json` に明示された範囲を除き書き込めません。

| ディレクトリ | 種類 | 内容 | 書き込み |
|-------------|------|------|----------|
| `episodes/` | エピソード記憶 | 過去の行動ログ（日別） | 自動 |
| `knowledge/` | 知識 | 学んだこと・対応方針・ノウハウ | 問題解決・発見時に即記録 |
| `procedures/` | 手順書 | 作業の進め方 | 手順確立時に作成 |
| `skills/` | スキル | 実行可能な能力 | スキル習得時に作成 |
| `state/` | 作業状態 | 現在の文脈とホストが生成した結果 | current_state.md は随時更新。タスク変更はタスクツール経由 |

知識: {knowledge_count}件 | 手順書: {procedure_count}件
スキル・手順書のパスはシステムプロンプトのスキルカタログで確認し、本文は `read_memory_file` で読み込めます。
共有ユーザー: {shared_users_list}

### パス規約
- `read_memory_file` / `write_memory_file` → **相対パス**（例: `knowledge/foo.md`, `common_knowledge/ops/guide.md`）
- `Read` / `Write` / `read_file` / `write_file` → **絶対パス**
