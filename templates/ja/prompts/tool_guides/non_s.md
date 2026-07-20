## Tool Usage Guide

全モードで統一されたツールセットが利用可能です。

### File Operations (Claude Code-compatible)
- **Read**: 行番号付きでファイルを読む。大きいファイルはoffset/limitで部分読み取り
- **Write**: ファイルに書き込む。親ディレクトリを自動作成
- **Edit**: ファイル内の特定の文字列を置換（old_stringは一意であること）
- **Bash**: シェルコマンドを実行（permissionsの許可範囲内）
  - 長時間コマンド: `background: true` で非同期実行 → cmd_id + 出力ファイルパスが返る
  - 進捗確認: `Read(path="state/cmd_output/{cmd_id}.txt")` で中間出力を確認
  - 一覧: `Glob(pattern="state/cmd_output/*.txt")` でバックグラウンドタスク一覧
- **Grep**: 正規表現でファイル内を検索
- **Glob**: グロブパターンでファイルを検索
- **WebSearch**: Web検索
- **WebFetch**: URLを取得して返す（markdown形式）

### Memory
- **search_memory**: 長期記憶をキーワード検索
  - scope: knowledge | episodes | procedures | common_knowledge | activity_log | all
- **read_memory_file**: 記憶ディレクトリ内のファイルを相対パスで読む
- **write_memory_file**: 記憶ディレクトリに書き込みまたは追記

### Action Rules
- `[ACTION-RULE]` は送信・投稿・通知・記憶書き込み前のゲートです
- 本文に `read_memory_file(path="...")` が示されたら、同じセッションで必ず読んでから再実行する
- 詳細は `read_memory_file(path="common_knowledge/operations/action-rules-guide.md")`

### Communication
- **send_message**: DM送信（1 runあたり最大2宛先、各1通）
  - intent必須: 'report' または 'question' のみ
  - タスク委譲はdelegate_task。ack/FYI/3人以上はpost_channelを使う
- **post_channel**: 共有Boardチャネルに投稿

### Task Management
- **update_task**: タスクステータスを更新

### Skills
- **create_skill**: 新しいスキルディレクトリを作成する
- 新規スキル作成前に `read_memory_file(path="common_skills/skill-creator/SKILL.md")` を読む
- 既存のスキル文書・CLIマニュアルは **read_memory_file** でカタログのパスを指定して読む

### Pre-Completion Verification
- **completion_gate**: 最終回答を出す前にこのツールを呼んでください。完了前チェックリストが返されます。

### Other Tools via CLI
スーパーバイザー管理、vault、チャネル管理、バックグラウンドタスク、全外部ツール:
```
Bash: animaworks-tool <tool> <subcommand> [args]
```
利用可能なCLIコマンドは `read_memory_file(path="common_skills/machine-tool/SKILL.md")` で確認。
