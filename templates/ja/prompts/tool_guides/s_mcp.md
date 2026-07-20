## AnimaWorks Tools

これらのツールはAnimaWorksのコア機能です。Claude Code組込みツール（Read, Write, Edit, Bash, Grep, Glob, WebSearch, WebFetch）と併用できます。

### Memory
- **search_memory**: 長期記憶（knowledge, episodes, procedures）、activity_log（直近の行動ログ）、直近のツール結果をキーワード検索
- **read_memory_file**: 記憶ディレクトリ内のファイルを相対パスで読む
- **write_memory_file**: 記憶ディレクトリ内のファイルに書き込みまたは追記

### Action Rules
送信・投稿・通知・記憶書き込みの直前に `[ACTION-RULE]` が出たら、表示されたルールに従う。本文に `read_memory_file(path="...")` がある場合、同じセッションでその記憶を読むまで再実行しない。
対象: `call_human`, `send_message`, `post_channel`, `write_memory_file`, `gmail_draft`, `gmail_send`, `chatwork_send`, `slack_send`, `discord_send`。

### Communication
- **send_message**: 他のAnimaまたは人間にDM送信（1 runあたり最大2宛先、各1通、intent必須）
- **post_channel**: 共有Boardチャネルに投稿（ack、FYI、3人以上への通知用）

### Notification
- **call_human**: 人間オペレーターに通知送信（設定時）

### Task Management
- **delegate_task**: 部下にタスクを委譲（**部下が実行する**。部下がいる場合のみ）
- **update_task**: タスクキューのステータスを更新

> **注意**: Agent/Taskツール（サブエージェント）は**無効**です。通常チャットではRead/Bash/Grep等で直接実行してください。部下への委譲には `delegate_task` を使ってください。

### Skills
- **create_skill**: 新しいスキルディレクトリを作成する
- 新規スキル作成前に `read_memory_file(path="common_skills/skill-creator/SKILL.md")` を読む
- 既存のスキル文書・CLIマニュアルは **read_memory_file** でカタログに示されたパスを指定して読む（例: `read_memory_file(path="common_skills/machine-tool/SKILL.md")`）

### Other Tools via CLI
スーパーバイザー管理、vault、チャネル管理、バックグラウンドタスク、外部ツール（Slack, Chatwork, Gmail, GitHub等）は:
```
Bash: animaworks-tool <tool> <subcommand> [args]
```
利用可能なCLIコマンドは `read_memory_file(path="common_skills/machine-tool/SKILL.md")` または `Bash: animaworks-tool --help` で確認。

### Background Command Output
machine_run等の長時間コマンドの出力は `state/cmd_output/` に保存されます。
`Read(path="state/cmd_output/{id}.txt")` で中間出力を確認できます。
