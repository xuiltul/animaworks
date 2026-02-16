# Cron: {name}

<!--
Cron supports two types:
1. LLM型 (type: llm) - Requires judgment and thinking
2. Command型 (type: command) - Deterministic bash/tool execution

Examples below demonstrate both types.
-->

## 毎朝の業務計画（毎日 9:00 JST）
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_task.md に書き出す。

## 週次振り返り（毎週金曜 17:00 JST）
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
（記憶の統合 = 脳科学でいう睡眠中の記憶固定化）

<!--
## バックアップ実行（毎日 2:00 JST）
type: command
command: /usr/local/bin/backup.sh

## Slack通知（平日 9:00 JST）
type: command
tool: slack_send_message
args:
  channel: "#general"
  message: "おはようございます！"
-->
