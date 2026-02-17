# Cron: {name}

<!--
=== Cron フォーマット仕様 ===

■ 基本構造
  ## タスク名
  schedule: <5フィールドcron式>
  type: llm | command
  （本文 or command/tool定義）

■ schedule: は必須
  各タスク（## 見出し）の直後に `schedule:` 行を書くこと。
  省略するとタスクが実行されない。

■ 5フィールドcron式
  schedule: 分 時 日 月 曜日
  ┌───── 分 (0-59)
  │ ┌───── 時 (0-23)
  │ │ ┌───── 日 (1-31)
  │ │ │ ┌───── 月 (1-12)
  │ │ │ │ ┌───── 曜日 (0=月 〜 6=日)
  │ │ │ │ │
  * * * * *

■ よく使うスケジュール例
  schedule: 0 9 * * *       # 毎朝 9:00
  schedule: */5 * * * *     # 5分ごと
  schedule: 0 9 * * 0-4     # 平日 9:00（月〜金）
  schedule: 0 17 * * 4      # 毎週金曜 17:00
  schedule: 0 2 * * *       # 毎日 2:00
  schedule: 30 12 1 * *     # 毎月1日 12:30

■ やってはいけない書き方
  ✗ ### cron式          ← 見出しレベルは ## のみ使用
  ✗ schedule: 毎朝9時   ← 自然言語NG、5フィールドcron式で記述
  ✗ （schedule: 行なし） ← 必ず記述すること

■ type の種類
  1. LLM型 (type: llm) - 判断・思考が必要なタスク
  2. Command型 (type: command) - 決定的なbash/tool実行

■ 詳細リファレンス
  → common_skills/cron-management.md を参照
-->

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_task.md に書き出す。

## 週次振り返り
schedule: 0 17 * * 4
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
（記憶の統合 = 脳科学でいう睡眠中の記憶固定化）

<!--
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh

## Slack通知
schedule: 0 9 * * 0-4
type: command
tool: slack_send_message
args:
  channel: "#general"
  message: "おはようございます！"
-->
