# Cron: {name}

<!--
=== Cron 格式规范 ===

■ 基本结构
  ## 任务名
  schedule: <5字段cron表达式>
  type: llm | command
  （正文 or command/tool定义）

■ schedule: 为必填
  在每个任务（## 标题）之后紧接着写 `schedule:` 行。
  若省略，任务将不会执行。

■ 5字段cron表达式
  schedule: 分 时 日 月 星期
  ┌───── 分 (0-59)
  │ ┌───── 时 (0-23)
  │ │ ┌───── 日 (1-31)
  │ │ │ ┌───── 月 (1-12)
  │ │ │ │ ┌───── 星期 (0=周一 〜 6=周日)
  │ │ │ │ │
  * * * * *

■ 常用调度示例
  schedule: 0 9 * * *       # 每天早上 9:00
  schedule: */5 * * * *     # 每5分钟
  schedule: 0 9 * * 0-4     # 工作日 9:00（周一〜周五）
  schedule: 0 17 * * 4      # 每周五 17:00
  schedule: 0 2 * * *       # 每天 2:00
  schedule: 30 12 1 * *     # 每月1日 12:30

■ 禁止的写法
  ✗ ### cron表达式          ← 标题层级仅使用 ##
  ✗ schedule: 每天早上9点   ← 自然语言不可，须用5字段cron表达式描述
  ✗ （无 schedule: 行） ← 必须书写

■ type 的种类
  1. LLM型 (type: llm) - 需要判断、思考的任务
  2. Command型 (type: command) - 确定性的bash/tool执行

■ 选项（仅command型）
  skip_pattern: <正则表达式>       — 若stdout匹配则跳过LLM分析
  trigger_heartbeat: false       — 即使有输出也不触发LLM分析

■ 详细参考
  → 参见 common_skills/cron-management.md
-->

## 每日早晨的业务计划
schedule: 0 9 * * *
type: llm
从长期记忆中确认昨日的进度，并规划今日的任务。
依据理念与目标判断优先级。
将结果写入 state/current_state.md。

## 每周回顾
schedule: 0 17 * * 4
type: llm
重读本周的episodes/，提取模式并整合到knowledge/中。
（记忆的整合 = 脑科学中所说的睡眠期间的记忆固化）

<!--
## 执行备份
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh

## Slack通知
schedule: 0 9 * * 0-4
type: command
tool: slack_send
args:
  channel: "#general"
  message: "早上好！"
-->
