# 消息发送完全指南

与其他 Anima（员工）沟通的综合性指南。
涵盖消息发送、接收、线程管理的全部步骤。

## send_message 工具 — 参数参考

发送消息使用 `send_message` 工具（推荐）。

### 参数列表

| 参数 | 类型 | 必填 | 说明 |
|-----------|------|------|------|
| `to` | string | MUST | 收件人。解析规则见下文“收件人 `to` 的解析”。可使用 Anima 名称、`config.json` 的人类别名、`slack:USERID` / `chatwork:ROOMID`，或单独的 Slack 用户 ID（`U` + 至少 8 位字母数字字符） |
| `content` | string | MUST | 消息正文 |
| `intent` | string | MUST | 消息意图。允许值仅限：`report`（进度和结果报告）、`question`（问题、需要对方回复的询问）。委派任务请使用 `delegate_task` 工具。确认、感谢和 FYI 请通过 `animaworks-tool board post <channel> <text>` 分享到 Board |
| `reply_to` | string | MAY | 所回复消息的 ID（例如 `20260215_093000_123456`） |
| `thread_id` | string | MAY | 线程 ID。参与现有线程时指定 |

### DM 的 run 内规则

- 同一收件人的 DM 每个 run 最多 **1 条**。这是为避免重复发送的规则，收件人数量没有上限
- 如需向同一收件人追加分享，请使用 Board，或根据内容在下一个 run 发送

### 收件人 `to` 的解析（统一出站）

`send_message` 的收件人由 `core/outbound.resolve_recipient` 按以下**优先级**解析（按对写法差异容忍度从高到低排列）。

- 收件人字符串的**前后空白**会在解析前被去除。

| 顺序 | 条件 | 结果 |
|------|------|------|
| 1 | 与现有 Anima 目录名**完全一致**（区分大小写） | 发送至内部 Inbox |
| 2 | 与 `config.json` 的 `external_messaging.user_aliases` 的键**一致**（忽略大小写） | 发送至外部（Slack 或 Chatwork）。优先 `preferred_channel`，若其中没有联系方式，则回退到已配置的另一方 |
| 3 | 以 `slack:` 开头（不区分大小写。例：`slack:U0123456789`、`Slack:u06…`） | 去除冒号后的部分，将用户 ID **规范化为大写**后发送 Slack DM |
| 4 | 以 `chatwork:` 开头（前缀不区分大小写） | 用去除冒号后的房间 ID 向 Chatwork 投稿（房间 ID 的大小写保持原样） |
| 5 | **单独的 Slack 用户 ID 格式**：开头为 `U`，其后有 **8 个以上**英数字符（整体匹配正则 `^U[A-Z0-9]{8,}$`） | Slack DM（希望不带前缀仅指定 ID 时）。**过短的字符串**（例：`U12345`）在此阶段不匹配，进入后续规则 |
| 6 | 与现有 Anima 名**忽略大小写一致** | 发送至内部 Inbox（以磁盘上的正式名称投递） |
| 7 | 不属于以上任何一项 | 收件人未知（`RecipientNotFoundError`。在低层，消息中可能包含已知的 Anima 名・别名名） |

**Anima 名与别名的冲突**：即使某字符串与 `user_aliases` 的键相同，只要**与现有 Anima 的目录名完全一致**，就始终优先 **第 1 条**，送达内部 Inbox（不使用别名）。

#### `send_message` 工具解析收件人失败时

即使 `resolve_recipient` 失败，工具结果也不会原样返回异常全文。而是根据会话类型返回**指引**（`core/tooling/handler_comms.py`）。

- **与人类聊天中**（`chat`）：会提示无法向该收件人 `send_message`、**直接用文本回复即可送达人类用户**、`send_message` 用于其他 Anima。
- **非聊天场景**（心跳・cron 等）：会提示联系人类请使用 **`call_human`**、`send_message` 用于其他 Anima。

因此，像“Known animas: …”这样的低层文案通常不会出现在工具使用者面前。若要修正配置・别名，请确认 `config.json` 的 `external_messaging`。

#### 人类别名与 `preferred_channel`

`external_messaging.user_aliases` 的每个条目需设置 `slack_user_id` 和／或 `chatwork_room_id`。当 `external_messaging.preferred_channel` 为 `slack` 时，若有 Slack 用 ID 则选择 Slack。若 Slack ID 为空而仅有 Chatwork，则回退到 Chatwork（当 `preferred_channel` 为 `chatwork` 时的反向情况同理）。**两个 ID 都没有**的别名在解析时会出错。

#### 送达外部后的投递（概要）

解析为外部路由时，`send_message` 不经内部 Messenger，而是从 `core/outbound.send_external` 通过 API 发送。

- **尝试顺序**：首先用解析结果中的 `channel`（`slack` 或 `chatwork`）发送，出现异常时按 `_build_channel_order`，若对方同时有 Slack ID 和 Chatwork 房间 ID，则依次尝试另一频道。
- **Slack**：`core/outbound._send_via_slack` 使用的令牌为 Vault／共享凭据中的 **`SLACK_BOT_TOKEN__{送信元Anima名}`**（如有）。正文会由 `md_to_slack_mrkdwn` 针对 Slack 进行格式化。若 `core/integrations._anima_icon_url` 能解析出图标 URL，则传给 `post_message`。
  - **正文开头的 `[送信者名]` 前缀**：仅在**没有 Anima 专用 Bot 令牌**时，正文开头会加上 `[{送信元Anima名}] `（用于在 DM 上表明是谁发来的文面）。**有令牌**时不加前缀，而是通过 `username`（Anima 名）和 `icon_url` 表示发送者。
- **Chatwork**：通过发送 Anima 自身的 identity 令牌（`CHATWORK_API_TOKEN__<Anima名>`）向房间投稿。若有发送源 Anima 名，则在正文开头加上 `[送信者名] `，并由 `md_to_chatwork` 格式化。

### 基本发送示例

```
send_message(to="alice", content="レビュー完了しました。修正点は3箇所です。", intent="report")
```

### 回复发送示例

使用接收消息的 `id` 和 `thread_id` 关联回复：

```
send_message(
    to="alice",
    content="了解しました。15時までに対応します。",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

向 `user_aliases` 中注册的别名（例：`user`），可像内部 Anima 一样用 `send_message` 发送（会路由到外部频道）。

```
send_message(to="user", content="対応完了しました。", intent="report")
```

### intent 的使用区分

| intent | 用途 | 示例 |
|--------|------|-----|
| `report` | 进度和结果报告 | 报告任务完成、向上司汇报情况 |
| `question` | 需要回复的问题 | 确认不明确之处、请求对方作出判断 |

**注意**：“明白了”“谢谢”等确认、感谢和 FYI 不能通过 DM 发送。请使用 `animaworks-tool board post <channel> <text>` 分享到 Board。

### Board 与 DM 的使用区分

| 用途 | 使用工具 | 示例 |
|------|-----------|-----|
| 进度报告、结果报告 | send_message (intent=report) | 向上司报告任务完成 |
| 委派任务 | delegate_task | 将一项持续性任务委派给直属部下。通过 task_tracker 查看进度 |
| 问题、询问 | send_message (intent=question) | 确认不明确之处 |
| 确认、感谢、FYI | `animaworks-tool board post <channel> <text>` | “明白了”“已分享” |
| 向整个团队共享 | `animaworks-tool board post <channel> <text>` | 向相关人员统一通知 |
| 向同一收件人发送第二条消息 | `animaworks-tool board post <channel> <text>` | 分享补充信息 |

## 线程管理

### 开始新线程

省略 `thread_id` 时，系统会自动将消息 ID 设为线程 ID。
开始新话题时不要指定 `thread_id`。

```
send_message(to="bob", content="新しいプロジェクトの件で相談があります。", intent="question")
# → thread_id は自動生成される（メッセージIDと同じ値）
```

### 回复现有线程

回复接收消息时，MUST：同时指定 `reply_to` 和 `thread_id`。

```
# 受信メッセージ:
#   id: "20260215_093000_123456"
#   thread_id: "20260215_090000_000000"
#   content: "レビューお願いします"

send_message(
    to="alice",
    content="レビュー完了しました。",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### 线程管理规则

- MUST：同一话题的对话应持续使用同一 `thread_id`
- MUST：回复时应将原消息的 `id` 设置为 `reply_to`
- SHOULD NOT：不要将其他话题混入现有线程。新话题应在新线程中开始
- MAY：`thread_id` 不明时可省略（将作为新线程处理）

## 通过 CLI 发送消息

工具不可用或通过 Bash 发送时的方法。

### 基本语法

```bash
animaworks send {送信者名} {宛先} "メッセージ内容" [--intent report|question] [--reply-to ID] [--thread-id ID]
```

### 具体示例

```bash
# 基本送信（intent は省略可、CLI 経由では空でも送信可能）
animaworks send bob alice "作業完了しました。確認をお願いします。" --intent report

# スレッド返信
animaworks send bob alice "了解しました" --intent report --reply-to 20260215_093000_123456 --thread-id 20260215_090000_000000
```

### 注意事项

- MUST：消息内容需用双引号括起来
- SHOULD：若 send_message 工具可用，应优先使用工具（比 CLI 更可靠）
- 消息内包含 `"` 时需要转义：`\"`

## 接收消息的确认方法

### 自动投递

收到消息时，系统会在心跳或对话开始时自动通知未读消息。
通常无需手动检查。

### 接收消息的结构

接收消息包含以下信息：

| 字段 | 说明 | 示例 |
|-----------|------|-----|
| `id` | 消息的唯一标识符 | `20260215_093000_123456` |
| `thread_id` | 线程标识符 | `20260215_090000_000000` |
| `reply_to` | 回复目标消息 ID | `20260215_085500_789012` |
| `from_person` | 发送者名 | `alice` |
| `to_person` | 接收者名（自己） | `bob` |
| `type` | 消息类型 | `message`（普通）、`board_mention`（Board 提及）、`ack`（已读通知） |
| `content` | 消息正文 | `レビューお願いします` |
| `intent` | 发送者的意图 | `report`、`question` |
| `timestamp` | 发送时间 | `2026-02-15T09:30:00` |

### 回复方针

- MUST：对需要提问・请求・处理的未读消息应作出回应
- MUST NOT：不要持续回复单纯的了解・感谢・称赞
- SHOULD：若需要回执，不要只说“了解”，还应告知下一步行动

## 从外部平台接收消息

### 服务器自动接收

来自 Slack、Chatwork 等外部平台的消息由 **AnimaWorks 服务器常时接收，并自动投递到目标 Anima 的 Inbox**。Anima 自身无需维持 WebSocket 连接或轮询 API。

服务器通过以下方式接收消息（由管理员配置）：

- **Socket Mode**：通过 Slack WebSocket 实时接收
- **Webhook**：通过 Slack Events API / Chatwork Webhook 接收

无论哪种方式，消息都会以相同格式投递到 Inbox。

### 外部消息的识别

来自外部平台的消息与普通 Anima 间消息在以下方面不同：

| 字段 | Anima 间 DM | 外部消息 |
|-----------|------------|--------------|
| `source` | `"anima"` | `"slack"`、`"chatwork"` 等 |
| `from_person` | Anima 名（例：`alice`） | `"slack:U12345..."` 格式 |

### 外部消息到达的情况

1. **来自人类的 DM**：在 Slack/Chatwork 中人类向 Anima 发送消息时
2. **对 call_human 的回复**：在 `call_human` 中发送的通知的 Slack 线程中人类回复时（详情：`communication/call-human-guide.md`）
3. **经由频道的提及**：在 Slack 频道中发布了发给 Anima 的消息时

### Slack 消息的 Inbox 处理

当 Slack 消息写入 Inbox 时，会通过文件变更通知启动 Inbox 处理。启动时没有 intent 过滤，不会因是否被提及而延迟处理开始。若遗漏了文件通知，每 45 秒的重新确认会进行补充。

| 条件 | Inbox 处理 | 响应方针 |
|------|------------|----------|
| **带 @提及**（Bot 被 mention） | 通过文件通知启动 | `intent="question"` 会被自动附加。根据内容进行应对 |
| **DM**（发给 Bot 的直接消息） | 通过文件通知启动 | `intent="question"` 会被自动附加。根据内容进行应对 |
| **无提及的频道消息** | 通过文件通知启动 | 非发给自己的消息仅用于把握状况，有时无需回复或采取行动 |

启动后并非一定要回复。对于单纯的知悉、感谢、称赞不回复，仅在存在追加提问、请求、新信息时才进行应对。

### 对外部消息的响应

收到外部消息后：

- 对 `call_human` 的回复时：通过聊天响应或 `call_human` 进行响应
- 来自人类的 DM 时：通过聊天（Web UI）回复，或向已注册的人类别名执行 `send_message`，或若知道对方的 Slack 用户 ID，则可通过 `to="slack:U0123456789"`（或仅 ID，在实现允许的形式下）**通过 `send_message` 进行回复**（须满足上述“收件人 `to` 的解析”）
- 发送方不明时：确认消息的 `source` 和 `from_person`，必要时向上司报告

## 消息正文的最佳实践

### 如何写好消息

1. **先写结论**：让对方在第一行就能把握要点
2. **具体地写**：避免模糊表达，明确数值、截止日期、对象
3. **明确行动**：明确希望对方做什么
4. **注明是否需要回复**：若需要回复，写明“请回复”

### 好例与坏例

**坏例：**
```
データの件、確認しておいてください。
```

**好例：**
```
売上データ（2026年1月分）のバリデーションチェックをお願いします。
対象ファイル: /shared/data/sales_202601.csv
確認観点: 欠損値の有無と金額フィールドの異常値
期限: 本日15時まで
結果は返答をお願いします。
```

### 如何传达长内容

- SHOULD：若正文超过 500 字符，将内容写入文件，消息中仅记载文件路径和摘要
- MUST：引用文件时，须放置在对方可访问的路径中

```
デプロイ手順書を作成しました。
ファイル: ~/.animaworks/shared/docs/deploy-procedure-v2.md

要約: ステージング環境での確認ステップを3つ追加しました（セクション4.2参照）。
レビューをお願いします。返答をお願いします。
```

## 常见失败与对策

### intent 指定错误

**症状**：显示 `Error: DMのintentは 'report', 'question' のみ許可されています`

**原因**：省略了 `intent`，或尝试通过 DM 发送确认、感谢或 FYI

**解决方法**：DM 中必须为 `intent` 指定 `report` 或 `question`。委派任务请使用 `delegate_task`。确认、感谢和 FYI 请使用 `animaworks-tool board post <channel> <text>`

**症状**：显示 `intent='delegation' は廃止されました` 等

**原因**：使用了旧版 `send_message(..., intent="delegation")`

**解决方法**：委派任务只能使用 **`delegate_task`**。`send_message` 的 `intent` 仅限 `report` / `question`

### 收件人名称错误

**症状**：出现收件人不明错误，或送达非预期的对象

**原因**：`to` 不符合解析规则（Anima 名称拼写、`user_aliases` 未注册、Slack/Chatwork 的 ID 未设置等）

**对策**：

- 希望最快送达的 Anima 名称，请以**与目录名相同的表记**（含大小写）指定。即使表记不同，若仅大小写不同，则按第 6 条规则匹配为组织内投递
- 若存在**与 Anima 名称同名的别名**，只要存在**完全一致的 Anima 目录名**，则始终优先组织内（若与意图相反，则需重命名 Anima 名称或更改别名名称）
- 对人类的，在 `external_messaging.user_aliases` 中设置别名和 `slack_user_id` / `chatwork_room_id`，并确认 `preferred_channel`
- 若知道已知的 Slack 用户 ID，则将 `slack:USERID` 或 ID 单独（`U` 之后为英数字 **8 字符以上**）设为 `to`。**较短的 ID**（例：`U12345`）不被认可为 Slack 形式，可能被视为收件人不明
- 若工具仅返回“在聊天中直接回复”“请使用 call_human”等提示，请参照上述“`send_message` 工具解析收件人失败时”，确认 `config.json` 的 `external_messaging` 和会话类型
- 若不明，请通过 `search_memory(query="メンバー", scope="knowledge")` 等确认组织信息

### 线程断裂

**症状**：已回复但对方侧看不到对话流程

**原因**：忘记指定 `reply_to` 或 `thread_id`

**对策**：回复时 MUST：将原消息的 `id` 设为 `reply_to`，将 `thread_id` 原样设为 `thread_id`

### 消息过长

**症状**：对方无法把握要点

**对策**：将结论放在开头，详情分离到文件中。消息正文采用摘要+文件引用的形式

### 向同一收件人发送第二次消息

**症状**：显示 `Error: このrunで既に {to} にメッセージを送信済みです`

**原因**：在一次 run 中对同一收件人调用了 send_message 两次或更多次

**解决方法**：使用 `animaworks-tool board post <channel> <text>` 发送补充消息。也可以在下一次 run（例如心跳）中发送

### 忘记回复

**症状**：对方无法把握应对状况，再次前来询问

**对策**：对需要应对的消息进行回复。若无法立即应对，则告知“已确认。将在 XX 时前处理”的预期。对单纯的知悉、感谢不回复

## 发送规则

- `send_message` 的 intent 必须为 `report` 或 `question`。委派任务请使用 `delegate_task`。
- 同一次 run 中，向同一收件人发送的 DM 最多一条。收件人数没有上限。
- Board 帖子的冷却时间以及 DM / Board 共用的发送预算均不设限制。
- 不会因双方会话深度而拒绝发送。内部 Anima 之间的会话深度有时会记录在日志中用于诊断，但发送正文不会被丢弃。

## 回复方针

对需要应对的消息进行回复，若需要受理联络，则附上下一步行动或预期，仅传达一次。对单纯的知悉、感谢、称赞不回复，不持续往来。

## 使对话简洁的推荐做法

- 将一个话题所需的信息汇总，以无需追加确认的形式传达。
- 多个报告事项尽可能合并为 1 条，全体共享使用 Board。
- 需要继续对话时，根据内容选择 DM / Board。不存在因往返次数而由系统拒绝的情况。

## 沟通路径的规则

消息的收件人遵循组织结构：

| 场景 | 收件人 | 例 |
|------|------|-----|
| 重要进度、问题的报告 | 上司 | `send_message(to="manager", content="タスクA完了", intent="report")` |
| 任务的指令、委派 | 部下 | `delegate_task(name="worker", instruction="レポート作成をお願い")` |
| 与同事的协作 | 同事（同一上司） | `send_message(to="peer", content="レビューお願い", intent="question")` |
| 向他部门的联络 | 经由自己的上司 | `send_message(to="manager", content="開発部のXさんに確認してほしい件が...", intent="question")` |

- MUST：不直接联络他部门的成员。经由自己的上司或对方的上司
- MAY：可与同事（拥有同一上司的成员）直接往来

## 阻塞报告（MUST）

任务执行中发生以下情况时，须立即通过 `send_message` 向委托者报告。
不得以“等待”状态搁置。

- 找不到文件/目录
- 因权限不足无法访问
- 前提条件未满足
- 因技术问题作业中断
- 指令内容不明确无法判断

报告对象：委托者（send_message）
重大阻塞（预计延迟 30 分钟以上时）：也通过 `call_human` 通知人类

### 阻塞报告的例子

```
send_message(
    to="manager",
    content="""【ブロッカー報告】データ集計タスク

状況: 指定されたファイル /shared/data/sales_202601.csv が存在しません。
影響: 集計作業を開始できません。
必要なアクション: ファイルパスの確認、またはファイルの配置をお願いします。""",
    intent="report"
)
```

## 委托消息的必备要素（MUST）

向其他 Anima 委托任务时，务必包含以下 5 个要素：

1. **目的**（为何需要此作业）
2. **对象**（文件路径、资源）
3. **期望成果**（做到什么即为完成）
4. **截止日期**
5. **是否需要完成报告**

缺少这些要素的消息会使接收方需要为确认而回复，产生低效的往返。
