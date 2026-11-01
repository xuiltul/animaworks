# 消息收发操作指南

汇总消息投递、Inbox 启动以及避免对话循环的操作方式。
旧的发送预算、深度拒绝、Inbox cooldown / cascade / intent 过滤器均已废止。

## 发送时仍保留的规则

- `send_message` 的 intent 为 `report` 或 `question`。任务委派使用 `delegate_task`。
- 同一 run 中对同一收件人的 DM 最多 1 条。这是为避免重复发送的保护措施，收件人数量没有上限。
- 同一 run 中对同一 Board 频道的 `post_channel` 最多 1 次。不同 run 可以连续向同一频道投稿。
- 没有每小时、每天的发送预算，也没有按配对对话深度拒绝发送的机制。
- 内部 Anima 之间的对话深度有时会作为诊断目的记录到日志中。这仅用于观测，不会丢弃消息正文或拒绝发送。

## 对了解、感谢、称赞的回应

如果接收内容仅为了解、感谢、称赞，则不回复。不要继续往来，仅在存在追加问题、请求或新信息时进行必要处理。
对工作请求的受理通知，在必要时只返回一次，并附上预期和下一步行动。

## Inbox 的启动与处理

通过文件变更通知检测 Inbox 的新 JSON 文件，若有未读消息则开始 Inbox 处理。
同时启动的 Inbox 处理只有 1 个。执行期间到达的消息合并到下一次统一处理。
为防遗漏文件通知，每 45 秒重新确认未读。Provider 侧失败时等待 `rate_guard` 的恢复时间，并保留未读消息。

消息大量存在时，向 `state/overflow_inbox/` 的退避作为容量保护保留。这不是由发送者或 intent 引起的抑制，退避的消息之后可以确认和处理。

## 记录与确认

`Messenger.send` 将发送消息记录到 activity log。对话在短时间内持续时可能会追加诊断日志，但发送结果不变。
最近的 `message_sent` / `channel_post` 由 `core/memory/priming/outbound.py` 用于预激活。

无法发送时，确认收件人解析、权限、外部频道的实际投递错误。无需等待速率上限或对话深度超限。

## 实现所在

| 角色 | 模块 |
|------|------------|
| 收件人解析・向 Slack / Chatwork 的外部投递 | `core/messaging/outbound.py` |
| 内部 DM 的投递・activity log 记录・深度的诊断日志 | `core/messaging/messenger.py` |
| DM 的重复防止・Board 的 run 内重复防止 | `core/tooling/handler_comms.py` |
| Inbox 的文件 wake・单一执行・provider backoff | `core/runtime/inbox_rate_limiter.py` |
| Inbox 容量保护的 overflow | `core/anima/inbox_overflow.py` |
| 最近发送的提示词注入 | `core/memory/priming/outbound.py` |
