# call_human 指南 — 向人类发送通知并接收回复

## 概述

`call_human` 是向人类管理员发送通知的工具。通知始终会送达 Web UI 的聊天界面和 Toast 通知，如果配置了 Slack 等通知频道，也会送达那里。**并非单向通信** — 当人类在 Slack 的线程中回复时，该回复会自动分发到发送方 Anima 的 Inbox。

## 发送

```
call_human(
    subject="件名",
    body="本文（状況・試したこと・依頼内容を含める）",
    priority="normal"  # "normal" | "high" | "urgent"
)
```

| 参数 | 必填 | 说明 |
|-----------|------|------|
| `subject` | MUST | 主题（简洁） |
| `body` | MUST | 正文（包含状况、已尝试的事项、请求） |
| `priority` | MAY | `normal`（默认）, `high`, `urgent` |

### 何时使用

- 紧急度为“高”的问题（数据丢失风险、安全、服务停止）
- 需要超出自己判断范围的决策时
- 顶层 Anima 没有上司时的升级

详细判断标准请参阅 `troubleshooting/escalation-flowchart.md`。

### 区分使用

- `call_human` 是与 `send_message` 不同的人类通知路径。除了需要紧急处理的情况外，顶层 Anima 向用户简短告知团队成果时也可使用

## 接收回复

### 机制

1. 通过 `call_human` 发送通知后，会送达 Web UI 的聊天，如果配置了 Slack，也会在 Slack 中发布消息
2. 人类在该消息的 **线程（thread）** 中回复
3. 回复会自动路由到发送方 Anima 的 Inbox
4. 在下一个 Inbox 处理周期（通常 2 秒内检测到）处理回复

### 回复消息的特征

接收到的回复消息具有以下属性:

- `source`: `"slack"`（经由 Slack 的回复）
- `from_person`: `"slack:U..."` 格式（Slack 用户 ID）
- 与普通 Inbox 消息一样处理

### 等待回复时

发送 `call_human` 后，如果需要等待回复:

1. 将等待状态记录到 `state/current_state.md`
2. 回复会在下一次 Inbox 处理时自动送达
3. 即使没有立即回复，人类回复时也会自动分发

### 响应回复

收到回复后，可以像普通 Inbox 消息一样响应。不过，由于回复对象是 Slack 用户，应使用聊天响应或进一步的 `call_human` 来应对，而不是 `send_message`。

## 注意事项

- 回复路由仅支持以 **Bot Token 模式**（`chat.postMessage`）发送的通知。Webhook 模式下回复不会送达（这是管理员配置的问题，并非 Anima 侧可控制）
- 回复映射保留 **7 天**。超过 7 天的线程回复不会送达
- 即使人类没有回复，必要时也可以再次通过 `call_human` 催促
