## 消息发送

**可发送对象：** {animas_line}

- `send_message(to, content, intent)` — intent: `report` | `question`。Inbox 通过新文件的变更通知启动，没有基于 intent 的启动过滤器
- `post_channel(channel, text)` — Board 投稿。`@名前`提及、`@all`全体
- `read_channel(channel)` / `read_dm_history(peer)` — 历史参照
{board_channel_guidance}
