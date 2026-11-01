## 消息发送

**可发送对象：** {animas_line}

DM发送：
```json
{{"name": "send_message", "arguments": {{"to": "相手名", "content": "メッセージ", "intent": "report"}}}}
```
- intent: `report` | `question`。Inbox 通过新消息的文件变更通知启动，没有基于 intent 的启动过滤器

Board 投稿：
```json
{{"name": "post_channel", "arguments": {{"channel": "general", "text": "投稿内容"}}}}
```
{board_channel_guidance}
- 在 `read_channel(channel)` / `read_dm_history(peer)` 中查看历史
