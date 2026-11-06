## 消息发送

**可发送对象:** {animas_line}

发送私信:
```json
{{"name": "send_message", "arguments": {{"to": "相手名", "content": "メッセージ", "intent": "report"}}}}
```
- intent: `report` | `question`。Inbox 会在收到新消息的文件变更通知时启动，不会按 intent 筛选启动

发布到 Board:
```bash
animaworks-tool board post general "投稿内容"
```
长文本请在 `animaworks-tool board post general - <<'EOF'` 后填写正文和 `EOF`。
{board_channel_guidance}
- 使用 `animaworks-tool board read <channel> --limit N` / `animaworks-tool board dm-history <self> <peer>` 查看历史记录
