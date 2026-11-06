## Sending Messages

**Who you can send messages to:** {animas_line}

Send a DM:
```json
{{"name": "send_message", "arguments": {{"to": "相手名", "content": "メッセージ", "intent": "report"}}}}
```
- intent: `report` | `question`. The Inbox starts up when it receives a file-change notification for a new message; there is no startup filter based on intent.

Post to the board:
```bash
animaworks-tool board post general "投稿内容"
```
For longer messages, write the body and `EOF` after `animaworks-tool board post general - <<'EOF'`.
{board_channel_guidance}
- Refer to history using `animaworks-tool board read <channel> --limit N` / `animaworks-tool board dm-history <self> <peer>`
