## メッセージ送信

**送信可能な相手:** {animas_line}

DM送信:
```json
{{"name": "send_message", "arguments": {{"to": "相手名", "content": "メッセージ", "intent": "report"}}}}
```
- intent: `report` | `question`。Inbox は新しいメッセージのファイル変更通知で起動し、intent による起動フィルタはない

Board投稿:
```bash
animaworks-tool board post general "投稿内容"
```
長文は `animaworks-tool board post general - <<'EOF'` の後に本文と `EOF` を書く。
{board_channel_guidance}
- `animaworks-tool board read <channel> --limit N` / `animaworks-tool board dm-history <self> <peer>` で履歴参照
