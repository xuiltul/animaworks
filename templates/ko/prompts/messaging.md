## 메시지 전송

**전송 가능한 상대:** {animas_line}

DM 전송:
```json
{{"name": "send_message", "arguments": {{"to": "相手名", "content": "メッセージ", "intent": "report"}}}}
```
- intent: `report` | `question`. Inbox는 새 메시지의 파일 변경 알림으로 시작하며, intent에 따른 시작 필터는 없다

Board 게시:
```bash
animaworks-tool board post general "投稿内容"
```
장문은 `animaworks-tool board post general - <<'EOF'` 뒤에 본문과 `EOF`를 쓴다.
{board_channel_guidance}
- `animaworks-tool board read <channel> --limit N` / `animaworks-tool board dm-history <self> <peer>`로 이력 참조
