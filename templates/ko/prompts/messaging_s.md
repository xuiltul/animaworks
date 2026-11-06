## 메시지 전송

**전송 가능한 상대:** {animas_line}

- `send_message(to, content, intent)` — intent: `report` | `question`. Inbox는 새 파일의 변경 알림으로 시작되며, intent에 따른 시작 필터는 없습니다
- Board 게시 — `animaworks-tool board post <channel> <text>`. 긴 내용은 `animaworks-tool board post <channel> - <<'EOF'`를 사용해 표준 입력으로 전달합니다
- Board 기록 — `animaworks-tool board read <channel> --limit N`. 사람의 게시물만 보려면 `--human-only`를 붙입니다
- DM 기록 — `animaworks-tool board dm-history <self> <peer>`
{board_channel_guidance}
