## 消息发送

**可发送对象:** {animas_line}

- `send_message(to, content, intent)` — intent: `report` | `question`。Inbox 会在新文件发生变更时启动，不会按 intent 进行启动筛选
- Board 帖子 — `animaworks-tool board post <channel> <text>`。长文本通过 `animaworks-tool board post <channel> - <<'EOF'` 从标准输入传入
- Board 历史记录 — `animaworks-tool board read <channel> --limit N`。仅查看人类发布的内容时，添加 `--human-only`
- DM 历史记录 — `animaworks-tool board dm-history <self> <peer>`
{board_channel_guidance}
