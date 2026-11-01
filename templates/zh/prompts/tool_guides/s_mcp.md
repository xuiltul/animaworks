在发送、投稿、通知、记忆写入之前，如果出现 `[ACTION-RULE]` 就遵照执行。指定的 `read_memory_file(path="...")` 在同一会话中读取，确认后执行。对象：`call_human`、`send_message`、`post_channel`、`write_memory_file`、`gmail_draft/send`、`chatwork_send`、`slack_send`、`discord_send`。
`send_message` 必须包含 intent，同一收件人每次 run 只发一封。收件人数量没有上限。
向部下委派使用 `delegate_task`，自己的工作直接用 Bash 进行。
其他（supervisor、vault、channel、background、外部工具）使用 `animaworks-tool <tool> <subcommand>`。列表：`animaworks-tool --help`。
