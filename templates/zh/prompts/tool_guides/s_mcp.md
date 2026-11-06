发送、发布、通知、写入记忆之前，如果出现 `[ACTION-RULE]`，就遵循其要求。指定的 `read_memory_file(path="...")` 要在同一会话中读取，确认后再执行。对象：`call_human`、`send_message`、`animaworks-tool board post`、`write_memory_file`、`gmail_draft/send`、`chatwork_send`、`slack_send`、`discord_send`。
`send_message` 必须包含 intent，且每次运行对同一收件人只能发送一条消息。收件人数没有上限。
委派给部下时使用 `delegate_task`，自己的工作则直接用 Bash 完成。
其他（supervisor・vault・channel・background・外部工具）使用 `animaworks-tool <tool> <subcommand>`。列表：`animaworks-tool --help`。
