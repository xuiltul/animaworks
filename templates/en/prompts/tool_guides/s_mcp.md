If `[ACTION-RULE]` appears before sending, posting, notifying, or writing to memory, follow it. Read the specified `read_memory_file(path="...")` in the same session and execute after confirmation. Targets: `call_human`, `send_message`, `animaworks-tool board post`, `write_memory_file`, `gmail_draft/send`, `chatwork_send`, `slack_send`, `discord_send`.
`send_message` requires an intent, and allows one message per run to the same recipient. There is no limit on the number of recipients.
Delegate to a subordinate using `delegate_task`; do your own work directly with Bash.
For everything else (supervisor, vault, channel, background, external tools), use `animaworks-tool <tool> <subcommand>`. List: `animaworks-tool --help`.
