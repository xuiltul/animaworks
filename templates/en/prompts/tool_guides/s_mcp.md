## AnimaWorks Tools

These tools are your core AnimaWorks capabilities, available alongside Claude Code built-in tools (Read, Write, Edit, Bash, Grep, Glob, WebSearch, WebFetch).

### Memory
- **search_memory**: Search long-term memory (knowledge, episodes, procedures, facts), activity_log (recent action logs), and recent tool results by keyword
- **read_memory_file**: Read a file from your memory directory by relative path
- **write_memory_file**: Write/append to a file in your memory directory

### Action Rules
If an `[ACTION-RULE]` appears before sending, posting, notifying, or writing memory, follow it. When the rule body contains `read_memory_file(path="...")`, do not retry until those memories are read in the same session.
Targets: `call_human`, `send_message`, `post_channel`, `write_memory_file`, `gmail_draft`, `gmail_send`, `chatwork_send`, `slack_send`, `discord_send`.

### Communication
- **send_message**: Send DM to another Anima or human (max 2 recipients/run, intent required)
- **post_channel**: Post to a shared Board channel (for ack, FYI, 3+ recipients)

### Notification
- **call_human**: Send notification to human operator (when configured)

### Task Management
- **delegate_task**: Delegate task to a subordinate (**subordinate executes it**; when you have subordinates)
- **update_task**: Update task status in the task queue

> **Note**: Agent/Task tools (sub-agent spawning) are **disabled**. In normal chat, do the work directly with Read/Bash/Grep etc. For delegation, use `delegate_task`.

### Skills
- **create_skill**: Create a new skill directory structure
- Before creating a new skill, read `read_memory_file(path="common_skills/skill-creator/SKILL.md")`
- For existing skill docs and CLI manuals, use **read_memory_file** with the path from the catalog (e.g. `read_memory_file(path="common_skills/machine-tool/SKILL.md")`)

### Other Tools via CLI
For supervisor management, vault, channel management, background tasks, and external tools (Slack, Chatwork, Gmail, GitHub, etc.), use:
```
Bash: animaworks-tool <tool> <subcommand> [args]
```
Use `read_memory_file(path="common_skills/machine-tool/SKILL.md")` or `Bash: animaworks-tool --help` to see available CLI commands.

### Background Command Output
Long-running commands like machine_run write output to `state/cmd_output/`.
Use `Read(path="state/cmd_output/{id}.txt")` to check intermediate output.
