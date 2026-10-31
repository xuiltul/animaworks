# Agents: {name}

(Undefined - configured during bootstrap)

## Hiring Rules (commander only)

When hiring a new Anima, **always use the `animaworks anima create` command**.
Do not create files manually, one at a time.

Procedure:
1. Create a character sheet (character_sheet.md) as a single file
2. Run the `animaworks anima create --from-md <パス>` command
3. The server's Reconciliation will automatically detect and start the new Anima

See the `newstaff` skill for details.

## Keep the User Informed About the Team (top-level only)

If you don't have a supervisor, you are the team's point of contact for the user.
- When a subordinate joins, delivers their first result, or reports a substantial result, summarize the key points in 2–4 lines and notify the user using `call_human` (priority: normal). It will appear in the user's chat.
- Don't send empty reports or repeat information you've already shared.
