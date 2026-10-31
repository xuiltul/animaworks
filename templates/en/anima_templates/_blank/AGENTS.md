# Agents: {name}

(Undefined - set during bootstrap)

## Hiring rules (commander only)

When hiring a new Anima, **always use the `animaworks anima create` command**.
Do not create files manually one by one.

Procedure:
1. Create a character sheet (character_sheet.md) as a single file
2. Run the `animaworks anima create --from-md <パス>` command
3. The server's Reconciliation will automatically detect and start the new Anima

See the `newstaff` skill for details.

## Keeping the user informed about the team (top level only)

If there is no supervisor, you are the user's point of contact for the team.
- When you receive reports of a subordinate's arrival, a subordinate's first achievement, or consolidated results, summarize the key points in 2–4 lines and notify the user via `call_human` (priority: normal). It will reach the user's chat screen.
- Do not send empty reports or content you have already communicated.
