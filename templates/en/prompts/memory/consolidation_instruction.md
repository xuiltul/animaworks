# Memory update (explicitly enabled daily maintenance)

## Scope and safeguards

{anima_name}, examine only these new records:

{episodes_summary}

Preserve confirmed human instructions, customer-specific facts, and reusable environment-specific procedures or lessons. Keep PR/commit approval status, one-off results and general tool usage in task history instead of generating long-term knowledge.

For an item worth preserving, find related files with `search_memory`, read their originals with `read_memory_file`, then update them. Create a file only when no existing file covers it. Retain source, date and confidence; do not convert inference into fact or combine details belonging to different customers or projects.

Preserve raw records, confirmed instructions, approval conditions and important tags. Rewriting identity.md, injection.md, permissions, or reorganizing the whole library is outside this task. If there is no useful change, write nothing. Use memory operations only; do not use `delegate_task`, `submit_tasks`, or `send_message`.

Briefly report changed files and their supporting evidence.
