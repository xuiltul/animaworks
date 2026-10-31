# Project memory update

## Scope and safeguards

{anima_name}, review only the following new records.

{episodes_summary}

Save only human-confirmed instructions, customer-specific facts, and reusable environment-specific procedures or lessons. Keep PR SHAs and approval status, one-time results, and descriptions of general operations in the case history; do not add them to long-term knowledge.

If there are items to save, use `search_memory` to find existing relevant files and `read_memory_file` to read the originals before updating them. Create new files only when no relevant item already exists. Retain the source, date, and confidence level; do not turn speculation into fact or mix details from different customers or cases.

Preserve original records, confirmed instructions, approval conditions, and important tags. `IDENTITY.md`・`SOUL.md`・`AGENTS.md`・Automatic permission rewriting and overall deduplication are not part of this task. If no changes are needed, do not rewrite anything. Complete this using memory operations only; do not use `delegate_task`・`submit_tasks`・`send_message`.

After completion, briefly report only the files changed and the rationale.
