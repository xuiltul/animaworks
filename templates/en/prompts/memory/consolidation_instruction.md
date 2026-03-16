# Memory Consolidation Task (Daily)

{anima_name}, it is time to organize your memory. Follow the workflow below.

## Today's Episodes

{episodes_summary}

## Resolved Events

{resolved_events_summary}

## Today's Activity Log (behavioral records)
{activity_log_summary}

※ The activity log records actions and does not include reasoning.
When extracting knowledge from it, note the following:
- Record in knowledge/ only what can be confidently judged as fact
- Record items requiring inference or interpretation with confidence: 0.5
- Add `source: "activity_log"` to frontmatter

{reflections_summary}

## Existing knowledge files list

{knowledge_files_list}

## Merge candidates (similar file pairs)

{merge_candidates}

## Workflow

### 1. Review episodes
Review today's episodes above. Identify those containing substantive information.

### 2. Match with existing knowledge
Use `search_memory` to find existing knowledge/ and procedures/ related to today's episodes.
If relevant files are found, use `read_memory_file` to review their content.

### 3. Update or create knowledge
- **Update existing files**: If today's experience suggests updates, read with `read_memory_file` and append or revise with `write_memory_file`
- **Create new knowledge**: If there are new patterns or lessons, create new files in knowledge/ with `write_memory_file`
- **Procedural knowledge**: Record repeatable steps or workflows in procedures/

### 4. Memory consolidation (MUST)
When merge candidates are provided, follow these steps **without fail**:
1. Use `read_memory_file` to review both file contents
2. Write the merged content to the better file with `write_memory_file` (or create a new file)
3. Archive the obsolete one with `archive_memory_file`
4. If the `[IMPORTANT]` tag exists, preserve it in the merged file

Even when no merge candidates are provided, check the existing knowledge files list above against
any knowledge you are about to create. If duplicates exist, prefer updating existing files over creating new ones.

- Update or archive outdated procedures/
- If there are contradictory knowledge items, keep the more accurate one and archive the older one

### 5. Quality check
- Verify that created or updated knowledge files do not contradict today's episode facts
- Use clear, topic-descriptive filenames (alphanumeric and hyphens recommended)

## Information to extract
- Specific configuration values, API keys, and where credentials are stored
- User and system identifiers (IDs, names, roles)
- Procedures, workflows, and process records
- Team structure, role assignments, and chain of command
- Technical decisions and their rationale
- Lessons and procedures from resolved events

## Critical constraints
- **You MUST perform this work yourself directly**. Do NOT use `delegate_task`, `submit_tasks`, or `send_message`. Do not delegate to subordinates or send any messages. Complete all work using only `search_memory`, `read_memory_file`, `write_memory_file`, and `archive_memory_file`

## Notes
- Do not convert greetings-only or substantively empty exchanges into knowledge
- [REFLECTION] tagged entries are conscious insights recorded by the agent; when extracted in the "Reflections" section above, prioritize these for knowledge extraction
- Entries tagged with `[IMPORTANT]` are critical lessons or failure records. You **MUST** extract them into knowledge/. If overlapping with existing knowledge, merge by appending to or updating the existing file. **Keep the `[IMPORTANT]` tag in the knowledge file body** (used for forgetting protection and RAG search boosting)
- When creating new knowledge/ files, add YAML frontmatter:
  ```
  ---
  created_at: "YYYY-MM-DDTHH:MM:SS"
  confidence: 0.7
  auto_consolidated: true
  success_count: 0
  failure_count: 0
  version: 1
  last_used: ""
  ---
  ```
- After completion, output a summary of what was done
