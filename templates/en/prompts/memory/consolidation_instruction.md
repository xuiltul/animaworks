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

### 4. Clean up unnecessary memory
- Merge duplicate knowledge/ files and archive the older one with `archive_memory_file`
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

## Notes
- Do not convert greetings-only or substantively empty exchanges into knowledge
- [REFLECTION] tagged entries are conscious insights recorded by the agent; prioritize these for knowledge extraction
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
