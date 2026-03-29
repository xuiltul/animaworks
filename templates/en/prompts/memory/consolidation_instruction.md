# Memory Consolidation Task (Daily)

{anima_name}, it is time to organize your memory. Follow the steps below.

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

## Error patterns (past 24 hours)

{error_patterns_summary}

---

## Workflow

### Step 1: Merge duplicate files (MUST — highest priority)

**When merge candidates are provided, process every pair.**
Additionally, review the file list above and find duplicate files covering the same topic yourself.

Merge procedure:
1. Use `read_memory_file` to review both contents
2. Combine the information and write to one file with `write_memory_file`
3. Archive the redundant one with `archive_memory_file`
4. If `[IMPORTANT]` tag exists, preserve it in the merged file

- "Merge later" or "too complex, skip" is NOT allowed. Complete all merges now
- Always prefer merging into existing files over creating new ones

### Step 2: Knowledge extraction from episodes

Review today's episodes; if substantive information exists:
1. Use `search_memory` to find related existing knowledge/ and procedures/
2. If relevant files found, review with `read_memory_file` and update with `write_memory_file`
3. Create new files only when no existing file covers the topic

### Step 2.5: Error pattern analysis

Review the "Error patterns" section above and if recurring patterns are found:
1. Use `search_memory` to find related existing procedures/
2. If existing procedures found, review with `read_memory_file` and update with `write_memory_file`
3. Create new files in `procedures/` only when no existing file covers the pattern
4. Single-occurrence errors should be ignored (noise)

Frontmatter for new procedure files:
```
---
created_at: "YYYY-MM-DDTHH:MM:SS"
confidence: 0.4
auto_consolidated: true
source: "error_trace_analysis"
version: 1
---
```

### Step 3: Quality check
- Verify updated or created content does not contradict episode facts
- Use clear, topic-descriptive filenames

## Information to extract
- Specific configuration values, credential locations
- User and system identifiers
- Procedures, workflows, and process records
- Team structure, role assignments, chain of command
- Technical decisions and their rationale
- Lessons and procedures from resolved events

## Critical constraints
- **You MUST perform this work yourself directly**. Do NOT use `delegate_task`, `submit_tasks`, or `send_message`. Complete all work using only memory operation tools
- **Do NOT skip Step 1 merging**. If duplicate files exist and you fail to merge them, the task is considered a failure

## Notes
- Do not convert greetings-only or substantively empty exchanges into knowledge
- [REFLECTION] tagged entries should be prioritized for knowledge extraction
- `[IMPORTANT]` tagged entries **MUST** be extracted into knowledge/. If overlapping with existing knowledge, merge by appending. **Keep the `[IMPORTANT]` tag in the file body**
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
- After completion, output a summary (include number of pairs merged and files archived)
