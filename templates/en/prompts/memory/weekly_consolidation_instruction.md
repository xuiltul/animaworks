# Memory Consolidation Task (Weekly)

{anima_name}, it is time to organize your memory for the past week.

## Current knowledge files ({total_knowledge_count} total)

{knowledge_files_list}

## Merge candidates (similar file pairs)

{merge_candidates}

## Critical constraints
- **You MUST perform this work yourself directly**. Do NOT use `delegate_task`, `submit_tasks`, or `send_message`. Do not delegate to subordinates or send any messages. Complete all work using only memory operation tools

## Workflow

### 1. Merge candidate consolidation (highest priority, MUST)
The merge candidates above are file pairs detected for consolidation based on RAG vector similarity.
**For every pair**, perform the following:
1. Use `read_memory_file` to review both file contents
2. Merge the content and write with `write_memory_file`
3. Archive the obsolete one with `archive_memory_file`
4. If the `[IMPORTANT]` tag exists, preserve it in the merged file

Skip this step if no merge candidates are provided.

### 2. Additional duplicate check
Review the file list above; even if not in the merge candidates, if similar files exist:
- Use `read_memory_file` to review the content
- If they should be merged, follow the same procedure

### 3. Conceptual integration of [IMPORTANT] knowledge
Consolidate `[IMPORTANT]`-tagged knowledge/ files that are 30+ days old. Transform specific incident records into universal principles and rules.

1. Use `search_memory` to find knowledge/ files containing `[IMPORTANT]`; review those 30+ days old with `read_memory_file`
2. Group by related themes and extract abstract principles from each group
3. Create `concept-{theme}.md` with `write_memory_file` (include `[IMPORTANT]` at the top)
4. Remove `[IMPORTANT]` tag from the original files (keep the files; they will naturally be forgotten)

Skip isolated `[IMPORTANT]` entries with no related group, items less than 30 days old, and already concept-level entries.

### 4. Procedure knowledge organization
Use `Glob` to check files in procedures/:
- Outdated procedures → Update to reflect current state or archive with `archive_memory_file`
- Unused procedures → Consider archiving
- Similar procedures → Merge

### 5. Compress old episodes
Use `Glob` to check episodes/; if there are files older than 30 days:
- Use `read_memory_file` to review the content
- Compress entries without the [IMPORTANT] tag to key points only and overwrite with `write_memory_file`

### 6. Resolve knowledge contradictions
Check for contradictory knowledge files:
- Keep the accurate one based on the latest information
- Archive the outdated one with `archive_memory_file`

After completion, output a summary of what was done.
