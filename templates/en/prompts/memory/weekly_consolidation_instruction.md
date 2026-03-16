# Memory Consolidation Task (Weekly)

{anima_name}, it is time to organize your memory for the past week.

## Workflow

### 1. Knowledge file inventory
Use `Glob` to check the list of files in knowledge/.
If there are many files, use `search_memory` to find similar topics and identify duplicate candidates.

### 2. Merge duplicate or similar files
When duplicate or similar knowledge/ files are found:
1. Use `read_memory_file` to review both contents
2. Create a new file with the merged content using `write_memory_file` (or append to the better one)
3. Archive the older one with `archive_memory_file`

**Note**: When merging, if the original file contains an `[IMPORTANT]` tag, always preserve it in the merged file (used for forgetting protection)

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

### 7. injection.md cleanup
Check the character count of injection.md. If it exceeds 5000 characters:
1. Use `read_memory_file(path="injection.md")` to review contents
2. Identify content that is not "role definition" or "absolute rules"
3. Business rules → Move to knowledge/ with `[IMPORTANT]` tag
4. Procedural content → Move to procedures/
5. Temporary directives (expired or already completed) → Delete
6. Overwrite injection.md with the cleaned-up version

After completion, output a summary of what was done.
