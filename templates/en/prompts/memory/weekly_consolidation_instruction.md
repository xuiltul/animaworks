# Memory Consolidation Task (Weekly)

{anima_name}, it is time to organize your memory for the past week.

## Workflow

### 1. Knowledge file inventory
Use `list_directory` to check the list of files in knowledge/.
If there are many files, use `search_memory` to find similar topics and identify duplicate candidates.

### 2. Merge duplicate or similar files
When duplicate or similar knowledge/ files are found:
1. Use `read_memory_file` to review both contents
2. Create a new file with the merged content using `write_memory_file` (or append to the better one)
3. Archive the older one with `archive_memory_file`

**Note**: When merging, if the original file contains an `[IMPORTANT]` tag, always preserve it in the merged file (used for forgetting protection)

### 3. Conceptual integration of [IMPORTANT] knowledge (amygdala memory → semantic consolidation)
Target: `[IMPORTANT]`-tagged knowledge/ files that are 30+ days old.

This corresponds to "systems consolidation" in neuroscience: transforming specific episodic memories into abstract principles and concepts.

**Steps:**
1. Use `search_memory` to find knowledge/ files containing `[IMPORTANT]`
2. Identify those created 30+ days ago; use `read_memory_file` to review their content
3. Group them by related themes (e.g., security, reporting, interpersonal)
4. Extract **abstract principles/lessons** from each group and create new concept files with `write_memory_file`:
   - Filename: `concept-{theme}.md` (e.g., `concept-security-principles.md`)
   - Include `[IMPORTANT]` at the beginning of the body (the concept itself remains important)
   - Write as **universal principles/rules**, not specific incidents
5. Remove the `[IMPORTANT]` tag from the original specific files (overwrite with `write_memory_file`)
   - Do NOT delete the files. They remain as normal knowledge and will naturally be forgotten over time

**Example:**
Before (specific episodes):
- `[IMPORTANT]` SQL data leaked via Chatwork
- `[IMPORTANT]` Forgot to report to supervisor
- `[IMPORTANT]` Sent code content through Chatwork

After (abstract concepts):
- `[IMPORTANT]` Never post code, SQL, or sensitive data in external chat tools
- `[IMPORTANT]` Always report incidents to supervisor immediately (MUST)

**Notes:**
- Isolated `[IMPORTANT]` entries with no related group: skip, leave for next cycle
- Already concept-level `[IMPORTANT]` entries: no need to re-integrate
- Files less than 30 days old: too fresh, skip

### 4. Procedure knowledge organization
Use `list_directory` to check files in procedures/:
- Outdated procedures → Update to reflect current state or archive with `archive_memory_file`
- Unused procedures → Consider archiving
- Similar procedures → Merge

### 5. Compress old episodes
Use `list_directory` to check episodes/; if there are files older than 30 days:
- Use `read_memory_file` to review the content
- Compress entries without the [IMPORTANT] tag to key points only and overwrite with `write_memory_file`

### 6. Resolve knowledge contradictions
Check for contradictory knowledge files:
- Keep the accurate one based on the latest information
- Archive the outdated one with `archive_memory_file`

After completion, output a summary of what was done.
