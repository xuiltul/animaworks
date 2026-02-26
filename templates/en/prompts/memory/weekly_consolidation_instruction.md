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

### 3. Procedure knowledge organization
Use `list_directory` to check files in procedures/:
- Outdated procedures → Update to reflect current state or archive with `archive_memory_file`
- Unused procedures → Consider archiving
- Similar procedures → Merge

### 4. Compress old episodes
Use `list_directory` to check episodes/; if there are files older than 30 days:
- Use `read_memory_file` to review the content
- Compress entries without the [IMPORTANT] tag to key points only and overwrite with `write_memory_file`

### 5. Resolve knowledge contradictions
Check for contradictory knowledge files:
- Keep the accurate one based on the latest information
- Archive the outdated one with `archive_memory_file`

After completion, output a summary of what was done.
