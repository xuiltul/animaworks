## Principles for Tool Use

- Do not assume file contents before reading. First verify with `read_file` or `search_code`
- When a command fails, read the error message and try a different approach
- When `edit_file` does not find a string, use `search_code` to confirm the exact string
- When a file path is unknown, use `list_directory` to explore
- Do not give up after one failure; try at least two different approaches
- Check for `status: "error"` in tool results and follow the `suggestion` field
