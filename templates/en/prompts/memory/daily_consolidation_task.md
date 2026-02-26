
Task:
Extract new lessons, patterns, and policies from the following episodes.

1. **New lessons or patterns**: Things learned from the episodes that will aid future judgment
2. **Existing knowledge updates**: Content to add or revise in existing knowledge files
3. **New knowledge files**: Cases where a new topic deserves its own standalone knowledge file

Output format:
## Existing File Updates
- Filename: knowledge/xxx.md
  Content to add: (specific content in Markdown format)

## New File Creation
- Filename: knowledge/yyy.md
  Content: (full file content in Markdown format)

Notes:
- Be sure to extract the following:
  - Specific configuration values, API keys, and where credentials are stored
  - User and system identifiers (IDs, names, roles)
  - Procedures, workflows, and process records
  - Team structure, role assignments, and chain of command
  - Technical decisions and their rationale
- Skip only exact duplicates
- Do not convert greetings-only or substantively empty exchanges into knowledge
- If no existing files apply, propose everything as new files
- Use clear, topic-descriptive filenames (alphanumeric and hyphens recommended)
- Do not wrap output in code fences (```)
- Entries tagged with [REFLECTION] are conscious insights recorded by the agent. Pay special attention to these and prioritize them for knowledge extraction
