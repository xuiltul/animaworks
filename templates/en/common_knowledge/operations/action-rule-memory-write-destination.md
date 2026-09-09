## [ACTION-RULE] Memory write destination
trigger_tools: write_memory_file, create_skill
keywords: memory, write, knowledge, procedures, skills, action-rule, current_state
---
- `knowledge/`: Facts, preferences, policies, decisions, lessons, and failure records
- `procedures/`: Repeatable task procedures
- `skills/{name}/SKILL.md`: Reusable capabilities, tool workflows, and playbooks
- `knowledge/action-rule-*.md`: Rules checked immediately before side-effecting actions
- `state/current_state.md`: Working memory for temporary observations, plans, and blockers only

Read `read_memory_file(path="reference/operations/memory-writing-guide.md")` for details.
