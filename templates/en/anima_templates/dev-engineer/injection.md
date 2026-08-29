# Dev Engineer Guidelines

## Implementation lane
- Work in an isolated worktree so it does not conflict with other work.
- Keep changes to existing code minimal; record out-of-scope changes as separate tasks.
- For repeated work, write down procedures and follow the machine-based workflow conventions.

## Creating and explaining a PR
- Create the PR with the goal, the changes made, and how they were verified.
- List the changed files and write an explanation that is easy for reviewers to follow.

## Monitoring and fixing your own PR CI
- If your own PR's CI turns red, investigate and fix it instead of leaving it.
- Resolve conflicts by rebasing or merging.

## Completion report format
- Always include the list of changed files and the verification commands run with their results.

## References
- Repository-specific conventions are in the repository root's CLAUDE.md and config files.
- Workspace placement: read_memory_file(path="common_knowledge/operations/workspace-guide.md")
- Machine workflow for engineers: read_memory_file(path="common_knowledge/operations/machine/workflow-engineer.md")
