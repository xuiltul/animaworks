# Development Engineer Guidelines

## Implementation Lane
- Keep work isolated in a worktree to avoid conflicts with other work. Create worktrees in `$ANIMAWORKS_SCRATCH_DIR`, and create them in `shared/worktrees/` only when handing work off to another Anima (`common_knowledge/operations/scratch-and-worktrees.md`).
- Keep changes to existing code to the minimum necessary, and record out-of-scope fixes as separate tasks.
- Document procedures for work that is repeated multiple times, and follow the conventions for workflows that make use of machines.

## Creating and Describing Pull Requests
- Create a PR that summarizes the purpose, changes, and validation methods.
- List the changed files and provide a clear description that makes the changes easy for reviewers to follow.

## Monitoring CI on Your PRs and Making Autonomous Fixes
- If CI on your PR fails, investigate the cause and fix it rather than leaving it unattended.
- Resolve conflicts by rebasing or merging.

## Completion Report Format
- Always include a list of changed files and the validation commands you ran, along with their results.

## References
- Refer to CLAUDE.md and the configuration files at the repository root for repository-specific conventions.
- Work setup: read_memory_file(path="common_knowledge/operations/workspace-guide.md")
