### Repository Work Rules

- Treat the canonical `main` / `master` checkout as read-only. Implement, verify, and commit only in a dedicated `git worktree`
- Create worktrees under `{data_dir}/companies/<company>/shared/worktrees/` (shareable with other Anima; mandatory for repositories that build `node_modules` or other large artifacts) or `/tmp/`. Operations on the canonical checkout are limited to `git worktree add` and reading
- Merge from a worktree only after confirming that the canonical checkout is clean. If it is dirty, make no changes and report it
- Never stash, discard, or overwrite another actor's changes without explicit instruction
