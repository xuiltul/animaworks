---
name: github-tool
description: >-
  GitHub integration tool for listing and creating issues and PRs via the gh CLI wrapper.
  Use when: creating or listing issues or pull requests, or checking repository work on GitHub.
tags: [development, github, external]
---

# GitHub Tool

External tool for GitHub issue and PR management via gh CLI.

## Invocation via Bash

Use **Bash** with `animaworks-tool github <subcommand> [args]`. See Actions below for syntax.

## Actions

### list_issues — List issues
```json
{"tool_name": "github", "action": "list_issues", "args": {"repo": "owner/repo", "state": "open", "limit": 20}}
```

### create_issue — Create issue
```json
{"tool_name": "github", "action": "create_issue", "args": {"title": "Title", "body": "Description", "labels": "bug,enhancement"}}
```

### list_prs — List pull requests
```json
{"tool_name": "github", "action": "list_prs", "args": {"repo": "owner/repo", "state": "open", "limit": 20}}
```

### create_pr — Create pull request
```json
{"tool_name": "github", "action": "create_pr", "args": {"title": "Title", "body": "Description", "head": "feature-branch", "base": "main", "draft": false}}
```
- `draft` (optional, default: false): Create as draft PR

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool github issues [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-issue --title TITLE --body BODY [--labels LABELS]
animaworks-tool github prs [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-pr --title TITLE --body BODY --head BRANCH [--base main]
```

## Notes

- gh CLI must be installed and authenticated
- Without --repo, uses the current directory's repository
