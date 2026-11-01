---
name: github-tool
description: >-
  GitHub 集成工具。通过 gh CLI 获取和创建 Issue、PR 列表。
  Use when: 需要创建或列出 Issue 和 PR、操作仓库、确认 GitHub 上的任务时。
tags: [development, github, external]
---


# GitHub 工具

通过 gh CLI 操作 GitHub 的 Issue 和 PR 的外部工具。

## 调用方法

**Bash**：使用 `animaworks-tool github <サブコマンド> [引数]` 执行

## 操作列表

### list_issues — Issue 列表
```bash
animaworks-tool github issues [--repo OWNER/REPO] [--state open] [--limit 20]
```

### create_issue — 创建 Issue
```bash
animaworks-tool github create-issue --title TITLE --body BODY [--labels LABELS]
```

### list_prs — PR 列表
```bash
animaworks-tool github prs [--repo OWNER/REPO] [--state open] [--limit 20]
```

### create_pr — 创建 PR
```bash
animaworks-tool github create-pr --title TITLE --body BODY --head BRANCH [--base main]
```
- `draft`（可选，默认：false）：是否作为草稿 PR 创建

## CLI 用法

```bash
animaworks-tool github issues [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-issue --title TITLE --body BODY [--labels LABELS]
animaworks-tool github prs [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-pr --title TITLE --body BODY --head BRANCH [--base main]
```

## 注意事项

- 需已安装 gh CLI 并完成认证
- 省略 --repo 时使用当前目录的仓库
