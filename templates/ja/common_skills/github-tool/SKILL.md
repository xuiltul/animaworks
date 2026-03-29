---
name: github-tool
description: >-
  GitHub連携ツール。Issue・PRの一覧取得・作成をgh CLI経由で行う。
  Use when: IssueやPRの作成・一覧、リポジトリ操作、GitHub上のタスク確認が必要なとき。
tags: [development, github, external]
---

# GitHub ツール

GitHubのIssue・PRをgh CLI経由で操作する外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool github <サブコマンド> [引数]` で実行

## アクション一覧

### list_issues — Issue一覧
```bash
animaworks-tool github issues [--repo OWNER/REPO] [--state open] [--limit 20]
```

### create_issue — Issue作成
```bash
animaworks-tool github create-issue --title TITLE --body BODY [--labels LABELS]
```

### list_prs — PR一覧
```bash
animaworks-tool github prs [--repo OWNER/REPO] [--state open] [--limit 20]
```

### create_pr — PR作成
```bash
animaworks-tool github create-pr --title TITLE --body BODY --head BRANCH [--base main]
```
- `draft` (任意, デフォルト: false): ドラフトPRとして作成するか

## CLI使用法

```bash
animaworks-tool github issues [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-issue --title TITLE --body BODY [--labels LABELS]
animaworks-tool github prs [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-pr --title TITLE --body BODY --head BRANCH [--base main]
```

## 注意事項

- gh CLI がインストール済みで認証済みであること
- --repo 省略時はカレントディレクトリのリポジトリを使用
