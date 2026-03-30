---
name: github-tool
description: >-
  GitHub 연동 도구. Issue·PR 목록·생성을 gh CLI 래퍼로 수행한다.
  Use when: Issue·PR 작성·목록, 리포지토리 작업 확인, GitHub에서 작업 상태 조회가 필요할 때.
tags: [development, github, external]
---

# GitHub 도구

GitHub의 Issue와 PR을 gh CLI를 통해 관리하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool github <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### list_issues — Issue 목록
```bash
animaworks-tool github issues [--repo OWNER/REPO] [--state open] [--limit 20]
```

### create_issue — Issue 생성
```bash
animaworks-tool github create-issue --title TITLE --body BODY [--labels LABELS]
```

### list_prs — PR 목록
```bash
animaworks-tool github prs [--repo OWNER/REPO] [--state open] [--limit 20]
```

### create_pr — PR 생성
```bash
animaworks-tool github create-pr --title TITLE --body BODY --head BRANCH [--base main]
```
- `draft` (선택, 기본값: false): 드래프트 PR로 생성 여부

## CLI 사용법

```bash
animaworks-tool github issues [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-issue --title TITLE --body BODY [--labels LABELS]
animaworks-tool github prs [--repo OWNER/REPO] [--state open] [--limit 20]
animaworks-tool github create-pr --title TITLE --body BODY --head BRANCH [--base main]
```

## 주의사항

- gh CLI가 설치 및 인증 완료 상태여야 합니다
- --repo 생략 시 현재 디렉토리의 리포지토리를 사용합니다
