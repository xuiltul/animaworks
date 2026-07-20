---
name: workspace-manager
description: >-
  워크스페이스(작업 디렉토리) 등록·목록·삭제·할당을 설정한다.
  Use when: 프로젝트 경로를 Anima에 연결, 별칭 관리, 작업 루트 전환이 필요할 때.
tags: [workspace, directory, project, management]
---

# 워크스페이스 관리

Anima가 작업하는 프로젝트 디렉토리(워크스페이스)를 관리하는 스킬입니다.

## 개념

Anima는 평소 "자신의 집" (~/.animaworks/animas/{name}/)에 있습니다.
프로젝트 작업을 할 때는 "직장"(워크스페이스)으로 이동하여 작업합니다.

워크스페이스는 조직 공유 레지스트리(config.json의 workspaces 섹션)에 등록하고, 에일리어스#해시로 참조합니다.

## 에일리어스와 해시

- 에일리어스: 사람이 부여하는 짧은 이름 (예: `myproject`)
- 해시: 경로의 SHA-256 앞 8자리가 자동 부여 (예: `3af4be6e`)
- 완전 형태: `myproject#3af4be6e` — 충돌 가능성이 제로
- 도구 인수에는 에일리어스만, 완전 형태, 해시만, 절대 경로 중 어느 것이든 사용 가능

## 조작 방법

### 등록

인간의 명시적 지시를 받은 최상위 Anima는 `grant_workspace_access`를 사용합니다:

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "make_default": true
}
```

이 도구는 조직 공유 레지스트리 등록, 대상 Anima의 `permissions.json.file_roots` 쓰기 권한 추가, 필요 시 `status.json.default_workspace` 갱신을 함께 수행합니다.

**주의**: 디렉토리가 존재하지 않으면 에러가 발생합니다.
**주의**: `read_memory_file(path="config.json")`은 자신의 Anima 디렉토리에 있는 `config.json`을 읽습니다. 조직 공유 레지스트리 등록에는 사용하지 않습니다.

### 목록 조회

조직 공유 레지스트리 목록은 `core.workspace.list_workspaces()`로 확인합니다. `read_memory_file(path="config.json")`은 사용하지 않습니다.

### 삭제

삭제는 관리자 작업으로 취급합니다. 일반 작업에서는 기존 에일리어스를 덮어쓰지 말고 새 에일리어스를 등록합니다.

### 자신의 기본 워크스페이스 변경

최상위 Anima는 `grant_workspace_access` 호출 시 `make_default: true`를 지정합니다.
최상위가 아닌 Anima는 스스로 워크스페이스 권한을 추가할 수 없습니다. 인간 지시를 통해 최상위 Anima에게 요청합니다.

### 부하에 할당 (슈퍼바이저용)

인간의 명시적 지시를 받은 최상위 Anima는 `target_anima`를 지정하여 부하 또는 자손 Anima에 권한을 부여할 수 있습니다:

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "target_anima": "ritsu",
  "make_default": true
}
```

## 도구에서의 사용

- **machine_run**: `working_directory`에 에일리어스 또는 완전 형태를 지정
- **submit_tasks**: 각 태스크의 `workspace` 필드에 에일리어스를 지정
- **delegate_task**: `workspace` 필드에 에일리어스를 지정

## 주의사항

- 디렉토리는 등록 시와 해석 시 양쪽에서 존재 확인이 이루어집니다
- 존재하지 않는 디렉토리를 등록하려 하면 에러가 발생합니다
- 에일리어스를 덮어쓰면 해시도 바뀌므로 이전 해시 참조는 해석 실패합니다
- 사람은 해시를 기억할 필요가 없습니다 — 에일리어스만으로 충분합니다
