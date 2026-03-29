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

- 에일리어스: 사람이 부여하는 짧은 이름 (예: `aischreiber`)
- 해시: 경로의 SHA-256 앞 8자리가 자동 부여 (예: `3af4be6e`)
- 완전 형태: `aischreiber#3af4be6e` — 충돌 가능성이 제로
- 도구 인수에는 에일리어스만, 완전 형태, 해시만, 절대 경로 중 어느 것이든 사용 가능

## 조작 방법

### 등록

config.json의 workspaces 섹션에 추가합니다:

1. `read_memory_file(path="config.json")`으로 현재 설정을 확인
2. workspaces 섹션에 에일리어스와 경로를 추가
3. `write_memory_file`로 저장

또는 Bash로 실행:
```bash
python3 -c "
from core.workspace import register_workspace
result = register_workspace('에일리어스명', '/absolute/path/to/project')
print(result)
"
```

**주의**: 디렉토리가 존재하지 않으면 에러가 발생합니다.

### 목록 조회

config.json의 workspaces 섹션을 읽습니다:
```
read_memory_file(path="config.json")
```

### 삭제

config.json의 workspaces 섹션에서 해당 항목을 삭제합니다.

### 자신의 기본 워크스페이스 변경

자신의 `status.json`의 `default_workspace` 필드를 업데이트합니다:
1. `read_memory_file(path="status.json")`으로 현재 내용을 확인
2. `default_workspace`에 에일리어스 (예: `aischreiber`) 또는 완전 형태 (예: `aischreiber#3af4be6e`)를 설정
3. `write_memory_file(path="status.json", content=...)`로 저장

예시:
```json
{
  "default_workspace": "aischreiber#3af4be6e"
}
```

### 부하에 할당 (슈퍼바이저용)

1. 워크스페이스를 등록 (위 참조)
2. 부하의 `status.json`의 `default_workspace` 필드를 업데이트:
   - `read_memory_file(path="../{subordinate}/status.json")`
   - `default_workspace`에 에일리어스를 설정
   - `write_memory_file(path="../{subordinate}/status.json", content=...)`

## 도구에서의 사용

- **machine_run**: `working_directory`에 에일리어스 또는 완전 형태를 지정
- **submit_tasks**: 각 태스크의 `workspace` 필드에 에일리어스를 지정
- **delegate_task**: `workspace` 필드에 에일리어스를 지정

## 주의사항

- 디렉토리는 등록 시와 해석 시 양쪽에서 존재 확인이 이루어집니다
- 존재하지 않는 디렉토리를 등록하려 하면 에러가 발생합니다
- 에일리어스를 덮어쓰면 해시도 바뀌므로 이전 해시 참조는 해석 실패합니다
- 사람은 해시를 기억할 필요가 없습니다 — 에일리어스만으로 충분합니다
