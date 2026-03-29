---
name: google-tasks-tool
description: >-
  Google Tasks 연동 도구. 태스크 리스트·태스크 목록·추가·갱신을 OAuth2로 수행한다.
  Use when: TODO 목록 조회, 태스크 추가, 완료 처리, 리스트 전환이 필요할 때.
tags: [tasks, google, todo, external]
---

# Google Tasks 도구

Google Tasks API로 태스크 리스트와 태스크를 조작하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool google_tasks <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### list_tasklists — 태스크 리스트 목록
```bash
animaworks-tool google_tasks tasklists [-n 50]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| max_results | integer | 50 | 최대 조회 건수 |

### list_tasks — 태스크 목록
```bash
animaworks-tool google_tasks list <태스크리스트ID> [-n 50] [--no-completed]
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| tasklist_id | string | Yes | 태스크 리스트 ID |
| max_results | integer | 50 | 최대 조회 건수 |
| show_completed | boolean | true | 완료된 태스크 포함 여부 |

### insert_task — 태스크 추가
```bash
animaworks-tool google_tasks add <태스크리스트ID> "태스크 제목" [--notes 메모] [--due 일시]
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| tasklist_id | string | Yes | 태스크 리스트 ID |
| title | string | Yes | 태스크 제목 |
| notes | string | No | 메모 |
| due | string | No | 기한 (RFC 3339) |

### insert_tasklist — 태스크 리스트 생성
```bash
animaworks-tool google_tasks new-list "리스트 이름"
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| title | string | Yes | 리스트 이름 |

### update_task — 태스크 업데이트
지정한 태스크의 제목, 메모, 기한, 완료 상태를 업데이트합니다 (지정한 항목만 업데이트).

```bash
animaworks-tool google_tasks update <태스크리스트ID> <태스크ID> [--title 제목] [--notes 메모] [--due 일시] [--status completed|needsAction]
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| tasklist_id | string | Yes | 태스크 리스트 ID |
| task_id | string | Yes | 태스크 ID |
| title | string | No | 새 제목 |
| notes | string | No | 메모 |
| due | string | No | 기한 (RFC 3339) |
| status | string | No | `needsAction`(미완료) 또는 `completed`(완료). title/notes/due/status 중 하나 이상을 지정해야 합니다. |

### update_tasklist — 태스크 리스트 이름 변경
```bash
animaworks-tool google_tasks update-list <태스크리스트ID> "새 리스트 이름"
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| tasklist_id | string | Yes | 태스크 리스트 ID |
| title | string | Yes | 새 리스트 이름 |

## 참고 사항

- 최초 사용 시 OAuth2 인증 플로우가 필요합니다
- credentials.json을 `~/.animaworks/credentials/google_tasks/`에 배치하세요 (Gmail/Calendar과 동일한 OAuth 클라이언트를 복사하여 사용 가능)
