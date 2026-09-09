# 워킹 메모리 (state/) 기술 레퍼런스

Anima의 작업 상태를 관리하는 `state/` 디렉토리의 상세 사양입니다.
프롬프트 주입 로직, 사이즈 제어, 마이그레이션, 잠금 제어를 포함합니다.

---

## state/ 디렉토리 구조

```
state/
├── current_state.md          # 워킹 메모리 (자유 형식 Markdown)
├── task_results/              # TaskExec 완료 결과
│   └── {task_id}/{attempt_token}.md
├── conversation.json          # 대화 상태
├── conversations/             # 스레드별 대화 파일
├── recovery_note.md           # 크래시 복구 노트
├── heartbeat_checkpoint.json  # Heartbeat 체크포인트
└── pending_procedures.json    # 보류 중인 절차 추적
```

---

## current_state.md

### 역할

Anima의 워킹 메모리입니다. "지금 무엇을 하고 있는지", "무엇을 관찰했는지", "어떤 블로커가 있는지"를 자유 형식으로 기록합니다. 태스크 관리용이 아니라 상황 인식을 위한 공간입니다.

태스크 추적은 호스트가 관리하는 정본 TaskStore가 담당합니다. `list_tasks`로 확인하고 태스크 도구로 변경하세요. DB나 큐 파일을 직접 수정하지 마세요.

### 사이즈 제어

| 파라미터 | 값 | 소스 |
|----------|-----|------|
| 표시 상한 | 3000자 | `_CURRENT_STATE_MAX_CHARS` (builder.py) |
| 디스크 trim 상한 | 8000자 (기본값) | `heartbeat.current_state_max_chars` (0 = 비활성) |
| Inbox 시 상한 | 500자 | builder.py 내 `min(_state_max, 500)` |

**세션 경계**:

- 일반 Heartbeat / cron / 대화 finalize에서는 `current_state.md`를 유지합니다
- 세션 요약에 현재 상태가 포함되어도 `current_state.md`가 비어 있거나 idle일 때만 기록합니다
- 활성 태스크가 없는 오래된 state는 TaskBoard housekeeping이 보관할 수 있습니다. 숨겨진 활성 태스크도 state를 보호합니다

**Heartbeat 중 선택적 정리**:

1. `heartbeat.current_state_max_chars`가 0보다 크고 Heartbeat 시작 전 `current_state.md`가 그 값을 초과하면 정리/압축 지시가 Heartbeat 프롬프트에 주입됩니다
2. Heartbeat 또는 cron 완료 후 `_enforce_state_size_limit()`가 실행됩니다
3. 설정된 상한을 초과한 내용은 당일 에피소드 메모리(`episodes/{date}.md`)에 `## current_state.md overflow archived`로 이동됩니다
4. 마지막 설정 글자 수를 유지하며, 줄바꿈 위치를 기준으로 조정합니다(첫 20% 이내에 줄바꿈이 있으면 그 지점에서 자름)

### 프롬프트 주입

| 트리거 | 동작 |
|--------|------|
| `chat` | 전문 주입 (3000자 상한, 스케일 적용) |
| `inbox` | 최대 500자로 제한 |
| `heartbeat` / `cron` | 전문 주입 (3000자 상한) |
| `task` | **주입하지 않음** (Minimal 티어) |

주입 시 `status: idle`만 존재하면 해당 섹션 자체가 생략됩니다.
그 외에는 `builder/task_in_progress` 템플릿으로 강조 헤더와 함께 주입됩니다.

### 잠금 제어

`core/anima.py`의 `_state_file_lock` (`asyncio.Lock`)이 `current_state.md`에 대한 동시 쓰기를 방지합니다.

`_is_state_file(path)`는 `state/current_state.md`에 대해서만 `True`를 반환합니다. `write_memory_file` 경유 쓰기 시 이 파일에 대해 잠금이 자동 획득됩니다.

### 경로 해석 (하위 호환)

`read_memory_file` / `write_memory_file`에서 `state/current_task.md`가 지정된 경우, 자동으로 `state/current_state.md`로 해석됩니다 (`handler_memory.py`).

---

## pending.md (폐지됨)

`state/pending.md`는 `current_state.md`에 통합된 후 자동 삭제됩니다.

### 마이그레이션 (MemoryManager 초기화 시)

1. `state/current_task.md`가 존재하고 `state/current_state.md`가 존재하지 않음 → 이름 변경
2. 둘 다 존재 → `current_state.md`를 우선, 경고 로그 출력
3. `state/pending.md`가 존재하고 내용이 있음 → `current_state.md`에 `## Migrated from pending.md`로 추가 후 삭제
4. `state/pending.md`가 비어 있음 → 삭제

### API

| 메서드 | 동작 |
|--------|------|
| `read_pending()` | 항상 빈 문자열 `""`을 반환. 비권장 경고를 로그 출력 |
| `update_pending()` | 아무것도 하지 않음 (no-op). 비권장 경고를 로그 출력 |

---

## 기존 태스크 파일

`state/task_queue.jsonl`과 `state/pending/`은 마이그레이션·내보내기 증거로만 보존합니다. 실행 중인 큐가 아닙니다. 운영자가 기존 쓰기 작업을 중지하고 백업과 함께 명시적으로 가져온 뒤 정본 런타임을 시작해야 합니다. 재개하려고 파일을 삭제하거나 재투입하거나 만들어 내지 마세요.

## 태스크 실행과 결과

호스트가 지시와 태스크를 원자적으로 저장하고 실행 가능한 작업을 가져와 모든 시도를 기록합니다. `in_progress`는 호스트 소유입니다. 에이전트는 `update_task`로 `done`, `pending`, `cancelled`를 선언합니다. `list_tasks(detail=true)`로 의존 관계와 주의 사유를 확인하세요. pending은 재시도를 뜻하지 않습니다. 원인을 해결한 뒤 `submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])`로 같은 태스크를 명시적으로 재개하세요.

수락한 결과 요약은 `state/task_results/{task_id}/{attempt_token}.md`에 저장됩니다(최대 2000자). 후속 태스크에는 호스트가 선택한 수락된 결과를 제공하며 오래된 파일의 존재만으로 완료를 판단하지 않습니다. 원본 기록을 보존하고 결과를 직접 써서 성공한 시도를 가장하지 마세요.

장시간 명령 도구는 별도 경로를 유지합니다. `animaworks-tool submit`은 `state/background_tasks/pending/`에 제출하고 BackgroundTaskManager가 명령 상태와 알림을 관리합니다. `operations/background-tasks.md`와 `operations/task-management.md`를 참조하세요.

## read_subordinate_state

상사가 `read_subordinate_state(name="부하명")`를 호출하면 부하의 `state/current_state.md`만 읽어옵니다 (`pending.md`는 포함되지 않음).
