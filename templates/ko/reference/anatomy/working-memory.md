# 워킹 메모리 (state/) 기술 레퍼런스

Anima의 작업 상태를 관리하는 `state/` 디렉토리의 상세 사양입니다.
프롬프트 주입 로직, 사이즈 제어, 마이그레이션, 잠금 제어를 포함합니다.

---

## state/ 디렉토리 구조

```
state/
├── current_state.md          # 워킹 메모리 (자유 형식 Markdown)
├── task_queue.jsonl           # 태스크 레지스트리 (append-only JSONL)
├── pending/                   # LLM 태스크 실행 큐 (JSON)
│   ├── {task_id}.json         # 제출된 태스크
│   ├── processing/            # 실행 중 (PendingTaskExecutor가 이동)
│   └── failed/                # 실패 태스크
├── task_results/              # TaskExec 완료 결과
│   └── {task_id}.md           # 결과 요약 (최대 2000자, 7일 TTL)
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

태스크의 공식적인 추적 및 관리는 `task_queue.jsonl` (Layer 2)이 담당합니다.

### 사이즈 제어

| 파라미터 | 값 | 소스 |
|----------|-----|------|
| 표시 상한 | 3000자 | `_CURRENT_STATE_MAX_CHARS` (builder.py) |
| 디스크 상한 | 3000자 | `_CURRENT_STATE_CLEANUP_THRESHOLD` (_anima_heartbeat.py) |
| Inbox 시 상한 | 500자 | builder.py 내 `min(_state_max, 500)` |

**Heartbeat 시 자동 정리**:

1. Heartbeat 시작 전에 `current_state.md`가 3000자를 초과할 경우, "정리하여 압축하라"는 지시가 Heartbeat 프롬프트에 주입됩니다
2. Heartbeat 완료 후 `_enforce_state_size_limit()`가 실행됩니다
3. 3000자 초과분은 해당 일자의 에피소드 기억 (`episodes/{date}.md`)에 `## current_state.md overflow archived`로 이동됩니다
4. 마지막 3000자를 유지하며 줄바꿈 위치에서 조정합니다 (앞쪽 20% 이내에 줄바꿈이 있으면 그 위치에서 잘라냄)

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

## task_queue.jsonl

태스크 레지스트리입니다. 상세는 `common_knowledge/anatomy/task-architecture.md` (Layer 2)를 참조하세요.

### 엔트리 스키마 (TaskEntry)

| 필드 | 타입 | 설명 |
|------|------|------|
| `task_id` | string | 고유 ID |
| `ts` | ISO8601 | 생성 일시 |
| `source` | `"human"` / `"anima"` | 태스크 출처 |
| `original_instruction` | string | 원본 지시문 |
| `assignee` | string | 담당 Anima명 |
| `status` | string | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `delegated` / `failed` |
| `summary` | string | 한 줄 요약 |
| `deadline` | ISO8601 / null | 기한 |
| `relay_chain` | array | 위임 체인 |
| `updated_at` | ISO8601 | 최종 업데이트 일시 |
| `meta` | object | `executor`, `batch_id`, `task_desc`, `origin` 등 |

---

## pending/ 디렉토리

LLM 태스크 실행 큐입니다. 상세는 `common_knowledge/anatomy/task-architecture.md` (Layer 1)를 참조하세요.

### 라이프사이클

```
pending/{task_id}.json → processing/{task_id}.json → 성공: 삭제 / 실패: failed/로 이동
```

- TTL: 24시간 (`_LLM_TASK_TTL_HOURS`). 초과한 태스크는 건너뜀
- 폴링 간격: 3초 (`_PENDING_WATCHER_POLL_INTERVAL`)
- `task_queue.jsonl`에서 `cancelled`인 태스크는 자동 건너뜀 → `failed/`로 이동

### JSON 스키마

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `task_type` | string | Yes | `"llm"` |
| `task_id` | string | Yes | 고유 ID |
| `batch_id` | string | No | 배치 ID (submit_tasks) |
| `title` | string | Yes | 제목 |
| `description` | string | Yes | 지시 내용 |
| `parallel` | boolean | No | 병렬 실행 가능 여부 |
| `depends_on` | array | No | 선행 태스크 ID |
| `context` | string | No | 추가 컨텍스트 |
| `acceptance_criteria` | array | No | 완료 조건 |
| `constraints` | array | No | 제약 |
| `file_paths` | array | No | 관련 파일 |
| `workspace` | string | No | 작업 디렉토리 (별칭) |
| `submitted_by` | string | Yes | 제출자 |
| `submitted_at` | ISO8601 | Yes | 제출 일시 |
| `source` | string | No | `"delegation"` 등 |

---

## task_results/ 디렉토리

TaskExec이 완료한 태스크의 결과 요약을 저장합니다.

| 파라미터 | 값 |
|----------|-----|
| 파일명 | `{task_id}.md` |
| 최대 글자 수 | 2000 (`_TASK_RESULT_MAX_CHARS`) |
| TTL | 7일 (하우스키핑에 의해 자동 삭제) |

의존 태스크 (`depends_on`)는 이 파일의 내용을 컨텍스트로 자동 수신합니다.

---

## read_subordinate_state

상사가 `read_subordinate_state(name="부하명")`를 호출하면 부하의 `state/current_state.md`만 읽어옵니다 (`pending.md`는 포함되지 않음).
