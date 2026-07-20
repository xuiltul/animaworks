# 태스크 관리

Digital Anima가 태스크를 수신하고, 추적하며, 완료하기 위한 운용 레퍼런스입니다.
태스크 진행 방법이 불확실할 때 검색하여 참조하세요.

## 태스크 관리의 기본 구조

태스크 상태는 `state/` 디렉토리 내 파일과 태스크 큐로 관리합니다.

| 리소스 | 역할 |
|--------|------|
| `state/current_state.md` | 워킹 메모리 (현재 상태, 관찰, 계획, 블로커) |
| `state/pending/` 디렉토리 | LLM 태스크 (JSON 형식). `submit_tasks`와 `delegate_task`가 기록. TaskExec 경로가 자동 가져오기 및 실행 |
| `state/task_queue.jsonl` | 영구 태스크 큐 (append-only JSONL). 사람과 Anima의 요청을 추적 |
| `state/task_results/` 디렉토리 | TaskExec 완료 결과 (`{task_id}.md`, 최대 2,000자). 의존 태스크에 자동 주입. 7일 TTL |

`state/current_state.md`는 항상 최신 상태를 유지해야 합니다 (MUST).
태스크 상태가 변경될 때마다 업데이트하세요.

### 3경로 실행 모델

AnimaWorks에서는 태스크가 3개의 독립 경로에서 처리됩니다:

| 경로 | 트리거 | 역할 | 실행 범위 |
|------|--------|------|-----------|
| **Inbox** | DM 수신 | Anima 간 메시지의 처리 및 회신 | 즉시, 경량 응답만 |
| **Heartbeat** | 정기 순회 | 상황 확인과 계획 수립 (Observe → Plan → Reflect) | 확인과 판단만. 실행은 `pending/`에 기록 |
| **TaskExec** | `pending/`에 태스크 출현 | LLM 태스크 실행 | 전체 실행 (도구 사용 포함) |

Heartbeat은 **실행하지 않습니다**. 실행이 필요한 태스크를 발견하면 부하가 있을 경우 `delegate_task`로 위임하거나, `submit_tasks`로 태스크를 투입하여 TaskExec 경로에 위임합니다.

**주의**: Agent/Task 도구(서브에이전트 생성)는 비활성화되어 있습니다. 백그라운드 실행에는 `submit_tasks`, 부하 위임에는 `delegate_task`를 사용하세요.

### 태스크 큐 (submit_tasks / update_task / 목록은 CLI)

영구 태스크 큐는 `state/task_queue.jsonl`에 append-only JSONL 형식으로 기록됩니다.
명시적인 백그라운드 실행이나 후속 추적이 필요한 경우 `submit_tasks`로 태스크를 등록하고, `update_task`로 상태를 업데이트합니다. 목록 조회는 CLI의 `animaworks-tool task list`를 사용합니다.
큐에 등록된 태스크는 시스템 프롬프트의 Priming 섹션에 요약 표시됩니다.

#### submit_tasks (태스크 등록 — 자신이 직접 실행)

> **중요**: `submit_tasks`로 투입한 태스크는 **자신의 TaskExec**이 실행합니다 (부하에게 전달되지 않습니다). 부하에게 태스크를 위임하려면 `delegate_task`를 사용하세요.

TaskExec로 넘길 태스크의 생성 및 등록에는 `submit_tasks`를 사용합니다. 일반 채팅에서 바로 처리할 수 있는 사람의 지시는 직접 실행하고, 후속 추적·병렬 실행·백그라운드 실행이 필요한 경우에만 `submit_tasks`, `update_task`, 또는 `state/current_state.md`로 기록합니다. 단일 태스크의 경우 tasks 배열에 1건만 지정합니다.

```
submit_tasks(batch_id="human-20260313", tasks=[
  {"task_id": "t1", "title": "월간 리포트 작성", "description": "월간 매출 리포트를 작성하여 aoi에게 제출해 주세요", "parallel": true}
])
```

| 파라미터 | 필수 | 설명 |
|----------|------|------|
| `batch_id` | MUST | 배치의 고유 식별자 |
| `tasks[].task_id` | MUST | 배치 내에서 고유한 태스크 ID |
| `tasks[].title` | MUST | 태스크 제목 (한 줄 요약) |
| `tasks[].description` | MUST | 원본 지시문 (위임 시 원문 인용을 포함) |
| `tasks[].parallel` | MAY | `true`로 병렬 실행 가능 (단일 태스크에서는 `true` 권장) |
| `tasks[].depends_on` | MAY | 선행 태스크 ID의 배열 |
| `tasks[].workspace` | MAY | 작업 디렉토리. 워크스페이스 에일리어스 (예: `myproject`)를 지정하면 TaskExec가 해당 디렉토리에서 실행. 생략 시 Anima 기본값 |

- 일반 채팅에서는 사람의 지시를 반드시 `submit_tasks`에 등록할 필요가 없습니다. 직접 처리할 수 없거나 병렬/백그라운드 실행 또는 후속 추적이 필요한 경우에만 등록합니다
- 사람 유래 태스크 (source=human 해당)는 최우선으로 처리 (MUST)
- 큐의 태스크는 Heartbeat에서 확인하고, 착수 시 `update_task`로 `in_progress`로 업데이트

#### update_task

태스크 상태를 업데이트합니다. 완료 시 `done`, 중단 시 `cancelled`, 실패 시 `failed`로 설정합니다.

```
update_task(task_id="abc123def456", status="in_progress")
update_task(task_id="abc123def456", status="done", summary="리포트 작성 완료")
```

| 파라미터 | 필수 | 설명 |
|----------|------|------|
| `task_id` | MUST | 태스크 ID (submit_tasks 시 반환된 ID) |
| `status` | MUST | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `failed` |
| `summary` | MAY | 업데이트 후 요약 |

#### 태스크 목록 조회 (CLI)

태스크 큐의 목록은 `animaworks-tool task list`로 조회합니다. 상태로 필터링이 가능합니다.

```
Bash: animaworks-tool task list                    # 전체
Bash: animaworks-tool task list --status pending   # 미착수만
Bash: animaworks-tool task list --status in_progress
Bash: animaworks-tool task list --status done
Bash: animaworks-tool task list --status failed
```

#### 태스크 큐의 상태와 마커

| 상태 | 의미 |
|------|------|
| `pending` | 미착수 |
| `in_progress` | 작업 중 |
| `done` | 완료 |
| `cancelled` | 취소 |
| `blocked` | 블록 중 |
| `failed` | 실패 (TaskExec 등에서 실행에 실패한 경우) |
| `delegated` | 위임 완료 (delegate_task로 부하에게 위임한 추적용) |

Priming 표시에서는 사람 유래 태스크 (source=human)에 🔴 HIGH 마커, 30분 이상 업데이트되지 않은 태스크에 ⚠️ STALE, 기한 초과 태스크에 🔴 OVERDUE 마커가 부여됩니다.

## current_state.md 사용법

`current_state.md`는 현재 상태, 관찰, 계획, 블로커를 기록하는 워킹 메모리입니다.
태스크 목록이 아닙니다. 태스크 추적은 `task_queue.jsonl`에서 수행합니다.

- **사이즈 상한**: 3,000자. 초과 시 Heartbeat에 의해 자동 정리
- **유휴 상태**: 태스크가 없는 경우 `status: idle`을 기록

### 형식

```markdown
status: in-progress
task: Slack 연동 기능 테스트
assigned_by: aoi
started: 2026-02-15 10:00
context: |
  aoi의 지시: Slack API 접속 테스트를 수행하고,
  #general 채널에 투고가 정상적으로 동작하는지 확인한다.
  테스트 완료 후 결과를 보고할 것.
blockers: 없음
```

### 필드 설명

| 필드 | 필수 | 설명 |
|------|------|------|
| `status` | MUST | 태스크 상태 (아래의 상태 전이 참조) |
| `task` | MUST | 태스크의 간결한 설명 (한 줄) |
| `assigned_by` | SHOULD | 누구에게서 받은 태스크인가. 자발적 태스크는 `self` |
| `started` | SHOULD | 착수 일시 |
| `context` | SHOULD | 태스크의 상세 및 배경 정보 |
| `blockers` | SHOULD | 블로커가 있으면 기재. 없으면 `없음` |

### 유휴 상태

태스크가 없는 경우 다음과 같이 기재합니다:

```markdown
status: idle
```

`idle`은 정상적인 상태로, 다음 태스크를 대기 중임을 의미합니다.
Heartbeat에서 확인했을 때 `idle`이면 특별한 액션은 불필요합니다 (`HEARTBEAT_OK`).

## 태스크 상태 전이

태스크는 `task_queue.jsonl`에서 추적됩니다. `current_state.md`는 진행 중 태스크의 작업 컨텍스트를 기록합니다.

```
submit_tasks 등록 → update_task(status="in_progress") → 작업 → update_task(status="done")
                                                          ↘ blocked → 보고 → 다른 태스크로 전환
```

### 상태 전이 절차

**착수**:
1. `task_queue.jsonl`에서 태스크를 선택하고 `update_task(task_id="...", status="in_progress")`로 업데이트
2. `current_state.md`에 `status: in-progress`로 작업 컨텍스트를 기재

**완료**:
1. `update_task(task_id="...", status="done", summary="...")`로 태스크 완료 처리
2. `current_state.md`를 `status: idle`로 복귀
3. 태스크 의뢰자에게 결과를 보고 (assigned_by가 타인인 경우 MUST)

**블록**:
1. `current_state.md`의 `status`를 `blocked`로 변경하고 `blockers`에 구체적인 이유를 기재 (MUST)
2. 블록 해소를 위한 액션을 실행 (아래의 블록 대응 플로우 참조)
3. 블록 해소에 시간이 걸리는 경우, `task_queue.jsonl`의 다음 태스크에 착수 가능 (MAY)

## 복수 태스크의 우선순위 관리

`task_queue.jsonl`에 여러 태스크가 존재할 때의 판단 기준:

1. **사람 유래 태스크 최우선**: source=human인 태스크는 최우선으로 처리 (MUST)
2. **상사 태스크 우선**: supervisor의 지시는 동일 수준의 다른 태스크보다 우선 (SHOULD)
3. **기한순**: deadline이 가까운 것부터 착수 (SHOULD)
4. **선입선출**: 동일 우선순위 및 동일 기한이면 수신 순서대로 처리 (MAY)

### 태스크 중단 시 절차

우선순위가 높은 태스크가 끼어든 경우:

1. 현재 태스크를 `update_task(status="pending")`으로 큐에 복귀
2. `current_state.md`에 진행 상황을 메모한 후, 새 태스크의 컨텍스트로 전환

## 블록된 태스크의 대응 플로우

태스크가 블록된 경우 다음 절차로 대응합니다.

### 단계 1: 블록 원인의 특정과 기록

current_state.md의 `blockers`에 구체적인 원인을 기재합니다 (MUST).

```markdown
status: blocked
task: AWS S3 버킷 설정
blockers: |
  AWS 크레덴셜이 미설정.
  config.json에 aws credential이 존재하지 않음.
  aoi에게 설정을 의뢰할 필요가 있음.
```

### 단계 2: 해소 액션

블록 원인에 따른 액션을 실행합니다:

| 원인 | 액션 |
|------|------|
| 정보 부족 | 의뢰자에게 질문 메시지를 전송 (SHOULD) |
| 권한 부족 | supervisor에게 권한 추가를 의뢰 (SHOULD) |
| 외부 의존 | 대기 중임을 의뢰자에게 보고 (SHOULD) |
| 기술적 문제 | knowledge/나 procedures/를 검색하여 해결책을 탐색. 찾을 수 없으면 보고 |

### 단계 3: 다른 태스크로 전환

블록 해소에 시간이 걸리는 경우, `task_queue.jsonl`의 다음 태스크에 착수해도 됩니다 (MAY).
블록된 태스크는 `update_task(status="blocked")`로 기록하고 블록 해소 후 재개합니다.

## 태스크 파일 템플릿

### current_state.md — 유휴 상태

```markdown
status: idle
```

### current_state.md — 작업 중

```markdown
status: in-progress
task: {태스크명}
assigned_by: {의뢰자명 or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {태스크의 상세 및 배경 정보}
blockers: 없음
```

### current_state.md — 블록 중

```markdown
status: blocked
task: {태스크명}
assigned_by: {의뢰자명 or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {태스크의 상세 및 배경 정보}
blockers: |
  {블록 이유의 구체적인 설명}
  {해소를 위해 실행한 액션}
```

## episodes/에의 태스크 로그 기록

태스크의 착수, 완료, 블록 등 상태 변화를 episodes/에 기록합니다 (SHOULD).
파일명은 `YYYY-MM-DD.md` (일별 로그)입니다.

```markdown
## 10:00 태스크 착수: Slack 연동 테스트

aoi의 지시를 받아 Slack API 접속 테스트를 시작.
permissions.json에서 slack: yes를 확인 완료.

## 11:30 태스크 완료: Slack 연동 테스트

Slack API 접속 테스트 완료. #general에의 투고 테스트도 성공.
결과를 aoi에게 보고 완료.

[IMPORTANT] Slack API 레이트 제한: 1분당 최대 1메시지 제한 있음.
버스트 전송 시 간격을 두어야 함.
```

중요한 교훈에는 `[IMPORTANT]` 태그를 부여합니다 (SHOULD). 이후 Heartbeat과 기억 통합에서 우선적으로 추출됩니다.

## 병렬 태스크 실행 (submit_tasks)

`submit_tasks` 도구를 사용하면 여러 태스크를 의존 관계와 함께 일괄 투입하여 병렬 실행할 수 있습니다.
TaskExec가 DAG (유향 비순환 그래프)로 의존 관계를 해결하고 독립 태스크를 동시 실행합니다.

### 사용법

```
submit_tasks(batch_id="build-20260301", tasks=[
  {{"task_id": "compile", "title": "컴파일", "description": "소스를 빌드", "parallel": true}},
  {{"task_id": "lint", "title": "Lint", "description": "정적 분석", "parallel": true}},
  {{"task_id": "package", "title": "패키지", "description": "빌드 산출물을 패키징",
   "depends_on": ["compile", "lint"]}}
])
```

| 파라미터 | 필수 | 설명 |
|----------|------|------|
| `batch_id` | MUST | 배치의 고유 식별자 |
| `tasks[].task_id` | MUST | 배치 내에서 고유한 태스크 ID |
| `tasks[].title` | MUST | 태스크 제목 |
| `tasks[].description` | MUST | 작업 내용 |
| `tasks[].parallel` | MAY | `true`로 병렬 실행 가능 (기본값: `false`) |
| `tasks[].depends_on` | MAY | 선행 태스크 ID의 배열 |
| `tasks[].workspace` | MAY | 작업 디렉토리. 워크스페이스 에일리어스를 지정하면 TaskExec가 해당 디렉토리에서 실행 |
| `tasks[].acceptance_criteria` | MAY | 완료 조건의 배열 |
| `tasks[].constraints` | MAY | 제약 조건의 배열 |
| `tasks[].file_paths` | MAY | 관련 파일 경로의 배열 |

### 실행 구조

1. `submit_tasks`가 밸리데이션 (ID 유일성, 의존 대상 존재, 순환 감지)을 수행
2. 태스크 파일이 `state/pending/`에 `batch_id` 포함으로 기록됨
3. submit_tasks 실행 후 TaskExec (PendingTaskExecutor)는 즉시 태스크를 감지 (wake로 폴링 대기 없음)
4. TaskExec가 배치를 감지하고 토폴로지컬 정렬로 실행 순서를 결정
5. 의존 없는 `parallel: true` 태스크는 세마포어 상한 내에서 동시 실행
6. 선행 태스크의 결과는 의존 태스크의 컨텍스트에 자동 주입
7. 선행 태스크가 실패하면 의존 태스크는 건너뜀
8. 태스크는 기록 후 24시간 이내에 실행되지 않으면 건너뜀 (TTL)

### 병렬 실행 상한

동시 실행 수는 `config.json`의 `background_task.max_parallel_llm_tasks` (기본값: 3, 1~10)로 제어됩니다.

### 태스크 결과의 저장

완료된 태스크의 결과 요약은 `state/task_results/{task_id}.md`에 저장됩니다 (최대 2,000자).
의존 태스크는 이 결과를 컨텍스트로 자동 수신합니다. 선행 태스크가 실패한 경우 의존 태스크는 건너뛰며 `FAILED: {사유}`가 기록됩니다.
각 태스크 완료 시, submit_tasks를 실행한 Anima에게 완료 통지가 DM으로 전송됩니다.

### submit_tasks와 delegate_task의 구분 사용

| 시나리오 | 방법 | 실행자 |
|----------|------|--------|
| 자신이 백그라운드에서 실행할 태스크 | `submit_tasks` | **자신** |
| 복수 독립 태스크를 자신이 병렬 실행 | `submit_tasks`에서 `parallel: true` | **자신** |
| 의존 관계가 있는 태스크군을 자신이 실행 | `submit_tasks`에서 `depends_on` 지정 | **자신** |
| **부하에게 작업을 맡기고 싶을 때** | **`delegate_task`** | **부하** |

**주의**: `state/pending/`에 JSON을 수동으로 기록해서는 안 됩니다. 반드시 `submit_tasks` 도구를 경유하여 투입하세요. `submit_tasks`는 Layer 1 (실행 큐)과 Layer 2 (태스크 레지스트리) 양쪽에 동시 등록하므로 태스크 추적 누락을 방지합니다.

## 태스크 위임 (delegate_task / Task tool) — 부하가 실행

> **중요**: `delegate_task`는 **부하의 TaskExec**이 태스크를 실행합니다 (자신은 실행하지 않습니다). 자신이 백그라운드에서 실행하려면 `submit_tasks`를 사용하세요.

부하를 가진 Anima (supervisor)는 `delegate_task` 도구로 태스크를 부하에게 위임할 수 있습니다.
S모드의 Chat 경로에서는 Task tool (및 Agent tool)으로도 위임이 가능합니다. Task tool은 부하를 지명하는 파라미터가 없으며, workload 최소이고 role 매치로 자동 선택됩니다.

### delegate_task의 동작

1. 부하의 태스크 큐에 태스크가 추가됨 (source="anima")
2. 부하의 `state/pending/`에 태스크 JSON이 기록되어 즉시 실행됨
3. 부하에게 DM이 자동 전송됨
4. 자신의 큐에 추적 엔트리가 생성됨 (status="delegated")

### 사용법

```
delegate_task(name="dave", instruction="API 테스트를 실시하고 결과를 보고해 주세요", deadline="2d", summary="API 테스트")
```

| 파라미터 | 필수 | 설명 |
|----------|------|------|
| `name` | MUST | 위임 대상 배하 Anima명 (자식, 손자, 증손자 등 모든 배하에 지정 가능) |
| `instruction` | MUST | 태스크 지시 내용 |
| `deadline` | MUST | 기한. 상대 형식 `30m` / `2h` / `1d` 또는 ISO8601 |
| `summary` | MAY | 태스크의 한 줄 요약 (생략 시 instruction의 첫 100자) |
| `workspace` | MAY | 작업 디렉토리. 워크스페이스 에일리어스를 지정하면 위임 대상이 해당 디렉토리에서 작업 |

### 위임 태스크의 추적

`task_tracker` 도구로 위임한 태스크의 진행 상황을 확인할 수 있습니다.
부하 측의 task_queue.jsonl에서 최신 상태를 대조하여 반환합니다.

```
task_tracker()                     # 활성 위임 태스크 목록 (기본값)
task_tracker(status="all")         # 완료 포함 전체
task_tracker(status="completed")   # 완료만
```

| status | 의미 |
|--------|------|
| `active` | 진행 중 (done/cancelled/failed 이외). 기본값 |
| `all` | 전체 |
| `completed` | 완료 (done/cancelled/failed)만 |

### 위임을 받은 측의 대응

1. DM으로 위임 메시지를 수신
2. 태스크 큐에 자동으로 태스크가 등록됨
3. 내용을 확인하고, 불명확한 점이 있으면 위임자에게 질문 (SHOULD)
4. 완료하면 위임자에게 결과를 보고 (MUST)
