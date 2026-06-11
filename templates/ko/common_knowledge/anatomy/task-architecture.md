# 태스크 아키텍처 — 3계층 모델

AnimaWorks의 태스크 관리는 3개의 계층으로 구성됩니다.
상위일수록 시스템이 엄격하게 관리하고, 하위일수록 Anima의 자유 재량에 맡겨집니다.

## 3계층 개요

```
┌─────────────────────────────────────────────────┐
│  Layer 1: 실행 큐 (Execution Queue)               │  ← 가장 엄격. 기계적으로 처리
│  state/pending/*.json                            │
├─────────────────────────────────────────────────┤
│  Layer 2: 태스크 레지스트리 (Task Registry)         │  ← 구조화. 도구 경유로 관리
│  state/task_queue.jsonl                          │
├─────────────────────────────────────────────────┤
│  Layer 3: 워킹 메모리 (Working Memory)              │  ← 자유 형식. 자체 관리
│  state/current_state.md                            │
└─────────────────────────────────────────────────┘
```

## Layer 1: 실행 큐 (state/pending/*.json)

메시지 큐 (SQS / RabbitMQ)에 해당합니다.

| 속성 | 설명 |
|------|------|
| 포맷 | JSON (스키마 고정) |
| 라이프사이클 | 투입 → 소비 → 삭제 (일시적) |
| 관리 주체 | 시스템 (PendingTaskExecutor가 자동 소비) |
| 기록 원본 | `submit_tasks`, `delegate_task`, SDK Task/Agent tool |
| 읽기 원본 | PendingTaskExecutor (3초 폴링) |

태스크의 전체 기술 (description, acceptance_criteria, constraints, depends_on, workspace 등)을 포함합니다.
PendingTaskExecutor가 검출하면 `processing/`으로 이동하여 실행하고, 완료 후 삭제합니다.
실패 시 `failed/`로 이동합니다.

태스크에 `workspace` 필드가 있는 경우, 레지스트리에서 해결한 절대 경로가 `working_directory`로 TaskExec 프롬프트에 주입됩니다. 해결 순서: 태스크의 `workspace` → `status.json`의 `default_workspace` → 없음.

Anima는 이 계층을 직접 조작하지 않습니다. 도구를 통해 간접적으로 기록합니다.

## Layer 2: 태스크 레지스트리 (state/task_queue.jsonl)

이슈 트래커 (Jira / GitHub Issues)에 해당합니다.

| 속성 | 설명 |
|------|------|
| 포맷 | append-only JSONL (TaskEntry 스키마) |
| 라이프사이클 | 등록 → 상태 전이 → compact로 아카이브 (영구적) |
| 관리 주체 | Anima (도구 경유) + 시스템 (Priming 주입, compact) |
| 기록 | `submit_tasks`, `update_task`, `delegate_task` |
| 읽기 | `format_for_priming`, Heartbeat compact (목록은 CLI: animaworks-tool task list) |

태스크의 요약 정보 (task_id, summary, status, deadline, assignee)를 보유합니다.
Priming의 Channel E에서 pending / in_progress 태스크가 시스템 프롬프트에 주입됩니다.
"무엇을 해야 하는지"의 공식 기록이며, 사람으로부터의 태스크 (source=human)는 반드시 여기에 등록합니다.

## Layer 3: 워킹 메모리 (state/current_state.md)

포스트잇 / 개인 메모에 해당합니다.

| 속성 | 설명 |
|------|------|
| 포맷 | Markdown (자유 형식) |
| 라이프사이클 | Anima가 자유롭게 생성 및 갱신. 일반 경계에서 유지되며 `heartbeat.current_state_max_chars` 설정 시에만 trim |
| 관리 주체 | Anima (완전한 재량) |
| 기록 | Anima (직접 파일 조작) |
| 읽기 | Anima 자신, Priming (current_state.md), supervisor (read_subordinate_state) |

`current_state.md`는 "지금 하고 있는 것", "관찰한 것", "블로커가 있는 것"을 기록하는 워킹 메모리입니다.
태스크 추적 및 관리는 Layer 2 (task_queue.jsonl)가 담당합니다.

> **pending.md는 폐지됨**: 이전의 `state/pending.md`는 `current_state.md`에 통합되어 자동 삭제됩니다. 백로그 관리는 Layer 2로 일원화되었습니다.

## 계층 간의 관계

### 데이터 흐름

```
사람의 지시 ─┬─► submit_tasks ──────────────────► Layer 2 (task_queue.jsonl)
             └─► Anima가 current_state.md에 기록 ► Layer 3

submit_tasks ─┬─► state/pending/*.json ──────► Layer 1 (실행 큐)
            └─► task_queue.jsonl에 등록 ────► Layer 2 (태스크 레지스트리)

delegate_task ─┬─► 부하의 state/pending/ ──► Layer 1
               ├─► 부하의 task_queue.jsonl ► Layer 2
               └─► 자신의 task_queue.jsonl ► Layer 2 (status=delegated)

PendingTaskExecutor ─┬─► 완료 → task_queue를 done으로 갱신
                     └─► 실패 → task_queue를 failed로 갱신
```

### 동기화 규칙

| 이벤트 | Layer 1 | Layer 2 | Layer 3 |
|--------|---------|---------|---------|
| submit_tasks 투입 | JSON 생성 | pending으로 등록 | — |
| delegate_task 투입 | JSON 생성 (부하) | 양측에 등록 | — |
| TaskExec 완료 | JSON 삭제 | done으로 갱신 | — |
| TaskExec 실패 | failed/로 이동 | failed로 갱신 | — |
| TaskExec 재시도 | JSON 재생성 | pending→in_progress | — |
| 세션 종료: 해결됨 | — | done으로 갱신 | — |
| 세션 종료: 새 태스크 | — | pending으로 등록 | — |
| Anima가 착수 | — | in_progress로 갱신 | current_state.md 갱신 |
| Anima가 완료 | — | done으로 갱신 | idle로 복귀 |
| Heartbeat 후 | — | compact 실행 | — |

### 각 계층이 "몰라도 되는" 것

- **Layer 1**은 Layer 2/3의 존재를 모릅니다 (PendingTaskExecutor는 JSON을 소비할 뿐)
- **Layer 3**은 Layer 1/2의 존재를 몰라도 됩니다 (Anima의 자유 메모)
- **Layer 2**가 Layer 1과 Layer 3을 연결하는 중심 추적 계층

## 설계 원칙

1. **모든 태스크는 Layer 2에 등록됨**: submit_tasks, delegate_task 어느 경로든 task_queue.jsonl에 엔트리가 존재함
2. **Layer 1은 일시적**: 실행 큐의 파일은 소비되면 사라짐. 영구적 기록은 Layer 2가 담당
3. **Layer 2가 SSoT**: 태스크의 "공식 상태"는 task_queue.jsonl의 상태로 판정
4. **Layer 3은 자유**: Anima의 워킹 메모리이며, 시스템은 제약을 부과하지 않음
5. **PendingTaskExecutor는 Layer 2를 갱신함**: 완료/실패 시 task_queue.jsonl의 상태를 동기화
