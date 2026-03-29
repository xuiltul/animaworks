---
name: subordinate-management
description: >-
  부하 Anima 슈퍼바이저 도구. 비활성화·복귀·모델 변경·재시작·태스크 위임·상태 조회·감사를 수행한다.
  Use when: 부하 비활성·재활성, 메인·BG 모델 변경, 프로세스 재시작, delegate_task, 조직 대시보드 확인이 필요할 때.
---

# 스킬: 부하 관리 (슈퍼바이저 도구)

부하를 가진 Anima에 자동으로 활성화되는 슈퍼바이저 도구입니다. 모든 도구는 전체 배하(자식, 손자, 증손자 등)에 대해 동작하며, 직속 부하와 간접 부하를 구분하지 않습니다.

## 사용 가능한 도구

### 전체 배하 (자식, 손자, 증손자... 모두)에 조작 가능

| 도구 | 용도 |
|------|------|
| `disable_subordinate` | 배하 정지 (status.json `enabled: false` → 프로세스 중지 + 자동 복귀 방지) |
| `enable_subordinate` | 정지 중인 배하 복귀 |
| `set_subordinate_model` | 배하의 메인 LLM 모델 변경 (status.json 업데이트; 반영에는 `restart_subordinate` 필요) |
| `set_subordinate_background_model` | 배하의 백그라운드 모델(heartbeat/cron용) 변경 (status.json 업데이트; 반영에는 `restart_subordinate` 필요; 빈 문자열로 클리어) |
| `restart_subordinate` | 배하 프로세스 재시작 (status.json `restart_requested` 플래그; Reconciliation이 약 30초 이내에 재시작) |
| `delegate_task` | 배하에 태스크 위임 (큐 추가 + DM 전송 + 자신 측 추적 항목 생성) |
| `org_dashboard` | 전체 배하의 프로세스 상태, 최근 활동, 현재 태스크, 태스크 수를 트리 형태로 표시 |
| `ping_subordinate` | 배하 생존 확인 (`name` 생략 시 전원 일괄, 지정 시 단일) |
| `read_subordinate_state` | 배하의 `current_state.md` 읽기 |
| `audit_subordinate` | 배하의 최근 활동 포괄 감사 (활동 요약, 태스크 상황, 에러 빈도, 도구 사용 통계, 통신 패턴) |

### 위임 태스크 추적

| 도구 | 용도 |
|------|------|
| `task_tracker` | `delegate_task`로 위임한 태스크의 진행 상황을 부하 측 큐에서 추적 (`status`: all / active / completed; 기본값: active) |

## 중요: disable_subordinate와 send_message의 차이

- **disable_subordinate**: status.json을 `enabled: false`로 변경. Reconciliation이 자동 복귀시키지 않음. **이것을 사용하세요**
- send_message로 "쉬어"라고 전달하는 것만으로는 **프로세스가 중지되지 않습니다**. 메시지를 보내도 Reconciliation이 재시작합니다

## 사용법

### 정지 및 복귀

여러 부하를 정지할 때는 각각 `disable_subordinate`를 호출합니다:

```
disable_subordinate(name="aoi", reason="업무 축소로 인한 일시 정지")
disable_subordinate(name="taro", reason="업무 축소로 인한 일시 정지")
enable_subordinate(name="aoi")
```

### 모델 변경 및 재시작

모델 변경은 status.json에 저장되지만, 실행 중인 프로세스에 반영하려면 `restart_subordinate`가 필요합니다:

```
set_subordinate_model(name="aoi", model="claude-sonnet-4-6", reason="부하 분산을 위해")
restart_subordinate(name="aoi", reason="모델 변경 반영")
```

백그라운드 모델(heartbeat/cron용)을 변경하는 경우:

```
set_subordinate_background_model(name="aoi", model="claude-sonnet-4-6", reason="heartbeat 부하 경감")
restart_subordinate(name="aoi", reason="백그라운드 모델 변경 반영")
```

백그라운드 모델을 클리어하고 메인 모델로 되돌리는 경우:

```
set_subordinate_background_model(name="aoi", model="", reason="메인 모델로 통일")
restart_subordinate(name="aoi")
```

### 상태 확인 및 감사

```
org_dashboard()                         # 전체 배하 대시보드
ping_subordinate()                      # 전체 배하 생존 확인
ping_subordinate(name="aoi")            # 단일 생존 확인
read_subordinate_state(name="aoi")      # 현재 태스크 및 보류 태스크 내용
audit_subordinate(name="aoi")           # 최근 1일 포괄 감사 리포트
audit_subordinate(name="aoi", days=7)   # 최근 7일 감사 (days: 1~30)
audit_subordinate(since="09:00")        # 전체 배하 오늘 9시 이후 감사
audit_subordinate(name="aoi", since="13:00")  # aoi 오늘 13시 이후
```

CLI에서도 실행 가능합니다 (S-mode의 Bash에서 사용 시 편리):

```bash
animaworks anima audit aoi              # 최근 1일 감사
animaworks anima audit aoi --days 7     # 최근 7일 감사
animaworks anima audit --all --since 09:00  # 전체 Anima, 오늘 9시 이후
```

### 태스크 위임

```
delegate_task(name="aoi", instruction="주간 리포트를 정리해 주세요", deadline="1d", summary="주간 리포트 작성")
# name, instruction, deadline은 필수. summary는 선택 (instruction의 첫 100자가 사용됨)
# workspace를 지정하면 위임 대상이 해당 워크스페이스에서 작업합니다 (workspace-manager 스킬 참조)
task_tracker(status="active")      # 위임 태스크 진행 확인 (status: all / active / completed)
```

부하에 대한 워크스페이스 할당(주 작업 디렉토리 지정)은 `workspace-manager` 스킬을 참조하세요.

## 권한

- **전체 배하 (재귀)**: 모든 도구가 사용 가능. 직속 부하와 손자 이하를 구분하지 않음
- 자기 자신에 대한 조작은 불가
