# Heartbeat과 Cron의 설정 및 운용

Heartbeat (정기 순회)과 Cron (정시 태스크)의 설정 방법과 운용 가이드입니다.
정기 실행의 동작을 변경하거나 새로운 정시 태스크를 추가할 때 참조하세요.

## Heartbeat이란

Heartbeat은 Digital Anima가 정기적으로 자동 기동하여 상황을 확인하고 계획을 세우는 메커니즘입니다.
사람이 정기적으로 수신함을 확인하고 진행 중인 작업을 검토하는 것과 같은 행동을 자동화한 것입니다.

### 중요: Heartbeat은 "확인과 계획"만 수행

Heartbeat는 의미 있는 변화를 확인하고 필요한 다음 조치를 판단합니다. 의식적인 성찰 보고는 필요하지 않습니다.

- MUST: Heartbeat에서는 상황 확인과 판단에 집중
- MUST NOT: Heartbeat 내에서 장시간 실행 태스크 (코딩, 대량 도구 호출 등)를 수행하지 않음
- MUST: 실행이 필요한 태스크를 발견하면 부하가 있을 경우 `delegate_task`로 위임하거나 `submit_tasks`로 태스크 투입

제출된 태스크는 **TaskExec 경로**에서 정기 Heartbeat와 독립적으로 실행됩니다.
TaskExec는 정본 TaskStore에서 실행 가능한 영속 태스크를 가져옵니다. 기동 알림과 복구는 호스트가 관리하며 기존 LLM JSON 파일은 감시하지 않습니다.

### Heartbeat과 대화의 병행 동작

Heartbeat과 사람과의 대화는 **별도 잠금**으로 관리되므로 동시에 동작할 수 있습니다.
Heartbeat 실행 중에도 사람의 메시지에는 즉시 응답 가능합니다.

### submit_tasks를 통한 태스크 투입

Heartbeat에서 실행할 태스크를 발견한 경우 `submit_tasks` 도구로 태스크를 투입합니다:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "API 테스트 실시",
   "description": "Slack API 접속 테스트를 실시하고, 전체 엔드포인트의 결과를 리포트로 정리한다. 완료 후 aoi에게 보고한다."}
])
```

`submit_tasks`는 검증 후 태스크와 완전한 실행 입력을 하나의 정본 TaskStore에 원자적으로 저장합니다.
실행 권한 획득과 시도 이력은 호스트가 관리합니다. `in_progress`는 읽기 전용이며 에이전트는 `update_task`로 `done`, `pending`, `cancelled`를 선언합니다. 중단된 pending 태스크는 같은 ID에 `resume: true`를 지정해 명시적으로 재개하고 새 태스크로 대체하지 않습니다.

**주의**: 저장소를 직접 수정하지 마세요. 기존 `state/task_queue.jsonl`과 `state/pending/`은 마이그레이션·내보내기 증거로 보존하며 실행 중인 제출 경로로 사용하지 않습니다.

단일 태스크에도 `submit_tasks` (tasks 배열 1건)를 사용합니다.
여러 독립 태스크는 `parallel: true`로 병렬 실행하고, 의존 관계가 있으면 `depends_on`을 지정합니다.
자세한 내용은 task-management를 참조하세요.

### Heartbeat 트리거 종류

Heartbeat에는 2종류의 트리거가 있습니다:

| 트리거 | 설명 |
|--------|------|
| 정기 Heartbeat | `config.json`의 `heartbeat.interval_minutes`에 따라 APScheduler가 정기적으로 기동 |
| 메시지 트리거 | Inbox에 미읽 메시지가 도착하면 즉시 기동 (Inbox 경로로 처리) |

메시지 트리거에는 다음의 세이프가드가 내장되어 있습니다:
- **쿨다운**: 이전 메시지 기동 완료 후 일정 시간 내에는 재기동하지 않음 (`config.json`의 `heartbeat.msg_heartbeat_cooldown_s`, 기본값 300초)
- **캐스케이드 감지**: 2자 간 일정 시간 내 왕복이 임계값을 초과하면 루프로 간주하여 억제 (`heartbeat.cascade_window_s` 기본값 30분, `heartbeat.cascade_threshold` 기본값 3)

## heartbeat.md 설정

`heartbeat.md`는 각 Anima의 설정 파일로, 활동 시간과 체크 항목을 정의합니다.
Heartbeat의 실행 간격은 `config.json`의 `heartbeat.interval_minutes`로 설정 가능합니다 (1~60분, 기본값 30). `heartbeat.md`에서는 변경할 수 없습니다.
각 Anima에는 이름 기반의 0~9분 오프셋이 부여되어 동시 기동을 분산합니다.
파일 경로: `~/.animaworks/animas/{name}/heartbeat.md`

### 형식

```markdown
# Heartbeat: {name}

## 활동 시간
24시간 (서버 설정 타임존)

## 체크리스트
- Inbox에 미읽 메시지가 있는가
- 진행 중 태스크에 블로커가 발생하지 않았는가
- 자신의 작업 영역에 새로운 파일이 생기지 않았는가
- 아무것도 없으면 아무것도 하지 않음 (HEARTBEAT_OK)

## 통지 규칙
- 긴급하다고 판단한 경우에만 관계자에게 통지
- 동일한 내용의 통지는 24시간 이내에 반복하지 않음
```

### 설정 필드

**실행 간격**:
- `config.json`의 `heartbeat.interval_minutes`로 설정 (1~60분, 기본값 30). `heartbeat.md`에서의 변경은 불가

**활동 시간** (SHOULD):
- `HH:MM - HH:MM` 형식으로 기재 (예: `9:00 - 22:00`)
- 이 시간 외에는 Heartbeat이 기동하지 않음
- 미설정 시 기본값: 24시간 (전 시간대)
- 타임존은 `config.json`의 `system.timezone`으로 설정 가능. 미설정 시 시스템 타임존을 자동 감지

**체크리스트** (MUST):
- Heartbeat 기동 시 에이전트가 확인할 항목
- 불릿 리스트 (`- `)로 기재
- 에이전트 프롬프트에 체크리스트 내용이 그대로 전달됨
- 커스터마이즈 가능: Anima의 역할에 맞게 항목을 추가 및 변경 가능

### 체크리스트 커스터마이즈 예시

기본 (전 Anima 공통):
```markdown
## 체크리스트
- Inbox에 미읽 메시지가 있는가
- 진행 중 태스크에 블로커가 없는가
- 아무것도 없으면 아무것도 하지 않음 (HEARTBEAT_OK)
```

개발 담당 예시:
```markdown
## 체크리스트
- Inbox에 미읽 메시지가 있는가
- 진행 중 태스크에 블로커가 발생하지 않았는가
- 모니터링 대상 GitHub 리포지토리에 새로운 Issue나 PR이 없는가
- CI/CD 실패 알림이 없는가
- 아무것도 없으면 아무것도 하지 않음 (HEARTBEAT_OK)
```

커뮤니케이션 담당 예시:
```markdown
## 체크리스트
- Inbox에 미읽 메시지가 있는가
- Slack의 미읽 멘션이 없는가
- 회신 대기 중인 메일이 없는가
- 진행 중 태스크에 블로커가 없는가
- 아무것도 없으면 아무것도 하지 않음 (HEARTBEAT_OK)
```

### 실행 모델 (비용 최적화)

`background_model`이 설정되어 있으면 Heartbeat / Inbox / Cron은 메인 모델 대신 해당 모델로 실행됩니다.
Chat (사람과의 대화)과 TaskExec (실제 작업)은 메인 모델을 유지합니다.

설정 방법: `animaworks anima set-background-model {name} claude-sonnet-4-6`
자세한 내용은 `reference/operations/model-guide.md`의 "백그라운드 모델" 섹션을 참조하세요.

### Heartbeat 내부 동작

- **크래시 복구**: 이전 Heartbeat이 실패한 경우 `state/recovery_note.md`에 오류 정보가 저장됩니다. 다음 기동 시 프롬프트에 주입되고, 복구 후 파일은 삭제됩니다.
- **성찰 기록**: Heartbeat 출력에 `[REFLECTION]...[/REFLECTION]` 블록이 있으면 activity_log에 `heartbeat_reflection`으로 기록되어 이후 Heartbeat 컨텍스트에 포함됩니다.
- **부하 체크**: 부하를 가진 Anima에는 Heartbeat과 Cron 프롬프트에 부하 상태 확인 지시가 자동 주입됩니다.

### Heartbeat 설정의 핫 리로드

heartbeat.md를 파일 시스템에서 업데이트하면 다음 Heartbeat 실행 시 `_check_schedule_freshness()`가 변경을 감지하고 SchedulerManager가 스케줄을 자동 리로드합니다.
서버 재시작은 불필요합니다 (MAY skip restart). APScheduler의 작업이 재등록됩니다.

## Per-Anima Heartbeat 간격 설정

### status.json에서의 설정

각 Anima의 `status.json`에 `heartbeat_interval_minutes`를 설정하면 Anima별 Heartbeat 간격을 지정할 수 있습니다.

```json
{
  "heartbeat_interval_minutes": 60
}
```

- 설정 가능 범위: 1~1440분 (1일)
- 미설정 시: `config.json`의 `heartbeat.interval_minutes` (기본값 30분)로 폴백
- Anima 스스로 `write_memory_file`로 `status.json`을 업데이트하여 자기 조정 가능

### 권장 가이드라인

| 상황 | 권장 간격 | 이유 |
|------|-----------|------|
| 활발한 개발 프로젝트 진행 중 | 15~30분 | 빈번한 상황 파악이 필요 |
| 일반 업무 | 30~60분 | 기본값. 균형 잡힌 빈도 |
| 저부하 및 대기 상태 | 60~120분 | 비용 절약. 태스크가 없으면 긴 간격으로 |
| 장기 휴면 및 비활성 | 120~1440분 | 최소한의 순회로 상황 파악 |

### Activity Level과의 관계

글로벌 Activity Level (10%~400%)이 설정되어 있는 경우, 실효 간격은 다음 공식으로 계산됩니다:

```
실효 간격 = 기본 간격 / (Activity Level / 100)
```

예: 기본 30분, Activity Level 50% → 실효 60분
예: 기본 30분, Activity Level 200% → 실효 15분

- 실효 간격의 하한은 5분 (아무리 부스트해도 5분 미만이 되지 않음)
- Activity Level 100% 이하에서는 max_turns도 비례하여 스케일 다운 (하한 3턴)
- Activity Level 100% 이상에서는 max_turns는 변경 없음 (간격만 단축)

### Activity Schedule (시간대별 자동 전환 / 나이트 모드)

Activity Level을 시간대에 따라 자동으로 전환하는 메커니즘입니다.
야간이나 휴일에 비용을 절감하거나 업무 시간대에만 활발하게 동작시키고 싶을 때 사용합니다.

#### 구조

- `config.json`의 `activity_schedule`에 시간대 엔트리를 설정
- 1분마다 현재 시각을 체크하여 해당 시간대의 레벨로 Activity Level을 자동 변경
- Activity Level이 변경되면 전체 Anima의 Heartbeat이 즉시 리스케줄링

#### 설정 형식

각 엔트리는 `start` (시작 시각), `end` (종료 시각), `level` (Activity Level %)의 3개 필드:

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- 시각은 `HH:MM` 형식 (24시간 표기)
- **자정 경계 지원**: `"22:00"` ~ `"06:00"`과 같이 start > end 지정이 가능 (심야 시간대를 커버)
- `level`은 10~400 범위
- 최대 24개 엔트리
- 빈 배열 `[]`로 스케줄 모드를 비활성화 (고정 Activity Level로 복귀)

#### 설정 방법

- **Settings UI**: 나이트 모드 체크박스 + 시간대 및 레벨 설정
- **API**: `PUT /api/settings/activity-schedule`에 위 JSON을 전송
- **설정 파일 직접 편집**: `config.json`의 `activity_schedule`을 편집 후 서버 재시작

#### 주의 사항

- Activity Level을 수동으로 변경하면 현재 시간대에 해당하는 스케줄 엔트리도 연동하여 업데이트됨
- 스케줄은 서버 기동 시에도 즉시 적용됨 (기동 시점의 시각으로 해당 레벨 설정)
- 어느 시간대에도 해당하지 않는 경우, 마지막으로 설정된 Activity Level이 유지됨

## Cron이란

Cron은 "정해진 시간에 자동 실행되는 태스크"입니다. Heartbeat이 "정기 순회"라면, Cron은 "정시 업무"입니다.

예:
- 매일 아침 9:00에 업무 계획 수립
- 매주 금요일 17:00에 주간 성찰
- 매일 2:00에 백업 스크립트 실행

## cron.md 설정

Cron 태스크는 `cron.md`에 Markdown + YAML 형식으로 정의합니다.
파일 경로: `~/.animaworks/animas/{name}/cron.md`

### 기본 형식

각 태스크는 `## 태스크명` 헤딩으로 시작하며, 본문 첫머리에 `schedule:` 디렉티브로 표준 5필드 cron 식을 기재합니다.

```markdown
# Cron: {name}

## 매일 아침 업무 계획
schedule: 0 9 * * *
type: llm
장기 기억에서 어제의 진행 상황을 확인하고 오늘의 태스크를 계획한다.
비전과 목표에 비추어 우선순위를 판단한다.
결과는 state/current_state.md에 기록한다.

## 주간 성찰
schedule: 0 17 * * 5
type: llm
이번 주의 episodes/를 되돌아보고 패턴을 추출하여 knowledge/에 통합한다.
```

구형 형식 (`## 태스크명 (매일 9:00 JST)`처럼 괄호 안에 스케줄을 기재하는 형식)은 `animaworks migrate-cron`으로 새 형식으로 변환할 수 있습니다.

### CronTask 스키마

각 태스크는 내부적으로 다음의 `CronTask` 모델로 파싱됩니다:

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | str | (필수) | 태스크명. `##` 헤딩에서 추출 |
| `schedule` | str | (필수) | 표준 5필드 cron 식. `schedule:` 디렉티브에서 추출 |
| `type` | str | `"llm"` | 태스크 유형: `"llm"` 또는 `"command"` |
| `description` | str | `""` | LLM형 지시문 (type: llm에서 사용) |
| `command` | str \| None | `None` | Command형: bash 커맨드 |
| `tool` | str \| None | `None` | Command형: 내부 도구명 |
| `args` | dict \| None | `None` | tool의 인자 (YAML 형식) |
| `skip_pattern` | str \| None | `None` | Command형: stdout이 이 정규식에 매치하면 follow-up LLM을 건너뜀 |
| `trigger_heartbeat` | bool | `True` | Command형: `False`이면 커맨드 출력 후 follow-up cron LLM을 건너뜀 |

## LLM형 Cron 태스크

`type: llm`은 에이전트 (LLM)가 판단과 추론을 수반하여 실행하는 태스크입니다.
description에 기재된 지시가 프롬프트로 에이전트에게 전달됩니다.

### 특징

- 에이전트가 도구를 사용하고, 기억을 검색하며, 판단을 내림
- 결과는 비정형 (태스크마다 다른 출력)
- 실행에 모델 API 호출이 필요 (비용 발생)

### 기술 예시

```markdown
## 매일 아침 업무 계획
schedule: 0 9 * * *
type: llm
어제의 episodes/를 되돌아보고 오늘의 태스크를 계획한다.
우선순위는 비전과 목표에 비추어 판단한다.
결과는 state/current_state.md에 기록한다.
정본 태스크 목록(`list_tasks`)의 미착수 태스크도 확인하고 필요하면 우선순위를 재검토한다.
```

description (`type:` 행 뒤의 본문)에는 다음을 포함하는 것이 좋습니다 (SHOULD):
- 무엇을 확인할 것인가 (입력)
- 어떻게 판단할 것인가 (기준)
- 무엇을 출력할 것인가 (산출물)

## Command형 Cron 태스크

`type: command`는 에이전트의 판단 없이 정해진 커맨드나 도구를 실행하는 태스크입니다.
확정적인 처리 (백업, 통지 전송 등)에 적합합니다.

### bash 커맨드형

```markdown
## 백업 실행
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

`command:` 뒤에 bash 커맨드를 한 줄로 기재합니다.
커맨드는 셸을 통해 실행됩니다.

### 내부 도구형

```markdown
## Slack 아침 인사
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "좋은 아침입니다! 오늘도 잘 부탁드립니다."
```

`tool:` 뒤에 내부 도구명, `args:` 뒤에 YAML 형식으로 인자를 기재합니다.
args는 YAML 인덴트 블록으로 파싱됩니다 (2스페이스 인덴트).

### Command형의 follow-up 제어

Command형 태스크는 커맨드가 정상 종료되고 stdout이 있는 경우, 그 출력을 LLM에 전달하여 follow-up 분석을 수행합니다 (heartbeat에 준하는 컨텍스트로 실행).

- **`trigger_heartbeat: false`** — follow-up LLM을 건너뜀 (출력 분석이 불필요한 경우)
- **`skip_pattern: <정규식>`** — stdout이 이 정규식에 매치하면 follow-up을 건너뜀

```markdown
## 로그 수집 (출력 분석 불필요)
schedule: 0 8 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/fetch-logs.sh

## 모니터링 체크 ("OK"일 때는 분석 불필요)
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

### LLM형과 Command형의 구분 사용

| 관점 | LLM형 | Command형 |
|------|-------|-----------|
| 판단이 필요한가 | 예 | 아니오 |
| API 비용 | 있음 | 없음 |
| 출력의 예측 가능성 | 비정형 | 확정적 |
| 적합한 태스크 | 계획 수립, 성찰, 문서 작성 | 백업, 통지 전송, 데이터 수집 |
| 오류 시 대응 | 에이전트가 자율적으로 처리 | 로그 기록만 |

판단이 어려운 경우의 가이드라인:
- "매번 같은 일을 하면 됨" → Command형 (SHOULD)
- "상황에 따라 판단이 달라짐" → LLM형 (SHOULD)
- "커맨드 실행 + 결과 해석" → LLM형으로 description에 커맨드 실행을 지시

## 스케줄 표기법

cron.md의 `schedule:` 디렉티브에는 **표준 5필드 cron 식**을 기재합니다.

### 표준 cron 식 (필수)

```
분 시 일 월 요일
```

예:
- `0 9 * * *` — 매일 9:00
- `0 9 * * 1-5` — 평일 9:00
- `*/30 9-17 * * *` — 9:00~17:00 30분마다
- `0 2 1 * *` — 매월 1일 2:00
- `0 17 * * 5` — 매주 금요일 17:00

타임존은 `config.json`의 `system.timezone`으로 설정 가능합니다. 미설정 시 시스템 타임존을 자동 감지합니다.

### 일본어 스케줄 표기에서의 마이그레이션

구형 형식 (`## 태스크명 (毎日 9:00 JST)`)으로 작성된 cron.md는 `animaworks migrate-cron`으로 표준 cron 식으로 변환할 수 있습니다. 변환 대응표:

| 자연어 표기 | cron 식 예 |
|-------------|-----------|
| `Every day HH:MM` | `0 9 * * *` |
| `Weekdays HH:MM` | `0 9 * * 1-5` |
| `Every {weekday} HH:MM` | `0 17 * * 5` (금요일) |
| `Every Nth of month HH:MM` | `0 9 1 * *` |
| `Every X minutes` | `*/5 * * * *` |
| `Every X hours` | `0 */2 * * *` |

격주, 매월 말일, 제N 요일 등은 자동 변환이 불가능합니다. 수동으로 cron 식을 기재하세요.

## cron_logs 확인 방법

Cron 태스크의 실행 결과는 서버 로그에 기록됩니다.
WebSocket을 통해 `anima.cron` 이벤트로 브로드캐스트도 됩니다.

로그 확인 방법:
- 서버 로그: `animaworks.lifecycle` 로거의 INFO 레벨
- Web UI: 대시보드의 액티비티 피드에 표시
- episodes/: LLM형 태스크의 경우 에이전트가 직접 episodes/에 로그를 기록 (SHOULD)

LLM형 태스크의 결과는 `CycleResult`로 기록되며, 다음 정보를 포함합니다:
- `trigger`: `"cron"`
- `action`: 에이전트의 행동 요약
- `summary`: 결과 요약 텍스트
- `duration_ms`: 실행 시간 (밀리초)
- `context_usage_ratio`: 컨텍스트 사용률

## 자주 사용하는 Cron 설정 예시

### 기본 세트 (전 Anima 권장)

```markdown
# Cron: {name}

## 매일 아침 업무 계획
schedule: 0 9 * * *
type: llm
episodes/에서 어제의 활동을 확인하고, 정본 태스크 목록(`list_tasks`)의 미착수 태스크를 확인한다.
오늘의 우선 태스크를 결정하고 state/current_state.md를 업데이트한다.

## 주간 성찰
schedule: 0 17 * * 5
type: llm
이번 주의 episodes/를 되돌아보고, 패턴과 교훈을 추출한다.
중요한 지식은 knowledge/에 기록한다.
반복적으로 수행한 작업이 있으면 procedures/에 절차화를 검토한다.
```

### 외부 연동 태스크

```markdown
## Slack 일일 보고 전송
schedule: 0 18 * * 1-5
type: command
tool: slack_send
args:
  channel: "#daily-report"
  message: "오늘 업무를 완료했습니다. 상세 내용은 내일 조회에서 공유합니다."

## GitHub Issue 확인
schedule: 0 10 * * 1-5
type: llm
담당 리포지토리의 새로운 Issue와 PR을 확인한다.
중요한 것이 있으면 supervisor에게 보고한다.
```

### 기억 유지보수

```markdown
## 지식 점검
schedule: 0 10 1 * *
type: llm
knowledge/의 전체 파일을 확인하고, 오래된 정보나 모순되는 기재를 정리한다.
중요도가 낮은 지식은 아카이브를 검토한다.

## 절차서 업데이트 확인
schedule: 0 10 * * 1
type: llm
procedures/의 절차서를 확인하고, 실제 운용과 괴리가 없는지 검토한다.
변경이 있으면 절차서를 업데이트한다.
```

### 주석 처리

실행하지 않을 태스크는 HTML 주석으로 감쌉니다:

```markdown
<!--
## 일시 정지 중인 태스크
schedule: 0 15 * * *
type: llm
이 태스크는 일시적으로 정지 중입니다.
-->
```

주석 내의 `## ` 헤딩은 파서에서 무시됩니다.

## Cron 설정의 핫 리로드

cron.md를 업데이트하면 heartbeat.md와 마찬가지로 스케줄이 자동 리로드됩니다.
Anima 스스로 cron.md를 수정한 경우에도 즉시 반영됩니다 (self-modify 패턴).

리로드 동작:
1. 해당 Anima의 기존 cron 작업을 전부 삭제
2. 업데이트된 cron.md를 파싱하여 새로운 작업을 등록
3. 로그에 `Schedule reloaded for '{name}'`가 출력

직접 cron.md를 수정할 때의 주의점:
- 헤딩 (`## 태스크명`) 바로 다음에 `schedule:` 디렉티브를 배치 (MUST)
- 스케줄은 표준 5필드 cron 식으로 기재 (MUST)
- type 행은 schedule 바로 다음에 배치 (SHOULD)
