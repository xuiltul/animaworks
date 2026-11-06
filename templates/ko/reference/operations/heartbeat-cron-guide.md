# 정기 실행 설정 및 운영

하트비트(정기 순회)와 Cron(정시 작업)의 설정 방법 및 운영 가이드.
정기 실행 동작을 변경하거나 새 정시 작업을 추가하려는 경우 참조한다.

## 하트비트란

하트비트는 Digital Anima가 정기적으로 자동 시작하여 상황을 확인하고 계획하는 기능이다.
사람이 주기적으로 받은 편지함을 확인하고 진행 중인 작업을 검토하는 것과 같은 행동을 자동화한 것이다.

### 중요: 하트비트는 ‘확인과 계획’만 수행

Heartbeat는 의미 있는 변화를 확인하고 필요한 다음 대응을 판단한다. 형식적인 회고 보고는 필요하지 않다.

- MUST: Heartbeat는 상황 확인과 판단에만 집중한다
- MUST NOT: 하트비트에서 장시간 실행 작업(코딩, 대량의 도구 호출 등)을 수행하지 않는다
- MUST: 실행이 필요한 작업을 발견하면 부하가 있는 경우 `delegate_task`로 위임하거나, `submit_tasks`로 작업을 투입한다

투입된 작업은 **TaskExec 경로**가 정기 Heartbeat와 독립적으로 실행한다.
TaskExec는 정본 TaskStore에서 실행 가능한 영구 작업을 가져온다. 기상 알림과 복구는 호스트가 관리하며, 이전 LLM JSON 파일은 모니터링하지 않는다.

### 하트비트와 대화의 병렬 동작

하트비트와 사람과의 대화는 **별도의 잠금**으로 관리되므로 동시에 동작할 수 있다.
하트비트 실행 중에도 사람의 메시지에 즉시 응답할 수 있다.

### submit_tasks를 통한 작업 투입

하트비트에서 실행해야 할 작업을 발견한 경우, `submit_tasks` 도구로 작업을 투입한다:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "APIテスト実施",
   "description": "Slack API接続テストを実施し、全エンドポイントの結果をレポートにまとめる。完了後 aoi に報告する。"}
])
```

`submit_tasks`는 검증 후 작업과 완전한 실행 입력을 하나의 정본 TaskStore에 일괄 저장한다.
실행 권한 획득과 시도 이력은 호스트가 관리한다. `in_progress`은 조회용이며, 에이전트는 `update_task`에서 `done` / `pending` / `cancelled`을 선언한다. 중단된 pending 작업은 동일한 ID에 `resume: true`을 지정해 명시적으로 재개하며, 다른 작업으로 대체하지 않는다.

**장시간 CLI 도구**(`animaworks-tool submit …`)는 `task_type="command"`로 TaskStore에 등록되고, PendingTaskExecutor가 시도를 가져와 백그라운드에서 실행한다. 완료 결과는 `state/background_tasks/{task_id}.json`과 알림에서 확인한다. 자세한 내용은 `operations/background-tasks.md`을 참조한다.

**주의**: 저장 위치를 직접 편집하지 않는다. 이전 `state/task_queue.jsonl`과 `state/pending/`는 마이그레이션 및 내보내기용 증거로 보존하며, 실행 중인 작업 투입 대상으로 사용하지 않는다.

단일 작업에도 `submit_tasks`(tasks 배열 1개 항목)를 사용한다.
서로 독립적인 여러 작업은 `parallel: true`로 병렬 실행하고, 의존 관계가 있는 경우 `depends_on`을 지정한다.
자세한 내용은 task-management를 참조한다.

### 하트비트 트리거 유형

하트비트에는 두 가지 트리거가 있다:

| 트리거 | 설명 |
|---------|------|
| 정기 하트비트 | `config.json`의 `heartbeat.interval_minutes`에 따라 APScheduler가 정기적으로 시작 |
| 메시지 트리거 | Inbox에 읽지 않은 메시지가 도착하면 즉시 시작(Inbox 경로로 처리) |

메시지 트리거는 Inbox의 JSON 파일 변경 알림으로 시작된다. 알림을 놓친 경우에 대비해 45초마다 읽지 않은 메시지를 다시 확인한다.
동시에 실행되는 Inbox 처리는 하나뿐이며, 실행 중 도착한 메시지는 다음 한 번의 처리에 모아서 처리한다. Provider 오류가 발생하면 `rate_guard`의 복구 시간을 기다리고, 읽지 않은 메시지는 남겨 둔다.
수신 intent에 따른 시작 필터나 메시지 기점의 cooldown / cascade 억제는 없다. 확인 및 감사만을 위한 메시지에 답장하지 않는 행동 규칙은 Inbox 프롬프트에서 지시한다.

## HEARTBEAT.md 설정

`HEARTBEAT.md`은 각 Anima의 설정 파일로, 활동 시간과 확인 항목을 정의한다.
하트비트 실행 간격은 `config.json`의 `heartbeat.interval_minutes`에서 설정할 수 있다(1~1440분, 기본값 30). `HEARTBEAT.md`에서는 변경할 수 없다.
각 Anima에는 이름을 기준으로 0~9분의 오프셋이 부여되어 동시 시작을 분산한다.
파일 경로: `~/.animaworks/animas/{name}/HEARTBEAT.md`

상급자가 하위 Anima의 `HEARTBEAT.md`를 편집하는 경우, 직접 파일을 조작하지 말고 `read_memory_file` / `write_memory_file`을 사용하여 `../{anima_name}/HEARTBEAT.md`과 같은 상대 경로로 지정한다.

### 형식

```markdown
# Heartbeat: {name}

## 活動時間
24時間（サーバー設定タイムゾーン）

## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 自分の作業領域に新しいファイルが置かれていないか
- 何もなければ何もしない（HEARTBEAT_OK）

## 通知ルール
- 緊急と判断した場合のみ関係者に通知
- 同じ内容の通知は24時間以内に繰り返さない
```

### 설정 필드

**실행 간격**:
- `config.json`의 `heartbeat.interval_minutes`에서 설정(1~1440분, 기본값 30). `HEARTBEAT.md`에서 변경할 수 없음

**활동 시간**(SHOULD):
- `HH:MM - HH:MM` 형식으로 기재한다(예: `9:00 - 22:00`)
- 이 시간대 외에는 하트비트가 시작되지 않는다
- 미설정 시 기본값: 24시간(모든 시간대)
- 시간대는 `config.json`의 `system.timezone`에서 설정할 수 있다. 미설정 시 시스템 시간대를 자동 감지한다

**체크리스트**(MUST):
- 하트비트 시작 시 에이전트가 확인할 항목
- 글머리 기호 목록(`- `로 시작)으로 기재한다
- 체크리스트 내용이 에이전트 프롬프트에 그대로 전달된다
- 사용자 지정 가능: Anima의 역할에 맞춰 항목을 추가하거나 변경해도 된다

### 체크리스트 사용자 지정 예시

기본값(모든 Anima 공통):
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

개발 담당 예시:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 監視対象のGitHubリポジトリに新しいIssueやPRがないか
- CI/CDの失敗アラートがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

커뮤니케이션 담당 예시:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- Slackの未読メンションがないか
- 返信待ちのメールがないか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

### 실행 모델(비용 최적화)

Heartbeat / Inbox / Cron은 `background_model`이 설정되어 있으면 메인 모델 대신 해당 모델로 실행된다.
Chat(사람과의 대화)과 TaskExec(실제 작업)는 메인 모델을 유지한다.

설정 방법: `animaworks anima set-background-model {名前} claude-sonnet-5-5`
자세한 내용은 `reference/operations/model-guide.md`의 ‘백그라운드 모델’ 섹션을 참조한다.

### 하트비트 내부 동작

- **충돌 복구**: 이전 하트비트가 실패한 경우 `state/recovery_note.md`에 오류 정보가 저장된다. 다음 시작 시 프롬프트에 주입되며, 복구 후 파일은 삭제된다.
- **회고 기록**: 하트비트 출력에 `[REFLECTION]...[/REFLECTION]` 블록이 있으면 activity_log에 `heartbeat_reflection`로 기록되어 이후 하트비트 컨텍스트에 포함된다.
- **부하 확인**: 부하가 있는 Anima의 경우 하트비트 및 Cron 프롬프트에 부하 상태 확인 지침이 자동으로 주입된다.
- **세션 시간 제한**(`config.json`의 `heartbeat`): `soft_timeout_seconds`(기본값 300초)이 지나면 마무리 알림을 주입하고, `hard_timeout_seconds`(기본값 600초)이 지나면 세션을 강제 종료한다.
- **유휴 상태 자동 압축**: `heartbeat.idle_compaction_minutes`(기본값 10분) — 스트림 종료 후 해당 시간이 지나면 유휴 자동 압축이 실행된다(실행 엔진 측 설정).

### 정기 하트비트 스케줄 방식

실효 간격(Activity Level 적용 후, 최소 5분으로 반올림)이 **60분 이하이며 60을 나누어떨어지게 하는 경우**, APScheduler의 `CronTrigger`에서 분 슬롯에 분산 등록된다(이름 기준 0~9분 오프셋과 조합).

그 외의 경우(예: 실효 간격 61분 또는 43분처럼 60을 나누어떨어지게 하지 않는 간격)는 **1분마다 폴링**(`_heartbeat_check`)하여 ‘이전 실행 이후 경과 시간’에 따라 실행한다. 오래된 `IntervalTrigger`에서 비롯된 오류를 피하기 위한 구현이다.

### 백그라운드 도구(core/tasks/background.py）

`core/tasks/background.py`은 하트비트/Cron 스케줄 자체가 아니라 **장시간 도구 호출의 백그라운드 실행**(상태의 JSON 영속화 포함)을 담당한다. 운영 세부 정보와 CLI 경로는 `operations/background-tasks.md`도 참조한다.

#### BackgroundTaskManager

- **저장 위치**: `state/background_tasks/{task_id}.json`. `task_id`은 UUID의 첫 12자(16진수)다. 각 파일에는 `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, `error`이 기록된다.
- **상태(`TaskStatus`)**: `pending` / `running` / `completed` / `failed`. `submit` / `submit_async`에서는 투입 직후부터 `running`로 JSON이 작성되고, 완료 또는 예외 발생 시 `completed` / `failed`으로 업데이트된다.
- **실행 API**: `submit(tool_name, tool_args, execute_fn)`은 동기 callable을 `asyncio`의 `run_in_executor`에서 스레드 풀로 실행한다. `submit_async`은 비동기 callable을 그대로 await한다. `get_task`(메모리 우선, 없으면 디스크), `list_tasks`(디스크의 JSON도 병합, 생성 시각 최신순), `active_count`(메모리 내 `running` 개수)을 제공한다.
- **완료 콜백**: `on_complete`에 전달한 비동기 함수는 작업 저장 후 호출된다. 콜백 내에서 예외가 발생해도 작업 결과는 보존되며 로그에만 기록된다(일반적으로 `state/background_notifications/`에 쓰기와 함께 사용되며, **다음 하트비트**에서 읽고 삭제하여 대화 컨텍스트에 포함하는 경로).
- **후보 판정 `is_eligible(tool_name)`**: 맵에 키가 있으면 백그라운드 대상이다. 키는 다음 두 가지 형식을 모두 허용한다. (1) **스키마 이름**(예: `generate_3d_model`) — Mode A의 외부 도구 디스패치 등. (2) **`ツール名:サブコマンド`**(예: `image_gen:pipeline`) — 각 도구 모듈의 `EXECUTION_PROFILE`에서 `background_eligible: true` 항목이 `get_eligible_tools_from_profiles()`에 의해 이 형식으로 등록된다(Mode S의 `submit` 경로 등).
- **후보 도구 맵 구성 `from_profiles()`**: 다음 3개 계층을 dict의 `update`로 병합하며, **나중 값이 우선**하여 덮어쓴다. (1) `_DEFAULT_ELIGIBLE_TOOLS`(코드 기본값) (2) 인수 `profiles`(`EXECUTION_PROFILE` 집계) (3) 인수 `config_eligible`(일반적으로 `config.json`의 `background_task.eligible_tools`에서 `threshold_s`을 확장한 `名前 → 秒`). 값은 프로필 연계를 위한 예상 초(정수)다.
- **코드 기본값 `_DEFAULT_ELIGIBLE_TOOLS`(초)**: `generate_character_assets` 30, `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` 각 30, `generate_3d_model` / `generate_rigged_model` / `generate_animations` 각 30, `local_llm` 60, `run_command` 60.
- **정리 `cleanup_old_tasks(max_age_hours=24)`**: `status`이 `completed` / `failed`에서 `completed_at`가 **인수로 지정한 시간(기본값 24시간)보다 오래된** JSON을 삭제한다. 또한 `running` 상태로 `created_at`부터 **48시간 초과** 경과한 파일은 충돌로 남은 고아 파일로 삭제한다. 반환값은 삭제 개수다.
- **보존 시간**: `cleanup_old_tasks`의 완료 작업 보존 시간은 인수 `max_age_hours`(기본값 24시간)으로 지정한다. 이에 해당하는 `config.json` 설정 키는 없다.

- **기타 서버 정시 작업**: Anima 단위의 `cron.md`과 별도로, 라이프사이클이 메모리 유지 관리, RAG 등의 시스템 Cron을 등록한다(예: 일일 통합 02:00, 일일 인덱싱 04:00). 시각은 설정된 시간대를 기준으로 한다.

### 하트비트 설정 핫 리로드

파일 시스템에서 HEARTBEAT.md을 업데이트하면 다음 하트비트 실행 시 `_check_schedule_freshness()`이 변경 사항을 감지하고 SchedulerManager가 스케줄을 자동으로 다시 불러온다.
서버를 재시작할 필요는 없다(MAY skip restart). APScheduler 작업이 다시 등록된다.

## Anima별 Heartbeat 간격 설정

### status.json에서 설정

root 관리자는 root CLI/API에서 Anima별 `heartbeat_interval_minutes`을 설정할 수 있다. 값은 root 소유 `status.json`에 저장되며, Anima 프로세스에서 편집하지 않는다.

```json
{
  "heartbeat_interval_minutes": 60
}
```

- 설정 가능 범위: 1~1440분(하루)
- 미설정 시: `config.json`의 `heartbeat.interval_minutes`(기본값 30분)으로 대체
- `status.json`는 root 소유다. 관리자는 서버가 중지된 동안 `animaworks config set animas.<name>.heartbeat_interval_minutes <minutes>`에서 개별 재정의를 설정할 수 있다. 서버 실행 중에는 CLI가 root API를 통해 적용한다. Anima 프로세스는 이 파일을 편집하지 않는다.

### 권장 지침

| 상황 | 권장 간격 | 이유 |
|------|----------|------|
| 활발한 개발 프로젝트 중 | 15〜30분 | 빈번한 상태 파악이 필요 |
| 일반 업무 | 30〜60분 | 기본값. 균형 잡힌 빈도 |
| 부하가 낮거나 대기 상태 | 60〜120분 | 비용 절약. 작업이 없으면 간격을 길게 |
| 장기간 휴면 또는 비활성 | 120〜1440분 | 최소한의 순회로 상태 파악 |

### Activity Level과의 관계

전역 Activity Level(10%〜400%)이 설정되어 있는 경우, 실제 간격은 다음 식으로 계산됩니다.

```
実効間隔 = ベース間隔 / (Activity Level / 100)
```

예: 기본 30분, Activity Level 50% → 실제 60분
예: 기본 30분, Activity Level 200% → 실제 15분

- 실제 간격의 하한은 5분입니다(아무리 부스트해도 5분 미만이 되지 않음)

### Activity Schedule(시간대별 자동 전환 / 나이트 모드)

시간대에 따라 Activity Level을 자동으로 전환하는 기능입니다.
야간이나 휴일에 비용을 절감하고 싶거나, 업무 시간대에만 활발히 작동하게 하고 싶은 경우에 사용합니다.

#### 작동 방식

- `config.json`의 `activity_schedule`에 시간대 항목을 설정합니다
- 1분마다 현재 시각을 확인하고, 해당하는 시간대의 레벨로 Activity Level을 자동 변경합니다
- Activity Level이 바뀌면 모든 Anima의 하트비트가 즉시 다시 예약됩니다

#### 설정 형식

각 항목은 `start`(시작 시각), `end`(종료 시각), `level`(Activity Level %)의 3개 필드로 구성됩니다.

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- 시각은 `HH:MM` 형식(24시간 표기)
- **날짜 경계 통과 지원**: `"22:00"`〜`"06:00"`처럼 start > end로 지정 가능(심야 시간대 적용)
- `level`은 10〜400 범위
- 최대 24개 항목까지 설정 가능
- 빈 배열 `[]`로 스케줄 모드를 비활성화할 수 있습니다(고정 Activity Level로 복귀)

#### 설정 방법

- **Settings UI**: 나이트 모드 체크박스 + 시간대 및 레벨 설정
- **API**: `PUT /api/settings/activity-schedule`에 위 JSON을 전송
- **root 설정 API**: Settings UI를 사용하거나 위 JSON을 `PUT /api/settings/activity-schedule`에 전송합니다. Anima 프로세스에서 `config.json`를 직접 편집하지 마세요

#### 주의 사항

- Activity Level을 수동으로 변경하면 현재 시간대에 해당하는 스케줄 항목도 함께 업데이트됩니다
- 서버 시작 시에도 스케줄이 즉시 적용됩니다(시작 시각에 해당하는 레벨로 설정)
- 어떤 시간대에도 해당하지 않으면 마지막으로 설정된 Activity Level이 유지됩니다

## Cron 작업이란

Cron은 “정해진 시간에 자동으로 실행되는 작업”입니다. 하트비트가 “정기 순회”라면 Cron은 “정시 업무”입니다.

예:
- 매일 아침 9:00에 업무 계획을 세우기
- 매주 금요일 17:00에 주간 회고하기
- 매일 2:00에 백업 스크립트를 실행하기

## cron.md 설정

Cron 작업은 `cron.md`에 Markdown + YAML 형식으로 정의합니다.
파일 경로: `~/.animaworks/animas/{name}/cron.md`

상급자가 하위 Anima의 `cron.md`를 편집할 때는 직접 파일을 조작하지 말고 `read_memory_file` / `write_memory_file`를 사용하며, `../{anima_name}/cron.md`와 같은 상대 경로로 지정합니다.

### 기본 형식

각 작업은 `## タスク名` 제목으로 시작하며, 본문 첫머리에 `schedule:` 지시어로 표준 5필드 cron 식을 기재합니다.

```markdown
# Cron: {name}

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。

## 週次振り返り
schedule: 0 17 * * 5
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
```

이전 형식(`## タスク名（毎日 9:00 JST）`처럼 괄호 안에 스케줄을 적는 형식)은 서버 시작 시 또는 `animaworks migrate`에서 새 형식으로 자동 변환됩니다.

### CronTask 스키마

각 작업은 내부적으로 다음 `CronTask` 모델로 파싱됩니다.

| 필드 | 유형 | 기본값 | 설명 |
|-----------|------|-----------|------|
| `name` | str | (필수) | 작업 이름. `##` 제목에서 추출 |
| `schedule` | str | (필수) | 표준 5필드 cron 식. `schedule:` 지시어에서 추출 |
| `type` | str | `"llm"` | 작업 유형: `"llm"` 또는 `"command"` |
| `description` | str | `""` | LLM 유형의 지침 문구(type: llm에서 사용) |
| `command` | str \| None | `None` | Command 유형의 bash 명령 |
| `tool` | str \| None | `None` | Command 유형의 내부 도구 이름 |
| `args` | dict \| None | `None` | tool 인수(YAML 형식) |
| `skip_pattern` | str \| None | `None` | Command 유형: stdout이 이 정규식과 일치하면 후속 LLM을 건너뜀 |
| `trigger_heartbeat` | bool | `True` | Command 유형: `False`이면 명령 출력 후의 후속 cron LLM을 건너뜀 |

## LLM 유형 Cron 작업

`type: llm`은 에이전트(LLM)가 판단하고 사고하면서 수행하는 작업입니다.
description에 작성한 지침이 프롬프트로 에이전트에 전달됩니다.

### 특징

- 에이전트가 도구를 사용하고, 기억을 검색하고, 판단을 내립니다
- 결과는 정형화되어 있지 않습니다(작업마다 출력이 다름)
- 실행하려면 모델 API 호출이 필요합니다(비용 발생)

### 작성 예시

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
昨日の episodes/ を読み返し、今日のタスクを計画する。
優先順位は理念と目標に照らして判断する。
結果は state/current_state.md に書き出す。
正本タスク一覧（`list_tasks`） の未着手タスクも確認し、必要なら優先度を見直す。
```

description(`type:` 행 뒤의 본문)에는 다음 내용을 포함해야(SHOULD) 합니다.
- 무엇을 확인할지(입력)
- 어떻게 판단할지(기준)
- 무엇을 출력할지(산출물)

## Command 유형 Cron 작업

`type: command`은 에이전트의 판단을 거치지 않고 정해진 명령이나 도구를 실행하는 작업입니다.
백업, 알림 전송 등 확정적인 처리에 적합합니다.

### bash 명령 유형

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

`command:`에 bash 명령을 한 줄로 작성합니다.
명령은 셸을 통해 실행됩니다.

### 내부 도구 유형

```markdown
## Slack朝の挨拶
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "おはようございます！本日もよろしくお願いします。"
```

`tool:`에 내부 도구 이름을, `args:`에 인수를 YAML 형식으로 작성합니다.
args는 YAML 들여쓰기 블록으로 파싱됩니다(들여쓰기 2칸).

### Command 유형의 후속 처리 제어

Command 유형 작업은 명령이 정상 종료되고 stdout이 있는 경우, 해당 출력을 LLM에 전달하여 후속 분석을 수행합니다(하트비트에 준하는 컨텍스트에서 실행).

- **`trigger_heartbeat: false`** — 후속 LLM을 건너뜁니다(출력 분석이 필요하지 않은 경우)
- **`skip_pattern: <正規表現>`** — stdout이 이 정규식과 일치하면 후속 처리를 건너뜁니다

```markdown
## ログ取得（出力分析不要）
schedule: 0 8 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/fetch-logs.sh

## 監視チェック（"OK" のときは分析不要）
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

### LLM 유형과 Command 유형의 구분

| 관점 | LLM 유형 | Command 유형 |
|------|--------|-----------|
| 판단이 필요한가 | 예 | 아니요 |
| API 비용 | 있음 | 없음 |
| 출력 예측 가능성 | 정형화되지 않음 | 확정적 |
| 적합한 작업 | 계획 수립, 회고, 글쓰기 | 백업, 알림 전송, 데이터 가져오기 |
| 오류 발생 시 대응 | 에이전트가 자율적으로 처리 | 로그에 기록만 함 |

선택이 어려울 때의 지침:
- “매번 같은 일을 하기만 함” → Command 유형(SHOULD)
- “상황에 따라 판단이 달라짐” → LLM 유형(SHOULD)
- “명령 실행 + 결과 해석” → LLM 유형으로 설정하고 description에서 명령 실행을 지시

## 스케줄 표기법

cron.md의 `schedule:` 지시어에는 **표준 5필드 cron 식**을 기재합니다.

### 표준 cron 식(필수)

```
分 時 日 月 曜日
```

예:
- `0 9 * * *` — 매일 9:00
- `0 9 * * 1-5` — 평일 9:00
- `*/30 9-17 * * *` — 9:00〜17:00에 30분마다
- `0 2 1 * *` — 매월 1일 2:00
- `0 17 * * 5` — 매주 금요일 17:00

시간대는 `config.json`의 `system.timezone`에서 설정할 수 있습니다. 설정하지 않으면 시스템 시간대를 자동으로 감지합니다.

### 일본어 스케줄에서 마이그레이션

이전 형식(`## タスク名（毎日 9:00 JST）`)으로 작성된 cron.md은 서버 시작 시 또는 `animaworks migrate`에서 표준 cron 식으로 자동 변환됩니다. 변환 대응표:

| 일본어 표기법 | cron 식 예시 |
|-----------|----------|
| `毎日 HH:MM` | `0 9 * * *` |
| `平日 HH:MM` | `0 9 * * 1-5` |
| `毎週{曜日} HH:MM` | `0 17 * * 5`(금요일) |
| `毎月N日 HH:MM` | `0 9 1 * *` |
| `X分毎` | `*/5 * * * *` |
| `X時間毎` | `0 */2 * * *` |

`隔週`, `毎月最終日`, `第N曜日`은 자동 변환할 수 없습니다. cron 식을 수동으로 기재하세요.

## cron_logs 확인 방법

Cron 작업의 실행 결과는 서버 로그에 기록됩니다.
WebSocket을 통해 `anima.cron` 이벤트로도 브로드캐스트됩니다.

로그 확인 방법:
- 서버 로그: `animaworks.lifecycle` 로거의 INFO 레벨
- Web UI: 대시보드의 활동 피드에 표시
- episodes/: LLM 유형 작업의 경우, 에이전트 자체가 episodes/에 로그를 작성해야(SHOULD) 합니다

LLM 유형 작업의 결과는 `CycleResult`로 기록되며, 다음 정보가 포함됩니다.
- `trigger`: `"cron"`
- `action`: 에이전트 행동 요약
- `summary`: 결과 요약 텍스트
- `duration_ms`: 실행 시간(밀리초)
- `context_usage_ratio`: 컨텍스트 사용률

## 자주 사용하는 Cron 설정 예시

### 기본 세트(모든 Anima에 권장)

```markdown
# Cron: {name}

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
episodes/ から昨日の行動を確認し、正本タスク一覧（`list_tasks`） の未着手タスクを見直す。
今日の優先タスクを決め、state/current_state.md を更新する。

## 週次振り返り
schedule: 0 17 * * 5
type: llm
今週の episodes/ を読み返し、パターンや教訓を抽出する。
重要な知見は knowledge/ に書き出す。
繰り返し行った作業があれば procedures/ に手順化を検討する。
```

### 외부 연동 작업

```markdown
## Slack日報送信
schedule: 0 18 * * 1-5
type: command
tool: slack_send
args:
  channel: "#daily-report"
  message: "本日の業務完了しました。詳細は明日の朝礼で共有します。"

## GitHub Issue 確認
schedule: 0 10 * * 1-5
type: llm
担当リポジトリの新しい Issue と PR を確認する。
重要なものがあれば supervisor に報告する。
```

### 기억 유지보수

```markdown
## 知識の棚卸し
schedule: 0 10 1 * *
type: llm
knowledge/ の全ファイルを確認し、古い情報や矛盾する記載を整理する。
重要度の低い知識はアーカイブを検討する。

## 手順書の更新確認
schedule: 0 10 * * 1
type: llm
procedures/ の手順書を確認し、実際の運用と乖離がないか見直す。
変更があれば手順書を更新する。
```

### 주석 처리

실행하지 않을 작업은 HTML 주석으로 감쌉니다.

```markdown
<!--
## 一時停止中のタスク
schedule: 0 15 * * *
type: llm
このタスクは一時的に停止中。
-->
```

주석 안의 `## ` 제목은 파서에서 무시됩니다.

## Cron 설정 핫 리로드

cron.md을 업데이트하면 HEARTBEAT.md과 마찬가지로 스케줄이 자동으로 다시 로드됩니다.
Anima 자체가 cron.md을 수정한 경우에도 즉시 반영됩니다(self-modify 패턴).
상급자는 하위 항목인 `cron.md` / `HEARTBEAT.md`을 write memory 도구로 편집할 수 있습니다. 인가된 상급자의 `AGENTS.md` 변경 요청도 root로 전달되어 인가됩니다. `status.json`을 변경할 때는 해당 상급자용 도구 또는 root CLI/API을 사용하세요. root 소유 파일에 Read / Write / Edit / apply_patch / `Path.write_text` / 셸 리디렉션 등으로 직접 기록하지 마세요.

다시 로드할 때의 동작:
1. 해당 Anima의 기존 cron 작업을 모두 삭제
2. 업데이트된 cron.md을 파싱하고 새 작업을 등록
3. 로그에 `Schedule reloaded for '{name}'`가 출력됨

직접 cron.md을 업데이트할 때의 주의 사항:
- 제목(`## タスク名`) 바로 뒤에 `schedule:` 지시어를 둡니다(MUST)
- 스케줄은 표준 5필드 cron 식으로 작성합니다(MUST)
- type 행은 schedule 바로 뒤에 둡니다(SHOULD)
