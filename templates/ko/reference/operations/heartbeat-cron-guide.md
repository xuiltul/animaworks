# 정기 실행의 설정과 운영

하트비트(정기 순회)와 Cron(정시 작업)의 설정 방법·운영 가이드.
정기 실행의 동작을 변경하고 싶은 경우나, 새로운 정시 작업을 추가하고 싶은 경우에 참조할 것.

## 하트비트란

하트비트는 Digital Anima가 정기적으로 자동 시작하여 상황을 확인·계획하는 구조.
사람이 정기적으로 수신함을 확인하고 진행 중인 일을 재검토하는 것과 같은 행동을 자동화한 것.

### 중요: 하트비트는 "확인과 계획"만

Heartbeat는 의미 있는 변화를 확인하고 필요한 다음 대응을 판단한다. 의례적인 회고 보고는 불필요.

- MUST: Heartbeat는 상황 확인과 판단에 집중할 것
- MUST NOT: 하트비트 내에서 장시간 실행 작업(코딩, 대량의 도구 호출 등)을 하지 않을 것
- MUST: 실행이 필요한 작업을 발견하면 부하가 있다면 `delegate_task`으로 위임하거나, `submit_tasks`로 작업을 투입할 것

투입된 작업은 **TaskExec 경로**가 정기 Heartbeat와 독립적으로 실행한다.
TaskExec는 정본 TaskStore에서 실행 가능한 영속 작업을 획득한다. 기상 알림과 복구는 호스트가 관리하며, 구 LLM JSON 파일은 모니터링하지 않는다.

### 하트비트와 대화의 병렬 동작

하트비트와 인간과의 대화는 **별도 락**으로 관리되므로 동시에 동작할 수 있다.
하트비트 실행 중에도 인간의 메시지에는 즉시 응답 가능.

### submit_tasks를 통한 작업 투입

하트비트에서 실행해야 할 작업을 발견한 경우, `submit_tasks` 도구로 작업을 투입한다:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "APIテスト実施",
   "description": "Slack API接続テストを実施し、全エンドポイントの結果をレポートにまとめる。完了後 aoi に報告する。"}
])
```

`submit_tasks`는 검증 후, 작업과 완전한 실행 입력을 하나의 정본 TaskStore에 일괄 저장한다.
실행 권한 획득과 시도 이력은 호스트가 관리한다. `in_progress`는 열람용이며, 에이전트는 `update_task`로 `done` / `pending` / `cancelled`을 선언한다. 중단된 pending 작업은 동일한 ID에 `resume: true`을 지정하여 명시적으로 재개하고, 다른 작업으로 대체하지 않는다.

**장시간 CLI 도구**(`animaworks-tool submit …`)는 `task_type="command"`으로 TaskStore에 등록되며, PendingTaskExecutor가 시도를 가져와 백그라운드로 실행한다. 완료 결과는 `state/background_tasks/{task_id}.json`과 알림으로 확인한다. 자세한 내용은 `operations/background-tasks.md`를 참조한다.

**주의**: 저장 위치를 직접 편집하지 않는다. 기존 `state/task_queue.jsonl`과 `state/pending/`는 마이그레이션·내보내기용 증적으로 보존하며, 운영 중 투입 대상으로 사용하지 않는다.

단일 작업이라도 `submit_tasks`(tasks 배열 1건)를 사용한다.
여러 독립 작업은 `parallel: true`으로 병렬 실행하고, 의존 관계가 있는 경우 `depends_on`을 지정한다.
자세한 내용은 task-management를 참조한다.

### 하트비트 트리거 유형

하트비트에는 두 가지 유형의 트리거가 있다:

| 트리거 | 설명 |
|---------|------|
| 정기 하트비트 | `config.json`의 `heartbeat.interval_minutes`에 따라 APScheduler가 주기적으로 시작 |
| 메시지 트리거 | Inbox에 읽지 않은 메시지가 도착했을 때 즉시 시작(Inbox 경로로 처리) |

메시지 트리거는 Inbox의 JSON 파일 변경 알림으로 시작한다. 알림을 놓칠 경우에 대비해 45초마다 읽지 않은 메시지를 다시 확인한다.
동시에 실행되는 Inbox 처리는 1개뿐이며, 실행 중 도착한 메시지는 다음 1회에 모아서 처리한다. Provider 오류 시 `rate_guard`의 복구 시간을 기다리고, 읽지 않은 메시지를 남긴다.
수신 intent에 의한 시작 필터나 메시지 기반 cooldown / cascade 억제는 없다. 이해·감사만 있는 메시지에 답장하지 않는 행동 규칙은 Inbox 프롬프트에서 지시한다.

## HEARTBEAT.md 설정

`HEARTBEAT.md`은 각 Anima의 설정 파일로, 활동 시간과 확인 항목을 정의한다.
하트비트 실행 간격은 `config.json`의 `heartbeat.interval_minutes`에서 설정할 수 있다(1~1440분, 기본값 30). `HEARTBEAT.md`에서는 변경할 수 없다.
각 Anima에는 이름을 기준으로 0~9분의 오프셋이 부여되어, 동시에 시작되는 것을 분산한다.
파일 경로: `~/.animaworks/animas/{name}/HEARTBEAT.md`

상급자가 하위 Anima의 `HEARTBEAT.md`를 편집할 때는 파일을 직접 조작하는 대신 `read_memory_file` / `write_memory_file`을 사용하고, `../{anima_name}/HEARTBEAT.md`과 같은 상대 경로로 지정한다.

### 포맷

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
- `config.json`의 `heartbeat.interval_minutes`에서 설정한다(1~1440분, 기본값 30). `HEARTBEAT.md`에서는 변경할 수 없다.

**활동 시간**（SHOULD）:
- `HH:MM - HH:MM` 형식으로 작성한다(예: `9:00 - 22:00`)
- 이 시간 외에는 하트비트가 시작되지 않는다
- 설정하지 않은 경우 기본값: 24시간(하루 종일)
- 시간대는 `config.json`의 `system.timezone`에서 설정할 수 있다. 설정하지 않은 경우 시스템 시간대를 자동으로 감지한다

**체크리스트**（MUST）:
- 하트비트 시작 시 에이전트가 확인하는 항목
- 글머리 기호 목록(`- `로 시작)으로 작성한다
- 체크리스트 내용은 에이전트의 프롬프트에 그대로 전달된다
- 사용자 지정 가능: Anima의 역할에 맞게 항목을 추가하거나 변경해도 된다

### 체크리스트의 커스터마이즈 예

기본(전 Anima 공통):
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

개발 담당의 예:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 監視対象のGitHubリポジトリに新しいIssueやPRがないか
- CI/CDの失敗アラートがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

커뮤니케이션 담당의 예:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- Slackの未読メンションがないか
- 返信待ちのメールがないか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

### 실행 모델 (비용 최적화)

Heartbeat / Inbox / Cron은 `background_model`가 설정되어 있는 경우, 메인 모델 대신 해당 모델로 실행된다.
Chat(인간과의 대화)과 TaskExec(실작업)은 메인 모델을 유지한다.

설정 방법: `animaworks anima set-background-model {名前} claude-sonnet-5-5`
자세한 내용은 `reference/operations/model-guide.md`의 "백그라운드 모델" 섹션을 참조.

### 하트비트 내부 동작

- **크래시 복구**: 이전 하트비트가 실패한 경우, `state/recovery_note.md`에 오류 정보가 저장된다. 다음 시작 시 프롬프트에 주입되며, 복구 후 파일은 삭제된다.
- **회고 기록**: 하트비트 출력에 `[REFLECTION]...[/REFLECTION]` 블록이 있으면 activity_log에 `heartbeat_reflection`로 기록되며, 이후 하트비트 컨텍스트에 포함된다.
- **부하 확인**: 부하를 가진 Anima에는 하트비트·Cron 프롬프트에 부하의 상태 확인 지침이 자동 주입된다.
- **세션 시간 제한**(`config.json`의 `heartbeat`): `soft_timeout_seconds`(기본 300초) 경과 시 랩업용 리마인더를 주입하고, `hard_timeout_seconds`(기본 600초)에서 세션을 강제 종료한다.
- **유휴 시 자동 컴팩트**: `heartbeat.idle_compaction_minutes`(기본 10분) — 스트림 종료 후 이 시간이 지나면 유휴 자동 컴팩션이 실행된다(실행 엔진 측 설정).
- **Board 게시**: 동일 run에서는 같은 채널에 1회까지. run을 넘는 게시 간격 제한은 없다.

### 정기 하트비트의 스케줄 방식

실효 간격(Activity Level 적용 후·하한 5분 반올림)이 **60분 이하이고 60을 나눌 수 있을 때**는 APScheduler의 `CronTrigger`으로 분 슬롯에 분산 등록된다(이름 기반 0~9분 오프셋과 조합).

그 외(예: 실효 61분, 또는 43분처럼 60을 나눌 수 없는 간격)는 **1분마다의 폴링**(`_heartbeat_check`)으로 "이전부터의 경과"에 따라 발화한다. 구 `IntervalTrigger` 유래의 문제를 회피하기 위한 구현.

### 백그라운드 도구와 DM 로그(core/tasks/background.py）

`core/tasks/background.py`은 하트비트/Cron의 스케줄 자체가 아니라, **장시간 도구 호출의 백그라운드 실행**(상태의 JSON 영속화 포함)과 **레거시 공유 DM 로그(`shared/dm_logs/`)의 로테이션**을 담당한다. 운영의 상세·CLI 경로는 `operations/background-tasks.md`도 참조.

#### BackgroundTaskManager

- **저장 위치**: `state/background_tasks/{task_id}.json`. `task_id`은 UUID의 앞 12자리(16진수)입니다. 각 파일에는 `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, `error`이 기록됩니다.
- **상태 (`TaskStatus`)**: `pending` / `running` / `completed` / `failed`. `submit` / `submit_async`에서는 작업이 제출된 직후부터 `running` 상태로 JSON이 기록되고, 완료 또는 예외 발생 시 `completed` / `failed`으로 업데이트됩니다.
- **실행 API**: `submit(tool_name, tool_args, execute_fn)`은 동기 callable을 `asyncio`의 `run_in_executor`에서 스레드 풀로 실행합니다. `submit_async`는 비동기 callable을 그대로 await합니다. `get_task`(메모리 우선, 없으면 디스크), `list_tasks`(디스크의 JSON도 병합, 생성 시각이 최신인 순), `active_count`(메모리의 `running` 개수)을 제공합니다.
- **완료 콜백**: `on_complete`에 전달한 비동기 함수는 작업을 저장한 뒤 호출됩니다. 콜백에서 예외가 발생해도 작업 결과는 유지되며 로그에만 기록됩니다(일반적으로 `state/background_notifications/`에 기록하고, **다음 하트비트**에서 읽고 삭제해 대화 컨텍스트에 포함하는 경로와 함께 사용).
- **후보 판정 `is_eligible(tool_name)`**: 맵에 키가 있으면 백그라운드 대상입니다. 키는 다음 두 가지 형식을 모두 지원합니다. (1) **스키마 이름**(예: `generate_3d_model`) — Mode A의 외부 도구 디스패치 등. (2) **`ツール名:サブコマンド`**(예: `image_gen:pipeline`) — 각 도구 모듈의 `EXECUTION_PROFILE`에서 `background_eligible: true` 항목을 `get_eligible_tools_from_profiles()`가 이 형식으로 등록합니다(Mode S의 `submit` 경로 등).
- **후보 도구 맵 구성 `from_profiles()`**: 다음 3개 계층을 dict의 `update`으로 병합하고, **나중 항목이 우선**하도록 덮어씁니다. (1) `_DEFAULT_ELIGIBLE_TOOLS`(코드 기본값) (2) 인수 `profiles`(`EXECUTION_PROFILE` 집계) (3) 인수 `config_eligible`(일반적으로 `config.json`의 `background_task.eligible_tools`에서 `threshold_s`을 펼친 `名前 → 秒`). 값은 프로필 연동에 사용할 예상 시간(정수)입니다.
- **코드 기본값 `_DEFAULT_ELIGIBLE_TOOLS`(초)**: `generate_character_assets` 30, `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` 각 30, `generate_3d_model` / `generate_rigged_model` / `generate_animations` 각 30, `local_llm` 60, `run_command` 60.
- **정리 `cleanup_old_tasks(max_age_hours=24)`**: `status`이 `completed` / `failed`에서 `completed_at`가 **인수로 지정한 시간(기본값 24시간)보다 오래된** JSON을 삭제합니다. 또한 `running` 상태로 `created_at`부터 **48시간 초과** 경과한 파일은 크래시 고아 파일로 삭제합니다. 반환값은 삭제한 항목 수입니다.
- **보존 시간**: `cleanup_old_tasks`의 완료 작업 보존 시간은 인수 `max_age_hours`(기본값 24시간)으로 지정합니다. 이에 대응하는 `config.json` 설정 키는 없습니다.

#### rotate_dm_logs（시스템 Cron）

- **실행 시점**: 라이프사이클(`core/lifecycle/system_crons.py` 등)의 시스템 Cron에서 **매일 04:30**(서버 설정 타임존). `server/supervisor/_mgr_scheduler.py` 쪽에서도 동일 ID의 작업이 등록되는 구성.
- **대상**: `shared/dm_logs/*.jsonl`(파일명에 `.archive.`을 포함하는 것은 건너뜀).
- **동작**: `core.time_utils`의 로컬 현재 시간 기준으로, 각 행 JSON의 `ts`(ISO 형식)을 파싱하여, **기본 7일**보다 오래된 항목을 `{stem}.{YYYYMMDD}.archive.jsonl`에 추가 아카이브한 후, 현재 파일에서 제거. `ts`의 해석에 실패한 행은 **현재 파일에 유지**(데이터 손실 방지).
- **기타 서버 정기 작업**: Anima 단위의 `cron.md`와 별도로, 라이프사이클이 메모리 유지보수·RAG 등의 시스템 Cron을 등록(예: 일일 통합 02:00, 일일 인덱스 04:00). 시간은 설정 타임존 기준. DM 로테이션은 위 04:30.

### 하트비트 설정 핫 리로드

파일 시스템에서 HEARTBEAT.md을 업데이트하면 다음 하트비트 실행 시 `_check_schedule_freshness()`이 변경 사항을 감지하고, SchedulerManager가 일정을 자동으로 다시 불러온다.
서버를 재시작할 필요가 없다(재시작을 건너뛸 수 있음). APScheduler 작업이 다시 등록된다.

## Anima별 하트비트 간격 설정

### status.json에서의 설정

root 관리자는 root CLI/API에서 Anima 개별의 `heartbeat_interval_minutes`을 설정할 수 있습니다. 값은 root 소유 `status.json`에 저장되며, Anima 프로세스에서는 편집하지 않습니다.

```json
{
  "heartbeat_interval_minutes": 60
}
```

- 설정 가능 범위: 1〜1440분 (1일)
- 미설정 시: `config.json`의 `heartbeat.interval_minutes` (기본 30분)로 폴백
- `status.json`는 root 소유입니다. 관리자는 서버 종료 중에 `animaworks config set animas.<name>.heartbeat_interval_minutes <minutes>`로 개별 오버라이드를 설정할 수 있습니다. 서버 시작 중에는 CLI가 root API를 통해 반영합니다. Anima 프로세스는 이 파일을 편집하지 않습니다.

### 권장 가이드라인

| 상황 | 권장 간격 | 이유 |
|------|----------|------|
| 활발한 개발 프로젝트 중 | 15~30분 | 빈번한 상황 파악 필요 |
| 일반 업무 | 30~60분 | 기본값. 균형 잡힌 빈도 |
| 저부하·대기 상태 | 60~120분 | 비용 절약. 작업이 없으면 길게 |
| 장기 휴면·비활성 | 120~1440분 | 최소한의 순회로 상황 파악 |

### Activity Level과의 관계

글로벌 Activity Level(10%~400%)이 설정된 경우, 실효 간격은 다음 식으로 계산된다:

```
実効間隔 = ベース間隔 / (Activity Level / 100)
```

예: 베이스 30분, Activity Level 50% → 실효 60분
예: 베이스 30분, Activity Level 200% → 실효 15분

- 실효 간격의 하한은 5분(아무리 부스트해도 5분 미만이 되지 않음)

### Activity Schedule(시간대별 자동 전환 / 나이트 모드)

Activity Level을 시간대에 따라 자동으로 전환하는 메커니즘.
야간이나 휴일에 비용을 줄이고 싶을 때, 또는 업무 시간에만 활발히 동작시키고 싶을 때 사용한다.

#### 메커니즘

- `config.json`의 `activity_schedule`에 시간대 엔트리를 설정한다
- 1분마다 현재 시간을 확인하고, 해당 시간대의 레벨로 Activity Level을 자동 변경
- Activity Level이 바뀌면 모든 Anima의 하트비트가 즉시 리스케줄된다

#### 설정 포맷

각 엔트리는 `start`(시작 시간), `end`(종료 시간), `level`(Activity Level %)의 3개 필드:

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- 시간은 `HH:MM` 형식(24시간 표기)
- **날짜 넘김 대응**: `"22:00"` ~ `"06:00"`처럼 start > end 지정 가능(심야 시간대 커버)
- `level`은 10~400 범위
- 최대 24개 엔트리까지
- 빈 배열 `[]`로 스케줄 모드 비활성화(고정 Activity Level로 복귀)

#### 설정 방법

- **Settings UI**: 나이트 모드 체크박스 + 시간대·레벨 설정
- **API**: `PUT /api/settings/activity-schedule`에 위 JSON을 전송
- **root 설정 API**: Settings UI를 사용하거나, 위 JSON을 `PUT /api/settings/activity-schedule`에 전송합니다. Anima 프로세스에서 `config.json`를 직접 편집하지 않습니다

#### 주의점

- Activity Level을 수동으로 변경하면 현재 시간대에 해당하는 스케줄 엔트리도 연동되어 업데이트된다
- 스케줄은 서버 시작 시에도 즉시 적용된다(시작 시점의 시간으로 해당 레벨 설정)
- 어떤 시간대에도 해당하지 않으면 마지막으로 설정된 Activity Level이 유지된다

## Cron 작업이란

Cron은 "정해진 시간에 자동 실행되는 작업". 하트비트가 "정기 순회"인 반면, Cron은 "정시 업무".

예:
- 매일 아침 9:00에 업무 계획을 세운다
- 매주 금요일 17:00에 주간 회고를 한다
- 매일 2:00에 백업 스크립트를 실행한다

## cron.md의 설정

Cron 작업은 `cron.md`에 Markdown + YAML 형식으로 정의한다.
파일 경로: `~/.animaworks/animas/{name}/cron.md`

상급자가 산하 Anima의 `cron.md`를 편집하는 경우, 직접 파일 조작이 아니라 `read_memory_file` / `write_memory_file`를 사용하고 `../{anima_name}/cron.md`와 같은 상대 경로로 지정한다.

### 기본 포맷

각 작업은 `## タスク名`의 제목으로 시작하고, 본문의 첫머리에 `schedule:` 디렉티브로 표준 5필드 cron식을 기재한다.

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

구형식(`## タスク名（毎日 9:00 JST）`처럼 괄호 안에 스케줄을 쓰는 형식)은 서버 시작 시 또는 `animaworks migrate`에서 신형식으로 자동 변환된다.

### CronTask의 스키마

각 작업은 내부적으로 다음 `CronTask` 모델로 파싱된다:

| 필드 | 타입 | 기본값 | 설명 |
|-----------|------|-----------|------|
| `name` | str | (필수) | 작업 이름. `##` 제목에서 추출 |
| `schedule` | str | (필수) | 표준 5필드 cron식. `schedule:` 디렉티브에서 추출 |
| `type` | str | `"llm"` | 작업 유형: `"llm"` 또는 `"command"` |
| `description` | str | `""` | LLM 유형의 지시문(type: llm에서 사용) |
| `command` | str \| None | `None` | Command 유형의 bash 명령어 |
| `tool` | str \| None | `None` | Command 유형의 내부 도구 이름 |
| `args` | dict \| None | `None` | tool의 인자(YAML 형식) |
| `skip_pattern` | str \| None | `None` | Command 유형: stdout이 이 정규식에 매치하면 follow-up LLM을 건너뜀 |
| `trigger_heartbeat` | bool | `True` | Command 유형: `False`이면 명령 출력 후 follow-up cron LLM을 건너뜀 |

## LLM 유형 Cron 작업

`type: llm`은 에이전트(LLM)가 판단·사고를 수반하여 실행하는 작업.
description에 쓰인 지침이 프롬프트로 에이전트에게 전달된다.

### 특징

- 에이전트가 도구를 사용하고, 기억을 검색하고, 판단을 내린다
- 결과는 비정형(작업마다 다른 출력)
- 실행에는 모델의 API 호출이 필요(비용 발생)

### 기재 예

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
昨日の episodes/ を読み返し、今日のタスクを計画する。
優先順位は理念と目標に照らして判断する。
結果は state/current_state.md に書き出す。
正本タスク一覧（`list_tasks`） の未着手タスクも確認し、必要なら優先度を見直す。
```

description(`type:` 행 이후의 본문)에는 다음을 포함해야 한다(SHOULD):
- 무엇을 확인할지(입력)
- 어떻게 판단할지(기준)
- 무엇을 출력할지(산출물)

## Command 유형 Cron 작업

`type: command`은 에이전트의 판단을 거치지 않고 정해진 명령이나 도구를 실행하는 작업.
결정적인 처리(백업, 알림 전송 등)에 적합하다.

### bash 명령 유형

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

`command:`에 bash 명령을 한 줄로 기재한다.
명령은 셸을 통해 실행된다.

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

`tool:`에 내부 도구 이름, `args:`에 YAML 형식으로 인자를 기재한다.
args는 YAML의 들여쓰기 블록으로 파싱된다(2칸 들여쓰기).

### Command 유형의 follow-up 제어

Command 유형 작업은 명령이 정상 종료되고 stdout이 있는 경우, 그 출력을 LLM에 전달하여 follow-up 분석을 수행한다(하트비트 상당의 컨텍스트로 실행).

- **`trigger_heartbeat: false`** — follow-up LLM을 건너뜀(출력 분석이 불필요한 경우)
- **`skip_pattern: <正規表現>`** — stdout이 이 정규식에 매치하면 follow-up을 건너뜀

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

### LLM 유형과 Command 유형의 용도 구분

| 관점 | LLM 유형 | Command 유형 |
|------|--------|-----------|
| 판단 필요 여부 | 예 | 아니요 |
| API 비용 | 있음 | 없음 |
| 출력의 예측 가능성 | 비정형 | 결정적 |
| 적합한 작업 | 계획 수립, 회고, 문서 작성 | 백업, 알림 전송, 데이터 획득 |
| 오류 시 대응 | 에이전트가 자율 대처 | 로그에 기록만 |

고민될 때의 가이드라인:
- "매번 같은 일만 하는 경우" → Command 유형(SHOULD)
- "상황에 따라 판단이 달라지는 경우" → LLM 유형(SHOULD)
- "명령 실행 + 결과 해석" → LLM 유형으로 description에 명령 실행을 지시

## 스케줄 표기법

cron.md의 `schedule:` 디렉티브에는 **표준 5필드 cron식**을 기재한다.

### 표준 cron식(필수)

```
分 時 日 月 曜日
```

예:
- `0 9 * * *` — 매일 9:00
- `0 9 * * 1-5` — 평일 9:00
- `*/30 9-17 * * *` — 9:00~17:00의 30분마다
- `0 2 1 * *` — 매월 1일 2:00
- `0 17 * * 5` — 매주 금요일 17:00

타임존은 `config.json`의 `system.timezone`에서 설정 가능. 미설정 시 시스템 타임존을 자동 감지.

### 일본어 스케줄에서의 전환

구형식(`## タスク名（毎日 9:00 JST）`)으로 작성된 cron.md은 서버 시작 시 또는 `animaworks migrate`에서 표준 cron식으로 자동 변환된다. 변환 대응표:

| 일본어 표기법 | cron식 예 |
|-----------|----------|
| `毎日 HH:MM` | `0 9 * * *` |
| `平日 HH:MM` | `0 9 * * 1-5` |
| `毎週{曜日} HH:MM` | `0 17 * * 5`(금요일) |
| `毎月N日 HH:MM` | `0 9 1 * *` |
| `X分毎` | `*/5 * * * *` |
| `X時間毎` | `0 */2 * * *` |

`隔週`, `毎月最終日`, `第N曜日`은 자동 변환 불가. 수동으로 cron식을 기재한다.

## cron_logs 확인 방법

Cron 작업의 실행 결과는 서버 로그에 기록된다.
WebSocket을 통해 `anima.cron` 이벤트로 브로드캐스트도 된다.

로그 확인 방법:
- 서버 로그: `animaworks.lifecycle` 로거의 INFO 레벨
- Web UI: 대시보드의 활동 피드에 표시
- episodes/: LLM 유형 작업의 경우, 에이전트 자신이 episodes/에 로그를 작성한다(SHOULD)

LLM 유형 작업의 결과는 `CycleResult`로 기록되며 다음 정보를 포함한다:
- `trigger`: `"cron"`
- `action`: 에이전트의 행동 요약
- `summary`: 결과의 요약 텍스트
- `duration_ms`: 실행 시간(밀리초)
- `context_usage_ratio`: 컨텍스트 사용률

## 자주 쓰는 Cron 설정 예

### 기본 세트(전 Anima 권장)

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

### 외부 연계 작업

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

실행하고 싶지 않은 작업은 HTML 주석으로 감싼다:

```markdown
<!--
## 一時停止中のタスク
schedule: 0 15 * * *
type: llm
このタスクは一時的に停止中。
-->
```

주석 내의 `## ` 제목은 파서에 무시된다.

## Cron 설정 핫 리로드

cron.md을 업데이트하면 HEARTBEAT.md과 마찬가지로 일정이 자동으로 다시 불러와진다.
Anima 자체가 cron.md을 수정한 경우에도 즉시 반영된다(self-modify 패턴).
상급자는 하위 Anima의 `cron.md` / `HEARTBEAT.md`을 write memory 도구로 편집할 수 있다. 허가된 상급자의 `AGENTS.md` 변경 요청도 root에 전달되어 인가된다. `status.json`을 변경하려면 해당 상급자용 도구 또는 root CLI/API을 사용한다. root 소유 파일을 Read / Write / Edit / apply_patch / `Path.write_text` / 셸 리디렉션 등으로 직접 작성하지 않는다.

다시 불러올 때의 동작:
1. 해당 Anima의 기존 cron 작업을 모두 삭제
2. 업데이트된 cron.md을 파싱하고 새 작업을 등록
3. 로그에 `Schedule reloaded for '{name}'`가 출력된다

직접 cron.md을 업데이트할 때의 주의 사항:
- 제목(`## タスク名`) 바로 뒤에 `schedule:` 디렉티브를 둔다(MUST)
- 스케줄은 표준 5필드 cron 식으로 작성한다(MUST)
- type 행은 schedule 바로 뒤에 둔다(SHOULD)
