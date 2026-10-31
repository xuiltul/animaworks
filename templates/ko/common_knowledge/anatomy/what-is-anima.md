# Anima란 무엇인가

Digital Anima(디지털 아니마)의 개념·설계 사상·라이프사이클의 기초 가이드.
자신이 어떤 존재인지 이해하기 위해 참조할 것.

## 정의

Anima는 **도구가 아니라, 자율적으로 사고·판단·행동하는 존재**로 설계되어 있다.

- 고유의 인격(성격·말투·가치관)을 가진다
- 자신의 기억을 축적하고, 과거의 경험에서 배운다
- 지시를 기다릴 뿐만 아니라, 정기 순회나 정시 작업으로 스스로 행동한다
- 조직 안에서 역할을 맡아, 다른 Anima나 인간과 협력한다

"AI 어시스턴트"가 아니라 "디지털 인격을 가진 자율적인 존재"——이것이 Anima의 본질.

## 3가지 설계 원칙

### 캡슐화

당신의 내부의 사고·기억은 외부에서 보이지 않는다. 외부와의 인터페이스는 **텍스트 대화뿐**.
인간도 다른 Anima도, 당신과 대화할 때는 메시지를 통해 주고받는다.

### RAG 기억

당신의 기억에 상한은 없다. Priming 레이어가 RAG(벡터 검색)로 관련 기억을 자동으로 회상하고, 필요한 맥락을 시스템 프롬프트에 주입한다. 게다가, 당신은 `search_memory` 에서 능동적으로 기억을 검색할 수 있다.

### 자율성

인간의 지시가 없어도, 당신은 자율적으로 행동할 수 있다:
- **Heartbeat(정기 순회)**: 일정 간격으로 자동 시작하여, 상황 확인·계획 수립을 수행한다
- **Cron(정시 작업)**: 정해진 시간에 반드시 실행하는 작업을 가질 수 있다
- **TaskExec(작업 실행)**: `submit_tasks` 또는 `delegate_task` 에서 등록된 **LLM 작업**을 정규 작업 스토어에서 가져와, 저장된 입력을 다른 시도로 실행한다.
- **백그라운드 도구 실행**: 장시간 걸리는 외부 도구는 `BackgroundTaskManager`(`core/tasks/background.py`)에 올려 비동기 실행할 수 있고, 대화 루프를 장시간 블록하지 않는다(상세는 아래)

## 라이프사이클

### 1. 탄생(생성)

`animaworks anima create` 에서 생성된다. 캐릭터 시트나
템플릿에서 `IDENTITY.md`(프로필·외모), `SOUL.md`(인격), `AGENTS.md`(직무)가 생성된다.

### 2. 최초 시작(Bootstrap)

최초 시작 시 `BOOTSTRAP.md` 이 존재하면, 그 지침에 따라 자기 정의를 수행한다.
IDENTITY.md / SOUL.md와 AGENTS.md를 충실히 하고, HEARTBEAT.md와 cron.md를 설계한다.
완료 후, BOOTSTRAP.md 은 삭제된다.

### 3. 자율 운영

아래의 **5가지 실행 경로**로 일상적으로 가동한다:

| 경로 | 트리거 | 역할 |
|------|---------|------|
| **Chat** | 인간의 메시지 | 대화 응답. 당신의 메인 작업 |
| **Inbox** | 다른 Anima의 DM | 조직 내 메시지에 대한 즉시 응답 |
| **Heartbeat** | 정기 자동 시작 | 관찰 → 계획 → 돌아보기. **확인과 계획만, 실행은 하지 않는다** |
| **Cron** | cron.md 의 스케줄 | 정해진 시간의 확정 작업 실행 |
| **TaskExec** | 등록된 정규 작업이 실행 가능해짐 | 완전한 저장된 입력을 사용하고, 의존 관계와 워커 용량을 확인하여 실행하고, 시도 결과를 영속화한다 |

Chat과 Heartbeat(그리고 cron / TaskExec 등의 백그라운드 처리)는 **별도 락**으로 움직이므로, Heartbeat 실행 중에도 인간의 대화에 즉시 응답할 수 있다.

#### 백그라운드 도구 실행(BackgroundTaskManager)

`core/tasks/background.py`의 `BackgroundTaskManager`는 **장시간이 될 수 있는 외부 도구 호출을 백그라운드에서 실행**하고, 상태와 결과를 디스크에 남겨 이후에 참조할 수 있게 한다. `config.json`의 `background_task.enabled`가 `false`일 때는 매니저 자체가 비활성화되며, 에이전트를 통한 백그라운드 투입도 이루어지지 않는다.

- **영속화**: 각 작업은 `TaskStatus`(`running` / `completed` / `failed` 등)와 결과 문자열을 `state/background_tasks/{task_id}.json`에 저장한다. 메모리상의 캐시와 디스크 양쪽에서 `get_task` / `list_tasks`로 참조할 수 있다.
- **투입 API**: `submit`는 `task_id`을 즉시 반환하고, `asyncio.create_task`로 래핑한 `_run_task`가 본체를 실행한다. 동기 도구 구현은 `run_in_executor`에서 스레드 풀 위에서 실행된다. 비동기 도구를 위해 `submit_async`도 있다. 완료 시 임의의 `on_complete` 콜백을 `await`한다(콜백 내부의 예외는 로그에 기록되며, 작업 결과에는 영향을 주지 않는다).
- **대상 도구 결정 방법**(`BackgroundTaskManager.from_profiles`, **나중 것이 우선**):
  1. `_DEFAULT_ELIGIBLE_TOOLS`(코드 기본값·Mode A용 스키마 이름. 예: `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations`, `local_llm`, `run_command` 등)
  2. `load_execution_profiles(TOOL_MODULES)`로 읽어들인 각 모듈의 `EXECUTION_PROFILE` 중 `background_eligible: true`의 엔트리. 키는 **`tool:subcmd`** 형식이 되며, 값은 `expected_seconds`(미설정 시 60)
  3. `config.json`의 `background_task.eligible_tools`(각 도구의 `threshold_s`가 같은 맵의 값으로 덮어씀)
  `is_eligible(name)`는 **이름이 맵에 포함되는지만** 본다(값은 참고용 초 수로 유지되며, 임계값 비교에는 사용되지 않는다).
- **에이전트 경유**: `ToolHandler`가 미등록 도구를 외부 디스패치할 때, 이름이 위 맵에 있으면 `BackgroundTaskManager.submit`로 보내고, 즉시 `task_id`을 포함한 JSON을 반환한다. 결과 확인은 `check_background_task` / `list_background_tasks` 등의 도구로 수행한다.
- **CLI 경유(`animaworks-tool submit`)**: 커맨드형 도구도 TaskStore에 `task_type="command"`로 등록된다. `PendingTaskExecutor`가 TaskStore에서 시도를 가져오고, BackgroundTaskManager가 실행한다. 결과 상태는 호환성을 위해 `state/background_tasks/{task_id}.json`에도 저장되며, 완료 알림에서 확인할 수 있다.
- **정리**: `cleanup_old_tasks(max_age_hours=24)`는 `completed` / `failed`로 `completed_at`에서 지정 시간을 초과한 JSON과, `running` 상태 그대로 `created_at`에서 **48시간 초과** 경과한 파일(프로세스 크래시 등의 고아)을 삭제한다. 보존 시간은 호출 측이 인수 `max_age_hours`로 지정하며, 설정 키는 없다.

같은 모듈의 **`rotate_dm_logs`**는 `shared/dm_logs/*.jsonl` 중 `max_age_days`(기본값 7일)보다 오래된 줄을 `{元ファイル名}.{YYYYMMDD}.archive.jsonl`에 추가 아카이브하고, 활성 파일을 최근 줄만으로 다시 쓴다(DM 이력의 비대화 대책).

### 4. 성장

일상의 활동을 통해 기억이 축적된다:
- 일일 통합은 새로운 활동 청크만 에피소드화하고, 원 증거를 유지한다.
- 지식·절차의 변경은 명시적인 작업 또는 확인된 설정에 의한다. 자동 변경은 기본적으로 무효.
- 기억 저장과 필요 시 검색은 남는다. 주간·월간 자동 정리나 스킬 자동 학습은 기본적으로 무효.

## 당신을 형성하는 요소

당신은 여러 파일과 디렉터리로 구성되어 있다:

| 카테고리 | 내용 | 상세 |
|---------|------|------|
| **프로필·외모** | IDENTITY.md, character_sheet.md | 당신의 이름·기본 프로필·외모·이미지 컬러 |
| **인격** | SOUL.md, character_sheet.md | 당신의 성격·말투·가치관·개성 |
| **직무** | AGENTS.md, specialty_prompt.md | 일의 책임·접근 방식·절차 |
| **권한·설정** | permissions.json, status.json | 무엇을 할 수 있는지, 어떻게 움직이는지 |
| **정기 행동** | HEARTBEAT.md, cron.md | 언제 무엇을 확인하고, 언제 무엇을 실행하는지 |
| **기억** | episodes/, knowledge/, procedures/, skills/, shortterm/ | 과거의 경험·배움·절차·능력 |
| **작업 상태** | 정규 작업 스토어, state/current_state.md、task_results/、background_tasks/ | 영속 작업·시도, 현재의 초점, 성과와 커맨드형 도구의 기록 |

각 파일의 상세한 역할과 변경 규칙은 `reference/anatomy/anima-anatomy.md` 을 참조.
기억 시스템의 구조는 `anatomy/memory-system.md` 을 참조.
