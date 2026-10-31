<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/lifecycle.md -->
<!-- i18n: source-sha256=d7c192cd9f539a0c0ddea724cc79608c33077c8e984c3fad725d8a2574a8a054 generated=2026-10-16 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: b304b7dc

# 시작 트리거와 라이프사이클

Anima는 server의 요청뿐만 아니라 메시지나 스케줄을 계기로 실행된다. root process가 상주하여 scheduler와 수신 처리를 관리하고, 실제 대화·자동 실행은 task runner의 전용 lane으로 분리한다.

## 트리거와 lane

| 트리거 | 실행되는 처리 |
|---|---|
| chat | 사용자와의 대화를 thread별로 처리한다. |
| inbox | 다른 Anima나 연계처에서 도착한 메시지를 전용 inbox lane에서 처리한다. |
| heartbeat | `HEARTBEAT.md`의 지침에 따른 정기적인 확인을 heartbeat task runner에서 수행한다. |
| cron | `cron.md`에 정의한 정기 처리를 cron task runner에서 실행한다. |
| task | TaskStore의 실행 가능한 작업을 background worker가 맡는다. |
| greet | 시작 후나 대화 시작 시의 인사를 전용 chat task runner에서 수행한다. |

채팅은 thread별 conversation lock으로 같은 대화의 경합을 방지한다. inbox와 scheduled background work는 별도의 제어 lock을 가진다. TaskExec는 전용 worker slot과 AgentCore를 사용하며, heartbeat/Cron의 실행 lock을 공유하지 않는다. 파일 `state/current_state.md`의 업데이트에는 process-safe한 state file lock이 사용된다.

## Heartbeat와 cron

heartbeat의 기본 간격은 전체 설정에서 30분이 기본값이며, `status.json`의 `heartbeat_interval_minutes`에서 Anima별로 덮어쓸 수 있다. 유효 범위는 1~1440분이다. 전체의 `activity_level`는 기본값 100%, 허용 범위 10~400%이며, 값이 높을수록 실제 간격이 짧아진다. `activity_schedule`을 사용하면 시간대별로 활동도를 전환할 수 있다. `HEARTBEAT.md`에서 활동 시간대를 설정한다.

`cron.md`는 제목 단위의 정의이며, `schedule`에 표준 5필드 cron 식을 적는다. `type`, 설명, 필요에 따라 command/tool과 인수를 설정하고, `core/runtime/schedule_parser.py`이 분석한다. 셋업이나 hot-reload 시에는 분석·등록하지 못한 스케줄만 `cron_health_*.md`로 알림한다. 등록된 cron의 실행 빈도나 실패를 정기적으로 모니터링하는 기능은 없다.

## 최초 시작과 1사이클

최초 시작에서는 `core/anima/bootstrap_state.py`이 `IDENTITY.md` / `SOUL.md`이나 `AGENTS.md` 등의 정의 상태를 확인하고, bootstrap의 시작·실패·완료를 기록한다. supervisor는 프로세스를 시작하고, IPC endpoint와 의존 서비스의 준비를 확인한 후 정상적인 처리를 시작한다.

1. trigger를 session type과 lane에 대응시키고, 필요한 lock 또는 worker slot을 획득한다.
2. `status.json`과 공통 설정을 해결하고, 기억, 권한, 현재 상태, 메시지 등에서 prompt를 준비한다.
3. 실행 모드에 맞는 engine adapter가 모델 호출과 tool 실행을 중개하고, 이벤트를 root에 반환한다.
4. 응답, 대화 기록, 작업 상태, 활동 기록을 확정하고, 후속 처리에 필요한 알림을 저장한다.
5. lock / worker slot을 해제하고, root의 scheduler와 inbox dispatcher가 다음 요청을 받을 수 있는 상태로 돌아간다.

## 설계 판단

- **cron은 각 Anima의 내부 시계이다.** 조직 전체의 스케줄러가 아니라, 사람이 자신의 일과를 가지는 것과 같이 각 Anima가 자신의 습관으로 가진다.
