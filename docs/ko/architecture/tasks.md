<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/tasks.md -->
<!-- i18n: source-sha256=2eafe0a8d1b890d7718379c0b58fba494ea12cf38ee02587bbcd8c19cd559a58 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: b304b7dc

# 작업 관리

작업의 원본은 `shared/taskboard.sqlite3`에 두는 TaskStore이다. `core/tasks/board/`은 canonical task, alias, attempt, lease를 읽고 쓰며, CLI와 Web UI에 동일한 작업 정보를 제공한다. 별도의 표시용 카드 계층은 두지 않는다.

## 상태와 실행 기록

일반적인 상태는 `pending`, `in_progress`, `delegated`, `done`, `cancelled`이다. 위임된 쪽의 실제 작업을 TaskStore에 저장하고, 위임 원본에는 alias row를 만들므로 같은 작업을 이중으로 세지 않고 추적할 수 있다. Web Task Board는 TODO, RUNNING, WAITING, DONE 열로 투영하며, 사람이 등록한 작업을 각 열 안에서 먼저 표시한다.

실행을 시작하면 task attempt가 생성되고, 고유 token과 일련번호로 실행 주체를 식별한다. 오래된 attempt의 업데이트는 거부되며, 결과나 종료 이유는 attempt 이력에 남는다. task lease는 actor가 일정 기간 작업을 조작할 권리를 확보하는 메커니즘으로, 완료·취소 시 해제된다. 미완료 실행을 재개시키는 알림은 wake-up outbox에 영속화되고, 처리 후 확인됨으로 표시된다.

TaskStore의 schema와 구현은 `core/tasks/board/tasks.py`, 목록의 투영은 `view.py`에 있다. `state/task_queue.jsonl`은 기존 데이터의 명시적 수집과 관련된 파일이며, 런타임의 원본은 아니다.

## CLI, 위임, background task

`animaworks task board`은 보드 목록, `list`과 `show`는 작업 정보, `add`은 신규 등록에 사용한다. `claim`와 `release`는 lease를 획득·해제하고, `done`, `cancel`, `note`은 결과나 이유 기록에 사용한다. `update`는 상태 변경, `resume`은 저장된 입력으로 재투입을 수행한다. 각 인자와 다른 CLI는 [CLI 참조](../reference/cli.md)를 참조한다.

Anima에서 다른 Anima로 작업을 넘길 경우 `delegate_task`을 사용한다. 위임 대상의 실행 결과는 alias에서 추적할 수 있다. 시간이 걸리는 `animaworks-tool` 작업은 `submit`으로 별도 background task로 시작할 수 있으며, 그 상태·결과는 `state/background_tasks/`에 저장된다. 완료 시 요청 원본에 알림이 간다. 인자 목록은 [도구 CLI 참조](../reference/tool-cli.md)를 참조한다.

외부 작업 collector는 GitHub, Slack, Chatwork, Gmail 등의 연동처에서 작업 후보를 수집하고, source별 상태와 함께 snapshot으로 제공한다. 이는 Anima의 실행 TaskStore와는 별개의 읽기용 데이터이다.
