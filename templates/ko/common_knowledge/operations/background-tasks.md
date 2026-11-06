# 백그라운드 작업 실행 가이드

## 개요

일부 외부 도구(이미지 생성, 3D 모델 생성, 로컬 LLM 추론, 음성 텍스트 변환 등)는
실행에 수 분에서 수십 분이 걸린다. 이를 직접 실행하면 실행 중 내내 잠금이 유지되어
메시지 수신이나 heartbeat가 종료된다.

`animaworks-tool submit`을 사용하면 작업을 백그라운드에서 실행하고,
자신은 즉시 다음 작업으로 넘어갈 수 있다.

`animaworks-tool submit`은 `task_type="command"`로 TaskStore에 등록되며,
Anima 자식 프로세스 내의 **PendingTaskExecutor**(`core/tasks/pending_executor.py`)가 시도를 가져온다.
**BackgroundTaskManager**(`core/tasks/background.py`)가 도구를 백그라운드로 실행하고,
시도는 TaskStore, 호환용 결과 JSON은 `state/background_tasks/{task_id}.json`에 저장한다.

## submit을 언제 사용할까

### 반드시 submit을 사용해야 하는 도구

도구 가이드(시스템 프롬프트)에 ⚠ 표시가 있는 하위 명령:

- `image_gen pipeline` / `fullbody` / `bustup` / `icon` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe audio`(하위 명령 이름은 `audio`)

각 도구의 `EXECUTION_PROFILE`에서 `background_eligible: true`인 것은 프로필을 통해
백그라운드 실행 후보로 등록된다(예: `chatwork sync` / `download` 등).
운영 방침은 계속해서 **⚠ 표시**를 우선한다.

### submit이 필요 없는 도구

실행 시간이 짧은(수십 초 미만 기준) 도구:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail`(일반적인 조작)
- `github`, `aws_collector`

### 판단 기준

- ⚠ 표시 있음 → 반드시 submit
- ⚠ 표시 없음 → 직접 실행(`animaworks-tool submit` 런타임 시, 프로필상 '단시간'인 경우 stderr에 경고가 나올 수 있음)

## 사용 방법

### 기본 문법

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### 실행 예

```bash
# 3Dモデル生成（Meshy API 等）
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# キャラクター画像一括生成（全ステップ）
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# ローカルLLM推論（Ollama）
animaworks-tool submit local_llm generate "要約してください: ..."

# 音声文字起こし（Whisper 等）— サブコマンドは audio
animaworks-tool submit transcribe audio "/path/to/audio.wav" --language ja
```

### 반환 값

submit은 즉시 JSON을 표준 출력으로 내보내고 종료한다(`task_id`은 12자리 hex):

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "バックグラウンドタスクを投入しました。完了時にinboxに通知されます。(task_id: a1b2c3d4e5f6)"
}
```

## 결과 수신

1. submit은 도구 이름·인자·Anima를 포함한 `task_type="command"`의 입력을 TaskStore에 등록한다. PendingTaskExecutor가 실행 시도를 가져와 대화 잠금 밖에서 BackgroundTaskManager에 실행을 요청한다.
2. TaskStore는 시도와 결과 참조를 기록한다. `state/background_tasks/{task_id}.json`은 기존 조회 API와의 호환을 위해 동일한 task_id로 `running` → `completed` / `failed`를 기록한다.
3. 완료 시 `_on_background_task_complete`가 **`state/background_notifications/{task_id}.md`**에 Markdown 알림을 작성한다.
4. 다음 **heartbeat**에서 `drain_background_notifications()`이 해당 `.md`을 읽고 삭제하여 컨텍스트에 주입한다.
5. Web UI의 WebSocket이나 `call_human` 계열의 인간 알림이 활성화되어 있으면, 같은 완료 시점에 거기에도 표시될 수 있다.

도구 **`list_background_tasks`**으로 메모리와 디스크를 병합한 목록을, **`check_background_task`**으로 `task_id` 지정 상태를 참조할 수 있다(모두 `BackgroundTaskManager`의 `list_tasks` / `get_task`에 해당).

## 실패 시 대응

- 알림에 '실패'라고 표시된 경우:
  1. 오류 내용을 확인한다
  2. 원인을 파악한다(API 키 미설정, 타임아웃, 인자 오류 등)
  3. 수정하여 다시 submit한다
  4. 해결할 수 없으면 상급자에게 보고한다

- 프로세스 비정상 종료 시 TaskStore의 시도 소유자를 확인하고, 소유자의 종료가 확정된 작업을 pending(확인 필요)으로 기록한다. 부작용이 이미 발생했을 수 있으므로 자동 재실행은 하지 않는다. 결과 JSON이 `running` 그대로 남아 있어도 완료로 판단하지 말고, 실제 상태를 확인한 후 명시적으로 재개한다.

## 흔한 실수

### 직접 실행해 버리는 경우

```bash
# 悪い例: 直接実行 → 長時間ロックされうる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

직접 실행해 버린 경우, 작업이 완료될 때까지 기다릴 수밖에 없다.
다음부터는 반드시 submit을 사용할 것.

### transcribe의 하위 명령 생략

```bash
# 悪い例: audio サブコマンドがないと意図したプロファイル判定にならない
animaworks-tool submit transcribe "/path/to/audio.wav"

# 良い例
animaworks-tool submit transcribe audio "/path/to/audio.wav"
```

### submit 후 결과를 계속 기다리는 경우

submit하면 즉시 다음 작업으로 넘어갈 것.
결과는 heartbeat용 알림 파일을 통해 가져오므로 폴링이나 대기는 필요 없다.

## 기술적 구조(참고)

### BackgroundTaskManager(`core/tasks/background.py`)

- **역할**: 장시간 도구 호출을 `asyncio` 작업으로 백그라운드 실행하고, 완료·실패 시 `on_complete`(임의의 비동기 콜백)를 `await`한다. 생성자에서 `state/background_tasks/`를 `mkdir(parents=True)`한다.
- **동기 도구**: `submit(tool_name, tool_args, execute_fn, task_id=None)` → `execute_fn(name, args) -> str | None`을 `run_in_executor(None, ...)`로 스레드 풀 실행. `task_id`가 생략되면 생성하고, CLI command task에서는 TaskStore와 동일한 ID를 전달한다.
- **비동기 도구**: `submit_async`(동일 시그니처로 `execute_fn`이 `Awaitable[str]`) → 이벤트 루프에서 `await execute_fn(...)`.
- **스케줄링**: `asyncio.create_task(..., name=f"bg-{task_id}")`로 래핑. 완료 시 `_async_tasks`에서 해당 항목을 제거.
- **영속화**: 각 변경 후 `_save_task`로 `state/background_tasks/{task_id}.json`에 `to_dict()`(`ensure_ascii=False`, `indent=2`). 손상된 JSON은 `_load_task`로 경고 로그 후 `None`.
- **조회**: `get_task`은 인메모리 우선, 없으면 디스크. `list_tasks(status=...)`는 디스크의 `*.json`와 병합하고, `created_at` 내림차순. `active_count`은 인메모리의 `RUNNING` 개수.
- **`on_complete`**: 콜백 내에서 예외가 발생해도 작업의 완료/실패 상태는 유지되며, 실패는 로그에 기록될 뿐이다.
- **자격 있는 도구 이름**(`is_eligible`)은 다음 **3계층**을 병합(나중 것이 우선). 키는 그대로 사전 조회(Mode A의 스키마 이름 `generate_3d_model`과 Mode S 제출용 `image_gen:3d`의 **둘 다**일 수 있음):
  1. 코드 내 기본값 `_DEFAULT_ELIGIBLE_TOOLS`(값은 기준 초. 현재 키):
     `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations`(각 30), `local_llm` / `run_command`(각 60)
  2. `BackgroundTaskManager.from_profiles`를 통해 각 모듈의 `EXECUTION_PROFILE`에서 `background_eligible: true`의 하위 명령을 추출(`core.integrations._base.get_eligible_tools_from_profiles`). 키는 `"{tool_name}:{subcmd}"`, 초는 `expected_seconds`(미설정 시 60)
  3. `config.json`의 `background_task.eligible_tools` — 각 키에 대해 `threshold_s`를 초로 덮어쓰기
- **비활성화**: `config.json`로 `background_task.enabled: false`로 설정하면 `BackgroundTaskManager` 자체가 생성되지 않는다. submit은 TaskStore에 등록되지만 실행 시도는 pending으로 돌아가고, 확인 필요 알림이 발생한다.
- **정리**: `cleanup_old_tasks(max_age_hours=24)`은 (1) `completed` / `failed`로 `completed_at`가 지정 시간을 초과한 JSON, (2) `running` 그대로 `created_at`이 **48시간 초과** 과거의 파일(프로세스 크래시 등의 고아)을 삭제한다. 반환 값은 삭제 개수. 보존 시간은 호출 측이 `max_age_hours`로 지정하며, 해당하는 `config.json` 설정 키는 없다.

### 명령형 작업(`animaworks-tool submit`)

1. `animaworks-tool submit`은 `task_type="command"`, 도구 이름, 인자, Anima, `task_id`을 포함한 입력을 TaskStore에 원자적으로 등록한다(`ANIMAWORKS_ANIMA_DIR` 필수).
2. TaskStore watcher는 기존 claim / attempt 메커니즘으로 작업을 가져온다. `BackgroundTaskManager`는 원래의 `task_id`을 결과 JSON에도 사용한다.
3. 명령형은 내부에서 `animaworks-tool … -j`을 **하위 프로세스**로 시작한다. **1회당 벽시계 타임아웃은 1800초(30분)**(`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`).
4. stdout 또는 오류는 `state/task_results/{task_id}/{attempt_token}.md`에 저장하고, TaskStore의 시도에 결과 참조를 기록한다. 상태·인자·시도의 정본은 TaskStore.
5. BackgroundTaskManager는 `state/background_tasks/{task_id}.json`에 `running` → `completed` / `failed`을 저장하고, `_on_background_task_complete`는 `state/background_notifications/{task_id}.md`에 알림을 작성한다. heartbeat가 알림을 가져온다.

### 기존 명령 설명자의 마이그레이션

이전의 `state/background_tasks/pending/*.json`은 새 런타임에서 모니터링하지 않는다. 종료 중에 `animaworks task-store migrate --anima NAME --backup PATH`을 실행하면, 처리되지 않은 최상위 설명자를 검증하여 TaskStore에 가져온다. 원본 파일은 증거로 남긴다. `processing/`와 `failed/`은 자동 재실행·마이그레이션하지 않고 경고와 함께 남기므로, 부작용을 확인한 후 운영자가 판단한다.

### LLM형 작업(정규 작업 스토어)

1. `submit_tasks` / `delegate_task`이 완전한 지침·맥락·완료 조건·제약·모델·의존 관계를 원자적으로 등록한다.
2. watcher는 실행 가능한 작업의 획득과 시도 생성을 같은 transaction에서 수행한다. 의존 대상과 설정된 워커 용량을 확인하고, 비병렬 제약은 같은 배치 내로 한정한다.
3. `done` / `cancelled` 선언 없이 시도가 끝나면 pending으로 돌아가지만 자동 재실행은 하지 않는다. 부작용을 확인하고, 필요하면 `submit_tasks(batch_id="resume", tasks=[{"task_id":"ID","resume":true}])`로 명시적으로 재개한다.
4. 결과는 `state/task_results/{task_id}.md`를 참조할 수 있는 경우가 있지만, 상태·입력·시도는 정규 스토어가 정본이다. 완료는 `update_task`으로 선언하고, LLM의 응답이나 파일 존재로 추측하지 않는다.
5. 확인 필요·완료 알림은 영속화하고, 정기 heartbeat에 의존하지 않고 전달한다. DM에서 실행 입력을 재구성하지 않는다.

SQLite나 기존 작업 파일을 직접 편집하지 말고 작업 도구를 사용한다. 위의 명령형 파이프라인과는 별개의 메커니즘이다.

### 파일의 라이프사이클

**명령형**(`animaworks-tool submit`):

```
TaskStore の command 入力 → claim / attempt
  → 成功: done | 失敗・中断: pending（要確認。自動再実行なし）
state/task_results/{task_id}/{attempt_token}.md  # attempt output / error
state/background_tasks/{task_id}.json           # 互換用の running → completed / failed 結果
```

**LLM형**(`submit_tasks` / `delegate_task`):

```
保存済み入力 → 実行可能pending → 取得済み試行（in_progress）
  → done/cancelled宣言、またはpending + 永続的な要確認通知
  → 明示resumeで保存済み入力を使う新しい試行
```

명령형과 LLM형은 다른 입력 형식이지만, 둘 다 TaskStore의 claim / attempt로 실행한다. 기존 LLM의 JSONL·`state/pending/` 설명자는 마이그레이션·export용이며, 실행 시작의 신호가 아니다.
