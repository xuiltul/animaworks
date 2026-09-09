# 백그라운드 태스크 실행 가이드

## 개요

일부 외부 도구 (이미지 생성, 3D 모델 생성, 로컬 LLM 추론, 음성 전사 등)는
실행에 수 분에서 수십 분이 소요됩니다. 이를 직접 실행하면 실행 도중 잠금이 유지되어
메시지 수신과 heartbeat이 중단됩니다.

`animaworks-tool submit`을 사용하면 태스크를 백그라운드에서 실행하여
즉시 다음 작업으로 이동할 수 있습니다.

## submit 사용 시점

### 반드시 submit을 사용해야 하는 도구

도구 가이드 (시스템 프롬프트)에서 ⚠ 마크가 붙은 서브커맨드:

- `image_gen pipeline` / `fullbody` / `bustup` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe`

### submit이 불필요한 도구

실행 시간이 짧은 (30초 미만) 도구:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail` (일반 작업)
- `github`, `aws_collector`

### 판단 기준

- ⚠ 마크 있음 → 반드시 submit
- ⚠ 마크 없음 → 직접 실행

## 사용법

### 기본 구문

```bash
animaworks-tool submit <도구명> <서브커맨드> [인자...]
```

### 실행 예시

```bash
# 3D 모델 생성 (Meshy API, 최대 10분)
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# 캐릭터 이미지 일괄 생성 (전 단계, 최대 30분)
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# 로컬 LLM 추론 (Ollama, 최대 5분)
animaworks-tool submit local_llm generate "요약해 주세요: ..."

# 음성 전사 (Whisper + Ollama 후처리, 최대 2분)
animaworks-tool submit transcribe "/path/to/audio.wav" --language ja
```

### 반환값

submit은 즉시 다음 JSON을 반환하고 종료합니다:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "백그라운드 태스크가 제출되었습니다. 완료 시 inbox에 통지됩니다."
}
```

## 결과 수신

1. submit 후 태스크는 백그라운드에서 실행됩니다
2. 완료되면 `state/background_notifications/{task_id}.md`에 결과가 기록됩니다
3. 다음 heartbeat에서 자동으로 이 통지를 확인할 수 있습니다
4. 통지에는 성공/실패 상태와 결과 요약이 포함됩니다

## 실패 시 대응

- 통지에 "실패"로 기재된 경우:
  1. 오류 내용을 확인합니다
  2. 원인을 특정합니다 (API 키 미설정, 타임아웃, 인자 오류 등)
  3. 수정 후 다시 submit합니다
  4. 해결할 수 없으면 상사에게 보고합니다

- 커맨드형 도구의 크래시 회수는 `state/background_tasks/pending/processing/` 파일 경로를 사용합니다. LLM 작업은 정규 태스크와 시도로 관리하며, 미완료 시 pending과 영속적인 확인 요청 알림을 남깁니다. 실제 부작용을 확인한 후 명시적으로 재개하세요. 기존 LLM 파일을 이동하거나 재생성하지 마세요.

## 자주 하는 실수

### 직접 실행해 버리는 경우

```bash
# 나쁜 예: 직접 실행 → 10분간 잠금 상태
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 좋은 예: submit으로 비동기 실행
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

직접 실행해 버린 경우, 태스크가 완료될 때까지 기다릴 수밖에 없습니다.
다음부터 반드시 submit을 사용하세요.

### submit 후 결과를 기다리는 경우

submit 후에는 즉시 다음 작업으로 이동하세요.
결과는 자동으로 통지되므로 폴링이나 대기는 불필요합니다.

## 기술적 구조 (참고)

PendingTaskExecutor는 2종류의 태스크를 감시하고 실행합니다.

### 커맨드형 태스크 (animaworks-tool submit)

1. `animaworks-tool submit`이 `state/background_tasks/pending/*.json`에 태스크 기술자를 기록합니다
2. PendingTaskExecutor의 watcher가 3초 간격으로 `state/background_tasks/pending/`을 감시합니다 (`wake()`로 즉시 체크도 가능)
3. 태스크를 감지하면 `pending/*.json`을 `pending/processing/`으로 이동하여 실행을 시작합니다
4. `execute_pending_task`가 BackgroundTaskManager.submit에 등록합니다. Anima의 잠금 외부에서 서브프로세스로 실행됩니다 (타임아웃 30분)
5. 등록 성공 시: processing 내 파일 삭제. 등록 실패 시: `pending/failed/`로 이동
6. 완료 시 `_on_background_task_complete` 콜백이 `state/background_notifications/{task_id}.md`에 통지를 기록합니다
7. 다음 heartbeat에서 `drain_background_notifications()`가 통지를 읽어 컨텍스트에 주입합니다

### LLM형 태스크 (정규 태스크 저장소)

1. `submit_tasks` / `delegate_task`가 완전한 지시, 문맥, 완료 조건, 제약, 모델, 의존 관계를 원자적으로 등록합니다.
2. watcher는 의존 관계와 설정된 워커 용량을 확인하고, 실행 가능한 태스크 획득과 시도 생성을 같은 트랜잭션으로 수행합니다. 비병렬 제한은 같은 배치 내에서만 적용됩니다.
3. `done` / `cancelled` 선언 없이 시도가 끝나면 pending으로 남지만 자동 재실행하지 않습니다. 부작용을 확인한 후 필요하면 `submit_tasks(batch_id="resume", tasks=[{"task_id":"ID","resume":true}])`로 명시적으로 재개하세요.
4. `state/task_results/{task_id}.md`는 결과 참조일 뿐이며 상태, 입력, 시도는 정규 저장소가 정본입니다. 완료는 `update_task`로 선언하고 LLM 응답이나 파일 존재로 추측하지 마세요.
5. 확인 요청·완료 알림은 영속화되며 주기적 heartbeat에 의존하지 않습니다. DM으로 실행 입력을 재구성하지 마세요.

SQLite나 기존 태스크 파일을 직접 편집하지 말고 태스크 도구를 사용하세요. 위의 커맨드형 파이프라인과는 별개입니다.

### 파일 라이프사이클

**커맨드형** (animaworks-tool submit):

```
state/background_tasks/pending/*.json
  → pending/processing/*.json
  → 성공: 삭제 | 실패: pending/failed/*.json
```

**LLM형** (submit_tasks / delegate_task):

```
저장된 입력 → 실행 가능 pending → 획득한 시도 (in_progress)
  → done/cancelled 선언 또는 pending + 영속적인 확인 요청 알림
  → 명시적 resume이 저장된 입력으로 새 시도 생성
```

위의 커맨드형 파일 회수는 LLM 태스크에 적용하지 않습니다. 기존 LLM JSONL·기술자는 마이그레이션/export 형식이며 실행 시작 신호가 아닙니다.
