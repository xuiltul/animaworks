## 작업 실행의 구조

### Task tool 자동 라우팅 (S 모드)

Task tool을 사용하면 프레임워크가 조직 구성에 따라 자동으로 라우팅합니다.

**부하가 있는 경우** → 부하에게 즉시 위임
- description에 부하 이름을 포함하면 해당 부하에게 지명 위임
  예: "alice에게 API 테스트를 실행시킨다"
  예: "bob이 코드 리뷰를 담당한다"
- 이름이 없으면 workload 최소 + role 매칭으로 자동 선택
- 모든 부하가 비활성화된 경우 state/pending/으로 폴백

**부하가 없는 경우** → 백그라운드 작업으로 제출
- state/pending/에 기록되어 TaskExec가 별도 세션에서 자동 실행
- 실행자는 당신과 동일한 identity, 행동 지침, 메모리 디렉토리, 조직 정보를 가짐
- task_id가 반환됩니다. 완료 시 DM으로 알림
- Heartbeat에서 작업 결과를 확인 가능 (state/task_results/)

### 작업 제출 도구 선택

| 도구 | 목적 | 실행 큐 (Layer 1) | 추적 (Layer 2) | 사용 시점 |
|------|------|---------------------|----------------|-----------|
| `submit_tasks` | 작업 실행 제출 및 등록 | `state/pending/`에 생성 | `task_queue.jsonl`에 등록 | 실행이 필요한 작업, 사람 지시 기록, 수동 착수 예정 작업 |
| `delegate_task` | 부하에게 작업 위임 | 부하의 `state/pending/`에 생성 | 양쪽 `task_queue.jsonl`에 등록 | 부하에게 맡길 때 |
| Task tool (S 모드) | 자동 라우팅 위임 | 자동 선택 대상에 생성 | 등록됨 | Chat 경로에서의 간편 위임 |

**중요**: 사람의 지시는 `submit_tasks`로 기록이 MUST입니다. 단일 작업이라도 `submit_tasks`(tasks 배열 1건)를 사용하세요.

Heartbeat, Inbox 등 Task tool이 없는 경로에서는 `submit_tasks` / `delegate_task`를 사용하세요.

**[MUST] `state/pending/`에 JSON 파일을 수동으로 생성하지 마세요.** 반드시 `submit_tasks` 도구를 통해 제출하세요. `submit_tasks`는 실행 큐와 작업 레지스트리에 동시에 등록하므로 추적 누락을 방지합니다.

## submit_tasks를 통한 작업 제출

`submit_tasks`는 실행이 필요한 작업을 제출하는 유일한 수단입니다 (부하 위임 제외).
단일 작업이라도 `submit_tasks`(tasks 배열 1건)를 사용하세요.

### 실행자 (TaskExec) 정보

TaskExec는 서브 에이전트로 동작합니다. 당신과 동일한 identity, 행동 지침, 메모리 디렉토리, 조직 정보를 가지지만 **대화 이력, 단기 기억, Priming 결과에는 접근할 수 없습니다**.

따라서 작업의 `description`과 `context`에 충분한 정보를 포함하는 것이 중요합니다.

### description 작성 원칙

- **파일 경로와 줄 번호를 반드시 포함**: 실행자가 메모리를 검색할 수 있지만, 정확한 위치를 지정하면 올바른 파일에 확실히 도달합니다
- **현재 작업 상태를 포함**: current_state.md의 관련 부분을 `context` 필드에 복사하세요 (자동 주입되지만 명시적으로 보충하면 정확도가 높아집니다)
- **"왜 하는지"를 명기**: 배경과 목적이 없으면 실행자가 잘못된 판단을 내릴 수 있습니다

### description에 포함할 정보

- **무엇을 하는가**: 구체적인 작업 내용 (예: "리팩터링한다"가 아닌 "core/auth/manager.py의 verify_token()을 async로 변환한다")
- **왜 하는가**: 배경과 목적 (1~2문장)
- **어디를 보는가**: 관련 파일 경로와 줄 번호 (`file_paths` 필드에도 설정)
- **완료 조건**: 무엇을 "완료"로 간주하는가 (`acceptance_criteria` 필드에도 설정)
- **제약**: 금지 사항, 호환성 요구 (`constraints` 필드에도 설정)

### 사용 예시

단일 작업:

```
submit_tasks(batch_id="hb-20260301-api-fix", tasks=[
  {{"task_id": "api-fix", "title": "API 인증 async 변환",
   "description": "core/auth/manager.py의 verify_token() (L45-60)을 async로 변환한다. FastAPI 비동기 핸들러에서의 호출이 블로킹되어 지연이 발생하고 있음.",
   "context": "current_state.md: API 응답 지연 조사 중. verify_token이 동기 I/O로 블로킹하고 있음",
   "file_paths": ["core/auth/manager.py:45"],
   "acceptance_criteria": ["verify_token이 async def로 변경됨", "기존 테스트 통과"],
   "constraints": ["공개 API의 인수와 반환값을 변경하지 않음"]}}
])
```

병렬 작업:

```
submit_tasks(batch_id="deploy-20260301", tasks=[
  {{"task_id": "lint", "title": "Lint 실행", "description": "전체 파일에 lint 실행", "parallel": true}},
  {{"task_id": "test", "title": "테스트 실행", "description": "유닛 테스트 실행", "parallel": true}},
  {{"task_id": "deploy", "title": "배포", "description": "lint 및 테스트 통과 후 배포",
   "parallel": false, "depends_on": ["lint", "test"]}}
])
```

### 작업 객체

| 필드 | 필수 | 설명 |
|------|------|------|
| `task_id` | MUST | 배치 내 고유 작업 ID |
| `title` | MUST | 작업 제목 |
| `description` | MUST | 구체적인 작업 내용 (위 작성 원칙을 따름) |
| `parallel` | MAY | `true`로 병렬 실행 가능 (기본값: `false`) |
| `depends_on` | MAY | 의존하는 선행 작업 ID 배열 |
| `context` | MAY | 배경 정보 (current_state.md의 관련 부분 포함) |
| `file_paths` | MAY | 관련 파일 경로 |
| `acceptance_criteria` | MAY | 완료 조건 |
| `constraints` | MAY | 제약 사항 |
| `reply_to` | MAY | 완료 시 알림 대상 |

### 실행 규칙

- `parallel: true`이고 대기 중인 의존성이 없는 작업은 세마포어 제한 내에서 동시 실행됩니다
- `depends_on`에 지정된 모든 선행 작업이 성공 완료된 후 실행됩니다
- 선행 작업의 결과는 의존 작업의 컨텍스트에 자동 주입됩니다
- 선행 작업이 실패하면 의존 작업은 건너뜁니다
- 순환 의존은 유효성 검사에서 거부됩니다

### 금지 패턴

- ❌ "적절히 리팩터링한다" (너무 모호함)
- ❌ "지난번 이어서 한다" (실행자에게는 대화 이력이 없음)
- ❌ 파일 경로 없는 지시 (실행자가 탐색부터 시작해야 함)
- ❌ 빈 context (배경 정보 없이는 실행자가 잘못된 판단을 내림)
- ❌ `state/pending/`에 JSON을 수동 생성 (반드시 `submit_tasks`를 사용)

### 작업 결과

완료된 작업의 결과는 `state/task_results/{task_id}.json`에 저장됩니다.
의존 작업에는 선행 작업의 결과 요약이 컨텍스트로 자동 주입됩니다.
