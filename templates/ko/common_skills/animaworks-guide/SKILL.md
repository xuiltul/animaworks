---
name: animaworks-guide
description: >-
  animaworks CLI 완전 레퍼런스. 서버·Anima·모델·태스크·설정·RAG·에셋·외부 도구 호출 구문을 정리한다.
  Use when: 서브커맨드 확인, 서버 시작·중지, Anima 생성·모델 변경·태스크·로그·설정·인덱스 조작이 필요할 때.
---

# AnimaWorks CLI 완전 레퍼런스

AnimaWorks의 모든 조작은 `animaworks` 커맨드로 수행합니다.
이 스킬은 모든 서브커맨드의 구문, 인수, 예시를 정리한 레퍼런스입니다.

운용 개념 및 규칙은 `common_knowledge/`를 참조하세요:
- 메시징 규칙 → `communication/messaging-guide.md`
- 태스크 관리 → `operations/task-management.md`
- 도구 체계 → `operations/tool-usage-overview.md`
- 조직 구조 → `reference/organization/structure.md`
- 모델 선택 및 설정 → `reference/operations/model-guide.md`

---

## 서버 조작 (기본적으로 사용하지 않는 것을 권장)

```bash
animaworks start                         # 서버 시작 (기본값: 0.0.0.0:18500)
animaworks start --port 8080             # 포트 지정
animaworks start --foreground            # 포그라운드 모드 (디버깅용)
animaworks stop                          # 서버 중지
animaworks restart                       # 서버 재시작
animaworks status                        # 시스템 상태 확인 (프로세스, Anima 목록)
animaworks reset                         # 런타임 디렉토리 삭제 + 재초기화
animaworks reset --restart               # 리셋 후 서버 자동 시작
```

---

## Anima 관리 (anima 서브커맨드)

### 목록, 상태, 상세 정보

```bash
animaworks anima list                    # 전체 Anima 목록 (이름, 활성/비활성, 모델, supervisor)
animaworks anima list --local            # API 미사용, 파일 시스템 직접 스캔
animaworks anima status                  # 전체 Anima 프로세스 상태 (State, Model, PID, Uptime)
animaworks anima status {name}           # 특정 Anima 프로세스 상태
animaworks anima info {name}             # 설정 상세 (모델, 역할, credential, voice 등)
animaworks anima info {name} --json      # JSON 출력
```

`anima info` 출력 항목:
- Anima 이름, Enabled, Role, Model, Execution Mode
- Credential, Fallback Model, Max Turns, Max Chains
- Context Threshold, Max Tokens, LLM Timeout
- Thinking 설정, Supervisor, Mode S Auth
- Voice 설정 (tts_provider, voice_id, speed, pitch)

### 생성

```bash
# 캐릭터 시트(MD)에서 생성 (권장)
animaworks anima create --from-md {file} [--role {role}] [--name {name}]

# 템플릿에서 생성
animaworks anima create --template {template_name} [--name {name}]

# 빈 상태로 생성
animaworks anima create --name {name}
```

### 활성화, 비활성화, 삭제

```bash
animaworks anima enable {name}           # 활성화 (일시 정지에서 복귀)
animaworks anima disable {name}          # 비활성화 (일시 정지)
animaworks anima delete {name}           # 삭제 (ZIP 아카이브 후)
animaworks anima delete {name} --no-archive  # 아카이브 없이 삭제
animaworks anima delete {name} --force   # 확인 없이 삭제
animaworks anima restart {name}          # 프로세스 재시작
animaworks anima audit {name}            # 부하의 최근 활동 포괄 감사 (기본값: 1일)
animaworks anima audit {name} --days 7   # 최근 7일 감사
```

### 모델 변경

```bash
animaworks anima set-model {name} {model_name}
animaworks anima set-model {name} {model_name} --credential {credential_name}
animaworks anima set-model --all {model_name}   # 전체 Anima 일괄 변경
```

서버가 실행 중인 경우 `anima restart {name}`이 필요합니다.

### 역할 변경

```bash
# 역할 변경 (템플릿 재적용 + 자동 restart)
animaworks anima set-role {name} {role}

# status.json의 role 필드만 변경 (템플릿 미변경)
animaworks anima set-role {name} {role} --status-only

# 파일 업데이트만, 재시작 없음
animaworks anima set-role {name} {role} --no-restart
```

set-role로 자동 업데이트되는 파일:
- `status.json` — role, model, max_turns를 역할 템플릿의 기본값으로 업데이트
- `specialty_prompt.md` — 역할별 전문 가이드라인으로 교체
- `permissions.json` — 역할별 도구 및 커맨드 허용 범위로 교체

유효한 역할: `engineer`, `researcher`, `manager`, `writer`, `ops`, `general`

### 핫 리로드

```bash
animaworks anima reload {name}           # status.json에서 모델 설정 재로드 (프로세스 재시작 없음)
animaworks anima reload --all            # 전체 Anima 리로드
```

---

## 모델 정보 (models 서브커맨드)

```bash
animaworks models list                   # 알려진 모델 목록 (이름, 실행 모드, 컨텍스트 윈도우, 설명)
animaworks models list --mode S          # 실행 모드로 필터 (S/A/B/C)
animaworks models list --json            # JSON 출력
animaworks models info {model_name}      # 특정 모델의 해석 정보 (실행 모드, 컨텍스트 윈도우, 임계값, 소스)
animaworks models show                   # models.json의 현재 내용 표시
animaworks models show --json            # 원시 JSON 출력
```

상세 → `reference/operations/model-guide.md`

---

## 채팅 및 메시징

```bash
# Anima와 채팅 (사람 → Anima)
animaworks chat {name} "메시지"
animaworks chat {name} "메시지" --from {sender_name}
animaworks chat {name} "메시지" --local  # API 미사용, 직접 실행

# Anima 간 메시지 전송
animaworks send {sender} {recipient} "메시지"
animaworks send {sender} {recipient} "메시지" --intent report
animaworks send {sender} {recipient} "메시지" --intent question
animaworks send {sender} {recipient} "메시지" --reply-to {message_id}
animaworks send {sender} {recipient} "메시지" --thread-id {thread_id}

# heartbeat 수동 트리거
animaworks heartbeat {name}
animaworks heartbeat {name} --local      # API 미사용, 직접 실행
```

---

## Board (공유 채널)

```bash
animaworks board read {channel}                         # 채널 메시지 읽기
animaworks board read {channel} --limit 50              # 최대 건수 지정
animaworks board read {channel} --human-only            # 사람의 메시지만
animaworks board post {sender} {channel} "텍스트"       # 채널에 게시
animaworks board dm-history {self} {peer}               # DM 히스토리 조회
animaworks board dm-history {self} {peer} --limit 50    # 건수 지정
```

---

## 설정 관리 (config 서브커맨드)

```bash
animaworks config list                   # 전체 설정값 목록
animaworks config list --section system  # 섹션으로 필터
animaworks config list --show-secrets    # API 키 값 표시
animaworks config get {key}              # 특정 설정값 조회 (도트 표기법: system.log_level)
animaworks config get {key} --show-secrets
animaworks config set {key} {value}      # 설정값 변경
animaworks config export-sections        # 템플릿 파일로 내보내기
animaworks config export-sections --dry-run
```

---

## 로그 조회 (logs)

```bash
animaworks logs {name}                   # 특정 Anima 로그 표시
animaworks logs --all                    # 서버 + 전체 Anima 로그 표시
animaworks logs {name} --lines 100       # 표시 행수 지정 (기본값: 50)
animaworks logs {name} --date 20260301   # 특정 날짜 로그 표시
```

---

## 비용 확인 (cost)

```bash
animaworks cost                          # 전체 Anima 토큰 사용량 및 비용
animaworks cost {name}                   # 특정 Anima 비용
animaworks cost --today                  # 오늘만
animaworks cost --days 7                 # 최근 7일 (기본값: 30일)
animaworks cost --json                   # JSON 출력
```

---

## 태스크 관리 (task 서브커맨드)

```bash
animaworks task list                     # 태스크 목록
animaworks task list --status pending    # 상태로 필터 (pending/in_progress/done/cancelled/blocked)
animaworks task add --assignee {name} --instruction "태스크 설명"
animaworks task add --assignee {name} --instruction "설명" --source human --deadline 2026-03-10T18:00:00
animaworks task update --task-id {id} --status done
animaworks task update --task-id {id} --status done --summary "완료 요약"
```

---

## RAG 인덱스 관리

```bash
animaworks index                         # 전체 Anima 인덱스 증분 업데이트
animaworks index --anima {name}          # 특정 Anima만
animaworks index --full                  # 전체 데이터 재인덱싱
animaworks index --dry-run               # 변경 내용 확인만 (실행 안 함)
```

---

## 에셋 조작

### 에셋 최적화

```bash
animaworks optimize-assets                              # 전체 Anima 3D 에셋 최적화
animaworks optimize-assets --anima {name}               # 특정 Anima만
animaworks optimize-assets --dry-run                    # 확인만
animaworks optimize-assets --simplify                   # 메시 단순화
animaworks optimize-assets --texture-compress           # 텍스처 압축
animaworks optimize-assets --texture-resize 512         # 텍스처 리사이즈
```

### 에셋 재생성

```bash
animaworks remake-assets {name} --style-from {reference}   # 스타일 전이로 에셋 재생성
animaworks remake-assets {name} --style-from {ref} --steps portrait,fullbody
animaworks remake-assets {name} --style-from {ref} --dry-run
animaworks remake-assets {name} --style-from {ref} --no-backup
```

---

## 외부 도구 실행 (animaworks-tool)

Anima가 외부 서비스(Slack, Gmail, GitHub 등)를 사용할 때의 커맨드입니다.

```bash
# 도움말 표시
animaworks-tool {tool_name} --help

# 실행
animaworks-tool {tool_name} {subcommand} [args...]

# 백그라운드 실행 (장시간 도구용)
animaworks-tool submit {tool_name} {subcommand} [args...]
```

### 예시

```bash
animaworks-tool web_search query "AnimaWorks framework"
animaworks-tool slack send --channel "#general" --text "좋은 아침입니다"
animaworks-tool github issues --repo owner/repo
animaworks-tool submit image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

submit 상세 → `common_knowledge/operations/background-tasks.md`

### 백그라운드 태스크 확인 (Anima 내부 도구)

submit으로 투입한 태스크의 진행 상황 확인에는 다음 내부 도구를 사용합니다:
- `list_background_tasks` — 실행 중 및 완료된 태스크 목록
- `check_background_task(task_id)` — 특정 태스크의 상태 및 결과 조회

이들은 CLI 커맨드가 아닌 Anima가 대화 중 사용하는 MCP 도구입니다.

---

## 초기화 및 마이그레이션

```bash
animaworks init                          # 런타임 디렉토리 초기화 (~/.animaworks/)
animaworks init --force                  # 재초기화, 기존 설정 덮어쓰기
animaworks init --skip-anima             # Anima 생성 건너뛰기
animaworks migrate-cron                  # cron.md를 일본어 형식에서 표준 cron으로 변환
```

---

## 글로벌 옵션

```bash
animaworks --gateway-url http://host:port {command}   # 서버 URL 지정
animaworks --data-dir /path/to/data {command}         # 런타임 디렉토리 지정
```
