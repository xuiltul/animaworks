# 공통 지식 — 목차・빠른 가이드

AnimaWorks의 모든 Anima가 공유하는 참조 문서의 목차.
막히거나 절차가 불명확할 때, 이 파일에서 해당 문서를 찾아
`read_memory_file(path="common_knowledge/...")`에서 세부 내용을 참조할 것.

> 💡 상세 기술 참조(구성 파일 사양・모델 설정・인증 설정 등)는 `reference/`로 이동했습니다.
> 목차: `reference/00_index.md`

---

## ⭐ 먼저 여기를 읽으세요

AnimaWorks를 처음 사용하거나 전체 구조를 정리하고 싶다면, 아래 1개 파일을 먼저 읽을 것.
Heartbeat / Cron / 팀 설계 / 기억 / 비용 최적화의 핵심이 한 장에 정리되어 있다.

| 파일 | 내용 |
|---------|------|
| **`anatomy/essentials.md`** | **AnimaWorks 에센셜 가이드** — 전체 구조・5가지 실행 경로・Heartbeat vs Cron・팀 설계・작업 진행 방식・기억 시스템・비용 최적화를 한 장으로 조망 |

읽은 후, 각 주제의 세부 내용은 아래 목차에서 찾아볼 것.

---

## 막혔을 때의 빠른 가이드

### 커뮤니케이션

| 문제 상황 | 참조처 |
|---------|--------|
| 메시지 보내는 방법을 모르겠음 | `reference/communication/messaging-guide.md` |
| Board(공유 채널) 사용법을 모르겠음 | `communication/board-guide.md` |
| 지침 전달 방법・보고 방법을 모르겠음 | `reference/communication/instruction-patterns.md` / `reference/communication/reporting-guide.md` |
| 위임・완료 보고・에스컬레이션의 필수 항목을 확인하고 싶음 | `communication/message-quality-protocol.md` |
| Inbox 시작이나 메시지 전송 동작을 확인하고 싶음 | `communication/sending-limits.md` |
| 사람에게 알림하는 방법을 모르겠음 | `communication/call-human-guide.md` |
| Slack 봇 토큰 설정을 모르겠음 | `reference/communication/slack-bot-token-guide.md` ※기술 참조 |

### 조직・계층

| 문제 상황 | 참조 대상 |
|---------|--------|
| 조직 구조・누구에게 연락해야 할지 모르겠다 | `reference/organization/structure.md` ※기술 참조 |
| 역할과 책임 범위를 확인하고 싶다 | `reference/organization/roles.md` |
| 계층 간 통신 규칙을 모르겠다 | `organization/hierarchy-rules.md` |

### 작업·운영

| 문제 | 참조 |
|---------|--------|
| 작업 관리 방법을 모르겠음 | `reference/operations/task-management.md` |
| 작업 보드(사람용 대시보드)를 사용하고 싶음 | `operations/task-board-guide.md` |
| 하트비트나 cron 설정 방법을 모르겠음 | `reference/operations/heartbeat-cron-guide.md` |
| 전송·게시·기억 쓰기 전에 확인 규칙을 추가하고 싶음 | `operations/action-rules-guide.md` |
| 장시간 도구 실행 방법을 모르겠음 | `operations/background-tasks.md` |
| 작업 공간 등록·사용 방법을 모르겠음 | `operations/workspace-guide.md` |
| worktree·임시 파일·캐시를 어디에 둘지 | `operations/scratch-and-worktrees.md` |
| 새 스킬을 만드는 방법·메타데이터를 확인하고 싶음 | `common_skills/skill-creator/SKILL.md` |
| 프로젝트 설정을 변경하고 싶음 | `reference/operations/project-setup.md` ※기술 참조 |

### 도구・모델・기술

| 문제 상황 | 참조 대상 |
|---------|--------|
| 도구 사용법・호출 방법을 모르겠다 | `reference/operations/tool-usage-overview.md` |
| 모델 선택・변경 방법을 모르겠다 | `reference/operations/model-guide.md` ※기술 참조 |
| Mode S의 인증 방식을 바꾸고 싶다 | `reference/operations/mode-s-auth-guide.md` ※기술 참조 |
| 음성 채팅 설정・사용법을 모르겠다 | `reference/operations/voice-chat-guide.md` ※기술 참조 |

### 자신에 대한 이해

| 문제 상황 | 참조 대상 |
|---------|--------|
| Anima가 무엇인지 알고 싶다 | `anatomy/what-is-anima.md` |
| 자신의 구성 파일 역할을 알고 싶다 | `reference/anatomy/anima-anatomy.md` ※기술 참조 |
| 기억의 구조・종류를 알고 싶다 | `reference/anatomy/memory-system.md` |

### 트러블슈팅

| 문제 상황 | 참조 대상 |
|---------|--------|
| 도구나 명령이 작동하지 않음 / 오류 발생 | `reference/troubleshooting/common-issues.md` |
| 작업이 차단됨 / 판단이 어려움 | `reference/troubleshooting/escalation-flowchart.md` |
| Gmail 도구 인증 설정이 잘 되지 않음 | `reference/troubleshooting/gmail-credential-setup.md` ※기술 참조 |

### 보안

| 문제 상황 | 참조 대상 |
|---------|--------|
| 외부 데이터의 신뢰성이 걱정된다 | `security/prompt-injection-awareness.md` |

### 활용 예시

| 문제 상황 | 참조 대상 |
|---------|--------|
| AnimaWorks로 무엇을 할 수 있는지 알고 싶다 | `reference/usecases/usecase-overview.md` |

**위에 해당하지 않는 경우** → `search_memory(query="キーワード", scope="common_knowledge")`에서 검색할 것

---

## 문서 목록

### anatomy/ — Anima의 구성 요소

| 파일 | 개요 |
|---------|------|
| ⭐ `essentials.md` | **에센셜 가이드** — AnimaWorks의 전체 구조를 한 장으로 조망(실행 경로・Heartbeat vs Cron・팀 설계・기억・비용 최적화) |
| `what-is-anima.md` | Anima란 무엇인가(개념・설계 철학・라이프사이클・실행 경로) |
| `anima-anatomy.md` | → `reference/anatomy/anima-anatomy.md`로 이동. 구성 파일 완전 가이드 |
| `memory-system.md` | 기억 시스템 가이드(기억의 종류・Priming・Consolidation・Forgetting・도구 구분 사용) |

### organization/ — 조직・구조

| 파일 | 개요 |
|---------|------|
| `structure.md` | → `reference/organization/structure.md`로 이동. 조직 구조의 메커니즘 |
| `roles.md` | 역할과 책임 범위(최상위 / 중간 관리 / 실행 Anima의 책무) |
| `hierarchy-rules.md` | 계층 간 규칙(통신 경로, 슈퍼바이저 도구, 긴급 시 예외) |

### communication/ — 커뮤니케이션

| 파일 | 개요 |
|---------|------|
| `messaging-guide.md` | 메시지 송수신 가이드(send_message 파라미터, 스레드 관리, 답변 방침) |
| `board-guide.md` | Board(공유 채널) 가이드(post_channel / read_channel 사용 구분, 게시 규칙) |
| `instruction-patterns.md` | 지침 전달 방법 패턴 모음(명확한 지침 작성법, 위임 패턴, 진행 상황 확인) |
| `reporting-guide.md` | 보고・에스컬레이션 방법(보고 시점, 형식, 긴급 vs 정기) |
| `message-quality-protocol.md` | 메시지 품질 프로토콜(위임 4항목・완료 보고 3항목・에스컬레이션 4항목 필수 체크) |
| `sending-limits.md` | Inbox 파일 wake・메시지 집약・run 내 중복 방지・루프 회피 |
| `call-human-guide.md` | 사람에게 알림 가이드(call_human 사용법, 답변 수신, 알림 채널 설정) |
| `slack-bot-token-guide.md` | → `reference/communication/slack-bot-token-guide.md`로 이동. Slack 봇 토큰 설정 가이드 |

### operations/ — 운영·작업 관리

| 파일 | 개요 |
|---------|------|
| `project-setup.md` | → `reference/operations/project-setup.md` 로 이동. 프로젝트 설정 방법 |
| `task-management.md` | 작업 관리(current_state.md 사용법과 작업 대기열, 상태 전환, 우선순위) |
| `task-board-guide.md` | 작업 보드(사람용 대시보드)의 구조와 운영 방법 |
| `heartbeat-cron-guide.md` | 정기 실행 설정 및 운영(하트비트 구조, cron 작업 정의, 자체 업데이트) |
| `action-rules-guide.md` | 작업 규칙(`[ACTION-RULE]`, `trigger_tools`, 전송 전 확인, 필수 `read_memory_file`) |
| `tool-usage-overview.md` | 도구 사용 개요(S/C/D/G/A/B 모드별 도구 체계, 내부/외부 도구, 호출 방법) |
| `background-tasks.md` | 백그라운드 작업 실행 가이드(submit 사용법, 판단 기준, 결과 수신 방법) |
| `workspace-guide.md` | 작업 공간 가이드(개념·등록·도구 사용·문제 해결) |
| `scratch-and-worktrees.md` | 작업 산출물 저장 위치(scratch·공유 worktree·캐시·자동 삭제) |
| `model-guide.md` | → `reference/operations/model-guide.md` 로 이동. 모델 선택·설정 가이드 |
| `mode-s-auth-guide.md` | → `reference/operations/mode-s-auth-guide.md` 로 이동. Mode S 인증 모드 설정 가이드 |
| `voice-chat-guide.md` | → `reference/operations/voice-chat-guide.md` 로 이동. 음성 채팅 가이드 |

### security/ — 보안

| 파일 | 개요 |
|---------|------|
| `prompt-injection-awareness.md` | 프롬프트 인젝션 방어 가이드(신뢰 수준, 경계 태그, untrusted 데이터 처리 규칙) |

### troubleshooting/ — 트러블슈팅

| 파일 | 개요 |
|---------|------|
| `common-issues.md` | 자주 발생하는 문제와 대처법(메시지 미도달, 전송 제한, 권한, 도구, 컨텍스트) |
| `escalation-flowchart.md` | 막혔을 때의 판단 플로차트(문제 분류, 긴급도 판정, 에스컬레이션 대상) |
| `gmail-credential-setup.md` | → `reference/troubleshooting/gmail-credential-setup.md`로 이동. Gmail Tool 인증 설정 가이드 |

### usecases/ — 유스케이스 가이드

| 파일 | 개요 |
|---------|------|
| `usecase-overview.md` | 유스케이스 가이드 개요(AnimaWorks로 할 수 있는 일・시작 방법・전체 테마 목록) |
| `usecase-communication.md` | 커뮤니케이션 자동화(채팅・메일 모니터링, 에스컬레이션, 정기 연락) |
| `usecase-development.md` | 소프트웨어 개발 지원(코드 리뷰, CI/CD 모니터링, Issue 구현, 버그 조사) |
| `usecase-monitoring.md` | 인프라・서비스 모니터링(생사 모니터링, 리소스 모니터링, SSL 인증서, 로그 분석) |
| `usecase-secretary.md` | 비서・사무 지원(일정 관리, 연락 조정, 일일 보고서 작성, 리마인더) |
| `usecase-research.md` | 조사・리서치・분석(웹 검색, 경쟁 분석, 시장 조사, 리포트 작성) |
| `usecase-knowledge.md` | 지식 관리・문서 정비(절차서 작성, FAQ 구축, 교훈 축적) |
| `usecase-customer-support.md` | 고객 지원(1차 대응, FAQ 자동 응답, 에스컬레이션 관리) |

---

## 키워드 색인

| 키워드 | 참조 항목 |
|-----------|--------|
| 기초, 입문, 전체 개요, 핵심, 시작 방법, 개요 | `anatomy/essentials.md` |
| 메시지, send_message, 전송, 답장, 스레드, inbox | `reference/communication/messaging-guide.md` |
| Board, 채널, post_channel, read_channel | `communication/board-guide.md` |
| DM 기록, read_dm_history, 지난 대화 | `communication/board-guide.md` |
| 지침, 위임, 작업 요청, 위임 | `reference/communication/instruction-patterns.md` |
| 보고, 일일 보고, 요약, 완료 보고, 에스컬레이션 | `reference/communication/reporting-guide.md` |
| 품질 프로토콜, 필수 항목, 검증 근거, 완료 조건, 위임 확인 | `communication/message-quality-protocol.md` |
| Inbox wake, 메시지 집계, 중복 전송 방지, 확인 응답, 대화 루프 | `communication/sending-limits.md` |
| call_human, 사람 알림, 사람에게 연락, 알림 채널 | `communication/call-human-guide.md` |
| Slack, 봇 토큰, SLACK_BOT_TOKEN, not_in_channel | `reference/communication/slack-bot-token-guide.md` |
| 조직, supervisor, 상급자, 부하, 동료 | `reference/organization/structure.md` |
| 역할, 책임, speciality, 전문성 | `reference/organization/roles.md` |
| 계층, 통신 경로, org_dashboard, ping_subordinate | `organization/hierarchy-rules.md` |
| delegate_task, 작업 위임, task_tracker | `organization/hierarchy-rules.md`, `reference/operations/task-management.md` |
| 작업, current_state, pending, 진행 상황, 우선순위 | `reference/operations/task-management.md` |
| 작업 대기열, submit_tasks, update_task, TaskExec, animaworks-tool task list | `reference/operations/task-management.md` |
| 작업 보드, 대시보드, 인간용 | `operations/task-board-guide.md` |
| 설정, config, status.json, SSoT, reload | `reference/operations/project-setup.md` |
| 하트비트, heartbeat, 정기 점검 | `reference/operations/heartbeat-cron-guide.md` |
| cron, 스케줄, 정기 작업 | `reference/operations/heartbeat-cron-guide.md` |
| 도구, animaworks-tool, MCP, skill | `reference/operations/tool-usage-overview.md` |
| 실행 모드, S-mode, C-mode, D-mode, G-mode, A-mode, B-mode | `reference/operations/tool-usage-overview.md` |
| 백그라운드, submit, 장시간 도구 | `operations/background-tasks.md` |
| 작업 공간, workspace, 작업 디렉터리, working_directory | `operations/workspace-guide.md` |
| scratch, worktree, 임시 파일, 캐시, ANIMAWORKS_SCRATCH_DIR | `operations/scratch-and-worktrees.md` |
| 모델, models.json, credential, set-model, 컨텍스트 윈도우 | `reference/operations/model-guide.md` |
| background_model, 백그라운드 모델, 비용 최적화 | `reference/operations/model-guide.md` |
| Mode S, 인증, API 직접 연결, Bedrock, Vertex AI, Max plan | `reference/operations/mode-s-auth-guide.md` |
| 음성, voice, STT, TTS, VOICEVOX, ElevenLabs | `reference/operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `reference/operations/voice-chat-guide.md` |
| Anima, 자신, 구성, 설계, 라이프사이클 | `anatomy/what-is-anima.md` |
| IDENTITY.md, SOUL.md, AGENTS.md, 인격, 프로필, 행동 지침, 불변, 가변 | `reference/anatomy/anima-anatomy.md` |
| permissions.json, bootstrap, HEARTBEAT.md, cron.md | `reference/anatomy/anima-anatomy.md` |
| 기억, memory, episodes, knowledge, procedures | `reference/anatomy/memory-system.md` |
| Priming, RAG, Consolidation, Forgetting, 망각 | `reference/anatomy/memory-system.md` |
| consolidation, 2-phase, multipass, 오류 추적 | `reference/anatomy/memory-system.md` |
| search_memory, write_memory_file, 기억 검색 | `reference/anatomy/memory-system.md` |
| skills, 스킬 검색, common_skills, search_memory scope="skills" | `reference/anatomy/memory-system.md`, `reference/operations/tool-usage-overview.md` |
| activity_log, BM25, RRF, 최근 로그 검색 | `reference/anatomy/memory-system.md`, `reference/troubleshooting/common-issues.md` |
| 프롬프트 인젝션, trust, untrusted, 경계 태그 | `security/prompt-injection-awareness.md` |
| 오류, 문제, 작동 안 함, 권한, 차단 명령어 | `reference/troubleshooting/common-issues.md` |
| 순서도, 판단, 고민, 긴급, 보안 | `reference/troubleshooting/escalation-flowchart.md` |
| Gmail, token.json, OAuth, pickle | `reference/troubleshooting/gmail-credential-setup.md` |
| 티어, 계층형, T1, T2, T3, T4 | `reference/troubleshooting/common-issues.md` |
| 사용 사례, 활용 예, 무엇을 할 수 있는지 | `reference/usecases/usecase-overview.md` |

---

## 사용 방법

```
# キーワードで検索
search_memory(query="メッセージ 送信", scope="common_knowledge")

# パスを直接指定
read_memory_file(path="reference/communication/messaging-guide.md")

# 技術リファレンスを参照
read_memory_file(path="reference/anatomy/anima-anatomy.md")

# このファイル自体を参照
read_memory_file(path="common_knowledge/00_index.md")
```
