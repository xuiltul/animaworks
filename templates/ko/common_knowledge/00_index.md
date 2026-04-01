# Common Knowledge — 목차 및 퀵 가이드

AnimaWorks의 모든 Anima가 공유하는 레퍼런스 문서 목차입니다.
막혔거나 절차가 불확실한 경우, 이 파일에서 해당 문서를 찾은 후
`read_memory_file(path="common_knowledge/...")`로 상세 내용을 확인하세요.

> 💡 상세 기술 레퍼런스(구성 파일 사양, 모델 설정, 인증 설정 등)는 `reference/`로 이동했습니다.
> 목차: `reference/00_index.md`

---

## 막혔을 때 퀵 가이드

### 커뮤니케이션

| 문제 | 참조 |
|------|------|
| 메시지 보내는 방법을 모르겠다 | `communication/messaging-guide.md` |
| Board(공유 channel) 사용법을 모르겠다 | `communication/board-guide.md` |
| 지시나 보고 방법을 모르겠다 | `communication/instruction-patterns.md` / `communication/reporting-guide.md` |
| 메시지 전송이 제한되었다 | `communication/sending-limits.md` |
| 사람에게 알리는 방법을 모르겠다 | `communication/call-human-guide.md` |
| Slack 봇 토큰 설정을 모르겠다 | `reference/communication/slack-bot-token-guide.md` (→ reference) |

### 조직 및 계층

| 문제 | 참조 |
|------|------|
| 조직 구조나 누구에게 연락해야 하는지 모르겠다 | `reference/organization/structure.md` (→ reference) |
| 역할과 책임 범위를 확인하고 싶다 | `organization/roles.md` |
| 계층 간 커뮤니케이션 규칙을 모르겠다 | `organization/hierarchy-rules.md` |

### 태스크 및 운영

| 문제 | 참조 |
|------|------|
| 태스크 관리 방법을 모르겠다 | `operations/task-management.md` |
| 태스크 보드(사람용 대시보드)를 사용하고 싶다 | `operations/task-board-guide.md` |
| heartbeat이나 cron 설정을 모르겠다 | `operations/heartbeat-cron-guide.md` |
| 장시간 실행 도구 사용법을 모르겠다 | `operations/background-tasks.md` |
| 워크스페이스 등록이나 사용법을 모르겠다 | `operations/workspace-guide.md` |
| 프로젝트 설정을 변경하고 싶다 | `reference/operations/project-setup.md` (→ reference) |

### 도구, 모델 및 기술

| 문제 | 참조 |
|------|------|
| 도구 사용법이나 호출 방법을 모르겠다 | `operations/tool-usage-overview.md` |
| machine 도구 사용법을 모르겠다 | `operations/machine/tool-usage.md` |
| 내 역할에서 machine 워크플로를 알고 싶다 | `operations/machine/workflow-{pdm,engineer,reviewer,tester}.md` |
| 목적별로 팀(역할·핸드오프)을 설계하고 싶다 | `team-design/guide.md` |
| 법무 팀(계약·감사·검증) 역할과 핸드오프를 알고 싶다 | `team-design/legal/team.md` |
| 재무 팀(분석·감사·데이터·시장 데이터) 역할과 핸드오프를 알고 싶다 | `team-design/finance/team.md` |
| 트레이딩 팀(전략·분석·엔지니어링·리스크) 역할과 핸드오프를 알고 싶다 | `team-design/trading/team.md` |
| 영업·마케팅 팀(콘텐츠 제작·리드 개발·파이프라인 관리) 역할과 핸드오프를 알고 싶다 | `team-design/sales-marketing/team.md` |
| 모델 선택이나 변경 방법을 모르겠다 | `reference/operations/model-guide.md` (→ reference) |
| Mode S의 인증 방식을 변경하고 싶다 | `reference/operations/mode-s-auth-guide.md` (→ reference) |
| 음성 채팅 설정이나 사용법을 모르겠다 | `reference/operations/voice-chat-guide.md` (→ reference) |

### 자기 자신에 대한 이해

| 문제 | 참조 |
|------|------|
| Anima가 무엇인지 알고 싶다 | `anatomy/what-is-anima.md` |
| 자신의 구성 파일 역할을 알고 싶다 | `reference/anatomy/anima-anatomy.md` (→ reference) |
| 기억 시스템의 구조와 종류를 알고 싶다 | `anatomy/memory-system.md` |

### 트러블슈팅

| 문제 | 참조 |
|------|------|
| 도구나 명령어가 동작하지 않는다 / 오류가 발생한다 | `troubleshooting/common-issues.md` |
| 태스크가 블로킹되었다 / 판단이 어렵다 | `troubleshooting/escalation-flowchart.md` |
| Gmail 도구 인증 설정이 안 된다 | `reference/troubleshooting/gmail-credential-setup.md` (→ reference) |

### 보안

| 문제 | 참조 |
|------|------|
| 외부 데이터의 신뢰성이 걱정된다 | `security/prompt-injection-awareness.md` |

### 활용 사례

| 문제 | 참조 |
|------|------|
| AnimaWorks로 무엇을 할 수 있는지 알고 싶다 | `usecases/usecase-overview.md` |

**위에 해당하지 않는 경우** → `search_memory(query="키워드", scope="common_knowledge")`로 검색하세요

---

## 문서 목록

### anatomy/ — Anima의 구성 요소

| 파일 | 개요 |
|------|------|
| `what-is-anima.md` | Anima란 무엇인가 (개념, 설계 철학, 라이프사이클, 실행 경로) |
| `anima-anatomy.md` | → `reference/anatomy/anima-anatomy.md`로 이동. 구성 파일 완전 가이드 |
| `memory-system.md` | 기억 시스템 가이드 (기억의 종류, Priming, Consolidation, Forgetting, 도구 활용) |

### organization/ — 조직 및 구조

| 파일 | 개요 |
|------|------|
| `structure.md` | → `reference/organization/structure.md`로 이동. 조직 구조의 작동 방식 |
| `roles.md` | 역할과 책임 범위 (최상위 / 중간 관리 / 실행 Anima의 책무) |
| `hierarchy-rules.md` | 계층 간 규칙 (커뮤니케이션 경로, supervisor 도구, 긴급 예외) |

### communication/ — 커뮤니케이션

| 파일 | 개요 |
|------|------|
| `messaging-guide.md` | 메시지 송수신 완전 가이드 (send_message 파라미터, 스레드 관리, 1라운드 규칙) |
| `board-guide.md` | Board(공유 channel) 가이드 (post_channel / read_channel 활용, 게시 규칙) |
| `instruction-patterns.md` | 지시 패턴 모음 (명확한 지시 작성법, 위임 패턴, 진행 확인) |
| `reporting-guide.md` | 보고 및 에스컬레이션 방법 (보고 시점, 포맷, 긴급 vs 정기) |
| `sending-limits.md` | 전송 제한 상세 (3계층 레이트 제한, 30/h 및 100/day 상한, 캐스케이드 감지) |
| `call-human-guide.md` | 사람 알림 가이드 (call_human 사용법, 답장 수신, 알림 channel 설정) |
| `slack-bot-token-guide.md` | → `reference/communication/slack-bot-token-guide.md`로 이동. Slack 봇 토큰 설정 가이드 |

### operations/ — 운영 및 태스크 관리

| 파일 | 개요 |
|------|------|
| `project-setup.md` | → `reference/operations/project-setup.md`로 이동. 프로젝트 설정 방법 |
| `task-management.md` | 태스크 관리 (current_state.md 사용법과 태스크 큐, 상태 전이, 우선순위) |
| `task-board-guide.md` | 태스크 보드(사람용 대시보드)의 구조와 운영 방법 |
| `heartbeat-cron-guide.md` | 정기 실행 설정 및 운영 (heartbeat 구조, cron 태스크 정의, 자체 갱신) |
| `tool-usage-overview.md` | 도구 사용 개요 (S/A/B 모드별 도구 체계, 내부/외부 도구, 호출 방법) |
| `background-tasks.md` | 백그라운드 태스크 실행 가이드 (submit 사용법, 판단 기준, 결과 수신 방법) |
| `workspace-guide.md` | 워크스페이스 가이드 (개념, 등록, 도구에서의 활용, 트러블슈팅) |
| `model-guide.md` | → `reference/operations/model-guide.md`로 이동. 모델 선택 및 설정 가이드 |
| `mode-s-auth-guide.md` | → `reference/operations/mode-s-auth-guide.md`로 이동. Mode S 인증 모드 설정 가이드 |
| `voice-chat-guide.md` | → `reference/operations/voice-chat-guide.md`로 이동. 음성 채팅 가이드 |

### operations/machine/ — machine 도구 워크플로

| 파일 | 개요 |
|------|------|
| `tool-usage.md` | machine 도구 사용 가이드(공통 원칙·메타 패턴·상태·레이트 제한) |
| `workflow-pdm.md` | machine 워크플로 — PdM(조사→계획서) ※현재 영문 |
| `workflow-engineer.md` | machine 워크플로 — Engineer(구체화→구현) ※현재 영문 |
| `workflow-reviewer.md` | machine 워크플로 — Reviewer(리뷰→메타 리뷰) ※현재 영문 |
| `workflow-tester.md` | machine 워크플로 — Tester(테스트 설계→실행→결과) ※현재 영문 |

### team-design/ — 목적별 팀 설계

| 파일 | 개요 |
|------|------|
| `guide.md` | Anima 팀 설계 기본 원칙(역할 분리·핸드오프·스케일·겸임) |
| `development/team.md` | 개발 풀 팀 — 4역할(PdM·Engineer·Reviewer·Tester)·핸드오프·스케일링 |
| `legal/team.md` | 법무 풀 팀 — 3역할(Director·Verifier·Researcher)·carry-forward tracker·핸드오프 |
| `finance/team.md` | 재무 풀 팀 — 4역할(Finance Director·Financial Auditor·Data Analyst·Market Data Collector)·Variance Tracker·핸드오프 |
| `trading/team.md` | 트레이딩 풀 팀 — 4역할(Strategy Director·Market Analyst·Trading Engineer·Risk Auditor)·Performance/Ops Tracker·핸드오프 |
| `sales-marketing/team.md` | 영업·마케팅 풀 팀 — 4역할(Director·Marketing Creator·SDR·Market Researcher)·Campaign Pipeline Tracker·Deal Pipeline Tracker·2실행 모드·스케일링 |

역할별 템플릿: `team-design/development/{pdm,engineer,reviewer,tester}/` — `injection.template.md`, `machine.md`, `checklist.md`

역할별 템플릿: `team-design/legal/{director,verifier,researcher}/` — `injection.template.md`, `machine.md`(researcher 제외), `checklist.md`

역할별 템플릿: `team-design/finance/{director,auditor,analyst,collector}/` — `injection.template.md`, `machine.md`(analyst·collector 제외), `checklist.md`

역할별 템플릿: `team-design/trading/{director,analyst,engineer,auditor}/` — `injection.template.md`, `machine.md`, `checklist.md`

역할별 템플릿: `team-design/sales-marketing/{director,creator,sdr,researcher}/` — `injection.template.md`, `machine.md`(researcher 제외), `checklist.md`

### security/ — 보안

| 파일 | 개요 |
|------|------|
| `prompt-injection-awareness.md` | 프롬프트 인젝션 방어 가이드 (신뢰 수준, 경계 태그, untrusted 데이터 처리 규칙) |

### troubleshooting/ — 트러블슈팅

| 파일 | 개요 |
|------|------|
| `common-issues.md` | 자주 발생하는 문제와 해결법 (메시지 미배달, 전송 제한, 권한, 도구, 컨텍스트) |
| `escalation-flowchart.md` | 막혔을 때 판단 플로차트 (문제 분류, 긴급도 판정, 에스컬레이션 대상) |
| `gmail-credential-setup.md` | → `reference/troubleshooting/gmail-credential-setup.md`로 이동. Gmail Tool 인증 설정 가이드 |

### usecases/ — 활용 사례 가이드

| 파일 | 개요 |
|------|------|
| `usecase-overview.md` | 활용 사례 가이드 개요 (AnimaWorks로 할 수 있는 것, 시작 방법, 전체 주제 목록) |
| `usecase-communication.md` | 커뮤니케이션 자동화 (채팅/이메일 모니터링, 에스컬레이션, 정기 연락) |
| `usecase-development.md` | 소프트웨어 개발 지원 (코드 리뷰, CI/CD 모니터링, Issue 구현, 버그 조사) |
| `usecase-monitoring.md` | 인프라 및 서비스 모니터링 (가용성 확인, 리소스 모니터링, SSL 인증서, 로그 분석) |
| `usecase-secretary.md` | 비서 및 사무 지원 (일정 관리, 연락 조정, 일일 보고 작성, 리마인더) |
| `usecase-research.md` | 조사 및 리서치 분석 (웹 검색, 경쟁 분석, 시장 조사, 보고서 작성) |
| `usecase-knowledge.md` | 지식 관리 및 문서 정비 (절차서 작성, FAQ 구축, 교훈 축적) |
| `usecase-customer-support.md` | 고객 지원 (1차 대응, FAQ 자동 응답, 에스컬레이션 관리) |

---

## 키워드 색인

| 키워드 | 참조 |
|--------|------|
| 메시지, send_message, 전송, 답장, 스레드, inbox | `communication/messaging-guide.md` |
| Board, channel, post_channel, read_channel | `communication/board-guide.md` |
| DM 이력, read_dm_history, 이전 대화 | `communication/board-guide.md` |
| 지시, 위임, 태스크 의뢰 | `communication/instruction-patterns.md` |
| 보고, 일일 보고, 요약, 에스컬레이션 | `communication/reporting-guide.md` |
| 레이트 제한, 전송 제한, 30통, 100통, 1라운드 규칙 | `communication/sending-limits.md` |
| call_human, 사람 알림, 사람에게 연락 | `communication/call-human-guide.md` |
| Slack, 봇 토큰, SLACK_BOT_TOKEN, not_in_channel | `reference/communication/slack-bot-token-guide.md` |
| 조직, supervisor, 상사, 부하, 동료 | `reference/organization/structure.md` |
| 역할, 책임, speciality, 전문 | `organization/roles.md` |
| 계층, 커뮤니케이션 경로, org_dashboard, ping_subordinate | `organization/hierarchy-rules.md` |
| delegate_task, 태스크 위임, task_tracker | `organization/hierarchy-rules.md`, `operations/task-management.md` |
| 태스크, current_state, pending, 진행, 우선순위 | `operations/task-management.md` |
| 태스크 큐, submit_tasks, update_task, TaskExec, animaworks-tool task list | `operations/task-management.md` |
| 태스크 보드, 대시보드, 사람용 | `operations/task-board-guide.md` |
| 설정, config, status.json, SSoT, reload | `reference/operations/project-setup.md` |
| heartbeat, 정기 점검 | `operations/heartbeat-cron-guide.md` |
| cron, 스케줄, 정시 태스크 | `operations/heartbeat-cron-guide.md` |
| 도구, animaworks-tool, MCP, skill | `operations/tool-usage-overview.md` |
| 실행 모드, S-mode, A-mode, B-mode, C-mode | `operations/tool-usage-overview.md` |
| 백그라운드, submit, 장시간 도구 | `operations/background-tasks.md` |
| 워크스페이스, workspace, 작업 디렉토리, working_directory | `operations/workspace-guide.md` |
| machine, machine run, 외부 에이전트, 계획서 | `operations/machine/tool-usage.md` |
| 조사, investigation, PdM, plan.md | `operations/machine/workflow-pdm.md` |
| impl-plan, 구체화, 구현 계획 | `operations/machine/workflow-engineer.md` |
| 리뷰, review, 메타 리뷰 | `operations/machine/workflow-reviewer.md` |
| 테스트, test, E2E, 테스터 | `operations/machine/workflow-tester.md` |
| 팀 설계, 역할 분리, 개발 팀, PdM, 핸드오프 | `team-design/guide.md`, `team-design/development/team.md` |
| 법무, 계약, 리스크, 감사, 검증, carry-forward, 낙관적 편향 | `team-design/legal/team.md` |
| 재무, 분석, Variance Tracker, Data Lineage, silent drop, Financial Auditor, Finance Director, Data Analyst, Market Data Collector | `team-design/finance/team.md` |
| 트레이딩, 전략, 백테스트, bot, P&L, 드로다운, Performance Tracker, Ops Issue Tracker, Strategy Director, Market Analyst, Trading Engineer, Risk Auditor, carry-forward | `team-design/trading/team.md` |
| 영업, 마케팅, 콘텐츠, 리드, 너처링, BANT, 파이프라인, campaign tracker, deal tracker, SDR, Brand Voice | `team-design/sales-marketing/team.md` |
| 모델, models.json, credential, set-model, 컨텍스트 윈도우 | `reference/operations/model-guide.md` |
| background_model, 백그라운드 모델, 비용 최적화 | `reference/operations/model-guide.md` |
| Mode S, 인증, API 직접, Bedrock, Vertex AI, Max plan | `reference/operations/mode-s-auth-guide.md` |
| 음성, voice, STT, TTS, VOICEVOX, ElevenLabs | `reference/operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `reference/operations/voice-chat-guide.md` |
| Anima, 자기 자신, 구성, 설계, 라이프사이클 | `anatomy/what-is-anima.md` |
| identity, injection, 인격, 행동 지침, 불변, 가변 | `reference/anatomy/anima-anatomy.md` |
| permissions.json, bootstrap, heartbeat.md, cron.md | `reference/anatomy/anima-anatomy.md` |
| 기억, memory, episodes, knowledge, procedures | `anatomy/memory-system.md` |
| Priming, RAG, Consolidation, Forgetting | `anatomy/memory-system.md` |
| search_memory, write_memory_file, 기억 검색 | `anatomy/memory-system.md` |
| 프롬프트 인젝션, trust, untrusted, 경계 태그 | `security/prompt-injection-awareness.md` |
| 오류, 문제, 동작하지 않음, 권한, 차단된 명령어 | `troubleshooting/common-issues.md` |
| 플로차트, 판단, 고민, 긴급, 보안 | `troubleshooting/escalation-flowchart.md` |
| Gmail, token.json, OAuth, pickle | `reference/troubleshooting/gmail-credential-setup.md` |
| 티어, tiered, T1, T2, T3, T4 | `troubleshooting/common-issues.md` |
| 활용 사례, 사례, 무엇을 할 수 있나 | `usecases/usecase-overview.md` |

---

## 사용법

```
# 키워드로 검색
search_memory(query="메시지 전송", scope="common_knowledge")

# 경로를 직접 지정
read_memory_file(path="common_knowledge/communication/messaging-guide.md")

# 기술 레퍼런스 참조
read_memory_file(path="reference/anatomy/anima-anatomy.md")

# 이 파일 자체를 참조
read_memory_file(path="common_knowledge/00_index.md")
```
