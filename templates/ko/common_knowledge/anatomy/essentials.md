# AnimaWorks 에센셜 가이드

[IMPORTANT] AnimaWorks의 전체 그림을 한 페이지로 파악하기 위한 통합 가이드.
Heartbeat / Cron / 팀 설계 / 메모리 / 비용 최적화의 핵심을 망라.
처음 읽는 경우나 개념 간 관계를 정리하고 싶을 때 가장 먼저 참조하세요.
각 주제의 상세 내용은 각 섹션 끝의 링크를 참조하세요.

---

## AnimaWorks란?

AI 에이전트를 "도구"가 아닌 **자율적인 인격**으로 운용하는 프레임워크.

- 각 Anima는 고유한 인격·기억·판단 기준을 가짐
- 사람의 지시 없이도 정기적으로 스스로 행동 (Heartbeat / Cron)
- 조직 내에서 역할을 맡아 다른 Anima나 사람과 협업
- 경험에서 배우고 기억을 축적하며 성장

---

## 5가지 실행 경로 — "언제, 어떻게 움직이나?"

Anima는 다음 5가지 경로로 가동됩니다. Chat 이외는 모두 자동으로 시작됩니다.

| 경로 | 언제 동작 | 무엇을 하나 | 누가 사용 |
|------|----------|-----------|----------|
| **Chat** | 사람이 메시지를 보냈을 때 | 대화 응답 | 사람 → Anima |
| **Inbox** | 다른 Anima로부터 DM이 왔을 때 | 조직 내 메시지에 즉시 응답 | Anima → Anima |
| **Heartbeat** | 정기 자동 시작 (기본 30분) | 관찰 → 계획 → 회고. **실행하지 않음** | 자동 |
| **Cron** | cron.md 스케줄 (예: 매일 9:00) | 정해진 시간에 확정된 태스크 실행 | 자동 |
| **TaskExec** | `state/pending/`에 태스크가 나타났을 때 | LLM 세션으로 실제 작업 실행 | 자동 (Heartbeat 또는 submit_tasks에서 투입) |

Chat과 Heartbeat는 **별도 잠금**으로 동작하므로, 순찰 중에도 사람의 대화에 즉시 응답 가능.

→ 상세: `anatomy/what-is-anima.md`

---

## Heartbeat vs Cron — 2가지 자율 행동

둘 다 "사람의 지시 없이 움직이는" 구조이지만 목적이 근본적으로 다릅니다.

| 관점 | Heartbeat (정기 순찰) | Cron (정시 태스크) |
|------|---------------------|------------------|
| **비유** | 정기적으로 사무실을 순찰하는 경비원 | 매일 아침 9시에 배달되는 신문 |
| **목적** | 상황 확인·계획 수립·회고 | 정해진 업무의 실행 |
| **실행하나?** | **안 함**. 태스크를 발견하면 `submit_tasks`나 `delegate_task`로 투입할 뿐 | **함**. LLM형이면 사고와 실행, Command형이면 즉시 실행 |
| **간격** | 고정 간격 (기본 30분, Activity Level로 변동) | cron식으로 유연 지정 (매일 9:00, 매주 금요일 17:00 등) |
| **설정 파일** | `heartbeat.md` (체크리스트) | `cron.md` (태스크 정의) |
| **전형적 예** | 미읽 메시지 확인, 블로커 감지, 진척 회고 | 아침 업무 계획, 주간 보고서, 백업 실행 |

**판단이 어려울 때**: "확인만 하고 판단한다" → Heartbeat 체크리스트에 추가. "정해진 시간에 뭔가를 실행한다" → Cron 태스크로 정의.

→ 상세: `operations/heartbeat-cron-guide.md`

---

## Cron의 2가지 타입 — LLM형 vs Command형

Cron 태스크는 사고가 필요한지 여부에 따라 2가지 타입이 있습니다.

| 관점 | LLM형 (`type: llm`) | Command형 (`type: command`) |
|------|---------------------|---------------------------|
| **비유** | "오늘 무엇을 우선해야 할지 생각해" | "매일 아침 이 버튼을 눌러" |
| **판단** | 있음 (상황에 따라 출력이 달라짐) | 없음 (매번 같은 것을 실행) |
| **API 비용** | 있음 (LLM 호출) | 명령 자체는 없음. 다만 **follow-up LLM이 시작될 수 있음** (아래 참조) |
| **출력** | 가변형 (태스크마다 다름) | 확정적 (명령의 stdout) |
| **적합한 태스크** | 계획 수립, 회고, 문서 작성, 기억 정리 | 백업, 알림 전송, 데이터 수집, 헬스 체크 |

### Command형의 follow-up LLM (중요)

Command형은 명령 자체는 기계적으로 실행하지만, **stdout에 반환값이 있으면 기본적으로 LLM이 시작되어 결과를 분석합니다** (follow-up). 즉, 항상 비용 제로는 아닐 수 있습니다.

```
명령 실행 → stdout 있음?
  → 없음 → 종료 (LLM 불필요)
  → 있음 → skip_pattern에 매치?
      → 매치 → 종료 (LLM 스킵)
      → 매치 안 함 → LLM이 시작되어 결과 해석·대처 판단
```

이 follow-up을 제어하는 옵션:
- **`trigger_heartbeat: false`** — follow-up LLM을 항상 스킵 (결과 분석이 불필요한 경우)
- **`skip_pattern: <정규표현식>`** — stdout가 매치할 때만 스킵 (정상 시에만 무시, 이상 시에는 LLM이 판단)

### 사용 분류 판단 기준

```
"매번 같은 것만 하면 되나?"
  → 예, 결과도 볼 필요 없다 → Command형 + trigger_heartbeat: false
  → 예, 다만 이상 시에만 판단이 필요 → Command형 + skip_pattern (정상 패턴)
  → 아니오 (상황에 따라 판단이 달라짐) → LLM형
  → 명령 실행 + 매번 결과 해석 필요 → LLM형 (description에 명령 실행을 지시)
```

### 기술 예

```markdown
## 아침 업무 계획 (LLM형)
schedule: 0 9 * * *
type: llm
episodes/에서 어제 진척을 확인하고 오늘 태스크를 계획한다.

## 백업 실행 (Command형·follow-up 불필요)
schedule: 0 2 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/backup.sh

## 모니터링 체크 (Command형·정상 시 스킵, 이상 시 LLM 판단)
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

→ 상세: `operations/heartbeat-cron-guide.md`

---

## 태스크 라우팅 — submit_tasks vs delegate_task

태스크를 실행으로 옮기는 방법은 2가지입니다.

| 관점 | `submit_tasks` | `delegate_task` |
|------|---------------|----------------|
| **누가 실행하나** | **자기 자신**의 TaskExec 경로 | **직속 부하** |
| **사용 장면** | 자신이 할 태스크를 비동기 실행하고 싶을 때 | 부하에게 위임하고 싶을 때 |
| **DAG/병렬** | `parallel: true`로 병렬, `depends_on`으로 의존 | 1건씩 위임 |
| **진척 추적** | task_queue.jsonl + Priming 표시 | `task_tracker`로 추적 |
| **전형적 예** | Heartbeat에서 발견한 태스크를 자신이 실행 | 상사가 부하에게 작업 위임 |

**판단 흐름:**
```
"이 태스크는 부하가 해야 하나?"
  → 예, 직속 부하가 있다 → delegate_task
  → 아니오, 자신이 한다 → submit_tasks
  → 부하가 없다 → submit_tasks
```

→ 상세: `operations/task-management.md`, `anatomy/task-architecture.md`

---

## 팀 설계 — 솔로에서 시작하여 스케일업

### 왜 역할을 분리하나?

| 이유 | 설명 |
|------|------|
| 컨텍스트 오염 방지 | 전 공정을 1인에게 맡기면 컨텍스트가 비대해져 판단 정확도가 떨어짐 |
| 구조적 품질 보증 | 실행자와 검증자를 분리 (셀프 리뷰의 맹점 제거) |
| 병렬 실행 | 독립적인 역할은 동시 실행으로 처리량 향상 |
| 전문성 심화 | 역할 고유의 체크리스트·기억으로 범용 에이전트보다 고품질 |

### 스케일링 단계

| 규모 | 구성 | 언제 사용 |
|------|------|----------|
| **솔로** | 1 Anima가 전 역할 겸임 | 소규모 태스크, 프로토타입, 시작 단계 |
| **페어** | PdM + Engineer | 중규모 정형 태스크 |
| **풀 팀** | PdM + Engineer + Reviewer + Tester | 본격적인 프로젝트 |
| **스케일드** | PdM + 복수 Engineer + 복수 Reviewer + Tester | 대규모·복수 모듈 |

### 판단 기준

- 실패 비용이 높다 → 역할 분리를 늘림
- "구현한 본인이 리뷰" 상태 → Reviewer 분리
- 병렬 작업 가능한 모듈이 많다 → Engineer 추가

---

## 기억 — 5가지를 구분하여 사용

| 기억 유형 | 디렉토리 | 한마디로 | 예 |
|----------|---------|---------|-----|
| **에피소드 기억** | `episodes/` | 언제 무엇을 했나 | "3/15에 Slack API 조사를 했다" |
| **의미 기억** | `knowledge/` | 배운 것 | "Slack API 레이트 제한은 100회/분" |
| **절차 기억** | `procedures/` | 어떻게 하나 | "Gmail 인증 셋업 절차" |
| **스킬** | `skills/` | 실행 가능한 절차서 | 이미지 생성 스킬, 조사 스킬 |
| **단기 기억** | `shortterm/` | 최근 맥락 | 지금 대화의 흐름 |

**Priming (자동 회상)**이 대화나 순찰 때마다 관련 기억을 자동으로 회상하여 시스템 프롬프트에 주입합니다. `search_memory`로 능동적 검색도 가능합니다.

**Consolidation (기억 통합)**이 일간으로 activity_log에서 에피소드를 추출하여 지식으로 승화합니다. 사용하지 않는 기억은 **Forgetting (능동적 망각)**으로 자동 정리됩니다.

→ 상세: `anatomy/memory-system.md`

---

## 비용 최적화 — background_model과 Activity Level

### background_model

Heartbeat / Inbox / Cron은 메인 모델과 별도의 경량 모델로 실행할 수 있습니다.

| 구분 | 사용 모델 | 대상 |
|------|----------|------|
| foreground | 메인 모델 (예: claude-opus-4-6) | Chat (사람과의 대화), TaskExec |
| background | background_model (예: claude-sonnet-4-6) | Heartbeat, Inbox, Cron |

설정: `animaworks anima set-background-model {이름} claude-sonnet-4-6`

### Activity Level

전체 활동 빈도를 10%~400%로 조정 가능. Heartbeat 간격에 직접 영향.

| Activity Level | Heartbeat 간격 (기본 30분 기준) | 용도 |
|---------------|-------------------------------|------|
| 200% | 15분 | 바쁜 시기·활발한 개발 |
| 100% (기본) | 30분 | 일반 운영 |
| 50% | 60분 | 저부하·비용 절감 |
| 30% | 100분 | 야간·휴일 |

**Activity Schedule**로 시간대별 자동 전환도 가능 (예: 9:00-22:00은 100%, 22:00-6:00은 30%).

→ 상세: `reference/operations/model-guide.md`, `operations/heartbeat-cron-guide.md`

---

## 조직의 기본 — 상사·부하·동료

Anima는 `status.json`의 `supervisor` 필드로 계층이 결정됩니다.

| 관계 | 정의 | 커뮤니케이션 |
|------|------|------------|
| **상사** | `supervisor`에 지정된 Anima | 진척 보고 (MUST), 문제 에스컬레이션 |
| **부하** | `supervisor`가 자신인 Anima | `delegate_task`로 위임, `org_dashboard`로 감시 |
| **동료** | 같은 `supervisor`를 가진 Anima | 직접 연락 OK |
| **다른 부서** | 위 어디에도 해당하지 않음 | 자신의 상사를 경유 (직접 연락은 원칙 금지) |
| **사람** | `supervisor: null` (최상위)인 경우 | `call_human`으로 알림 |

→ 상세: `organization/hierarchy-rules.md`, `organization/roles.md`

---

## 막혔을 때의 첫 번째 조치

| 상황 | 할 일 |
|------|------|
| 조작 방법을 모르겠다 | `search_memory(query="키워드", scope="common_knowledge")` |
| 태스크가 블로킹되었다 | `troubleshooting/escalation-flowchart.md` 참조 |
| 도구가 동작하지 않는다 | `troubleshooting/common-issues.md` 참조 |
| 무엇을 해야 할지 모르겠다 | Heartbeat 체크리스트 실행. current_state.md와 task_queue 확인 |
| 판단이 어렵다 | 상사에게 `send_message(intent="question")`으로 상담 |

→ 전체 문서 목차: `common_knowledge/00_index.md`
