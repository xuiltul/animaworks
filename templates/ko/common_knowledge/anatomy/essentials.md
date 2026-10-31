# AnimaWorks 필수 가이드

[IMPORTANT] AnimaWorks의 전체 모습을 한 장으로 파악하기 위한 통합 가이드.
Heartbeat / Cron / 팀 설계 / 기억 / 비용 최적화의 핵심을 망라.
처음 읽는 경우나 개념 간의 관계를 정리하고 싶을 때 먼저 참조할 것.
각 토픽의 상세 내용은 마지막의 링크를 참조.

---

## AnimaWorks란

AI 에이전트를 '도구'가 아닌 **자율적인 인격**으로 운영하는 프레임워크.

- 각 Anima는 고유한 인격·기억·판단 기준을 가짐
- 인간의 지시가 없어도 정기적으로 스스로 행동함 (Heartbeat / Cron)
- 조직 안에서 역할을 맡아 다른 Anima나 인간과 협업함
- 경험에서 배우고 기억을 축적하며 성장함

---

## 5가지 실행 경로 — '언제·어떻게 움직이는가'

Anima는 다음 5가지 경로로 가동한다. Chat 외에는 모두 자동으로 시작된다.

| 경로 | 언제 움직이는가 | 무엇을 하는가 | 누가 사용하는가 |
|------|---------|---------|---------|
| **Chat** | 인간이 메시지를 보냈을 때 | 대화 응답 | 인간 → Anima |
| **Inbox** | 다른 Anima에게서 DM이 왔을 때 | 조직 내 메시지에 즉시 응답 | Anima → Anima |
| **Heartbeat** | 정기 자동 시작 (기본 30분) | 관찰 → 계획 → 돌아보기. **실행은 하지 않음** | 자동 |
| **Cron** | cron.md의 스케줄 (예: 매일 아침 9:00) | 정해진 시간의 확정 작업 실행 | 자동 |
| **TaskExec** | 정규 작업이 실행 가능하고 의존 대상이 완료되었을 때 | 저장된 입력을 가져온 LLM 시도로 실행 | submit_tasks 또는 delegate_task로 등록 |

Chat과 Heartbeat는 **별도 락**으로 움직이므로 순회 중에도 인간의 대화에 즉시 응답할 수 있다.

→ 상세: `anatomy/what-is-anima.md`

---

## Heartbeat vs Cron — 2가지 자율 행동

둘 다 '인간의 지시 없이 움직이는' 구조지만, 목적이 근본적으로 다르다.

| 관점 | Heartbeat (정기 순회) | Cron (정시 작업) |
|------|---------------------|------------------|
| **비유** | 정기적으로 사무실을 순찰하는 경비원 | 매일 아침 9시에 도착하는 신문 배달 |
| **목적** | 상황 확인·계획 수립·돌아보기 | 정해진 업무의 실행 |
| **실행하는가** | **아니요**. 작업을 발견하면 `submit_tasks` 또는 `delegate_task`로 투입만 함 | **예**. LLM형이면 사고와 실행, Command형이면 즉시 실행 |
| **간격** | 고정 간격 (기본 30분, Activity Level에 따라 변동) | cron식으로 유연하게 지정 (매일 9:00, 매주 금요일 17:00 등) |
| **설정 파일** | `HEARTBEAT.md` (체크리스트) | `cron.md` (작업 정의) |
| **전형적 예** | 읽지 않은 메시지 확인, 블로커 감지, 진행 상황 돌아보기 | 아침 업무 계획, 주간 리포트, 백업 실행 |

**판단이 어려울 때**: '확인만 하고 판단하기' → Heartbeat의 체크리스트에 추가. '정해진 시간에 무언가를 하기' → Cron 작업으로 정의.

→ 상세: `operations/heartbeat-cron-guide.md`

---

## Cron의 2가지 유형 — LLM형 vs Command형

Cron 작업에는 사고가 필요한지 여부에 따라 2가지 유형이 있다.

| 관점 | LLM형 (`type: llm`) | Command형 (`type: command`) |
|------|---------------------|---------------------------|
| **비유** | '오늘 무엇을 우선해야 할지 생각해줘' | '매일 아침 이 버튼을 눌러줘' |
| **판단** | 있음 (상황에 따라 출력이 달라짐) | 없음 (매번 같은 것을 실행) |
| **API 비용** | 있음 (LLM 호출) | 명령 자체는 없음. 단, **follow-up LLM이 시작되는 경우 있음** (아래 참조) |
| **출력** | 비정형 (작업마다 다름) | 확정적 (명령의 stdout) |
| **적합한 작업** | 계획 수립, 돌아보기, 문서 작성, 기억 정리 | 백업, 알림 전송, 데이터 획득, 헬스 체크 |

### Command형의 follow-up LLM (중요)

Command형은 명령 자체는 기계적으로 실행하지만, **stdout에 반환값이 있는 경우 기본적으로 LLM이 시작되어 결과를 분석한다** (follow-up). 즉 완전히 비용이 0인 것은 아닐 수 있다.

```
コマンド実行 → stdout あり？
  → なし → 終了（LLM 不要）
  → あり → skip_pattern にマッチする？
      → マッチ → 終了（LLM スキップ）
      → マッチしない → LLM が起動して結果を解釈・対処判断
```

이 follow-up을 제어하는 옵션:
- **`trigger_heartbeat: false`** — follow-up LLM을 항상 스킵 (결과 분석이 필요 없는 경우)
- **`skip_pattern: <正規表現>`** — stdout이 매치되었을 때만 스킵 (정상 시에만 무시, 이상 시에는 LLM에 판단시키기)

### 용도 구분의 판단 기준

```
「毎回同じことをするだけ？」
  → はい、結果も見なくてよい → Command型 + trigger_heartbeat: false
  → はい、ただし異常時だけ判断が必要 → Command型 + skip_pattern（正常パターン）
  → いいえ（状況に応じて判断が変わる） → LLM型
  → コマンド実行 + 毎回結果の解釈が必要 → LLM型（description にコマンド実行を指示）
```

### 기재 예

```markdown
## 毎朝の業務計画（LLM型）
schedule: 0 9 * * *
type: llm
episodes/ から昨日の進捗を確認し、今日のタスクを計画する。

## バックアップ実行（Command型・follow-up不要）
schedule: 0 2 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/backup.sh

## 監視チェック（Command型・正常時はスキップ、異常時はLLMが判断）
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

→ 상세: `operations/heartbeat-cron-guide.md`

---


## 작업을 흘려보내는 방법 — submit_tasks vs delegate_task

작업을 실행으로 옮기는 방법은 2가지가 있다.

| 관점 | `submit_tasks` | `delegate_task` |
|------|---------------|----------------|
| **누가 실행하는가** | **자기 자신**의 TaskExec 경로 | **직속 부하** |
| **사용하는 상황** | 자신이 해야 할 작업을 비동기 실행하고 싶을 때 | 부하에게 위임하고 싶을 때 |
| **DAG/병렬** | `parallel: true`로 병렬, `depends_on`로 의존 | 1건씩 위임 |
| **진행 상황 추적** | 정규 작업 스토어를 참조하는 `list_tasks` / TaskBoard | `task_tracker`로 추적 |
| **전형적 예** | Heartbeat에서 발견한 작업을 자신이 실행 | 상급자가 부하에게 작업을 위임 |

**판단 플로우**:
```
「このタスクは部下がやるべき？」
  → はい、直属の部下がいる → delegate_task
  → いいえ、自分でやる → submit_tasks
  → 部下がいない → submit_tasks
```

→ 상세: `operations/task-management.md`, `anatomy/task-architecture.md`

---

## 팀 설계 — 솔로에서 시작해 스케일하기

### 왜 역할을 나누는가

| 이유 | 설명 |
|------|------|
| 컨텍스트 오염 방지 | 전체 공정을 한 사람이 맡으면 컨텍스트가 비대해져 판단 정확도가 떨어짐 |
| 품질의 구조적 보장 | 실행하는 자와 검증하는 자를 분리 (셀프 리뷰의 맹점을 배제) |
| 병렬 실행 | 독립된 역할은 동시 실행으로 처리량 향상 |
| 전문성의 심화 | 역할 고유의 체크리스트·기억으로 범용 에이전트보다 높은 품질 |

### 스케일링의 단계

| 규모 | 구성 | 언제 사용하는가 |
|------|------|---------|
| **솔로** | 1 Anima가 전체 역할 겸임 | 소규모 작업, 프로토타입, 시작한 지 얼마 안 됨 |
| **페어** | PdM + Engineer | 중간 규모의 정형 작업 |
| **풀 팀** | PdM + Engineer + Reviewer + Tester | 본격적인 프로젝트 |
| **스케일드** | PdM + 복수 Engineer + 복수 Reviewer + Tester | 대규모·복수 모듈 |

### 판단의 기준

- 실패 비용이 높음 → 역할 분리를 늘림
- '구현한 본인이 리뷰'가 되고 있음 → Reviewer를 분리
- 병렬 작업이 가능한 모듈이 많음 → Engineer를 늘림

---

## 기억 — 5가지 종류를 구분해 사용

| 기억의 종류 | 디렉터리 | 한마디로 | 예 |
|-----------|------------|--------|-----|
| **에피소드 기억** | `episodes/` | 언제 무엇을 했는가 | '3/15에 Slack API 조사를 했다' |
| **의미 기억** | `knowledge/` | 배운 것 | 'Slack API의 레이트 제한은 100회/분' |
| **절차 기억** | `procedures/` | 어떻게 하는가 | 'Gmail 인증의 설정 절차' |
| **스킬** | `skills/` | 실행 가능한 절차서 | 이미지 생성 스킬, 조사 스킬 |
| **단기 기억** | `shortterm/` | 최근의 맥락 | 지금 대화의 흐름 |

**Priming (자동 회상)** 이 대화나 순회 때마다 관련 기억을 자동으로 회상하고, 필요한 맥락을 시스템 프롬프트에 주입한다. 추가로 `search_memory`로 능동적으로 검색도 가능하다.

**Consolidation (기억 통합)** 은 새로운 활동 청크만 에피소드화하고, 변화 없는 재실행에서는 생성하지 않는다. 지식의 자동 변경·주간 월간 정리·스킬 자동 학습은 기본적으로 비활성화. 기억 저장과 필요 시 검색은 남는다.

→ 상세: `anatomy/memory-system.md`

---

## 비용 최적화 — background_model과 Activity Level

### background_model

Heartbeat / Cron은 명시적으로 설정된 background_model을 사용할 수 있다. Inbox는 메인 모델을 사용하며, 작업별 모델 지정은 해당 작업에서 우선된다. 저렴하다는 이유만으로 자동 선택하지 않는다.

| 구분 | 사용 모델 | 대상 |
|------|-----------|------|
| foreground | 메인 모델, 또는 명시된 작업별 모델 | Chat, Inbox, TaskExec |
| background | 명시 설정된 background_model, 미설정 시 메인 모델 | Heartbeat, Cron |

설정: `animaworks anima set-background-model {名前} claude-sonnet-5-5`

### Activity Level

전체 활동 빈도를 10%~400%로 조정할 수 있다. Heartbeat 간격에 직접 영향을 준다.

| Activity Level | Heartbeat 간격 (기본 30분 기준) | 용도 |
|---------------|-------------------------------|------|
| 200% | 15분 | 바쁜 시기·활발한 개발 |
| 100% (기본) | 30분 | 일반 운영 |
| 50% | 60분 | 저부하·비용 절약 |
| 30% | 100분 | 야간·휴일 |

**Activity Schedule**로 시간대별 자동 전환도 가능하다 (예: 9:00-22:00는 100%, 22:00-6:00는 30%).

→ 상세: `reference/operations/model-guide.md`, `operations/heartbeat-cron-guide.md`

---

## 조직의 기본 — 상급자·부하·동료

Anima는 `status.json`의 `supervisor` 필드로 계층이 결정된다.

| 관계 | 정의 | 커뮤니케이션 |
|------|------|------------------|
| **상급자** | `supervisor`에 지정된 Anima | 진행 상황 보고 (MUST), 문제 에스컬레이션 |
| **부하** | `supervisor`이 자신의 Anima | `delegate_task`로 위임, `org_dashboard`로 모니터링 |
| **동료** | 같은 `supervisor`을 가진 Anima | 직접 연락 OK |
| **타 부서** | 위 어느 것에도 해당하지 않음 | 자신의 상급자 경유 (직접 연락은 원칙적으로 금지) |
| **인간** | `supervisor: null` (최상위 레벨)인 경우 | `call_human`로 알림 |

→ 상세: `organization/hierarchy-rules.md`, `organization/roles.md`

---

## 막혔을 때의 첫 번째 조치

| 상황 | 할 일 |
|------|---------|
| 조작 방법을 모르겠음 | `search_memory(query="キーワード", scope="common_knowledge")` |
| 작업이 블록되었음 | `troubleshooting/escalation-flowchart.md`을 참조 |
| 도구가 작동하지 않음 | `troubleshooting/common-issues.md`을 참조 |
| 무엇을 해야 할지 모르겠음 | current_state.md과 `list_tasks`을 확인. 필요하면 설정된 Heartbeat 체크리스트를 참조 |
| 판단이 어려움 | 상급자에게 `send_message(intent="question")`로 상담 |

→ 전체 문서 목차: `common_knowledge/00_index.md`
