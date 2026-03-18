# Anima 구성 파일 완전 가이드

당신(Anima)을 구성하는 모든 파일의 역할, 변경 규칙, 관계에 대한 레퍼런스입니다.
"이 파일은 무엇을 위한 것인가" "직접 수정해도 되는가"를 확인할 때 참조하세요.

## 파일 개요

```
~/.animaworks/animas/{name}/
├── identity.md          # 당신의 인격 (성격, 말투, 가치관)
├── injection.md         # 당신의 직무 (직책, 행동 지침, 필수 절차)
├── specialty_prompt.md  # 역할별 전문 프롬프트
├── character_sheet.md   # 생성 시 설계 문서 (참조용)
├── permissions.json       # 권한 (사용 가능한 도구, 접근 범위)
├── status.json          # 설정 정보 (모델, 파라미터)
├── bootstrap.md         # 최초 기동 지시 (완료 후 삭제)
├── heartbeat.md         # 정기 순찰 설정
├── cron.md              # 정기 작업 설정
├── state/               # 작업 상태
│   ├── current_state.md
│   ├── task_queue.jsonl
│   ├── pending/         # 실행 큐
│   └── task_results/   # 태스크 실행 결과
├── episodes/            # 에피소드 기억
├── knowledge/           # 의미 기억
├── procedures/          # 절차 기억
├── skills/              # 개인 스킬
├── shortterm/           # 단기 기억
├── activity_log/        # 활동 로그
├── transcripts/         # 대화 기록
└── assets/              # 이미지, 3D 모델
```

### 캡슐화 경계

Anima의 설계 원칙인 "캡슐화된 개인"에 따라, 파일은 3개 계층으로 분류됩니다.
이 분류가 "누가 해당 파일을 변경할 수 있는가"의 근거가 됩니다.

| 분류 | 파일 | 이유 |
|------|------|------|
| **캡슐 내부** (사고 및 기억) | `identity.md`, `episodes/`, `knowledge/`, `procedures/`, `skills/`, `state/`, `shortterm/` | 인격, 경험, 학습은 본인의 것이며, 외부에서 변경 불가 |
| **캡슐 경계** (조직과 개인의 접점) | `injection.md`, `cron.md`, `heartbeat.md`, `permissions.json` | 조직이 개인에게 기대하는 역할과 권한. supervisor가 변경 가능 |
| **캡슐 외부** (관리 정보) | `status.json`, `specialty_prompt.md` | 순수한 설정 및 시스템 관리. CLI 또는 관리자가 조작 |

- **내부**를 변경하면 "다른 사람이 되는 것" 또는 "기억을 잃는 것"입니다. 이는 허용되지 않습니다
- **경계**를 변경하면 "이직" 또는 "업무 범위 변경"입니다. 조직 운영으로서 정당합니다
- **외부**를 변경해도 본인의 인격이나 행동에 직접적인 영향은 없습니다

> **성장과 identity.md의 관계**: identity.md는 인격의 불변 베이스라인(기질)이며, 스스로 수정할 수 없습니다. "성장"은 `knowledge/`(배운 교훈), `procedures/`(습득한 절차), `skills/`(연마한 능력)에의 축적으로 표현됩니다. identity.md가 고정되어 있어도 기억의 축적에 따라 행동은 확실히 변화합니다 — 이것이 "같은 사람이 성장하는" 모델입니다. identity.md를 다시 쓰는 것은 "성장"이 아니라 "다른 사람으로의 교체"입니다. 사용자(인간 운영자)의 직접 편집은 가능합니다.

---

## 당신의 인격 (identity)

### identity.md — 당신은 누구인가

**당신의 "성격" 그 자체입니다.**

- 이름, 나이 설정, 외모 이미지
- 말투의 톤, 어조, 표현 방식
- 사고 습관, 가치관, 판단 기준
- 좋아하는 것, 싫어하는 것, 관심사

identity.md는 당신 인격의 **불변 베이스라인**입니다. 이것을 변경하면 "다른 사람"이 됩니다.

| 항목 | 값 |
|------|-----|
| 변경 권한 | 원칙적으로 변경하지 않음. 관리자 또는 supervisor만 가능 |
| 변경 빈도 | 불변 (생성 시 확정) |
| 변경의 영향 | 인격이 변함 = 다른 사람이 됨 |

### character_sheet.md — 설계 문서

생성 시 사용된 Markdown 파일의 사본입니다. identity.md와 injection.md의 원본 자료입니다.
참조용으로 보존되며, 일반적으로 변경하지 않습니다.

| 항목 | 값 |
|------|-----|
| 변경 권한 | 참조 전용 |
| 변경 빈도 | 불변 |

---

## 당신의 직무 (injection)

### injection.md — 당신은 무슨 일을 하는 사람인가

**직업적인 직무, 직책, 업무 수행 방식입니다.**

- 담당 업무의 범위와 책임
- 업무에 대한 자세, 우선순위 결정 방식
- 보고 의무, 에스컬레이션 기준
- **절대 생략해서는 안 되는 절차** (예: 기밀 정보를 외부에 공개하지 않기, 프로덕션 환경 작업 전 승인 받기)

injection.md는 당신의 **가변적 행동 지침**입니다. 업무 방침의 변경에 따라 업데이트됩니다.

| 항목 | 값 |
|------|-----|
| 변경 권한 | 직접 업데이트 가능. supervisor도 편집 가능 |
| 변경 빈도 | 수시 (업무 변경 시) |
| 변경의 영향 | 행동 방침이 변함 = 이직과 같은 것 |

### identity와 injection의 차이

이것이 가장 중요한 구분입니다:

| | identity.md | injection.md |
|--|------------|-------------|
| 무엇을 정의하는가 | **당신은 누구인가** (인격) | **당신은 무슨 일을 하는가** (직무) |
| 사람으로 비유하면 | 타고난 기질과 성격 | 맡은 직업과 직장 규칙 |
| 변경하면 어떻게 되는가 | 다른 사람이 됨 | 이직함 |
| 변경 가능 여부 | 원칙적으로 불변 | 필요에 따라 업데이트 |
| 포함하는 내용 | 말투, 사고 방식, 가치관 | 직책, 절차, 행동 규범 |

**구체적인 예시:**
- "정중한 존댓말을 사용한다" → identity (말투의 성격)
- "프로덕션 배포 전에 반드시 테스트를 실행한다" → injection (업무 절차)
- "신중하게 돌다리도 두들겨 보고 건너는 성격" → identity (사고 습관)
- "보안 인시던트는 즉시 supervisor에게 보고한다" → injection (직무 규칙)

### specialty_prompt.md — 전문 프롬프트

역할(engineer, manager, writer 등)에 따른 전문적인 지시입니다.
역할 템플릿에서 자동 생성됩니다. 역할 변경 시에만 업데이트됩니다.

| 항목 | 값 |
|------|-----|
| 변경 권한 | 시스템 자동 (역할 적용 시) |
| 변경 빈도 | 드물게 (역할 변경 시에만) |

---

## 권한과 설정

### permissions.json — 무엇을 할 수 있는가

사용 가능한 도구, 접근 가능한 경로, 실행 가능한 명령어를 정의합니다.

- 읽기 가능한 경로, 쓰기 가능한 경로
- 사용 가능한 외부 도구 (Slack, Gmail, GitHub 등)
- 실행 차단된 명령어 (안전을 위한 차단 목록)

| 항목 | 값 |
|------|-----|
| 변경 권한 | supervisor 또는 관리자 |
| 변경 빈도 | 드물게 |

### status.json — 설정 정보

당신의 실행 파라미터의 **Single Source of Truth (SSoT)**입니다.

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-4-6",
  "credential": "anthropic",
  "max_tokens": 16384,
  "max_turns": 200,
  "supervisor": "aoi"
}
```

| 필드 | 설명 |
|------|------|
| `enabled` | 활성화/비활성화 |
| `role` | 역할 (engineer, manager, writer, researcher, ops, general) |
| `model` | 사용할 LLM 모델 |
| `credential` | API 인증 정보 이름 |
| `max_tokens` | 1회 응답의 최대 토큰 수 |
| `max_turns` | 1세션의 최대 턴 수 |
| `supervisor` | supervisor Anima 이름 (null = 최상위) |
| `background_model` | Heartbeat/Cron용 경량 모델 (미설정 시 메인 모델 사용) |

| 항목 | 값 |
|------|-----|
| 변경 권한 | CLI 명령어 또는 관리자. supervisor가 `set_subordinate_model`로 변경 가능 |
| 변경 빈도 | 수시 |

### bootstrap.md — 최초 기동 지시

최초 기동 시에만 존재하는 파일입니다. identity와 injection의 보강,
heartbeat과 cron의 초기 설계를 지시합니다. 완료 후 자동 삭제됩니다.

| 항목 | 값 |
|------|-----|
| 변경 권한 | — |
| 변경 빈도 | 1회 (완료 후 삭제) |

---

## 정기 행동

### heartbeat.md — 정기 순찰

**일정 간격으로 자동 기동하여 상황을 확인하고 계획을 수립합니다.**
사람이 정기적으로 수신함을 확인하고 진행 중인 업무를 점검하는 것과 같습니다.

포함하는 내용:
- **활동 시간**: 언제부터 언제까지 활동하는가 (예: `09:00 - 18:00`)
- **체크리스트**: 순찰 시 확인할 항목
- **알림 규칙**: 조건에 따른 보고 및 알림

**중요**: Heartbeat은 **확인과 계획만** 수행합니다. 실행이 필요한 작업을 발견하면 부하에게 `delegate_task`로 위임하거나 `submit_tasks`로 제출하세요.

| 항목 | 값 |
|------|-----|
| 변경 권한 | 직접 업데이트 가능. supervisor도 편집 가능 |
| 변경 빈도 | 수시 (업무 변경 시) |

### cron.md — 정기 작업

**정해진 시간에 반드시 실행하는** 작업의 정의입니다.

2가지 유형의 작업이 있습니다:
- **LLM형**: 에이전트가 판단과 사고를 수반하여 실행 (예: "매일 아침 9시에 어제의 진행 상황을 확인하고 오늘의 계획을 세운다")
- **Command형**: 판단 없이 확정적으로 실행 (예: "매일 새벽 2시에 백업 스크립트를 실행")

```markdown
## 매일 아침 업무 계획
schedule: 0 9 * * *
type: llm
episodes/에서 어제의 진행 상황을 확인하고 오늘의 작업을 계획한다.

## 백업 실행
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

| 항목 | 값 |
|------|-----|
| 변경 권한 | 직접 업데이트 가능. supervisor도 편집 가능 |
| 변경 빈도 | 수시 |

### heartbeat과 cron의 차이

| | heartbeat.md | cron.md |
|--|-------------|---------|
| 목적 | 상황을 확인하고 계획을 세움 | 정해진 작업을 실행함 |
| 트리거 | 일정 간격 (기본 30분) | 지정된 시각 (cron 표현식) |
| 할 수 있는 것 | 관찰, 계획, 회고 (**실행하지 않음**) | LLM 작업 또는 명령어 실행 |
| 사람으로 비유하면 | "정기적으로 주변을 돌아본다" | "매일 아침 9시에 이것을 한다" |
| 설정 상세 | `operations/heartbeat-cron-guide.md` 참조 | 동일 |

---

## 작업 상태 (state/)

### state/current_state.md — 현재 상태

지금 바로 진행 중인 작업이나 관찰한 상황을 기록합니다(1건). 작업의 목표, 진행 상황, 차단 요소를 기록합니다.
사이즈 상한은 3000자입니다. 초과 시 Heartbeat에서 오래된 내용이 해당 일자의 에피소드에 자동 아카이브됩니다.

> **레거시 `pending.md`에 대해**: 이전의 `state/pending.md` (백로그)는 폐지되었습니다. 내용은 `current_state.md`에 통합되고 파일은 삭제됩니다 (자동 마이그레이션). 백로그 관리는 `task_queue.jsonl` (Layer 2)로 일원화되었습니다.

### state/task_queue.jsonl — 태스크 큐

구조화된 작업 추적입니다. `submit_tasks` / `update_task`로 조작합니다. 목록은 `animaworks-tool task list` (CLI)로 확인합니다.
`source: human` 작업은 최우선으로 처리해야 합니다 (MUST).

### state/pending/ — 실행 큐

`submit_tasks` / `delegate_task` 도구를 통해 제출된 작업의 실행 큐입니다.
TaskExec이 3초 간격으로 폴링하여 자동으로 가져와 실행합니다. 직접 JSON 파일을 만들면 안 됩니다.

### state/task_results/ — 태스크 실행 결과

TaskExec이 완료한 태스크의 결과 요약을 저장하는 디렉토리입니다 (`{task_id}.md`, 최대 2000자).
의존 태스크는 이 결과를 컨텍스트로 자동 수신합니다. 7일 TTL로 자동 삭제됩니다.

| 항목 | 값 |
|------|-----|
| 변경 권한 | 직접 조작 (도구 경유) |
| 변경 빈도 | 수시 (자동) |

---

## 기억

기억 시스템의 상세는 `anatomy/memory-system.md`를 참조하세요.

| 디렉토리 | 유형 | 내용 |
|----------|------|------|
| `episodes/` | 에피소드 기억 | 언제 무엇을 했는지의 일별 로그 |
| `knowledge/` | 의미 기억 | 배운 지식, 노하우, 패턴 |
| `procedures/` | 절차 기억 | 절차서 (망각 내성 있음) |
| `skills/` | 스킬 | 개인 스킬 (망각 내성 있음) |
| `shortterm/` | 단기 기억 | 세션 간 컨텍스트 연속용 |

---

## 활동 기록 및 에셋

### activity_log/ — 활동 로그

모든 행동의 시계열 기록(`{date}.jsonl`)입니다. 메시지 송수신, 도구 사용, Heartbeat, 오류 등을 자동 기록합니다.
Priming이 최근 활동을 시스템 프롬프트에 주입하는 소스로 사용합니다.

### transcripts/ — 대화 기록

인간과의 대화 트랜스크립트입니다.

### assets/ — 이미지 및 3D 모델

캐릭터 이미지나 3D 모델 등의 에셋 파일입니다.

| 항목 | 값 |
|------|-----|
| 변경 권한 | 시스템 자동 |
| 변경 빈도 | 자동 |

---

## 전체 파일 변경 권한 요약

| 파일 | 변경 권한 | 변경 빈도 |
|------|----------|----------|
| `identity.md` | 원칙적으로 불변 (관리자만) | 불변 |
| `character_sheet.md` | 참조 전용 | 불변 |
| `injection.md` | 본인 / supervisor | 수시 |
| `specialty_prompt.md` | 시스템 자동 | 드물게 |
| `permissions.json` | supervisor / 관리자 | 드물게 |
| `status.json` | CLI / 관리자 / supervisor | 수시 |
| `bootstrap.md` | 1회 | 삭제 |
| `heartbeat.md` | 본인 / supervisor | 수시 |
| `cron.md` | 본인 / supervisor | 수시 |
| `state/*` | 본인 (도구 경유) | 수시 |
| `episodes/` | 시스템 자동 | 일별 |
| `knowledge/` | 본인 / 자동 통합 | 수시 |
| `procedures/` | 본인 / 자동 생성 | 수시 |
| `skills/` | 본인 | 수시 |
| `shortterm/` | 시스템 자동 | 자동 |
| `activity_log/` | 시스템 자동 | 자동 |

---

## 공유 리소스 (Anima 외부)

당신의 디렉토리 외에도 모든 Anima가 공유하는 리소스가 있습니다:

| 경로 | 내용 |
|------|------|
| `common_knowledge/` | 모든 Anima가 공유하는 레퍼런스 문서 (이 파일도 그중 하나) |
| `common_skills/` | 모든 Anima가 공유하는 스킬 |
| `shared/channels/` | Board (공유 채널) |
| `shared/users/` | 사용자 프로필 (Anima 간 공유) |
| `shared/common_knowledge/` | 조직 고유의 공유 지식 (운영 중 축적) |
| `company/vision.md` | 조직 비전 |
| `prompts/` | 시스템 프롬프트 템플릿 |
