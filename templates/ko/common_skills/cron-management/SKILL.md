---
name: cron-management
description: >-
  cron.md를 올바른 형식으로 읽고 쓰는 스킬. 정기 태스크 추가·수정·삭제 절차를 제공한다.
  Use when: cron.md 편집, cron 식 추가, LLM·커맨드형 작업 추가·삭제, 정기 작업 점검이 필요할 때.
---

## cron.md 구조

### 전체 구성

```markdown
# Cron: {자신의_이름}

## 태스크명 1
schedule: 0 9 * * *
type: llm
태스크 설명...

## 태스크명 2
schedule: */5 * * * *
type: command
command: /path/to/script.sh
```

### 반드시 지켜야 할 규칙

1. **각 태스크는 `## 태스크명`으로 시작** (H2 헤더. H3이나 H1은 사용하지 않음)
2. **`schedule:` 행은 필수** (`schedule:` 키워드로 시작. `###` 헤더가 아님)
3. **스케줄은 표준 5필드 cron 표현식만 가능** (`09:00`이나 `매주 금요일 17:00`은 불가)
4. **`type:` 행은 필수** (`llm` 또는 `command`)
5. **태스크 사이에 빈 줄을 삽입** (가독성을 위해)

### 잘못된 형식

```markdown
❌ ### */5 * * * *           ← H3 헤더에 cron 표현식을 쓰면 안 됨
❌ ### 09:00                 ← 자연어 시간 표기는 불가
❌ ### 매주 금요일 17:00      ← 한국어 스케줄 표기는 불가
❌ cron: 0 9 * * *           ← 키 이름은 "schedule:"이어야 함 ("cron:"이 아님)
❌ interval: 5m              ← interval 형식은 불가
❌ schedule: 0 9 * * * *     ← 6필드는 불가 (5필드만 가능)
```

### 올바른 형식

```markdown
✅ schedule: 0 9 * * *       ← "schedule:" + 스페이스 + 5필드 cron 표현식
✅ schedule: */5 * * * *
✅ schedule: 30 21 * * *
✅ schedule: 0 17 * * 5
```

---

## 5필드 cron 표현식 레퍼런스

### 필드 구성

```
schedule: 분 시 일 월 요일
```

| 필드 | 위치 | 범위 | 설명 |
|------|------|------|------|
| 분 | 1 | 0-59 | 몇 분에 실행할지 |
| 시 | 2 | 0-23 | 몇 시에 실행할지 (24시간제) |
| 일 | 3 | 1-31 | 며칠에 실행할지 |
| 월 | 4 | 1-12 | 몇 월에 실행할지 |
| 요일 | 5 | 0-6 | 무슨 요일에 실행할지 (0=월, 6=일) |

**주의**: 요일은 **0=월요일, 6=일요일** (APScheduler 사양. 일반적인 cron의 0=일요일과 다름)

### 특수 문자

| 문자 | 의미 | 예시 |
|------|------|------|
| `*` | 모든 값 | `* * * * *` = 매분 |
| `*/n` | n 간격 | `*/5 * * * *` = 5분마다 |
| `n-m` | 범위 | `0 9-17 * * *` = 9시~17시 매 정시 |
| `n,m` | 리스트 | `0 9,12,18 * * *` = 9시, 12시, 18시 |
| `n-m/s` | 범위+간격 | `0 9-17/2 * * *` = 9시~17시 2시간마다 |

### 자주 사용하는 스케줄 예시

#### 매일

| 목적 | cron 표현식 | 설명 |
|------|------------|------|
| 매일 오전 9:00 | `0 9 * * *` | 분=0, 시=9 |
| 매일 오전 9:30 | `30 9 * * *` | 분=30, 시=9 |
| 매일 정오 | `0 12 * * *` | 분=0, 시=12 |
| 매일 오후 6:00 | `0 18 * * *` | 분=0, 시=18 |
| 매일 오후 9:30 | `30 21 * * *` | 분=30, 시=21 |
| 매일 새벽 2:00 | `0 2 * * *` | 분=0, 시=2 |

#### 간격

| 목적 | cron 표현식 | 설명 |
|------|------------|------|
| 5분마다 | `*/5 * * * *` | 분=*/5 (0,5,10,...,55) |
| 10분마다 | `*/10 * * * *` | 분=*/10 |
| 15분마다 | `*/15 * * * *` | 분=*/15 |
| 30분마다 | `*/30 * * * *` | 분=*/30 |
| 1시간마다 | `0 * * * *` | 매시 0분 |
| 2시간마다 | `0 */2 * * *` | 0시, 2시, ..., 22시 |
| 업무 시간 중 5분마다 | `*/5 9-17 * * *` | 9:00~17:55 |
| 업무 시간 중 1시간마다 | `0 9-17 * * *` | 9:00~17:00 |

#### 요일

| 목적 | cron 표현식 | 설명 |
|------|------------|------|
| 평일 오전 9:00 | `0 9 * * 0-4` | 요일=0-4 (월~금) |
| 매주 월요일 9:00 | `0 9 * * 0` | 요일=0 (월) |
| 매주 금요일 오후 5:00 | `0 17 * * 4` | 요일=4 (금) |
| 매주 금요일 오후 6:00 | `0 18 * * 4` | 요일=4 (금) |
| 평일 업무 시간 30분마다 | `*/30 9-17 * * 0-4` | 월~금 9:00~17:30 |
| 주말 오전 10:00 | `0 10 * * 5,6` | 요일=5,6 (토, 일) |

#### 월별

| 목적 | cron 표현식 | 설명 |
|------|------------|------|
| 매월 1일 오전 9:00 | `0 9 1 * *` | 일=1 |
| 매월 15일 정오 | `0 12 15 * *` | 일=15 |
| 매월 말일 근처(28일) 오후 5:00 | `0 17 28 * *` | 일=28 (근사치) |
| 분기 시작일 오전 9:00 (1,4,7,10월) | `0 9 1 1,4,7,10 *` | 월=1,4,7,10 |

---

## 태스크 유형 상세

### type: llm — LLM 판단 태스크

사고와 판단이 필요한 태스크. `schedule:`과 `type: llm` 뒤에 자유롭게 태스크 내용을 작성합니다.

```markdown
## 매일 아침 업무 계획
schedule: 0 9 * * *
type: llm
장기 기억에서 어제의 진행 상황을 확인하고 오늘의 태스크를 계획합니다.
비전과 목표에 비추어 우선순위를 판단합니다.
결과를 state/current_state.md에 기록합니다.
```

- 설명 텍스트가 그대로 LLM 프롬프트로 전달됩니다
- 구체적인 출력(무엇을 기록할지)을 명시하면 효과적입니다
- 여러 줄 가능

### type: command — 커맨드 실행 태스크

결정적으로 실행할 bash 커맨드나 도구 호출입니다.

#### 패턴 A: bash 커맨드

```markdown
## 백업 실행
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- `command:`에 실행할 커맨드를 한 줄로 작성합니다
- 셸 리디렉션 (`>`, `>>`, `|`)을 사용할 수 있습니다
- 여러 줄 커맨드는 비권장 (한 줄로 통합하거나 스크립트 파일을 사용하세요)

#### 패턴 B: 도구 호출

```markdown
## Slack 아침 인사
schedule: 0 9 * * 0-4
type: command
tool: slack_send
args:
  channel: "#general"
  message: "좋은 아침입니다!"
```

- `tool:`에 도구명 (`get_tool_schemas()`의 스키마명)을 작성합니다
- `args:` 이하는 YAML 블록 형식으로 2스페이스 인덴트
- `dispatch(tool_name, args)`를 통해 실행됩니다

### 옵션: skip_pattern

커맨드의 표준 출력이 특정 패턴에 매칭될 때 LLM 분석 세션을 건너뜁니다.

```markdown
## Chatwork 미회신 확인
schedule: */5 * * * *
type: command
command: chatwork_cli.py unreplied --json
skip_pattern: ^\[\]$
```

- `skip_pattern:`에는 정규 표현식을 작성합니다
- 위 예시에서는 미회신이 0건(`[]`)인 경우 건너뜁니다

### 옵션: trigger_heartbeat

커맨드 출력 시 LLM 분석 세션을 트리거할지를 태스크 단위로 제어합니다.

```markdown
## Chatwork 미회신 확인
schedule: */15 * * * *
type: command
command: animaworks-tool chatwork unreplied
skip_pattern: ^\[\]$
trigger_heartbeat: false
```

- `trigger_heartbeat: false` — 커맨드 출력이 있어도 LLM 분석 세션을 건너뜀
- `trigger_heartbeat: true` (기본값) — 커맨드 출력이 있으면 cron LLM 세션에서 분석 및 대응
- `false`, `no`, `0`을 지정하면 LLM 분석 억제. 그 외는 true 처리
- `skip_pattern`과 병용 가능. `trigger_heartbeat: false`는 `skip_pattern`보다 먼저 평가됨
- LLM 분석 세션은 heartbeat 동등의 컨텍스트(기억, Knowledge, 조직 정보)를 가짐

---

## 유형 선택 판단 기준

### type: command를 사용할 경우
- 실행할 커맨드가 완전히 확정된 경우
- 파라미터가 고정 (리전, 클러스터명, 프로파일 등)
- 결과 판단은 cron LLM 세션에 맡김

### type: llm을 사용할 경우
- 상황에 따라 실행 내용을 바꿔야 하는 경우
- 여러 도구를 조합한 조사가 필요한 경우
- 인간적인 판단 및 분석이 실행 단계에서 필요한 경우

### 금지 패턴
- type: llm에 확정 커맨드를 포함 → 해당 커맨드는 type: command로 해야 함
- "이대로 실행하라"고 작성해 놓고 type: llm 사용 → LLM은 커맨드를 정확히 재현할 수 없음. type: command를 사용하세요

### Anima는 "커맨드를 외우는 사람"이 아닙니다
type: command는 사람이 스크립트를 저장하는 것과 같습니다.
Anima의 가치는 결과를 보고 판단하는 능력에 있습니다.
확정적인 실행은 프레임워크에 맡기고,
Anima에게는 판단, 분석, 보고에 집중시키세요.

---

## cron.md 조작 절차

### 새 태스크 추가

1. 자신의 `cron.md`를 읽어들입니다
2. 파일 끝에 새 섹션을 추가합니다
3. **작성 전에 형식을 확인합니다** (아래 체크리스트 참조)
4. 파일을 저장합니다

```markdown
## 새 태스크명
schedule: <5필드 cron 표현식>
type: llm|command
<설명 또는 command/tool 행>
```

### 기존 태스크 변경

1. `cron.md`를 읽어들입니다
2. 해당 섹션 (`## 태스크명`에서 다음 `##` 직전까지)을 찾습니다
3. 변경할 행 (`schedule:`, `type:`, 설명 등)을 편집합니다
4. 파일을 저장합니다

### 태스크 삭제

1. `cron.md`를 읽어들입니다
2. 해당 섹션 전체 (`## 태스크명`에서 다음 `##` 직전까지)를 삭제합니다
3. 파일을 저장합니다

### 태스크 일시 비활성화

HTML 주석으로 감싸면 파서가 건너뜁니다:

```markdown
<!--
## 일시 정지 중인 태스크
schedule: 0 9 * * *
type: llm
이 태스크는 일시적으로 정지 중입니다.
-->
```

---

## 작성 전 체크리스트

cron.md를 업데이트하기 전에 다음을 **반드시** 확인하세요:

- [ ] 각 태스크가 `## 태스크명`으로 시작하는가 (`###`이나 `#`이 아닌지)
- [ ] `schedule:` 행이 있는가 (`###` 헤더나 자연어가 아닌지)
- [ ] 스케줄이 5필드 cron 표현식인가 (`분 시 일 월 요일`)
- [ ] 각 필드 값이 유효 범위 내인가 (분: 0-59, 시: 0-23, 일: 1-31, 월: 1-12, 요일: 0-6)
- [ ] `type:` 행이 있는가 (`llm` 또는 `command`)
- [ ] command 유형의 경우 `command:` 또는 `tool:`이 있는가
- [ ] tool 유형의 경우 `args:` 인덴트가 올바른가 (2스페이스)
- [ ] 태스크 사이에 빈 줄이 있는가

### 검증 방법

작성 후 다음 커맨드로 올바르게 파싱되는지 확인할 수 있습니다:

```bash
python -c "
from core.schedule_parser import parse_cron_md, parse_schedule
import pathlib

content = pathlib.Path('$ANIMAWORKS_ANIMA_DIR/cron.md').read_text()
tasks = parse_cron_md(content)
for t in tasks:
    trigger = parse_schedule(t.schedule)
    status = '✅' if trigger else '❌ 파싱 실패'
    print(f'{status} {t.name}: schedule=\"{t.schedule}\" type={t.type}')
"
```

모든 태스크에 ✅가 표시되면 정상입니다. ❌가 나오면 스케줄 표현식을 수정하세요.

---

## 전체 기술 예시

```markdown
# Cron: example_anima

## 매일 아침 업무 계획
schedule: 0 9 * * *
type: llm
장기 기억에서 어제의 진행 상황을 확인하고 오늘의 태스크를 계획합니다.
비전과 목표에 비추어 우선순위를 판단합니다.
결과를 state/current_state.md에 기록합니다.

## Chatwork 미회신 확인
schedule: */5 9-18 * * 0-4
type: command
command: chatwork-cli unreplied --json > $ANIMAWORKS_ANIMA_DIR/state/chatwork_unreplied.json
skip_pattern: ^\[\]$
trigger_heartbeat: false

## Slack 아침 인사
schedule: 0 9 * * 0-4
type: command
tool: slack_send
args:
  channel: "#general"
  message: "좋은 아침입니다! 오늘도 잘 부탁드립니다."

## 주간 회고
schedule: 0 17 * * 4
type: llm
이번 주의 episodes/를 다시 읽고 패턴을 추출하여 knowledge/에 통합합니다.
개선점이 있으면 procedures/에 절차를 추가합니다.

## 월간 리포트
schedule: 0 10 1 * *
type: llm
지난달의 episodes/와 knowledge/를 분석하여 월간 요약 리포트를 작성합니다.
리포트는 knowledge/monthly_report_YYYY-MM.md로 저장합니다.
```

---

## 자주 하는 실수와 수정

| 실수 | 올바른 형식 | 원인 |
|------|-----------|------|
| `### */5 * * * *` | `schedule: */5 * * * *` | H3 헤더는 태스크 구분에 사용할 수 없음 |
| `### 09:00` | `schedule: 0 9 * * *` | 자연어 시간은 파싱 불가 |
| `### 매주 금요일 17:00` | `schedule: 0 17 * * 4` | 한국어 표기는 파싱 불가 |
| `schedule: 9:00` | `schedule: 0 9 * * *` | HH:MM 형식은 5필드 cron이 아님 |
| `schedule: every 5 minutes` | `schedule: */5 * * * *` | 영어 표기는 파싱 불가 |
| `schedule: 0 9 * * 7` | `schedule: 0 9 * * 6` | 요일 7은 범위 밖 (0-6) |
| `schedule: 0 9 * * SUN` | `schedule: 0 9 * * 6` | 요일명은 사용 불가 (숫자만) |
| `schedule: 0 25 * * *` | `schedule: 0 23 * * *` | 시간 범위는 0-23 |
| `schedule: 60 * * * *` | `schedule: 0 * * * *` | 분 범위는 0-59 |

---

## 주의사항

- cron.md 변경은 서버 재시작 또는 다음 Heartbeat에서 반영됩니다 (즉시 적용되지 않음)
- 타임존은 `config.json`의 `system.timezone`으로 설정 가능. 미설정 시 시스템 타임존 자동 감지
- 동시각에 여러 태스크가 겹치면 병렬 실행됩니다
- command 유형 태스크 실패 시에도 서버는 중지되지 않습니다 (로그에 기록됨)
- LLM 유형 태스크 실행에는 현재 모델 설정(status.json)이 사용됩니다
- **다른 Anima의 tools/ 디렉토리에 접근할 때는 사전에 권한을 확인하세요**
