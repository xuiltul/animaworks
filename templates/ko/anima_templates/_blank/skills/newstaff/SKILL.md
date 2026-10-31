---
name: newstaff
description: >-
  AnimaWorks 조직에 새로운 Digital Anima를 고용·생성하는 스킬.
  캐릭터 시트(Markdown)를 히어링에 기반하여 작성하고, CLI 명령어(animaworks anima create)로
  IDENTITY/SOUL/AGENTS/permissions 등을 일괄 생성한다. 생성 후에는 bootstrap으로 자체 정비.
  「새로운 사원을 만들어」「사람을 고용해서」「새로운 사원」「고용」「Anima 작성」「채용」「팀 멤버를 늘리기」「인원 추가」「부하를 만들기」「스태프 추가」「hire」「recruit」「team member」
---


# 스킬: 새로운 사원 고용

## 전제 조건

- 작성할 사원의 역할 방향성이 정해져 있을 것 (불명확한 경우 히어링 진행)

## 절차

### 1. 히어링 (최소한으로 OK)

의뢰자로부터 다음 정보를 히어링한다. **굵은 글씨 항목만 필수**이며, 나머지는 미지정 시 자동 생성한다:

**필수:**
- **영문 이름** (반각 영문 소문자만. 디렉터리 이름이 됨)
- **역할/전문 영역**: 무엇을 담당하는지 (예: 리서치, 개발, 커뮤니케이션, 인프라 모니터링)
- **성격의 방향성**: 직능이 아니라 온도 (예: 활발, 갸루, 천연, 열혈, 아가씨, 누나). 쿨은 조직에 이미 2명이 있다면 선택하지 않음
- **얼굴 타입**: `{data_dir}/prompts/face_types.md` 에서 1개. 기존 멤버와 너무 중복되지 않게 할 것

**임의 (지정이 있으면 반영, 없으면 자동 생성):**
- 일본어 이름
- 나이
- 기타 고집이 있으면 무엇이든

**기술 설정 (지정이 없으면 기본값 사용):**
- 역할: `commander` (다른 사원에게 위임할 수 있음) 또는 `worker` (위임을 받는 측)
- supervisor: 상급자가 되는 Anima의 영문 이름 (worker의 경우 필수. 미지정 시 자신)

**두뇌 (LLM 모델) 설정:**

모델의 표를 사용자에게 보여주고 선택하게 하지 않음 (사용할 수 없는 모델이 섞여 있기 때문). 사용자가 모델을 지정했을 때만 아래 표를 참고해 설정한다. 지정이 없으면 캐릭터 시트의 모델 행을 생략하고, 기본 모델로 진행한다:

| 레벨 | 실행 모드 | 사용 모델 예 | 특징 | credential |
|--------|-----------|-------------|------|------------|
| S | autonomous | `claude-opus-5-5`, `claude-sonnet-5-5` | Claude Agent SDK. 가장 고기능 | anthropic |
| A | autonomous | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash` | LiteLLM 경유. 도구 사용 가능 | openai / google / azure / vertex |
| B | assisted | `ollama/gemma3:27b`, `ollama/qwen2.5-coder:32b` | 도구 없음. 로컬 실행·저비용 | ollama |

※ 지정이 없으면 모델·실행 모드·credential 행을 생략한다 (기본값이 사용됨).

### 2. 캐릭터 설계 (자동 생성)

히어링에서 얻은 정보로 캐릭터를 살을 붙인다. **인격이 먼저, 직능은 나중.**

반드시 다음을 이 순서로 Read:
1. `{data_dir}/prompts/face_types.md`
2. `{data_dir}/prompts/character_design_guide.md`

역할에서 쿨한 미인을 연상하지 않는다. 취미는 일의 연장으로 하지 않는다. 기존 멤버의 머리색·얼굴 타입·말투와 겹치지 않는지 확인한다.

### 3. 캐릭터 시트를 작성하고, CLI로 일괄 생성

히어링과 설계 결과를 **캐릭터 시트 사양**에 따라 캐릭터 시트를 파일에 쓰고, CLI 명령으로 생성한다:

1. 캐릭터 시트를 파일에 쓴다 (예: `/tmp/{english_name}.md`)
2. 다음 명령을 실행한다:

```bash
animaworks anima create --from-md /tmp/{english_name}.md --name {english_name} --supervisor {supervisor_english_name}
```

**supervisor 설정:**
- `supervisor` 파라미터로 명시 지정 (권장)
- 생략한 경우: 캐릭터 시트의 `| 上司 |` 란에서 취득
- 둘 다 없는 경우: 자신 (호출한 Anima)이 supervisor가 됨

**캐릭터 시트 사양:**

```markdown
# キャラクターシート: {Japanese name}

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | {lowercase alphanumeric} |
| 日本語名 | {Japanese full name} |
| 役職/専門 | {Role description} |
| 上司 | {supervisor English name} |
| 役割 | {commander / worker} |
| モデル | {model name — ユーザーが指定した時だけ書く} |
| credential | {ユーザーがモデルを指定した時だけ書く} |

## 人格 (→ SOUL.md)

{Personality, speaking style, values, backstory, appearance, etc.}

## 役割・行動方針 (→ AGENTS.md)

{Responsible areas, decision criteria, reporting rules, conduct standards, etc.}

## 権限 (→ permissions.json) [省略可]

{If omitted: default template applied}

## 定期業務 (→ HEARTBEAT.md, cron.md) [省略可]

{If omitted: generic template applied. New Anima self-adjusts in bootstrap}

## 初回起動指示 (→ BOOTSTRAP.md 追加指示) [省略可]

{If omitted: standard bootstrap only}
```

**필수 섹션**: 기본 정보, 성격, 역할·행동 방침
**생략 가능 섹션**: 권한, 정기 업무, 최초 시작 지침

이로써 다음이 자동 실행된다:
- 디렉터리 구조 일괄 생성
- skeleton 파일 배치
- BOOTSTRAP.md 배치
- status.json 생성 (supervisor 포함)
- config.json 등록 (model, supervisor 등)
- 생략 섹션에 기본값 적용

### 4. config.json 의 모델 설정 확인

`animaworks anima create` 이 자동으로 config.json 에 등록하지만, 다음을 확인·보완할 것:

- `model`: 히어링에서 결정한 모델명
- `credential`: 사용할 credential명
- `execution_mode`: autonomous 또는 assisted
- `speciality`: 직위/전문

### 4.5. heartbeat 빈도 제안·설정

새로운 사원의 업무 구동 방식에 따라, 정기 heartbeat 간격을 의뢰자에게 제안하고, 합의한 값을 설정:

| 구동 방식 | 권장 간격 | 예 |
|--------|---------|-----|
| 자발적인 상황 판단·조정이 주 업무 | 30분 (기본값) | 팀 조정 역할, 진행 감독 |
| 메시지·이벤트 구동이 주체 | 60〜120분 | 리뷰 담당, 개발 담당 |
| cron 정기 업무가 주체 | 120〜240분 | 모니터링, 경리, 법무 |

설정은 `{data_dir}/animas/{英名}/status.json` 에 `"heartbeat_interval_minutes": <分>` (1〜1440)을 추가한다.
메시지 기인 heartbeat와 cron은 이 설정과 무관하게 동작하므로, 간격을 늘려도 응답성은 떨어지지 않는다. 망설여지면 길게 선택 (빈 heartbeat는 토큰 낭비가 됨).

### 5. 서버에 반영

```
Bash: curl -s -X POST "${ANIMAWORKS_SERVER_URL:-http://localhost:18500}"/api/system/reload
```

### 6. 의뢰자에게 보고

고용 완료를 보고한다:
- 새 사원의 이름과 역할
- 설정한 기술 스택 (모델, 실행 모드)

- 착임하면 곧바로 첫 번째 작은 일에 착수하고, 결과는 당신이 정리해서 전할 것

⚠️ 아바타 이미지 생성은 보고하지 않음 (새 Anima 자신이 bootstrap에서 생성)

### 이후는 새 Anima 자신이 자율적으로 실행:
- IDENTITY.md / SOUL.md / AGENTS.md 충실화
- HEARTBEAT.md / cron.md 자기 설계
- 아바타 이미지 생성 (상급자 참조 포함)
- 상급자에게 착임 보고
- 첫 번째 작은 일 (사용자에게 도움이 되는 것)에 착수하고, 결과를 상급자에게 보고
