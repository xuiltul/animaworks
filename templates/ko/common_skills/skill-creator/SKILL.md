---
name: skill-creator
description: >-
  Markdown 스킬을 만드는 메타 스킬. SKILL.md frontmatter·본문과 Progressive Disclosure·create_skill 절차를 다룬다.
  Use when: 신규 스킬 추가, skill 도구용 기술 규칙 확인, references·templates 포함 생성이 필요할 때.
---

# skill-creator

## 스킬 파일 구조

스킬 파일은 YAML frontmatter와 Markdown 본문으로 구성된다.
필수 필드: `name`, `description`.
선택 필드: `allowed_tools`, `tags` 등.

```yaml
---
name: skill-name
description: >-
  스킬이 하는 일을 간결히 서술(3인칭).
  Use when: 이 스킬을 쓰는 상황을 쉼표로 구분해 나열.
---
```

`description`은 발견·선택에 쓰이는 핵심 필드이며, 모델이 관련성을 판단하는 데 사용된다.
본문은 `skill` 도구로 이름 지정 로드 시 주입된다.

**작성 형식**: `references/description_guide.md`의 **`Use when:`** 패턴(Agent Skills 표준)을 따른다.
편집 후 **`python scripts/lint_skill.py path/to/SKILL.md`** 로 검증한다.

## `description` 작성

구 방식의 **`「」` 키워드 나열**은 쓰지 않는다. 짧은 3인칭 요약 + **`Use when:`** 로 구체적인 동사·명사를 쓴다.

규칙·예시·체크리스트는 **`references/description_guide.md`** 를 본다 (250자, XML 금지 등).

### 도메인 고유·구체적

일반적인 표현은 오탐을 유발한다. 도구명·조작·대상을 스킬에 맞게 명시한다.

## Progressive Disclosure

스킬 정보는 3단계로 공개된다.

| 레벨 | 내용 | 표시 시점 |
|------|------|----------|
| Level 1 | `name` + `description` | 스킬 목록·도구 설명(예산 내) |
| Level 2 | 본문 | `skill(skill_name=...)` 실행 시 주입 |
| Level 3 | 외부 파일 | 본문 지시에 따라 `references/`·`templates/` 로드 |

Level 1은 간결하게, 절차는 Level 2에, 긴 자료는 Level 3으로 분리한다.

## 생성 절차

### Step 1: 확인

- 무엇을 자동화·문서화할지
- 개인 스킬 vs 공통 스킬(절차는 `procedures/*.md` 별도)
- **`Use when:`** 에 넣을 이용 시나리오

### Step 2: 설계

- **name**: 케밥 케이스(예: `my-skill`); 외부 도구 가이드는 `*-tool` 명명 검토
- **description**: 3인칭 요약 + **`Use when:`** (`references/description_guide.md` 참고)
- **body**: 섹션 구성, `{{now_local}}` 등 빌트인
- **references** / **templates**: 선택
- **allowed_tools**: 선택(소프트 제약)

### Step 3: 생성

```
create_skill(skill_name="{name}", description="{description}", body="{body}")
```

공통 스킬:

```
create_skill(skill_name="{name}", description="{description}", body="{body}", location="common")
```

신규 스킬은 `create_skill` 사용을 권장한다. 플랫 `skills/foo.md` 만으로는 스킬 도구 해석이 안 될 수 있다.

### Step 4: 확인

- `skills/{name}/SKILL.md` 재확인 또는 `skill(skill_name="{name}")`
- **`python scripts/lint_skill.py`** 실행(권장)

## 체크리스트

- [ ] `---` 로 구분된 YAML frontmatter
- [ ] `name`, `description` 존재
- [ ] **`Use when:`** 포함, **`「」`** 키워드 나열 없음
- [ ] 도메인 고유·구체적 표현(모호한 「관리」「확인」만 쓰지 않기)
- [ ] 본문에 실행 가능한 단계
- [ ] 설명을 `## 개요` 에만 두지 않고 frontmatter 를 정으로 사용
- [ ] 가능하면 `create_skill` 로 `{name}/SKILL.md` 생성

## 템플릿

동봉된 `templates/skill_template.md` 를 쓰거나 아래를 복사한다:

```markdown
---
name: {{skill_name}}
description: >-
  {{1행: 기능 요약}}
  Use when: {{쉼표로 구분한 이용 시나리오}}
---

# {{skill_name}}

## 절차

1. ...
2. ...

## 주의사항

- ...
```

## 주의사항

- 스킬은 Markdown 절차서이며 Python 도구와 다르다
- 필수 frontmatter: `name`, `description`
- 선택: `allowed_tools`, `tags`
- 본문은 가능하면 약 150행 이내, 긴 참조는 `references/` 활용
