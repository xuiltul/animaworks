# description 작성 가이드 (Agent Skills 표준)

## 기본 규칙

`description`은 스킬 발견·선택에 쓰이는 가장 중요한 필드입니다.
LLM이 이를 읽고 「이 스킬이 지금 대화와 관련 있는가」를 판단합니다.

### 형식

```yaml
description: >-
  [1행: 스킬이 하는 일을 간결히 서술(3인칭)]
  Use when: [쉼표로 구분한 이용 시나리오]
```

### 규칙

1. **250자 이내** — 카탈로그 표시에서 250자를 넘는 부분은 잘림
2. **3인칭** — 「~할 수 있다」「~를 수행한다」 형태. 「나는」「당신은」 불가
3. **`Use when:` 필수** — LLM이 사용 여부를 판단하는 단서
4. **구체적인 동사·명사** — 「로그인」「스크린샷」「메일 발송」 등
5. **XML 태그 불가** — `<` `>` 는 보안상 금지
6. **`「」` 키워드 나열 금지** — 구 방식. LLM 추론에 맡김

### 좋은 예

```yaml
description: >-
  헤드리스 브라우저 CLI. 웹 페이지 열람·조작·로그인·스크린샷 촬영이 가능하다.
  Use when: 브라우저로 사이트를 열 때, 웹앱 조작·확인, 로그인, 스크린샷, UI 확인이 필요할 때.
```

```yaml
description: >-
  Gmail operations via CLI. Send, receive, search emails, and manage labels.
  Use when: sending emails, checking inbox, searching mail, managing Gmail labels or filters.
```

### 나쁜 예

```yaml
# ❌ 「」 키워드 나열 (구 방식)
description: >-
  브라우저 CLI.
  「브라우저로 확인」「스크샷」「브라우저 조작」

# ❌ 너무 모호함
description: Helps with documents

# ❌ 1인칭
description: I can help you process PDF files

# ❌ 250자 초과
description: >-
  (매우 긴 설명…)
```

### 체크리스트

- [ ] 250자 이내인가
- [ ] 3인칭인가
- [ ] `Use when:` 가 포함되었는가
- [ ] 구체적인 동사·명사가 있는가
- [ ] `「」` 키워드 나열이 없는가
- [ ] XML 태그가 없는가

### linter로 검증

스킬 작성 후 linter로 검증할 수 있다:

```bash
python scripts/lint_skill.py /path/to/SKILL.md
```
