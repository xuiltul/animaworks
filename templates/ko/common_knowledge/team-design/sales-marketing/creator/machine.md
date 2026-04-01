# Marketing Creator — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 씁니다** — 인라인 짧은 지시 문자열로의 실행은 금지입니다. 계획서 파일을 넘깁니다
2. **출력은 드래프트** — machine 출력은 반드시 본인이 검증한 뒤 `status: draft`로 Director에게 납품합니다
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{개요}.{type}.md`(`/tmp/` 금지)
4. **레이트 제한**: chat 5회/session, heartbeat 2회
5. **machine은 인프라에 접근할 수 없습니다** — 기억·메시지·조직 정보는 계획서에 포함합니다

---

## 개요

Marketing Creator는 `content-plan.md`를 받아 machine으로 콘텐츠를 제작하고 셀프 체크 후 Director에게 납품합니다.

---

## Phase 1: 콘텐츠 제작

### Step 1: content-plan.md 확인

Director로부터 받은 `content-plan.md` 내용을 확인합니다.
- 목적·타겟·핵심 메시지
- 퍼널 스테이지(TOFU/MOFU/BOFU)
- 구성 지시·톤·분량 안
- 컴플라이언스 주의

불명한 점이 있으면 Director에게 확인합니다(추측으로 제작하지 않음).

### Step 2: machine에 콘텐츠 제작 요청

`content-plan.md`와 Brand Voice 가이드를 입력으로 콘텐츠 제작을 machine에 의뢰합니다.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{content-request.md})" \
  -d /path/to/workspace
```

**제작 지시에 포함할 것**:
- `content-plan.md` 전체 내용
- Brand Voice 가이드(톤·금지 표현·용어 통일)
- 퍼널 스테이지에 맞는 CTA 요건
- 출력 형식 지정

### Step 3: draft-content.md 검증

Creator가 machine 출력을 읽고 `creator/checklist.md`에 따라 셀프 체크합니다.

문제가 있으면 Creator 본인이 수정하고 `status: draft`로 `draft-content.md`를 저장합니다.

```bash
write_memory_file(path="state/plans/{date}_{제목}.draft-content.md", content="...")
```

### Step 4: Director에게 납품

`draft-content.md`를 `send_message(intent: report)`로 Director에게 보고합니다.

## Phase 2: 수정 대응

### Step 5: 반려에 대응

Director로부터 반려가 있으면 수정 지시를 입력으로 machine에 수정판 제작을 의뢰합니다.

수정판을 다시 `creator/checklist.md`로 셀프 체크하고 `draft-content.md`를 갱신해 Director에게 재납품합니다.

---

## 드래프트 콘텐츠 템플릿(draft-content.md)

```markdown
# 콘텐츠 드래프트: {제목}

status: draft
plan_ref: {content-plan.md 경로}
version: {v1 | v2 | ...}
author: {anima 이름}
date: {YYYY-MM-DD}

## 본문

{콘텐츠 본문}

## 셀프 체크 결과

- [ ] 핵심 메시지가 반영되어 있음
- [ ] 타겟에 적절한 톤임
- [ ] Brand Voice에 준수함
- [ ] 컴플라이언스 주의를 준수함
- [ ] 퍼널 스테이지에 맞는 CTA가 있음

## 수정 이력

| 판 | 날짜 | 수정 내용 |
|----|------|---------|
| v1 | {날짜} | 초고 |
```

---

## 제약 사항

- Brand Voice 준수를 machine 지시에 포함합니다(MUST)
- machine 출력을 셀프 체크 없이 납품해서는 안 됩니다(NEVER)
- `content-plan.md` 지시에서 벗어난 제작을 해서는 안 됩니다(NEVER)
