# SDR — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 씁니다** — 인라인 짧은 지시 문자열로의 실행은 금지입니다. 계획서 파일을 넘깁니다
2. **출력은 드래프트** — machine 출력은 반드시 본인이 검증한 뒤 발송합니다
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{개요}.{type}.md`(`/tmp/` 금지)
4. **레이트 제한**: chat 5회/session, heartbeat 2회
5. **machine은 인프라에 접근할 수 없습니다** — 리드 정보·너처링 상황은 계획서에 포함합니다

---

## 개요

SDR은 두 가지 상황에서 machine을 활용합니다.
- 리드 발견 시 초동 컨택 메시지 드래프트
- 너처링 대상에 대한 팔로업 메일 드래프트

---

## Phase 1: 리드 초동 드래프트

### Step 1: 리드 정보 정리

발견한 리드 정보를 정리합니다.
- 발견 경위(SNS / 인바운드 / 이벤트 등)
- BANT 평가 결과
- 리드 프로필·관심 사항

### Step 2: machine에 초동 메시지 드래프트 요청

리드 정보와 메시지 방침을 입력으로 초동 컨택 메시지를 machine에 의뢰합니다.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{outreach-request.md})" \
  -d /path/to/workspace
```

### Step 3: 드래프트 검증

SDR이 machine 출력을 읽고 `sdr/checklist.md`에 따라 셀프 체크합니다.

- [ ] 톤이 적절한가
- [ ] 컴플라이언스상 문제가 없는가(정보통신망법 옵트인 확인 등)
- [ ] 퍼스널라이즈가 적절한가

문제가 있으면 수정하고 검증 완료 후 발송합니다.

## Phase 2: 너처링 메일

### Step 4: 너처링 상황 정리

대상 리드 상황을 정리합니다.
- 지금까지의 교환
- BANT 평가 변화
- 리드 반응·관심 변화

### Step 5: machine에 팔로업 메일 드래프트 요청

너처링 상황을 입력으로 팔로업 메일을 machine에 의뢰합니다.

### Step 6: 드래프트 검증 후 발송

SDR이 machine 출력을 `sdr/checklist.md`에 따라 검증하고 문제 없으면 발송합니다.

---

## 리드 리포트 템플릿(lead-report.md)

```markdown
# 리드 리포트: {기업명 또는 개인명}

status: {new | qualified | disqualified | nurturing}
source: {inbound | outbound | sns | referral | event | other}
author: {anima 이름}
discovered: {YYYY-MM-DD}

## BANT 평가

| 항목 | 평가 | 근거 |
|------|------|------|
| Budget(예산) | {있음 / 불명 / 없음} | {근거} |
| Authority(결재권) | {있음 / 불명 / 없음} | {근거} |
| Need(과제·니즈) | {명확 / 잠재 / 없음} | {근거} |
| Timeline(도입 시기) | {구체적 / 미정 / 없음} | {근거} |

## 커스텀 필드

{도입 시 팀 고유 평가 항목 추가}

## 리드 개요

{발견 경위, 교환 요약, 주목 포인트}

## 권장 액션

{Director 제안: 상담화 / 너처링 지속 / 보류 / 추가 조사}
```

---

## 제약 사항

- machine 출력을 그대로 발송해서는 안 됩니다(NEVER — 반드시 셀프 체크 후 발송)
- 컴플라이언스상 우려가 있는 메시지는 Director 확인 후 발송합니다(MUST)
- 옵트아웃 수단을 포함하지 않은 메일 발송은 금지입니다(NEVER)
