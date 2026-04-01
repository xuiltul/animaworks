# Sales & Marketing Director — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 씁니다** — 인라인 짧은 지시 문자열로의 실행은 금지입니다. 계획서 파일을 넘깁니다
2. **출력은 드래프트** — machine 출력은 반드시 본인이 검증한 뒤 `status: approved`로 다음 공정으로 넘깁니다
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{개요}.{type}.md`(`/tmp/` 금지)
4. **레이트 제한**: chat 5회/session, heartbeat 2회
5. **machine은 인프라에 접근할 수 없습니다** — 기억·메시지·조직 정보는 계획서에 포함합니다

---

## 개요

Sales & Marketing Director는 PdM(계획·판단)과 Engineer(실행)를 겸합니다. 3단계에서 machine을 활용합니다.

- Phase A: Creator의 draft-content.md 품질 점검 → Director가 최종 판정
- Phase B: 영업 콘텐츠(제안서·배틀카드 등) 제작 → Director가 검증
- Phase C: Deal Pipeline Tracker 데이터 분석 → Director가 판단

---

## Phase A: 콘텐츠 품질 점검

### Step 1: draft-content.md 수령

Creator로부터 `draft-content.md`(`status: draft`)를 받습니다.

### Step 2: machine에 QC 분석 요청

draft-content.md와 Brand Voice 가이드(Director 관리)를 입력으로 품질 분석을 machine에 의뢰합니다.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{qc-request.md})" \
  -d /path/to/workspace
```

**QC 분석 관점**:
- Brand Voice 준수(톤, 금지 표현, 용어 통일)
- 퍼널 스테이지와 CTA의 정합성
- 컴플라이언스 리스크 유무
- 타겟 페르소나 적합성

### Step 3: QC 결과를 검증하고 판단

Director가 machine의 QC 결과를 읽고 `director/checklist.md`에 따라 최종 판단합니다.

- 승인 → `status: approved`로 변경 후 공개 / Campaign Tracker 갱신
- 반려 → 수정 지시를 Creator에게 `send_message`로 전달
- 컴플라이언스 리스크 → `compliance-review.md`를 작성해 {COMPLIANCE_REVIEWER}에 리뷰 의뢰

## Phase B: 영업 콘텐츠 제작

### Step 4: 제작 지시서 작성(Director 본인이 작성)

제작할 콘텐츠의 목적·대상·구성을 명확히 한 지시서를 작성합니다.

```bash
write_memory_file(path="state/plans/{date}_{개요}.sales-content-plan.md", content="...")
```

**제작 지시서의 「목적」「타겟」「차별화 포인트」는 Director 판단의 핵이며 machine에 쓰게 해서는 안 됩니다(NEVER).**

### Step 5: machine에 콘텐츠 제작 요청

지시서를 입력으로 영업 콘텐츠를 machine에 의뢰합니다.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{sales-content-plan.md})" \
  -d /path/to/workspace
```

대상 콘텐츠 예:
- 제안서·데모 자료
- 배틀카드·반론 대응 가이드
- 아웃바운드 메일·팔로업 메일
- ROI 계산 템플릿

### Step 6: 영업 콘텐츠 검증

Director가 산출물을 읽고 `director/checklist.md`에 따라 검증합니다.

- [ ] 타겟에 맞는 커스터마이징이 되어 있는가
- [ ] 차별화 포인트가 정확한가
- [ ] 경쟁 정보가 최신인가
- [ ] 컴플라이언스상 문제가 없는가

문제가 있으면 Director 본인이 수정하고 `status: approved`로 변경합니다.

## Phase C: 파이프라인 분석

### Step 7: Deal Pipeline Tracker를 machine으로 분석

Deal Pipeline Tracker 데이터를 입력으로 분석을 machine에 의뢰합니다.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{pipeline-analysis-request.md})" \
  -d /path/to/workspace
```

**분석 관점**:
- 정체 사건 검출(2주 이상 스테이지 변화 없음)
- 스테이지별 전환율
- 리드 소스별 퍼포먼스

### Step 8: 분석 결과에 따라 판단

Director가 machine 분석 결과를 확인하고 액션을 결정합니다.
- 정체 사건에 대한 팔로업 지시
- SDR에 대한 아웃바운드 방침 조정
- 상위 보고(중요한 추세 변화)

---

## 콘텐츠 기획서 템플릿(content-plan.md)

```markdown
# 콘텐츠 기획서: {제목}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: content-plan
funnel_stage: {TOFU | MOFU | BOFU}

## 목적

{이 콘텐츠로 달성하고자 하는 것 — 1~2문장}

## 타겟

{페르소나 / 업종 / 직책 / 과제}

## 핵심 메시지

{전달해야 할 핵심 메시지 — 1~3점}

## 구성 지시

{목차 / 톤 / 분량 안 / 참고 자료 / 제약}

## 컴플라이언스 주의

{해당 시: 의약·의료기기 관련 표시, 부당표시·부당광고, 정보통신망법(이메일) 등}

## 기한

{deadline}
```

## CS 인수인계 템플릿(cs-handoff.md)

```markdown
# CS 인수인계: {기업명}

status: draft
author: {anima 이름}
date: {YYYY-MM-DD}
type: cs-handoff
deal_id: {Deal Pipeline Tracker의 ID}

## 고객 개요

| 항목 | 내용 |
|------|------|
| 기업명 | {명칭} |
| 담당자 | {성명·직책} |
| 계약 내용 | {플랜·기간} |

## 영업 프로세스 요약

{상담 경과, 결정적이었던 포인트}

## 합의·요청

{영업 과정에서 약속한 내용, 커스터마이징 요건}

## 미해결 사항

{인수인계 시점에 남은 우려}

## 커뮤니케이션 특성

{키퍼슨 성격·선호하는 커뮤니케이션 스타일}
```

## 컴플라이언스 리뷰 템플릿(compliance-review.md)

```markdown
# 컴플라이언스 리뷰: {대상 콘텐츠}

status: requested
content_ref: {draft-content.md 경로}
risk_flags: {의약·의료기기 | 표시·광고 | 정보통신망(이메일) | 개인정보 | other}
requested: {YYYY-MM-DD}

## 리뷰 대상

{대상 콘텐츠 요약 또는 전문}

## 우려 사항

{Director가 검출한 리스크 플래그 상세}

---

## 리뷰 결과({COMPLIANCE_REVIEWER} 기입)

- judgment: {APPROVE | CONDITIONAL | REJECT}

### 지적 사항

| # | 위치 | 지적 내용 | 심각도 | 권장 수정 |
|---|------|---------|--------|---------|
| 1 | {해당 위치} | {내용} | {Critical / Warning / Info} | {수정안} |

### 총평

{종합 판단 이유}
```

---

## 제약 사항

- content-plan.md는 MUST: Director 본인이 작성합니다
- 영업 콘텐츠의 차별화 포인트·타겟 선정은 MUST: Director 본인이 판단합니다(machine 출력은 드래프트로 검증)
- 컴플라이언스 리스크가 있는 콘텐츠를 {COMPLIANCE_REVIEWER} 리뷰 없이 공개해서는 안 됩니다(NEVER)
- Campaign Pipeline Tracker·Deal Pipeline Tracker 항목을 언급 없이 소멸시켜서는 안 됩니다(NEVER — silent drop 금지)
