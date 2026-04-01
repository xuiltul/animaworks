# 영업·마케팅 풀 팀 — 팀 개요

## 4개 역할 구성

| 역할 | 책임 | 권장 `--role` | `speciality` 예 | 상세 |
|--------|------|--------------|-----------------|------|
| **Sales & Marketing Director** | 전략 수립·영업 실행[machine]·파이프라인 관리·콘텐츠 QC·최종 승인 | `manager` | `sales-marketing-director` | `sales-marketing/director/` |
| **Marketing Creator** | 마케팅 콘텐츠 제작[machine]·Brand Voice 준수 | `writer` | `marketing-creator` | `sales-marketing/creator/` |
| **SDR (Sales Development)** | 리드 개발·너처링·인게이지먼트·인바운드 대응 | `general` | `sales-development` | `sales-marketing/sdr/` |
| **Market Researcher** | 시장 조사·경쟁 분석·잠재 고객 프로파일링 | `researcher` | `market-researcher` | `sales-marketing/researcher/` |

한 Anima에 전 과정을 몰아넣으면 콘텐츠 품질의 자기 평가 편향·리드 선별의 느슨함·영업과 마케팅의 우선순위 경쟁에 따른 컨텍스트 오염이 발생합니다.

각 역할 디렉터리에는 `injection.template.md`(injection.md 초안), `machine.md`(machine 활용 패턴, 해당 역할만), `checklist.md`(품질 체크리스트)가 있습니다.

> 기본 원칙 상세: `team-design/guide.md`

## 2가지 실행 모드

### Campaign mode(계획 기반)

```
Director → content-plan.md (approved) → delegate_task
  → Marketing Creator → machine으로 제작 → draft-content.md (draft)
    → Director → checklist + machine QC → 승인 / 반려 / {COMPLIANCE_REVIEWER} 컴플라이언스 리뷰
```

콘텐츠 마케팅의 표준 플로우입니다. Director가 기획하고, Creator가 제작하며, Director가 검증합니다.

### Engagement mode(SDR의 자율 순회)

```
SDR → SNS/메일/인바운드 모니터링 (cron)
  → 리드 발견 → Director에게 report + lead-report.md
  → 너처링 대상 → machine으로 드래프트 → SDR이 검증·발송
  → CS 문의 → CS 팀으로 에스컬레이션
```

SDR이 cron으로 채널을 자율적으로 순회하며 리드를 발견·육성합니다.

## 핸드오프 체인

```
Director → content-plan.md (approved)
  → delegate_task
    → Creator: 콘텐츠 제작 (machine 활용)
    → Researcher: 시장 조사 (직접 도구 사용)
      → draft-content.md / research-report.md (reviewed)
        → Director → machine QC + checklist → 승인 / 반려
          └─ 컴플라이언스 리스크 → {COMPLIANCE_REVIEWER}에 리뷰 의뢰 (cross-team)
          └─ 승인 → 공개 → Campaign Tracker 갱신

SDR → 자율 순회 (cron)
  → lead-report.md → Director → Deal Tracker 갱신
  → 계약 성립 → cs-handoff.md → CS 팀
```

### 인수인계 문서

| 송신 → 수신 | 문서 | 조건 |
|----------------|------------|------|
| Director → Creator | `content-plan.md` | `status: approved` |
| Creator → Director | `draft-content.md` | `status: draft` |
| Director → SDR | 아웃바운드 지시 | `delegate_task` |
| SDR → Director | `lead-report.md` | 리드 발견 시 |
| Director → {COMPLIANCE_REVIEWER} | `compliance-review.md` | 컴플라이언스 리스크 플래그 |
| {COMPLIANCE_REVIEWER} → Director | 동일 파일에 리뷰 결과 추가 | `status: reviewed` |
| Director → Researcher | 조사 의뢰 | `delegate_task` |
| Researcher → Director | `research-report.md` | `status: approved` |
| Director → CS 팀 | `cs-handoff.md` | 계약 성립 시 |

### 운영 규칙

- **수정 사이클**: Critical → 전면 재제작 / Warning → 차분 수정만 / 3왕복으로 해소되지 않음 → 사람에게 에스컬레이션
- **Campaign Pipeline Tracker**: 콘텐츠의 스테이지를 추적합니다. silent drop 금지
- **Deal Pipeline Tracker**: 상담의 스테이지를 추적합니다. silent drop 금지
- **컴플라이언스 에스컬레이션**: Creator/SDR 1차 필터 → Director 2차 판정 → {COMPLIANCE_REVIEWER} 크로스팀 리뷰
- **프로덕트 마케팅**: 신기능 정보는 상위(COO 등) 경로로 Director에게 전달
- **machine 실패 시**: `current_state.md`에 기록 → 다음 heartbeat에서 재평가

## 스케일링

| 규모 | 구성 | 비고 |
|------|------|------|
| 솔로 | Director가 전 역할 겸임(checklist로 품질 보증) | SNS 게시, 간단 리서치 |
| 페어 | Director + Creator | 콘텐츠 마케팅 중심 |
| 트리오 | Director + Creator + SDR | 아웃바운드 영업 포함 |
| 풀 팀 | 본 템플릿대로 4명 | 풀 퍼널 마케팅 + 영업 |

## 다른 팀과의 대응 관계

| 개발 팀 역할 | 법무 팀 역할 | 영업·MKT 팀 역할 | 대응하는 이유 |
|----------------|----------------|-------------------|-------------|
| PdM(계획·판단) | Director(분석 계획·판단) | Director(전략·영업 실행) | 「무엇을 할지」를 결정하는 사령탑 |
| Engineer(구현) | Director + machine | Director + machine(영업 제작) | machine으로 제작을 실행. 독립 Anima 불필요 |
| Reviewer(정적 검증) | Verifier(독립 검증) | {COMPLIANCE_REVIEWER}(컴플라이언스) | 독립된 관점의 검증. 크로스팀 |
| Tester(동적 검증) | Researcher(근거 검증) | Researcher(시장 조사) | 외부 데이터로 뒷받침 |
| — | — | Creator(콘텐츠 제작) | 마케팅 고유. 대량 콘텐츠 제작 특화 |
| — | — | SDR(리드 개발) | 영업 고유. 실시간 모니터링·인게이지먼트 |

## Campaign Pipeline Tracker — 캠페인 추적표

콘텐츠/캠페인의 제작 스테이지를 추적합니다. silent drop을 구조적으로 방지합니다.

### 추적 규칙

- 새로운 콘텐츠 기획이 발생하면 이 표에 등록합니다
- 다음 Heartbeat / 리뷰 시 전 항목의 스테이지를 갱신합니다
- 정체(2주 이상 스테이지 변화 없음)는 Director에게 보고합니다
- silent drop(언급 없이 소멸)은 금지입니다

### 템플릿

```markdown
# 캠페인 추적표: {팀명}

| # | 기획명 | 타입 | 퍼널 | 스테이지 | 담당 | 시작일 | 기한 | 비고 |
|---|--------|-------|---------|---------|------|--------|------|------|
| CP-1 | {명칭} | {blog/email/...} | {TOFU/MOFU/BOFU} | {스테이지} | {Creator/Director} | {날짜} | {날짜} | {특기} |

스테이지 범례:
- 기획 중: content-plan.md 작성 중
- 리서치: Researcher에 조사 의뢰 중
- 제작 중: Creator가 machine으로 제작 중
- QC: Director가 품질 점검 중
- 컴플라이언스: {COMPLIANCE_REVIEWER} 리뷰 중
- 승인됨: 공개 대기
- 공개됨: 배포 완료
- 효과 측정: 퍼포먼스 집계 중
```

## Deal Pipeline Tracker — 상담 추적표

개별 상담의 세일즈 스테이지를 추적합니다. silent drop을 구조적으로 방지합니다.

### 추적 규칙

- SDR이 리드를 발견하면 이 표에 등록합니다
- 다음 Heartbeat / 리뷰 시 전 항목의 스테이지를 갱신합니다
- 정체(2주 이상 스테이지 변화 없음)는 원인 분석을 수행합니다
- silent drop(언급 없이 소멸)은 금지입니다

### 템플릿

```markdown
# 상담 추적표: {팀명}

| # | 기업명 | 소스 | 스테이지 | 담당 | 시작일 | 갱신일 | 비고 |
|---|--------|-------|---------|------|--------|--------|------|
| D-1 | {명칭} | {inbound/outbound/...} | {스테이지} | {SDR/Director} | {날짜} | {날짜} | {특기} |

스테이지 범례:
- Lead: 리드 획득(미선별)
- Qualified: BANT 평가 통과
- Discovery: 니즈 심화 중
- Proposal: 제안서 제출됨
- Negotiation: 조건 협상 중
- Won: 수주
- Lost: 실주(사유를 비고에 기록)
- CS Handoff: CS 팀 인수인계 완료
```
