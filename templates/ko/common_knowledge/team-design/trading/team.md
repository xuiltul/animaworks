# 트레이딩 풀 팀 — 팀 개요

## 4개 역할 구성

| 역할 | 책무 | 권장 `--role` | `speciality` 예 | 상세 |
|------|------|--------------|-----------------|------|
| **Strategy Director** | 전략 설계·리스크 한도 설정·PDCA 총괄·최종 판단 | `manager` | `trading-director` | `trading/director/` |
| **Market Analyst** | 시장 분석·시그널 생성·모델 개발 | `researcher` | `market-analyst` | `trading/analyst/` |
| **Trading Engineer** | bot 구현·백테스트·실행 기반·데이터 파이프라인 | `engineer` | `trading-engineer` | `trading/engineer/` |
| **Risk Auditor** | 독립 P&L 검증·운용 건전성 감사·carry-forward 추적 | `engineer` or `ops` | `risk-auditor` | `trading/auditor/` |

한 Anima에 전 과정을 몰아두면 손익의 낙관적 편향(손실 경시)·운용 검증의 부재(bot 정지를 눈치채지 못함)·이슈 추적의 소실(지갑 미확인 방치)이 발생한다.

각 역할 디렉터리에 `injection.template.md`(injection.md 초안), `machine.md`(machine 활용 패턴), `checklist.md`(품질 체크리스트)가 있다.

> 기본 원칙 상세: `team-design/guide.md`

## 핸드오프 체인

```
Director → strategy-plan.md (approved) + performance-tracker 참조
  → delegate_task
    → Engineer: bot 구현 / 백테스트 (machine 활용)
    → Analyst: 시장 분석 / 시그널 검증 (machine 활용)
      → backtest-report.md + market-analysis.md (reviewed)
        → Auditor (P&L 검증 + 운용 건전성 감사) ← 독립 검증
          └─ 문제 있음 → Director에 반려
          └─ APPROVE → Director → performance-tracker 갱신 → 상위 보고 / call_human
```

### 인계 문서

| 송신 → 수신 | 문서 | 조건 |
|------------|------|------|
| Director → Engineer | `strategy-plan.md` | `status: approved` |
| Director → Analyst | `strategy-plan.md`(분석 관점 지시) | `status: approved` |
| Engineer → Auditor | `backtest-report.md` | `status: reviewed` |
| Analyst → Director | `market-analysis.md` | `status: reviewed` |
| Auditor → Director | `performance-review.md` + `ops-health-report.md` | `status: approved` |

### 운영 규칙

- **수정 사이클**: Critical(드로다운 임계 초과·bot 정지·자산 불일치) → 즉시 대응 + Auditor 재검증 / Warning → 차분 확인만 / 3왕복으로 해소되지 않으면 → 사람에게 에스컬레이션
- **Performance Tracker**: 전략 버전을 가로질러 P&L·승률·Sharpe·최대 DD를 추적한다. 전회에 플래그한 문제가 다음에 언급 없이 사라지는 것(silent drop)은 금지
- **Ops Issue Tracker**: 운용상 이슈를 carry-forward로 추적한다. silent drop 금지
- **PDCA 사이클**: Plan=Director(전략 설계), Do=Engineer(구현)+ Analyst(분석), Check=Auditor(독립 검증), Act=Director(판단·수정 지시)
- **machine 실패 시**: `current_state.md`에 기록 → 다음 heartbeat에서 재평가

## 스케일링

| 규모 | 구성 | 비고 |
|------|------|------|
| 솔로 | Director가 전 역할 겸임(checklist로 품질 보장) | 페이퍼 트레이드 검증, 단일 전략 |
| 페어 | Director + Auditor | 중리스크, 소수 전략 라이브 운용 |
| 트리오 | Director + Engineer + Auditor | bot 개발 단계(Analyst는 Director가 겸임) |
| 풀 팀 | 본 템플릿대로 4명 | 복수 전략 본운용 |

## 개발 팀·법무 팀과의 대응 관계

| 개발 팀 역할 | 법무 팀 역할 | 트레이딩 팀 역할 | 대응하는 이유 |
|-------------|-------------|-----------------|--------------|
| PdM(조사·계획·판단) | Director(분석 계획·판단) | Director(전략 설계·PDCA 판단) | 「무엇을 할지」를 결정하는 사령탑 |
| Engineer(구현) | Director + machine | Engineer(bot 구현·백테스트) | 코드를 작성하고 동작하는 것을 만든다 |
| Reviewer(정적 검증) | Verifier(독립 검증) | Auditor(P&L 검증 + 운용 건전성) | 「실행과 검증의 분리」의 핵. 가장 중요한 분리 지점 |
| Tester(동적 검증) | Researcher(근거 검증) | Analyst(시장 분석·시그널 품질) | 외부 데이터에 기반한 뒷받침·품질 확인 |

## Strategy Performance Tracker — 전략 성과 추적표

전략 버전 갱신마다 이 표를 갱신한다. 전회에 플래그한 문제가 다음에 언급 없이 사라지는 것(silent drop)을 구조적으로 방지한다.

### 추적 규칙

- 전략 버전 갱신(파라미터 변경 포함)마다 행을 추가한다
- 드로다운이 임계 `{max_drawdown_pct}`를 초과하면 즉시 Director에 보고한다
- 「지속 모니터링」이 아닌 상태가 붙은 문제는 다음 리뷰에서 반드시 언급한다
- silent drop(언급 없이 소멸)은 금지

### 템플릿

```markdown
# 전략 성과 추적표: {전략명}

| # | 기간 | 버전 | P&L | 승률 | Sharpe | 최대 DD | 상태 | 비고 |
|---|------|----------|-----|------|--------|--------|----------|------|
| S-1 | {시작〜종료} | {v1} | {금액} | {%} | {값} | {%} | {평가} | {특기사항} |
| S-2 | {시작〜종료} | {v2} | {금액} | {%} | {값} | {%} | {평가} | {특기사항} |

상태 범례:
- 본운용: 라이브 환경에서 운용 중
- 페이퍼 트레이드: 검증 중
- 파라미터 조정 중: 개선 사이클 진행 중
- 정지: 임계 초과 또는 엣지 소멸로 정지
- 폐기: 전략 자체를 폐기
```

## Ops Issue Tracker — 운용 이슈 추적표

운용상 이슈를 carry-forward로 추적한다. silent drop을 구조적으로 방지한다.

### 추적 규칙

- 운용 이슈(bot 정지·API 장애·자산 불일치 등)를 검출하면 이 표에 등록한다
- 다음 Heartbeat / 리뷰 시 전 항목의 상태를 갱신한다
- 「해소」가 아닌 항목은 다음에 반드시 언급한다
- silent drop(언급 없이 소멸)은 금지

### 템플릿

```markdown
# 운용 이슈 추적표: {팀명}

| # | 검출일 | 이슈 | 심각도 | 상태 | 담당자 | 해소일 | 비고 |
|---|--------|------|--------|----------|--------|--------|------|
| O-1 | {날짜} | {이슈 내용} | Critical | {대응 상황} | {담당} | {날짜} | {특기사항} |
| O-2 | {날짜} | {이슈 내용} | Warning | {대응 상황} | {담당} | {날짜} | {특기사항} |

상태 범례:
- 미대응: 검출됨·미착수
- 대응 중: 수정 작업 진행 중
- 해소: 문제가 제거됨
- 재발: 한 번 해소되었으나 재발
- 지속 모니터링: 임시 대응 완료·근본 대응은 미완
```
