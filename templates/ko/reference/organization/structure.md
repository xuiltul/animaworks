# 조직 구조의 동작 방식

AnimaWorks의 조직 구조는 각 Anima의 `status.json` (또는 `identity.md`)을 Single Source of Truth (SSoT)로 하여 구축됩니다.
`core/org_sync.py`가 디스크 상의 **supervisor**를 `config.json`에 동기화하고, 프롬프트 구축 시 활용됩니다.
본 문서에서는 조직 구조가 어떻게 정의, 해석, 표시되는지를 설명합니다.

## 데이터 소스와 우선순위

### supervisor (상사)

조직의 상하 관계는 각 Anima의 `supervisor`로 정의됩니다. 읽기 우선순위:

1. **status.json** — `"supervisor"` 키 (권장)
2. **identity.md** — 테이블 행 `| 上司 | name |` (일본어만 해당. `core/config/models.py`의 `read_anima_supervisor`가 해석)

`supervisor`가 미설정, 빈 값, 또는 "なし", "(なし)", "（なし）", "-", "---" 중 하나이면 최상위입니다.
`config.json`의 `animas.<name>.supervisor`는 org_sync에 의해 **디스크에서 동기화**되므로 수동 편집은 덮어쓰여집니다.

### speciality (전문 영역)

전문 영역은 `core/prompt/builder.py`의 `_scan_all_animas()`에 의해 다음 우선순위로 해석됩니다:

1. **status.json** — `"speciality"` 키 (자유 텍스트)
2. **config.json** — `animas.<name>.speciality` (status.json에 `speciality` 키가 없는 경우의 폴백)
3. **status.json** — `"role"` 키 (위 방법으로 해석되지 않는 경우의 최종 폴백. 역할 이름: engineer, researcher, manager, writer, ops, general)

**참고:** org_sync는 **speciality를 동기화하지 않습니다**. speciality는 프롬프트 구축 시마다 디스크와 config에서 해석됩니다.
`animaworks anima create --from-md`로 생성한 Anima는 `status.json`에 `role`이 들어가지만 `speciality`는 들어가지 않습니다.
커스텀 표시(예: "개발 리드")가 필요하면, `status.json`에 `"speciality": "개발 리드"`를 수동으로 추가하세요.

## org_sync를 통한 config.json 동기화

`core/org_sync.py`의 `sync_org_structure()`가 다음을 수행합니다:

1. 각 Anima 디렉토리(`identity.md`가 존재하는 것만)에서 `status.json` / `identity.md`를 읽어 supervisor를 추출 (`read_anima_supervisor`)
2. 순환 참조를 감지 (감지된 Anima는 동기화 대상에서 제외)
3. `config.json`의 `animas.<name>.supervisor`를 디스크의 값에 맞춰 업데이트 (**supervisor만**)
4. 디스크에 존재하지 않는 Anima의 config 항목을 삭제 (prune)

**동기화 대상:** supervisor만. speciality는 org_sync에서 업데이트되지 않습니다.

**실행 타이밍:**

- 서버 기동 시 (`animaworks start`의 Anima 프로세스 기동 후)
- Anima가 reconciliation으로 추가되었을 때 (`on_anima_added` 콜백)

## supervisor에 의한 계층 정의

- `supervisor: null` 또는 미설정 → 해당 Anima는 최상위
- `supervisor: "alice"` → alice가 상사

status.json에서의 설정 예시 (권장):

```json
{
  "enabled": true,
  "supervisor": null,
  "speciality": "경영 전략 및 전체 총괄"
}
```

```json
{
  "enabled": true,
  "supervisor": "alice",
  "speciality": "개발 리드"
}
```

이 설정으로 다음 계층이 구축됩니다:

```
alice (경영 전략 및 전체 총괄)
├── bob (개발 리드)
│   └── dave (백엔드 개발)
└── carol (디자인 및 UX)
```

중요한 제약:
- supervisor에 지정하는 이름은 알려진 Anima 이름(영문)이어야 합니다
- 순환 참조(alice → bob → alice)는 감지되어 동기화 대상에서 제외됩니다
- 1명의 Anima가 가질 수 있는 supervisor는 1명뿐입니다

## 조직 컨텍스트 구축 프로세스

`core/prompt/builder.py`의 `_build_org_context()`가 디렉토리 스캔과 config.json 머지 결과에서 다음 정보를 산출합니다:

1. **상사 (supervisor)**: 자신의 supervisor 값. 미설정이면 "당신이 최상위입니다"
2. **부하 (subordinates)**: supervisor가 자신의 이름인 모든 Anima
3. **동료 (peers)**: 자신과 같은 supervisor를 가진 Anima (자신 제외)

산출 결과는 시스템 프롬프트에 "당신의 조직 내 위치"로서 주입됩니다:

```
## 당신의 조직 내 위치

당신의 전문: 개발 리드

상사: alice (경영 전략 및 전체 총괄)
부하: dave (백엔드 개발)
동료 (같은 상사를 가진 멤버): carol (디자인 및 UX)
```

## 자신의 위치 파악하기

시스템 프롬프트의 "당신의 조직 내 위치" 섹션에서 다음을 확인할 수 있습니다:

| 항목 | 의미 | 행동에 미치는 영향 |
|------|------|-------------------|
| 당신의 전문 | speciality 값 | 이 분야에 관한 질문과 판단에 대해 자신이 책임을 집니다 |
| 상사 | 보고 대상 Anima | 진행 보고 및 문제 에스컬레이션 대상 |
| 부하 | 자신 아래의 Anima | 작업 위임 및 진행 상황 확인 대상 |
| 동료 | 같은 상사를 가진 동료 | 관련 업무에서 직접 협력하는 상대 |

### 확인해야 할 포인트

- 상사가 "(없음 — 당신이 최상위입니다)"이면, 조직의 최상위로서 전체 책임을 집니다
- 부하가 "(없음)"이면, 작업 실행자로서 직접 손을 움직입니다
- 동료가 있으면 관련 업무에서 직접 조율할 수 있습니다

## 조직 변경 시 동작

조직 구조 변경은 다음 절차로 반영됩니다:

1. 대상 Anima의 `status.json`을 편집 (`supervisor` / `speciality` 변경)
2. **supervisor를 변경한 경우:** 서버를 재시작하거나 다음 org_sync 실행을 기다림 (org_sync가 config.json에 supervisor를 동기화)
3. **speciality를 변경한 경우:** 프롬프트 구축 시 status.json에서 매번 읽어오므로 서버 재시작 불필요. 다음 채팅/heartbeat에서 반영

주의 사항:
- `config.json`의 `animas.<name>.supervisor`를 직접 편집해도 org_sync 실행 시 디스크의 값으로 덮어쓰여집니다 (speciality는 덮어쓰여지지 않음)
- 조직 변경 후에는 영향받는 Anima에게 메시지로 알리는 것을 권장합니다 (SHOULD)

## 조직 구조 패턴 예시

아래는 각 Anima의 `status.json`에 설정하는 예시입니다. org_sync가 `supervisor`를 config.json에 동기화합니다. `speciality`는 프롬프트 구축 시 status.json / config에서 매번 해석됩니다.

### 패턴 1: 플랫 조직

전원이 최상위. 상하 관계 없음.

각 Anima의 status.json:
```json
{ "supervisor": null, "speciality": "기획" }
{ "supervisor": null, "speciality": "개발" }
{ "supervisor": null, "speciality": "디자인" }
```

```
alice (기획)
bob (개발)
carol (디자인)
```

특징:
- 전원이 대등한 입장에서 직접 소통 가능
- 소규모 팀이나 각자 독립적인 업무를 가진 경우에 적합
- 전원의 동료는 "(없음)" (같은 supervisor를 공유하지 않으므로)

### 패턴 2: 계층형 조직

명확한 상하 관계가 있습니다. 가장 일반적인 패턴입니다.

각 Anima의 status.json에 `supervisor`와 `speciality`를 설정:

```
alice (CEO 및 전체 총괄)
├── bob (개발 부장)
│   ├── dave (백엔드)
│   └── eve (프론트엔드)
└── carol (영업 부장)
    └── frank (고객 대응)
```

특징:
- bob과 carol은 동료 (같은 supervisor = alice)
- dave와 eve는 동료 (같은 supervisor = bob)
- dave에서 frank로의 연락은 bob → alice → carol → frank 경로를 따름 (타 부서 규칙)

### 패턴 3: 전문가 + 매니저형

소수의 매니저가 다수의 전문가를 총괄합니다.

```
manager (프로젝트 관리)
├── dev1 (API 개발)
├── dev2 (DB 설계)
├── dev3 (인프라)
└── qa (품질 보증)
```

특징:
- 전 멤버가 동료 관계. 직접 협력이 용이
- manager가 전체 작업 배분과 진행 관리를 담당
- 스타트업이나 프로젝트 팀에 적합

## speciality 활용

`speciality`는 `status.json`의 `speciality` 키에 자유 텍스트로 기술합니다. 미설정 시 `role`(역할 이름)이 폴백으로 표시됩니다.

- 조직 컨텍스트에서 각 Anima 이름 옆에 표시됩니다 (예: `bob (개발 리드)` 또는 `bob (engineer)`)
- 다른 Anima가 작업의 상담 상대나 위임 대상을 판단하는 단서가 됩니다
- 미설정 시 "(미설정)"으로 표시됩니다

**Anima 생성 시 동작 (`core/anima_factory.py`):**
- `animaworks anima create --from-md PATH [--role ROLE] [--supervisor NAME] [--name NAME]`으로 생성하면, `status.json`에 `supervisor`와 `role`이 기록됩니다
- **supervisor**: `--supervisor` 옵션이 지정되어 있으면 그것을 우선. 미지정 시 캐릭터 시트의 기본 정보 테이블(`| 上司 | name |`)에서 해석
- **speciality**: 캐릭터 시트의 기본 정보 테이블에는 포함되지 않으며, `_create_status_json`도 speciality를 기록하지 않으므로 생성 시 자동 설정되지 않음
- 커스텀 전문 표시가 필요하면 생성 후 `status.json`에 `"speciality": "개발 리드"` 등을 수동으로 추가
- `create_from_template` / `create_blank`으로 생성한 경우에도 동일하게 speciality는 status.json에 자동 설정되지 않음 (템플릿에 status.json이 포함되어 있으면 그 내용이 복사됨)

효과적인 speciality 작성법:
- 구체적이고 짧게: `백엔드 개발`, `고객 지원`, `데이터 분석`
- 애매하지 않게: `여러 가지` → `기획, 조율, 진행 관리`
- 여러 전문 영역이 있는 경우 가운데점으로 구분: `UI 설계 · 프론트엔드 개발`
