# 조직 구조의 작동 방식

AnimaWorks에서 조직 구조는 각 Anima의 `status.json`(또는 `IDENTITY.md`)을 Single Source of Truth(SSoT)로 하여 구축된다.
`core/org/org_sync.py`가 디스크 상의 **supervisor**를 `config.json`에 동기화하고, 프롬프트 구축 시에 활용된다.
본 문서에서는 조직 구조가 어떻게 정의·해석·표시되는지 설명한다.

## 데이터 소스와 우선순위

### supervisor（상급자）

조직의 상하 관계는 각 Anima의 `supervisor`에서 정의된다. 읽기 우선순위:

1. **status.json** — `"supervisor"` 키(권장)
2. **IDENTITY.md** — 표 형식 `| 上司 | name |`의 행(일본어만. `core/config/models.py`의 `read_anima_supervisor`가 해석)

`supervisor`가 미설정·비어 있음·「없음」「(없음)」「（없음）」「-」「---」인 경우는 최상위 레벨이 된다.
`animaworks config set animas.<name>.supervisor <supervisor>`은 root 소유 status/config를 일괄 업데이트한다. `config.json`만 직접 변경하면 org_sync에 의해 디스크에서 덮어써지므로 CLI/API를 사용한다.

### speciality（전문）

전문 영역은 `core/prompt/builder.py`의 `_scan_all_animas()`에 따라 다음 우선순위로 해결된다:

1. **status.json** — `"speciality"` 키(자유 텍스트)
2. **config.json** — `animas.<name>.speciality`(status.json에 `speciality` 키가 없는 경우의 폴백)
3. **status.json** — `"role"` 키(위에서 해결되지 않는 경우의 최종 폴백. 역할 이름: engineer, researcher, manager, writer, ops, general)

**주의:** org_sync는 **speciality를 동기화하지 않는다**. speciality는 프롬프트 구축 시 디스크와 config에서 그때마다 해결된다.
`animaworks anima create --from-md`로 생성한 Anima는 `status.json`에 `role`가 들어가지만 `speciality`는 들어가지 않는다.
커스텀 표시(예: 「개발 리드」)로 하고 싶은 경우는 `animaworks config set animas.<name>.speciality "開発リード"`를 사용한다. root 소유 status/config의 양쪽이 모두 업데이트된다.

## org_sync에 의한 config.json 동기화

`core/org/org_sync.py`의 `sync_org_structure()`이 다음을 수행한다:

1. 각 Anima 디렉터리(`IDENTITY.md`가 존재하는 것만)에서 `status.json` / `IDENTITY.md`를 읽고 supervisor를 추출(`read_anima_supervisor`)
2. 순환 참조를 검출(검출된 Anima는 동기화 대상 외)
3. `config.json`의 `animas.<name>.supervisor`를 디스크의 값에 맞춰 업데이트(**supervisor만**)
4. 디스크에 존재하지 않는 Anima의 config 엔트리를 삭제(prune)

**동기화되는 항목:** supervisor만. speciality는 org_sync에서 업데이트되지 않는다.

**실행 타이밍:**

- 서버 시작 시(`animaworks start`의 Anima 프로세스 시작 후)
- Anima가 reconciliation에서 추가되었을 때(`on_anima_added` 콜백)

## supervisor에 의한 계층 정의

- `supervisor: null` 또는 미설정 → 해당 Anima는 최상위 레벨
- `supervisor: "alice"` → alice가 상급자

root 소유의 status 값 예(표시용. 직접 편집하지 말고 CLI/API로 설정):

```json
{
  "enabled": true,
  "supervisor": null,
  "speciality": "経営戦略・全体統括"
}
```

```json
{
  "enabled": true,
  "supervisor": "alice",
  "speciality": "開発リード"
}
```

이 설정으로 다음 계층이 구축된다:

```
alice（経営戦略・全体統括）
├── bob（開発リード）
│   └── dave（バックエンド開発）
└── carol（デザイン・UX）
```

중요한 제약:
- supervisor에 지정하는 이름은 알려진 Anima 이름(영문 이름)이어야 한다
- 순환 참조(alice → bob → alice)는 감지되어 동기화 대상에서 제외된다
- 1명의 Anima가 가질 수 있는 supervisor는 1명뿐이다

## 조직 컨텍스트 구축 프로세스

`core/prompt/builder.py`의 `_build_org_context()`가 디렉터리 스캔과 config.json의 병합 결과에서 다음 정보를 산출한다:

1. **상급자(supervisor)**: 자신의 supervisor 값. 미설정이면 「당신이 최상위입니다」
2. **부하(subordinates)**: supervisor가 자신의 이름으로 되어 있는 모든 Anima
3. **동료(peers)**: 자신과 같은 supervisor를 가진 Anima(자신 제외)

산출 결과는 시스템 프롬프트에 「당신의 조직상의 위치」로 주입된다:

```
## あなたの組織上の位置

あなたの専門: 開発リード

上司: alice (経営戦略・全体統括)
部下: dave (バックエンド開発)
同僚（同じ上司を持つメンバー）: carol (デザイン・UX)
```

## 자신의 위치 읽는 방법

시스템 프롬프트의 「당신의 조직상의 위치」 섹션에서 다음을 확인할 수 있다:

| 항목 | 의미 | 행동에의 영향 |
|------|------|-------------|
| 당신의 전문 | speciality의 값 | 이 분야에 관한 질문이나 판단은 자신이 책임을 진다 |
| 상급자 | 보고 대상 Anima | 진행 상황 보고·문제의 에스컬레이션 대상 |
| 부하 | 자신의 휘하 Anima | 작업의 위임 대상·진행 상황 확인 대상 |
| 동료 | 같은 상급자를 가진 동료 | 관련 업무에서 직접 연계하는 상대 |

### 확인해야 할 포인트

- 상급자가 「(없음 — 당신이 최상위입니다)」라면, 당신은 조직의 최상위로서 전체 책임을 진다
- 부하가 「(없음)」이라면, 당신은 작업 실행자로서 스스로 움직인다
- 동료가 있으면, 관련 업무에서 직접 조정할 수 있다

## 조직 변경 시 동작

조직 구조의 변경은 다음 절차로 반영된다:

1. 조직 설정은 root 소유의 CLI/API에서 변경한다(예: `animaworks config set animas.<name>.supervisor <supervisor>` / `animaworks config set animas.<name>.speciality <speciality>`).
2. CLI/API은 `status.json`와 `config.json`의 양쪽을 업데이트하고, org_sync가 계층을 동기화한다. 다음 reconciliation / prompt에서 새 값이 사용된다.
3. **speciality의 변경:** 프롬프트 구축 시 읽히므로 Anima의 재시작은 불필요. 다음 채팅/하트비트에서 반영된다.

주의점:
- Anima 프로세스에서 `status.json` / `config.json`를 직접 편집하지 않는다. 작성자는 root와 서버 정지 중의 CLI뿐이다.
- 조직 변경 후에는 영향을 받는 Anima에 메시지로 알림을 보내는 것을 SHOULD(권장)

## 조직 구조의 패턴 예

다음은 조직 설정의 예. root 소유 CLI/API에서 설정하고, org_sync가 `supervisor`을 동기화한다. `speciality`은 프롬프트 구축 시 해결된다.

### 패턴 1: 플랫 조직

전원이 최상위. 상하 관계 없음.

각 Anima의 status.json:
```json
{ "supervisor": null, "speciality": "企画" }
{ "supervisor": null, "speciality": "開発" }
{ "supervisor": null, "speciality": "デザイン" }
```

```
alice（企画）
bob（開発）
carol（デザイン）
```

특징:
- 전원이 대등한 입장에서 직접 주고받을 수 있다
- 소규모 팀이나, 각자가 독립된 업무를 가진 경우에 적합하다
- 전원의 동료는 「(없음)」(같은 supervisor를 공유하지 않기 때문)

### 패턴 2: 계층형 조직

명확한 상하 관계가 있다. 가장 일반적인 패턴.

각 계층 필드는 root 소유 CLI/API에서 설정한다(예: `animaworks config set animas.dave.supervisor bob`):

```
alice（CEO・全体統括）
├── bob（開発部長）
│   ├── dave（バックエンド）
│   └── eve（フロントエンド）
└── carol（営業部長）
    └── frank（顧客対応）
```

특징:
- bob과 carol은 동료(같은 supervisor = alice)
- dave와 eve는 동료(같은 supervisor = bob)
- dave에서 frank로의 연락은 bob → alice → carol → frank의 경로를 따른다(타 부서 규칙)

### 패턴 3: 전문가＋매니저형

소수의 매니저가 다수의 전문가를 총괄한다.

```
manager（プロジェクト管理）
├── dev1（API開発）
├── dev2（DB設計）
├── dev3（インフラ）
└── qa（品質保証）
```

특징:
- 전 멤버가 동료 관계. 직접 연계가 용이
- manager가 전체의 작업 배분과 진행 상황 관리를 담당
- 스타트업이나 프로젝트 팀에 적합하다

## speciality의 활용

`speciality`은 root 소유의 `status.json`에 자유 텍스트로 저장된다. `animaworks config set animas.<name>.speciality <value>`로 설정한다. 미설정 시 `role`(역할 이름)이 폴백으로 표시된다.

- 조직 컨텍스트에서 각 Anima의 이름 옆에 표시된다(예: `bob (開発リード)` 또는 `bob (engineer)`)
- 다른 Anima가 작업의 상담 대상이나 위임 대상을 판단하는 단서가 된다
- 미설정인 경우는 「(미설정)」으로 표시된다

**Anima 생성 시 동작(`core/anima/factory.py`):**
- `animaworks anima create --from-md PATH [--role ROLE] [--supervisor NAME] [--name NAME]`로 생성하면 `status.json`에 `supervisor`와 `role`이 기록된다
- **supervisor**: `--supervisor` 옵션이 지정되어 있으면 그것을 우선. 미지정인 경우는 캐릭터 시트의 기본 정보 테이블(`| 上司 | name |`)에서 해석
- **speciality**: 캐릭터 시트의 기본 정보 테이블에는 포함되지 않고, `_create_status_json`도 speciality를 기록하지 않으므로 생성 시 자동 설정되지 않는다
- 커스텀 전문 표시가 필요한 경우는 생성 후 `animaworks config set animas.<name>.speciality "開発リード"`로 설정한다. 설정 파일을 직접 편집하지 않는다
- `create_from_template` / `create_blank`로 생성한 경우도 마찬가지로 speciality는 status.json에 자동 설정되지 않는다(템플릿에 status.json가 포함되는 경우는 그 내용이 복사된다)

효과적인 speciality 작성법:
- 구체적이고 짧게: `バックエンド開発` `顧客サポート` `データ分析`
- 너무 모호하지 않게: `いろいろ` → `企画・調整・進行管理`
- 여러 전문이 있는 경우는 가운뎃점으로 구분: `UI設計・フロントエンド開発`
