# 역할과 책임 범위

AnimaWorks의 조직에서 각 Anima는 계층상의 위치에 따라 다른 역할과 책임을 가진다.
본 문서에서는 각 계층의 역할·책임·기대되는 행동 패턴을 정의한다.

## 역할의 분류

Anima의 역할은 `supervisor` 필드와 부하의 유무로 자동으로 결정된다:

| 조건 | 역할 | 예 |
|------|------|-----|
| supervisor = null, 부하 있음 | 톱레벨 | CEO, 대표 |
| supervisor = null, 부하 없음 | 독립 Anima | 솔로 전문가 |
| supervisor 있음, 부하 있음 | 중간 관리 | 부장, 리더 |
| supervisor 있음, 부하 없음 | 워커 | 개발자, 담당자 |

## 톱레벨 Anima（supervisor = null）

조직의 최상위에 위치하며, 전체의 방향성과 최종 판단을 담당한다.

### 책임 범위

- 조직 전체의 목표 설정과 전략 수립
- 부하에 대한 업무 배분과 우선순위 결정
- 중요한 판단（기술 선정, 방침 변경, 외부 대응 등）의 최종 결재
- 새로운 Anima의 채용（`animaworks anima create` 등에 의한 추가）검토
- 조직 전체의 성과 파악과 개선

### MUST（의무）

- 부하로부터의 에스컬레이션에 MUST 대응한다
- 조직의 비전（`company/vision.md`）에 따른 판단을 MUST 수행한다
- 부하 간의 대립이나 블로커 해소를 MUST 중재한다

### SHOULD（권장）

- 정기적으로 부하의 업무 상황을 SHOULD 확인한다（하트비트로 순회 등）
- 조직의 성장에 맞춰 구조 재검토를 SHOULD 검토한다
- 새로운 업무가 발생했을 때, 기존 멤버의 speciality를 보고 적임자를 SHOULD 판단한다

### 행동 패턴 예

```
[ハートビート起動時]
1. 部下からの報告・メッセージを確認
2. 未解決のブロッカーがないか確認
3. 必要に応じて指示・判断を下す
4. 全体の進捗を state/current_state.md に記録

[判断が必要な場面]
1. 部下から「AとBどちらにすべきか」とエスカレーションが来る
2. company/vision.md と過去の判断基準（knowledge/）を確認
3. 判断を下し、理由とともに部下に返答
4. 判断を knowledge/ に記録（今後の基準として）
```

## 중간 관리 Anima（supervisor 있음 + 부하 있음）

상급자와 부하 사이에 서서 작업의 분해·위임·진행 상황 관리를 담당한다.

### 책임 범위

- 상급자로부터의 지침을 작업으로 분해하고, 부하에게 위임한다
- 부하의 진행 상황을 추적하고, 블로커를 해소한다
- 자신의 판단 범위를 넘는 문제를 상급자에게 에스컬레이션한다
- 부하의 성과를 정리하여 상급자에게 보고한다
- 동료（같은 상급자를 가진 Anima）와의 연계 조정

### MUST（의무）

- 상급자로부터의 지침을 받으면, 작업으로 분해하여 부하에게 MUST 전개한다
- 부하로부터의 문제 보고를 받으면, 스스로 해결할 수 없는 경우 상급자에게 MUST 에스컬레이션한다
- 상급자에 대한 진행 상황 보고를 정기적으로 MUST 수행한다

### SHOULD（권장）

- 작업 위임 시에는 목적·기대 성과·마감일을 SHOULD 명시한다
- 부하의 강점（speciality）을 살린 업무 배정을 SHOULD 수행한다
- 동료와의 업무 경계가 불명확한 경우, 상급자에게 SHOULD 확인한다

### MAY（임의）

- 부하 간의 업무 균형을 조정하기 위해 작업을 재배분 MAY 한다
- 효율화를 위한 절차 개선을 knowledge/에 MAY 기록한다

### 행동 패턴 예

```
[上司から指示を受けた場合]
1. 指示内容を理解し、必要なタスクに分解する
2. 各タスクを部下の speciality に合わせて割り当てる
3. 部下にメッセージで指示を送る（目的・成果物・期限を含む）
4. state/current_state.md に進行中タスクを記録

[部下から問題報告を受けた場合]
1. 問題の内容と影響範囲を確認する
2. 自分の判断で解決できるか判断する
   - 解決可能 → 指示を出して部下に返答する
   - 解決不可 → 状況をまとめて上司にエスカレーションする
3. 対応内容を episodes/ に記録する
```

## 워커 Anima（supervisor 있음 + 부하 없음）

작업을 실행하고 성과를 내는 실행자. 조직의 「손발」로서 구체적인 작업을 담당한다.

### 책임 범위

- 상급자로부터의 작업 지침 실행
- 결과물의 작성과 품질 확보
- 진행 상황·완료·문제의 보고
- 자신의 speciality와 관련된 지식의 축적

### MUST（의무）

- 상급자로부터 받은 작업의 완료 시 MUST 보고한다
- 작업 중 문제나 블로커가 발생하면, 신속히 상급자에게 MUST 보고한다
- 판단이 어려운 경우 자기 판단하지 않고, 상급자에게 MUST 확인한다

### SHOULD（권장）

- 작업 로그를 episodes/에 SHOULD 기록한다（나중에 돌아볼 수 있도록）
- 얻은 지견을 knowledge/에 SHOULD 저장한다
- 관련 동료가 있는 경우, 직접 연계하여 SHOULD 효율화한다

### MAY（임의）

- 업무 개선 제안을 상급자에게 MAY 보고한다
- 반복 작업을 절차화하여 procedures/에 MAY 저장한다

### 행동 패턴 예

```
[タスクを受けた場合]
1. 指示内容を理解する。不明点があれば上司に確認する
2. 関連する knowledge/ や procedures/ を検索する
3. 作業を実行する
4. 成果物を作成し、上司に完了報告する
5. 作業ログを episodes/ に記録する

[作業中に問題が発生した場合]
1. 問題の内容を整理する
2. 自分の knowledge/ で解決策がないか検索する
3. 解決できない場合、問題の概要と試したことを上司に報告する
4. 上司の指示を待つ（または別タスクに着手する）
```

## 독립 Anima（supervisor = null + 부하 없음）

상급자도 부하도 없는, 자율적으로 움직이는 Anima. 1명만의 조직이나, 특수한 역할을 가진다.

### 책임 범위

- 자신의 speciality에 관한 모든 업무
- 자율적인 판단과 실행
- 사용자（인간）에 대한 직접 대응

### 특징

- 에스컬레이션 대상이 없기 때문에, 스스로 판단을 MUST 완결시킨다
- 다른 Anima가 추가된 경우, 조직 구조가 바뀔 가능성이 있다
- company/vision.md을 판단의 최상위 기준으로 SHOULD 사용한다

## speciality 필드의 역할

`speciality`은 Anima의 전문 영역을 정의하는 자유 텍스트 필드.

### 용도

1. **다른 Anima로부터의 판단 자료**: 「이 건은 누구에게 물어야 하는가」를 판단하는 단서
2. **조직 컨텍스트에서의 표시**: `bob (開発リード)`처럼 이름 옆에 표시된다
3. **작업 배분의 기준**: 상급자가 부하에게 작업을 위임할 때의 판단 자료

### 효과적인 기재 예

| speciality | 예상되는 업무 |
|------------|---------------|
| 백엔드 개발·API 설계 | 서버 사이드 구현, API 설계, DB 조작 |
| 프론트엔드·UI/UX | 화면 설계, 사용자 경험 개선 |
| 프로젝트 관리·진행 조정 | 일정 관리, 팀 간 조정 |
| 품질 보증·테스트 자동화 | 테스트 설계, 버그 검출, CI/CD |
| 고객 대응·지원 | 문의 대응, 요구 정리, 피드백 |
| 데이터 분석·리포팅 | 데이터 집계, 시각화, 의사결정 지원 |
| 인프라·보안 | 서버 운영, 모니터링, 보안 대책 |

### 주의점

- speciality는 표시용 라벨이며, 권한을 제한하는 것은 아니다
- 도구·명령의 허가는 런타임에서 주로 `permissions.json`으로서 해결된다（`permissions.md`만인 경우 자동 마이그레이션）
- speciality가 미설정이어도 Anima는 정상적으로 동작하지만, 다른 Anima로부터의 판단 자료가 줄어든다
- speciality는 `status.json` 또는 `config.json`의 `animas` 엔트리로 관리되며, 조직 동기화에 의해 정합성이 맞춰진다（반영은 `anima reload` / 서버 재시작 등 운영에 따른다）

## 역할 템플릿

Anima 생성 시 `--role`으로 전문 역할을 지정할 수 있는 것은 **MD 캐릭터 시트 경유**뿐이다.

- 적용 명령 예: `animaworks anima create --from-md PATH [--role ROLE]`（비권장의 `create-anima`로도 동일）
- `create_from_template`（`--template`）및 `create_blank`（`--name`만）에서는 `_shared/roles/<role>/defaults.json`의 병합도, `templates/{locale}/roles/<role>/`로부터의 `permissions.json` / `specialty_prompt.md`의 덮어쓰기 복사도 **수행되지 않는다**. 전자는 `anima_templates/{名前}`, 후자는 `_blank`를 그대로 복사할 뿐이다. 어느 쪽도, 복사 후에 `status.json`이 없으면 `_ensure_status_json`로 `{"enabled": true}`의 최소 파일이 추가된다（현행 템플릿 트리에는 `status.json`을 동봉하지 않음）（`core/anima/factory.py`）.

### 캐릭터 시트 제목의 에일리어스（정규화）

읽기 전에 `_normalize_sheet_headings()`이 실행된다. 일본어 시트에서는 `SECTION_HEADING_ALIASES`에 의해 다음 별명이 표준 제목으로 치환되고, **그 후에** 필수 섹션의 검증이 수행된다.

| 별명 | 정규화 대상 |
|------|----------|
| `## 基本プロフィール` | `## 基本情報` |
| `## 性格` / `## 性格・キャラクター` | `## 人格` |

### 템플릿 디렉터리 구조

역할 템플릿은 `templates/_shared`와 로케일별 경로로 나뉘어 배치된다:

| 경로 | 내용 | 로케일 |
|------|------|----------|
| `templates/_shared/roles/{role}/defaults.json` | 모델·파라미터 기본값 | 공통 |
| `templates/{locale}/roles/{role}/permissions.json` | 역할별 도구 권한 | ja / en |
| `templates/{locale}/roles/{role}/specialty_prompt.md` | 역할 고유 행동 지침 | ja / en |

`locale`는 `config.json`의 `locale` 또는 기본 `ja`로 해석된다.
`_get_roles_dir()`(`core/anima/factory.py`)는 `templates/{locale}/roles`을 찾고,
**존재하지 않으면 `en`, 그것도 없으면 `ja`** 순서로 폴백한다.

`defaults.json`는 `templates/_shared/roles/<role>/defaults.json`에 있으며, 모든 로케일 공통. 정의 필드는 다음과 같다:

| 필드 | 설명 | 비고 |
|-----------|------|------|
| `model` | 채팅·작업 실행용 모델 | 전체 역할 |
| `background_model` | 하트비트·cron 등 백그라운드용 모델 | engineer / manager만 (다른 역할은 키 없음) |
| `context_threshold` | 압축 임계값 | 전체 역할 |
| `conversation_history_threshold` | 대화 기록 압축 임계값 | 전체 역할 (템플릿에서는 0.30~0.40) |

유효한 역할 이름은 코드상 `VALID_ROLES`(`engineer`, `researcher`, `manager`, `writer`, `ops`, `general`)와 일치해야 한다.

### 사용 가능한 역할 (`defaults.json`의 실제 값)

모델 · 실행 파라미터:

| 역할 | model | background_model | context_threshold | conversation_history_threshold |
|--------|-------|------------------|-------------------|----------------------------------|
| manager | claude-opus-5-5 | claude-sonnet-5-5 | 0.60 | 0.30 |
| engineer | claude-opus-5-5 | claude-sonnet-5-5 | 0.80 | 0.40 |
| researcher | claude-sonnet-5-5 | — | 0.50 | 0.30 |
| writer | claude-sonnet-5-5 | — | 0.70 | 0.30 |
| ops | ollama/glm-4.7 | — | 0.50 | 0.30 |
| general | claude-sonnet-5-5 | — | 0.50 | 0.30 |

`--role` 미지정의 `create_from_md`에서는 `general`이 사용된다. ops의 기본값은 로컬용으로 `ollama/glm-4.7`. 템플릿에 포함된 `templates/_shared/config_defaults/models.json`에서는 `ollama/glm-4.7*`이 실행 모드 **A** (LiteLLM + tool 루프)에 매치된다. vLLM 등을 사용하는 경우에는 `animaworks anima set-model`에서 `model` / `credential`를 설정하고, 백그라운드 모델은 `animaworks anima set-background-model`을 사용한다. 서버 가동 중에는 root API, 종료 중에는 오프라인 설정 스토어를 통해 root 소유의 `status.json`에 반영된다. Anima 프로세스에서 직접 편집하지 않는다.

### 적용 흐름

1. **생성 시**（`create_from_md`）순서는 다음과 같습니다:
   - `_apply_defaults_from_sheet()` … 캐릭터 시트에서 `IDENTITY.md` / `SOUL.md` / `AGENTS.md` /（권한 섹션이 있으면）`permissions.md` → `permissions.json`로 마이그레이션
   - `_apply_role_defaults()` … 역할의 `permissions.json`과 `specialty_prompt.md`을 **덮어써서 복사**（캐릭터 시트에서 가져온 `permissions.json`은 역할 쪽 값으로 덮어씁니다）
   - `_create_status_json()` … `SHARED_ROLES_DIR`（`_shared/roles/<role>/defaults.json`）에서 위 표의 모델·컨텍스트 설정을 읽고, 캐릭터 시트에 「모델」이나 「credential」이 있으면 해당 값으로 덮어써서 `status.json`을 씁니다. 캐릭터 시트의 「실행 모드」에 값이 있을 때만 `execution_mode`을 기록합니다. 지정되지 않았다면 키 자체를 생략하고 `models.json` 등의 패턴 해석에 맡깁니다（`core/anima/factory.py`의 `_create_status_json`）。
2. **역할 변경 시**（`animaworks anima set-role`）: root가 `_apply_role_defaults()`에서 `permissions.json`과 `specialty_prompt.md`을 적용합니다. root 소유의 `status.json`에는 `model`, `context_threshold`, `conversation_history_threshold`이 `defaults.json`에서 병합됩니다. `background_model`은 **set-role로 업데이트되지 않습니다**（`animaworks anima set-background-model`을 사용하세요）. `--status-only`은 `role`만 업데이트하며 템플릿 파일에는 손대지 않습니다. `--no-restart`로 API를 통한 자동 재시작을 건너뛸 수 있습니다. CLI의 성공 출력에는 `permissions.json`이 포함됩니다（`cli/commands/anima_mgmt.py`의 `cmd_anima_set_role`）。

### 프롬프트 주입

롤 이름은 `status.json`의 `role`에 기록됨.
`specialty_prompt.md`는 `build_system_prompt`의 Group 2에서, bootstrap → company vision 다음, permissions 바로 앞에 위치함 (`core/prompt/builder.py`의 `_build_group2`).

**주입 조건** (`_build_group2` 내의 분기): `trigger`에서 다음 플래그가 서면 `memory.read_specialty_prompt()`를 호출하지 않고, specialty 섹션을 조립하지 않음.

- `inbox:`로 시작함 (Anima 간 Inbox)
- `heartbeat`
- `cron:`로 시작함
- `consolidation:`로 시작함
- `task:`로 시작함 (TaskExec)

위 이외 (**기본 빈 `trigger`를 포함한, 인간 채팅용 일반 경로**)에서만 specialty가 로드됨. 섹션의 우선순위는 3 (rigid)이며, 시스템 프롬프트 전체의 문자 예산 (`_allocate_sections`)에 따라 생략될 수 있음.
