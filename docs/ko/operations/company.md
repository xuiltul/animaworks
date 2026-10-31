<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/company.md -->
<!-- i18n: source-sha256=2327b7793a54d6271a2504245e4bb7059a9dd22cee520d3789b3fb4b120e5ece generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 581e20f1

# 회사 관리

회사 기능은 하나의 AnimaWorks 런타임 내에서 Anima, 공유 자산, 채널을 회사 단위로 정리하는 메커니즘이다. 회사 이름이나 표시 이름에는 가상의 예시만 사용한다.

## 데이터 배치와 소속

회사별 데이터는 `companies/<company>/`에 두며, Anima의 소속은 각 `animas/<name>/status.json`의 `company`가 나타낸다. 회사 디렉터리에는 표시 정보, 회사의 지식·스킬, 공유 영역 등이 포함된다. `companies/<company>/animas/`는 소속 Anima를 참조하는 뷰이며, Anima의 실체는 `animas/` 쪽에 있다.

정보의 공유 범위는 다음과 같이 나뉜다.

- 개별 지식·스킬은 해당 Anima의 영역에 둔다.
- 회사에 공유하는 지식·스킬이나 작업 파일은 해당 회사의 영역에 둔다.
- 전사 공통으로 해도 되는 내용만 공통 영역에 둔다.

회사가 다른 Anima 간의 직접 통신이나 위임은 경계 판정의 대상이 된다. 타사 영역에 대한 읽기·쓰기도 제한된다. 회사 경계를 확인할 수 없는 작업은 거부된다. 소속을 설정하지 않은 Anima는 회사 분리의 대상이 아니므로, 격리가 필요하면 명시적으로 회사에 할당한다.

## CLI

`cli/commands/company_cmd.py`이 `animaworks company`의 각 명령을 등록한다.

```sh
animaworks company create example-company --display-name "Example Company"
animaworks company list
animaworks company assign sample-anima --to example-company
animaworks company assign sample-anima --unassign
```

`create`은 회사 영역을 생성하고 부족한 골격을 보완한다. `list`는 회사와 소속 상태를 표시한다. `assign`는 Anima의 소속을 변경한다. `--unassign`은 소속을 해제한다.

기존 자산을 회사 영역으로 옮길 경우 `animaworks company adopt <path> --to <company>`을 사용한다. 이동 전에 백업이 생성되며, 기본적으로 이전 위치에 상대 링크가 남는다. dry-run이 가능한 `split`은 manifest에서 회사 생성·소속 변경·자산 이동을 한꺼번에 계획하고, `--execute`를 붙인 경우에만 적용한다. `export`은 회사 이전용 bundle을 생성한다. 이동·분할·export는 범위를 확인한 후 실행하고, 완료 후에는 백업과 출력 내용을 대조한다.

## 채널의 회사 소속

open channel의 회사 범위는 채널 메타데이터의 `company`에서 관리한다. 외부 메시징에서 생성하는 board에는 연동별 `default_channel_company`을 지정할 수 있다. 기존 채널을 변경하는 경우 대상 채널의 회사와 멤버를 확인한다. 설정의 모든 항목은 [설정 참조](../reference/config.md)를 참조한다.

## GitHub 계정 할당

`config.json`의 `github_identities`는 GitHub CLI에서 어떤 계정의 자격 증명을 사용할지 선택하는 대응표이다. `anima:<name>`은 Anima 단위의 지정, 회사 slug는 회사 공통의 기본 지정으로 사용되며, Anima 단위의 지정이 우선된다.

```json
{
  "github_identities": {
    "example-company": "example-github-account",
    "anima:sample-anima": "sample-github-account"
  }
}
```

해당하는 GitHub CLI 계정에 로그인되어 있는지 확인한다. 토큰 값은 설정 파일로 복제하지 않고, 명령이 필요한 계정에서 자격 증명을 가져온다. 해결할 수 없으면 일반적인 GitHub CLI의 활성 계정으로 돌아간다.

## 운영상 확인

소속 변경은 `status.json`을 정본으로 하여 반영된다. 회사 공유 자산이나 검색 색인을 변경한 경우에는 소속처와 참조 범위를 확인하고, 필요에 따라 공유 색인을 업데이트한다. 공통 영역에는 회사 고유 정보, 개인 정보, 자격 증명을 두지 않는다.
