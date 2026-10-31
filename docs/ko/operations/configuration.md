<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/configuration.md -->
<!-- i18n: source-sha256=7af4d2067fc6d903b130aec1b7be2e9f06d27f3968f81963e44a6b4116d85779 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 581e20f1

# 설정 관리

AnimaWorks의 설정은 런타임 전체의 동작, 모델 해석, Anima별 상태, 권한 정책으로 나뉜다. 용도가 다른 파일을 하나로 묶지 말고, 변경 대상과 반영 방법을 선택한다.

## 설정 파일의 역할

| 파일 | 주요 역할 | 범위 |
|---|---|---|
| `config.json` | 서버, 연동, 메모리, GPU, 공통 기본값 등의 런타임 설정 | 런타임 전체. |
| `models.json` | 모델 이름 패턴별 실행 모드 및 모델 메타데이터 | 모델 해석. |
| `animas/<name>/status.json` | Anima별 모델, 실행 모드, 소속, 상태 등 | 개별 Anima. |
| `permissions.global.json` | 모든 Anima에 적용하는 명령 금지 규칙 및 입력 감지 설정 | 런타임 전체의 보안 경계. |
| `animas/<name>/permissions.json` | 개별 Anima의 파일, 명령, 외부 도구 권한 | 개별 Anima. |

파일의 실제 위치는 `ANIMAWORKS_DATA_DIR`가 설정되어 있으면 해당 디렉터리, 미설정이면 기본 데이터 디렉터리이다. 경로와 항목의 전체 목록은 [설정 참조](../reference/config.md)를 참조한다.

## 우선순위

런타임 설정의 공통 값은 `config.json`이 가지며, Anima 고유의 지정은 그 위에 덮어쓰기로 적용된다. 모델의 실행 모드는 Anima의 `status.json`에 명시한 `execution_mode`, `models.json`의 패턴, 호환용 `config.json`의 `model_modes`, 코드 기본값 순서로 해석된다. 모델 이름이 어떤 패턴에도 일치하지 않으면 `A`이 된다.

`permissions.global.json`과 `permissions.json`은 일반 설정과 별개의 권한 정책이다. 개별 deny는 허용보다 우선한다. 글로벌 설정은 서버 시작 시 읽히고, 실행 중 정책으로 캐시된다.

## 편집 방법

`animaworks config get <dot.path>`, `animaworks config set <dot.path> <value>`, `animaworks config list`로 `config.json`을 참조·편집할 수 있다. 값은 진릿값·숫자·`null`로 해석된다. 비밀 값을 터미널이나 로그에 무심코 표시하지 않는다.

Anima의 모델 변경에는 `animaworks anima set-model <name> <model>`를 사용한다. `models.json`은 모델 이름의 대응표이다. 대응하는 편집 명령이 없는 항목을 수동으로 변경하는 경우, 파일을 백업하고 구문과 값을 확인한 뒤 저장한다. 권한 파일은 접근 경계이므로 의도하지 않은 허용 범위 확대가 없는지 반드시 검토한다.

## 변경 반영

- `status.json`의 Anima 설정 변경은 `animaworks anima reload <name>`로 실행 중인 Anima에 다시 읽게 한다. 프로세스나 실행 환경의 초기화를 수반하는 변경은 `animaworks anima restart <name>`를 사용한다.
- `config.json`의 일반 설정이나 연결 설정은 대상 설정이 지원하는 hot reload를 사용하거나 서버를 재시작한다. 반영 방법을 판단할 수 없으면 재시작한다.
- `permissions.global.json`는 시작 시 캐시된다. 변경을 반영하려면 서버를 재시작한다.
- `models.json`의 모델 패턴은 캐시되지만 파일 업데이트를 감지하여 다시 읽는다. 변경 후 대상 Anima의 다음 실행에서 해석을 확인한다.

재시작 대상을 잘못 선택하면 설정 파일만 변경되고 실행 중인 연결이나 프로세스에 반영되지 않을 수 있다. 변경 후에는 `animaworks anima status <name>`이나 서버의 health/status을 확인하여 의도한 값이 유효한지 확인한다.
