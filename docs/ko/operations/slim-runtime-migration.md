<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/slim-runtime-migration.md -->
<!-- i18n: source-sha256=6efe1c3dda4226990ef09df2a9fb50b937e6ddd8f9b83b0238a10ab3d6ea2bd1 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 581e20f1

# Task Store로의 런타임 마이그레이션

이 절차는 기존 작업 투입 데이터를 `shared/taskboard.sqlite3`의 영구 Task Store로 마이그레이션할 때 사용한다. 새 런타임에서 이전 형식의 입력이 발견되어도 자동으로 실행 대상에 섞지 않는다. 마이그레이션이 필요한 담당만 종료·대조하여 전환한다.

## 마이그레이션 전 확인

- 이전 코드와 새 코드를 동일한 데이터 디렉터리에 동시에 연결하지 않는다.
- `animaworks task-store status --anima <name>`에서 대상 Anima의 상태를 확인한다.
- `animaworks task-store quiesce --anima <name>`에서 새 claim을 중지한다. 실행 중인 시도는 종료될 때까지 기다리고, 서버와 대상 worker를 종료한다.
- SQLite의 backup 출력 위치를 새 경로로 지정하고, 여유 용량과 접근 권한을 확인한다.

## 마이그레이션 및 대조

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store status --anima sample
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store quiesce --anima sample
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store migrate \
  --anima sample --backup /path/to/backup/before.sqlite3
```

마이그레이션 처리는 입력 형식, 실행 중 lease, 충돌 등을 검사한 후 저장한다. 비정상적인 행, 결과가 불명확한 실행 중 작업, 충돌한 입력이 있으면 종료한다. 표시된 오류를 해결한 후 다시 상태를 조사한다. 원본 파일은 증적으로 보관하며, 마이그레이션이 완료되었다는 이유로 삭제하지 않는다.

마이그레이션 건수와 작업 ID를 대조하고, 지침, 제약, model, workspace, 의존 관계, 추적 정보가 의도한 내용인지 확인한다. 결과가 확정되지 않은 작업은 외부 상태를 확인하고 자동으로 재투입하지 않는다. 절차서, Anima별 지침, 공유 템플릿에 이전 투입 방법이 남아 있지 않은지도 검토한다.

대조가 끝난 대상만 재개한다.

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store resume --anima sample
```

그 후 새 런타임을 시작하여 Task Store의 상태와 실제 처리를 확인한다. 한 번에 모든 Anima를 전환할 필요는 없다.

## 작업 상태와 재개

Task Store는 task ID와 시도 ID를 사용하여 claim과 업데이트를 관리한다. worker가 소유한 실행 중 상태를 다른 실행 경로에서 덮어쓰지 않는다. 완료 또는 취소된 작업은 동일한 ID로 재투입하지 않고, 별도의 요청으로 새 ID를 사용한다. 미완료 작업을 재개할 때는 저장된 입력을 참조하고, 중복된 시도가 존재하지 않는지 확인한다.

DB의 claim은 외부 서비스에 대한 부작용이 정확히 한 번만 발생하는 것을 보장하지 않는다. 전송이나 변경의 결과가 불명확한 경우 외부의 실제 상태를 확인한 후 판단한다.

## 롤백

완료된 작업을 다시 실행하지 않기 위해, 이전 DB backup을 그대로 복원하여 시작하지 않는다. 새 런타임을 종료하고, 현재 Task Store 상태를 다른 위치로 export한 후 필요한 설정·산출물을 개별적으로 대조한다. 롤백 시에도 이전·새 프로세스가 동일한 데이터 디렉터리를 동시에 사용하지 않도록 한다.

자세한 CLI 옵션과 최신 상태는 `animaworks task-store --help` 및 [CLI 참조](../reference/cli.md)를 확인한다.
