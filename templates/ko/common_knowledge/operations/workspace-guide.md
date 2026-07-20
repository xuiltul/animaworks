# 워크스페이스 가이드

Anima가 작업하는 프로젝트 디렉터리(워크스페이스)의 개념과 사용법입니다.

## 워크스페이스란

### 집과 사무실의 개념

Anima는 평소 "자신의 집"(`~/.animaworks/animas/{name}/`)에 있습니다.
여기에는 identity, 메모리, 설정 등 Anima 고유의 데이터가 저장됩니다.

프로젝트 작업(코드 변경, 조사, 빌드 등)을 할 때는
"사무실"(워크스페이스)로 이동하여 작업합니다.
워크스페이스는 프로젝트의 소스 코드와 산출물이 위치하는 디렉터리입니다.

### 레지스트리와 별칭

워크스페이스는 조직 공유 레지스트리(`config.json`의 `workspaces` 섹션)에 등록됩니다.
사람이 부여한 짧은 이름(별칭)과 경로의 SHA-256 처음 8자리(해시)로 고유하게 참조할 수 있습니다.

| 형식 | 예시 | 용도 |
|------|------|------|
| 별칭만 | `myproject` | 일반적인 참조 (충돌 시 해시 형식 사용 권장) |
| 전체 형식 | `myproject#3af4be6e` | 충돌 없는 엄밀한 참조 |
| 해시만 | `3af4be6e` | 별칭을 모르는 경우 |
| 절대 경로 | `/home/user/dev/myproject` | 직접 지정 (레지스트리 미등록도 가능) |

## 도구에서의 사용

### machine_run (Machine Tool)

`working_directory`에 별칭, 전체 형식, 해시 또는 절대 경로를 지정할 수 있습니다.

```bash
animaworks-tool machine run "코드를 리팩터링해 줘" -d myproject
animaworks-tool machine run "테스트를 실행해 줘" -d myproject#3af4be6e
animaworks-tool machine run "빌드해 줘" -d /home/user/dev/myproject
```

### submit_tasks

각 태스크의 `workspace` 필드에 별칭을 지정하면,
TaskExec가 해당 워크스페이스를 작업 디렉터리로 사용합니다.

```
submit_tasks(batch_id="build", tasks=[
  {"task_id": "t1", "title": "Compile", "description": "...", "workspace": "myproject", "parallel": true}
])
```

### delegate_task

`workspace` 필드에 별칭을 지정하면,
위임받은 부하가 해당 워크스페이스에서 작업합니다.

```
delegate_task(name="aoi", instruction="API 테스트를 실행해 줘", deadline="2d", workspace="myproject")
```

## 등록과 할당

### 등록 절차

상세 내용은 `common_skills/workspace-manager` 스킬을 참조하세요.
요점:

1. `config.json`의 `workspaces` 섹션에 별칭과 경로를 추가
2. 또는 `core.workspace.register_workspace`를 Python에서 호출
3. 디렉터리는 등록 시 존재 여부가 확인됩니다 (존재하지 않으면 에러)

### 부하에게 할당

supervisor는 부하의 `status.json`의 `default_workspace` 필드를 업데이트하여
주요 작업 디렉터리를 할당합니다. `workspace-manager` 스킬을 참조하세요.

## 자주 발생하는 문제

### 디렉터리가 존재하지 않음

- **등록 시**: 존재하지 않는 경로를 등록하려 하면 에러가 발생합니다
- **사용 시**: 등록 후 디렉터리가 삭제된 경우, 해석 시 에러가 발생합니다
- **대처**: 경로를 확인하고 올바른 절대 경로로 재등록하세요

### 별칭을 찾을 수 없음

- **원인**: 별칭이 레지스트리에 등록되지 않았거나 오타
- **대처**: `read_memory_file(path="config.json")`로 `workspaces` 섹션을 확인하고 올바른 별칭을 사용하세요

### 해시가 변경됨

- **원인**: 별칭을 덮어써서 경로를 변경한 경우, 해시도 변경됩니다
- **대처**: 전체 형식(`alias#hash`)을 사용 중이었다면 새 해시로 업데이트하세요. 별칭만 사용하고 있었다면 영향 없음
