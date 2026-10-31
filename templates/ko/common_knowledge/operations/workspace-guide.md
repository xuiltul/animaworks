# 작업 공간 가이드

Anima가 작업하는 프로젝트 디렉터리(작업 공간)의 개념과 사용법.

작업 중인 worktree, 빌드 및 임시 파일의 위치는 `operations/scratch-and-worktrees.md`을 참조한다.

## 작업 공간이란

### 집과 작업장의 개념

Anima는 평소 "자신의 집"(`~/.animaworks/animas/{name}/`)에 있다.
여기에는 IDENTITY.md / SOUL.md, 기억, 설정 등 Anima 고유의 데이터가 저장된다.

프로젝트 작업(코드 변경, 조사, 빌드 등)을 할 때는
"작업장"(작업 공간)으로 나가서 작업한다.
작업 공간은 프로젝트의 소스 코드와 산출물이 있는 디렉터리이다.

### 레지스트리와 별칭

작업 공간은 조직 공유 레지스트리(`config.json`의 `workspaces` 섹션)에 등록된다.
사람이 붙이는 짧은 이름(별칭)과 경로의 SHA-256 앞 8자리(해시)로 고유하게 참조할 수 있다.

| 형식 | 예 | 용도 |
|------|-----|------|
| 별칭만 | `myproject` | 일반적인 참조(충돌 시 해시 포함 권장) |
| 완전형 | `myproject#3af4be6e` | 충돌 없는 엄격한 참조 |
| 해시만 | `3af4be6e` | 별칭을 모를 경우 |
| 절대 경로 | `/home/user/dev/myproject` | 직접 지정(레지스트리 미등록도 가능) |

## 도구에서의 사용

### submit_tasks

각 작업의 `workspace` 필드에 별칭을 지정하면,
TaskExec가 해당 작업 공간을 작업 디렉터리로 사용한다.

```
submit_tasks(batch_id="build", tasks=[
  {"task_id": "t1", "title": "コンパイル", "description": "...", "workspace": "myproject", "parallel": true}
])
```

### delegate_task

`workspace` 필드에 별칭을 지정하면,
위임받은 부하가 해당 작업 공간에서 작업한다.

```
delegate_task(name="aoi", instruction="API テストを実施して", workspace="myproject")
```

## 등록과 할당

### 등록 절차

자세한 내용은 `common_skills/workspace-manager` 스킬을 참조할 것.
요점:

1. `config.json`의 `workspaces` 섹션에 별칭과 경로를 추가
2. 또는 `core.org.workspace.register_workspace`을 Python에서 호출
3. 디렉터리는 등록 시 존재 확인됨(존재하지 않으면 오류)

### 부하에게 할당

슈퍼바이저는 부하의 `status.json`의 `default_workspace` 필드를 업데이트하고,
주요 작업 디렉터리를 할당한다. `workspace-manager` 스킬을 참조.

## 자주 있는 문제

### 디렉터리가 존재하지 않음

- **등록 시**: 존재하지 않는 경로를 등록하려고 하면 오류가 발생
- **사용 시**: 등록되어 있어도 나중에 디렉터리가 삭제된 경우, 해결 시 오류가 발생
- **대처**: 경로를 확인하고 올바른 절대 경로로 재등록

### 별칭을 찾을 수 없음

- **원인**: 별칭이 레지스트리에 등록되어 있지 않거나 오타
- **대처**: `read_memory_file(path="config.json")`에서 `workspaces` 섹션을 확인하고 올바른 별칭을 사용

### 해시가 변경됨

- **원인**: 별칭을 덮어써서 경로를 변경한 경우, 해시도 변경됨
- **대처**: 완전형(`alias#hash`)을 사용하고 있었다면 새 해시로 업데이트. 별칭만 사용하고 있었다면 영향 없음
