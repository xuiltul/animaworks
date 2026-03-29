---
name: tool-creator
description: >-
  AnimaWorks용 Python 외부 도구 모듈을 만드는 메타 스킬. ExternalToolDispatcher·get_credential·permissions를 다룬다.
  Use when: core/tools에 모듈 추가, Web API 래퍼 구현, animaworks-tool로 호출할 커스텀 도구 개발이 필요할 때.
---

# tool-creator

## 개요

AnimaWorks의 도구는 3가지로 분류됩니다:

| 유형 | 배치 위치 | 탐색 방법 |
|------|----------|----------|
| **코어 도구** | `core/tools/*.py` | `discover_core_tools()` → TOOL_MODULES (시작 시 고정) |
| **공유 도구** | `{data_dir}/common_tools/*.py` | discover_common_tools() |
| **개인 도구** | `{anima_dir}/tools/*.py` | discover_personal_tools() |

`{data_dir}`은 통상 `~/.animaworks/`입니다. 개인 및 공유 도구는 `ExternalToolDispatcher`가 `refresh_tools`로 재스캔하여 핫 리로드 가능합니다. ToolHandler는 `write_memory_file`로 `tools/*.py`에 쓸 때 permissions.json의 도구 생성 권한을 확인합니다.

개인 및 공유 도구는 **Bash**를 통해 `animaworks-tool <도구> <서브커맨드> [인수]`로 호출합니다. 스키마명 형식은 `{tool_name}_{action}` (예: `my_tool` + `query` → `my_tool_query`)입니다.

## 절차

### Step 1: 도구 설계

1. 도구명 결정 (스네이크 케이스, 예: `my_api_tool`)
2. 제공할 스키마(조작)를 정의
3. 필요한 파라미터를 정의

### Step 2: 모듈 파일 생성

아래 템플릿에 따라 Python 파일을 생성합니다.

#### 단일 스키마 도구 (간단)

파일명 `my_tool.py`의 경우, `animaworks-tool my_tool action [args]`로 호출되어 스키마명 `my_tool_action`이 dispatch에 전달됩니다.

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("animaworks.tools")


def get_tool_schemas() -> list[dict]:
    """도구 스키마를 반환합니다 (필수)."""
    return [
        {
            "name": "my_tool_action",
            "description": "이 도구가 수행하는 작업 설명",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "파라미터 설명",
                    },
                    "param2": {
                        "type": "integer",
                        "description": "선택 파라미터",
                        "default": 10,
                    },
                },
                "required": ["param1"],
            },
        }
    ]


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """스키마명에 따른 처리를 실행합니다 (권장)."""
    args.pop("anima_dir", None)  # 프레임워크에서 주입되지만 이 도구에서는 미사용
    if name == "my_tool_action":
        return _do_action(
            param1=args["param1"],
            param2=args.get("param2", 10),
        )
    raise ValueError(f"Unknown tool: {name}")


def _do_action(param1: str, param2: int = 10) -> dict[str, Any]:
    """실제 처리 로직."""
    # 여기에 구현
    return {"result": f"Processed {param1} with {param2}"}
```

#### 다중 스키마 도구 (API 연동 등)

스키마명은 `{tool_name}_{action}` 형식입니다. `animaworks-tool myapi query [args]` 호출 시 `myapi_query`로 dispatch에 전달됩니다. 파일명: `myapi.py`.

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("animaworks.tools")


def get_tool_schemas() -> list[dict]:
    return [
        {
            "name": "myapi_query",
            "description": "API에 쿼리를 보내고 결과를 가져옵니다",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색 쿼리"},
                    "limit": {"type": "integer", "description": "최대 건수", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "myapi_post",
            "description": "API에 데이터를 전송합니다",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "전송 데이터"},
                },
                "required": ["data"],
            },
        },
    ]


class MyAPIClient:
    """API 클라이언트."""

    def __init__(self) -> None:
        from core.tools._base import get_credential
        self._api_key = get_credential(
            "myapi", "myapi_tool", env_var="MYAPI_KEY",
        )

    def query(self, query: str, limit: int = 10) -> list[dict]:
        import httpx
        resp = httpx.get(
            "https://api.example.com/search",
            params={"q": query, "limit": limit},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def post(self, data: str) -> dict:
        import httpx
        resp = httpx.post(
            "https://api.example.com/data",
            json={"data": data},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def dispatch(name: str, args: dict[str, Any]) -> Any:
    args.pop("anima_dir", None)  # 프레임워크에서 주입되지만 이 도구에서는 미사용
    client = MyAPIClient()
    if name == "myapi_query":
        return client.query(
            query=args["query"],
            limit=args.get("limit", 10),
        )
    elif name == "myapi_post":
        return client.post(data=args["data"])
    raise ValueError(f"Unknown tool: {name}")
```

### Step 3: 파일 저장

개인 도구로 저장 (`write_memory_file`의 path는 anima_dir 기준 상대 경로):

```
write_memory_file(path="tools/my_tool.py", content=<코드>)
```

`tools/`에 쓰려면 permissions.json의 "도구 생성" 섹션에서 **개인 도구** 허가가 필요합니다.

### Step 4: 도구 활성화

저장 후 `refresh_tools`를 호출하여 핫 리로드합니다:

```
refresh_tools()
```

세션을 재시작하지 않아도 즉시 도구를 사용할 수 있습니다. 개인 도구는 permissions.json의 외부 도구 섹션에 등록할 필요가 없으며, `refresh_tools`로 탐색되면 **Bash** + `animaworks-tool <도구> <서브커맨드>`로 호출 가능합니다.

### Step 5: 공유 (선택)

다른 Anima도 사용하게 하려면 도구를 공유합니다:

```
share_tool(tool_name="my_tool")
```

이렇게 하면 `~/.animaworks/common_tools/`에 복사되어 모든 Anima에서 사용 가능해집니다. 공유에는 permissions.json에서 **공유 도구** 허가가 필요합니다.

## 필수 인터페이스

| 함수 | 필수 | 설명 |
|------|------|------|
| `get_tool_schemas()` | 필수 | 도구 스키마 목록을 반환. `name`, `description`, `input_schema` (또는 `parameters`) 포함 필요 |
| `dispatch(name, args)` | 권장 | 스키마명에 따른 디스패치. ExternalToolDispatcher가 우선 호출. `args`에서 `args.pop("anima_dir", None)`으로 제거 필요 |
| 스키마명과 동명의 함수 | 대안 | `dispatch()` 대신 사용 가능 |
| `cli_main(argv)` | 선택 | `animaworks-tool <tool_name>`으로 단독 실행용 |
| `EXECUTION_PROFILE` | 선택 | 장시간 실행 도구용. `animaworks-tool submit`으로 백그라운드 투입 가능 |

## Bash 호출

Anima는 **Bash**를 통해 `animaworks-tool <도구> <서브커맨드> [인수]`로 개인/공유 도구를 호출합니다:

```bash
animaworks-tool myapi query "검색어" [--limit 10]
```

`schema_name = tool_name + "_" + action`이 `dispatch(name, args)`에 전달됩니다. 위 예시에서 `name="myapi_query"`입니다.

## 스키마 정의 규약

`input_schema`와 `parameters` 모두 지원되며 `core/tooling/schemas._normalise_schema`에서 정규화됩니다.

```python
{
    "name": "tool_action_name",       # 스네이크 케이스. {tool_name}_{action} 형식
    "description": "1~2문장 설명",     # LLM이 도구 선택에 사용
    "input_schema": {                  # JSON Schema 형식 (parameters도 가능)
        "type": "object",
        "properties": { ... },
        "required": [ ... ],
    },
}
```

## 인증 정보 취득 (get_credential)

API 키 등은 `get_credential()`을 통해 취득합니다. 하드코딩하지 마세요.

```python
from core.tools._base import get_credential

api_key = get_credential(
    credential_name="myapi",   # config.json의 credentials 키
    tool_name="myapi_tool",   # 에러 메시지용
    key_name="api_key",       # 기본값. keys 내 다른 키도 지정 가능
    env_var="MYAPI_KEY",      # 폴백 환경변수
)
```

**해석 순서**: config.json → vault.json (암호화 볼트) → shared/credentials.json → 환경변수. 어디에도 없으면 ToolConfigError.

## permissions.json의 도구 생성 허가

도구 생성 및 공유를 위해 permissions.json에 다음을 추가합니다:

```markdown
## 도구 생성
- 개인 도구: yes
- 공유 도구: yes
```

`yes` 대신 `OK`, `enabled`, `true`도 유효합니다.

## 검증 체크리스트

- [ ] 파일명: 스네이크 케이스, `.py` 확장자 (예: `my_tool.py`)
- [ ] `from __future__ import annotations`가 파일 최상단에 있는가
- [ ] `get_tool_schemas()`가 존재하고 리스트를 반환하는가
- [ ] 스키마명이 `{tool_name}_{action}` 형식인가 (animaworks-tool 연동)
- [ ] 스키마에 `name`, `description`, `input_schema` (또는 `parameters`)가 있는가
- [ ] `dispatch()` 또는 스키마명과 동명의 함수가 존재하는가
- [ ] `dispatch()`에서 `args.pop("anima_dir", None)`을 하는가 (args를 다른 함수에 전달할 때)
- [ ] 모든 스키마에 대응하는 핸들러가 있는가
- [ ] 에러 시 적절한 예외를 발생시키는가
- [ ] 외부 API에 타임아웃을 설정했는가 (권장: 30초)

## 보안 가이드라인

1. **인증 정보**: `get_credential()`을 통해 취득. 하드코딩 금지

2. **접근 제한**: 다른 Anima의 디렉토리에 접근하지 않음

3. **타임아웃**: 외부 API에 반드시 타임아웃 설정 (권장: 30초)

4. **로깅**: `logging.getLogger("animaworks.tools")` 사용

5. **의존성**: 외부 라이브러리는 함수 내에서 임포트 (지연 임포트)

## 주의사항

- 도구는 Python 코드이므로 스킬(Markdown 절차서)과 다릅니다
- 도구 생성에는 permissions.json의 "도구 생성" 섹션에서 **개인 도구: yes** 허가가 필요합니다
- 공유 도구화에는 **공유 도구: yes** 허가가 필요합니다
- 생성한 도구는 `refresh_tools` 호출로 즉시 탐색됩니다 (핫 리로드)
- 개인/공유 도구는 permissions.json의 "외부 도구" 섹션에 등록 불필요. `refresh_tools`로 탐색되면 Bash + animaworks-tool에서 사용 가능
- 스키마명은 `{tool_name}_{action}` 형식. 다른 도구와 충돌하지 않도록 고유하게 유지하세요
- 코어 도구와 동명의 개인/공유 도구는 섀도잉되어 건너뛰어집니다 (`core/tools/__init__.py`)
