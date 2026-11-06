---
name: tool-creator
description: >-
  AnimaWorks용 Python 외부 도구 모듈을 만드는 메타 스킬.core/integrations연동·get_credential·permissions를 다룬다.
  Use when: core/integrations에 새 모듈을 추가하거나, Web API 래퍼를 구현하거나, animaworks-tool에서 호출할 사용자 지정 도구를 개발해야 할 때.
---


# tool-creator

## 개요

AnimaWorks의 도구는 3가지로 나뉜다:

| 종류 | 배치 위치 | 발견 방법 |
|------|--------|----------|
| **코어 도구** | `core/integrations/*.py`（`_` 접두사 파일은 제외） | `discover_core_tools()` → `TOOL_MODULES`（패키지 import） |
| **공유 도구** | `{data_dir}/common_tools/*.py` | `discover_common_tools()` |
| **개인 도구** | `{anima_dir}/tools/*.py` | `discover_personal_tools()` |

`{data_dir}`는 보통 `~/.animaworks/`.

- **디스패치**: `ExternalToolDispatcher`는 `_DISPATCH_TABLE`를 폐지하고, 각 모듈의 `dispatch(name, args)`(또는 스키마 이름과 동일한 함수)로 통일하고 있다(`core/tooling/dispatch.py`).
- **병합**: `AgentCore` 시작 시의 `_discover_personal_tools()`와 `refresh_tools`는 모두 **공통→개인** 순서로 병합하고, **개인이 동일 이름을 덮어쓴다**(`{**common, **personal}`). 병합 결과는 `ExternalToolDispatcher`의 `_personal_tools`에 유지된다(이름은 historical이지만 **공통 도구도 포함**).
- **코어와의 충돌**: 코어 `TOOL_MODULES`와 동일 이름의 파일은 공통·개인 발견 시 **스킵**된다(경고 로그만).
- **도구 파일 쓰기**: `write_memory_file`에서 `tools/*.py`에 쓸 때는 `permissions`(**`permissions.json` 우선**)의 **tool_creation.personal**을 충족해야 한다(`core/tooling/handler_memory.py`).

## 실행 경로(LLM에서 어떻게 호출되는가)

| 모드 | 전형적 경로 |
|--------|-----------|
| **A(LiteLLM 등)** | 통합 도구 **`use_tool(tool_name, action, args)`** → 모듈의 `dispatch`(`core/tooling/handler.py`). 상세는 **`read_memory_file`**에서 각 도구의 스킬을 읽는 설계(`core/tooling/schemas/skill.py`의 `USE_TOOL`). |
| **S(Agent SDK)** | Claude Code 내장 **Bash**로 `animaworks-tool <ツール> …`, 또는 MCP 경유(MCP에 올라가는 것은 엄선된 서브셋만. 아래 「코어 도구를 리포지토리에 추가하는 경우」 참조). |
| **Anthropic 폴백 등** | `build_tool_list`에서 `include_use_tool=False`의 구성이 있을 수 있음 → 외부는 **Bash + animaworks-tool**이나 스킬 전제. |

시작 시에는 위 병합된 맵이 `ToolHandler`에 전달되므로, **프로세스 시작 전에 둔** 공통·개인 도구는 처음부터 `use_tool` / `ExternalToolDispatcher`에서 참조할 수 있다. **세션 중에 새로 추가한** `.py`만, `refresh_tools`으로 재스캔하지 않으면 `use_tool`가 도구 이름을 인식하지 못한다(맵 미업데이트 때문).

## 절차

### Step 1: 도구 설계

1. 도구 이름(모듈 이름)을 정한다(스네이크 케이스, 예: `my_api_tool`). `animaworks-tool my_api_tool …`의 첫 번째 인자가 된다.
2. **액션**(서브커맨드)을 정한다. 스키마 이름은 원칙적으로 **`{tool_name}_{action}`**(예: `myapi_query`). `use_tool`에서는 `tool_name="myapi"`, `action="query"`.
3. 파라미터를 JSON Schema로 정의한다(`input_schema` 또는 `parameters`).

### Step 2: 모듈 파일 작성

#### 단일 액션 예

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_tool_schemas() -> list[dict]:
    """ツールスキーマを返す。個人・共有ツールでは必須推奨（スキーマ読み込み・ログ用）。"""
    return [
        {
            "name": "my_tool_action",
            "description": "このツールが何をするかの説明",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "パラメータの説明",
                    },
                    "param2": {
                        "type": "integer",
                        "description": "オプションパラメータ",
                        "default": 10,
                    },
                },
                "required": ["param1"],
            },
        }
    ]


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """スキーマ名に応じた処理を実行する（推奨）。"""
    args.pop("anima_dir", None)  # フレームワークから注入。必要なら Path(anima_dir) で利用
    if name == "my_tool_action":
        return _do_action(
            param1=args["param1"],
            param2=args.get("param2", 10),
        )
    raise ValueError(f"Unknown tool: {name}")


def _do_action(param1: str, param2: int = 10) -> dict[str, Any]:
    return {"result": f"Processed {param1} with {param2}"}
```

`animaworks-tool`에서 호출하는 경우, 이 다음에 **`cli_main`를 반드시 구현**한다(아래 「`cli_main`」 절).

#### 복수 액션 + 인증 (API 연동)

`get_credential(credential_name, tool_name, key_name="api_key", env_var=...)`의 해결 순서는 **`config.json`의 `credentials.{credential_name}`** (`api_key` 또는 `keys[key_name]`) → **`vault.json`의 `shared` 섹션** (키 이름은 인수 `env_var`로 전달한 문자열) → **`shared/credentials.json` (레거시, 키는 `env_var`)** → **환경 변수 `env_var`** (`core/credentials.py`).

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_tool_schemas() -> list[dict]:
    return [
        {
            "name": "myapi_query",
            "description": "APIにクエリを送信して結果を取得する",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "limit": {"type": "integer", "description": "最大件数", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "myapi_post",
            "description": "APIにデータを送信する",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "送信データ"},
                },
                "required": ["data"],
            },
        },
    ]


class MyAPIClient:
    def __init__(self) -> None:
        from core.credentials import get_credential

        self._api_key = get_credential(
            "myapi",
            "myapi_tool",
            env_var="MYAPI_KEY",
        )

    def query(self, query: str, limit: int = 10) -> list[dict]:
        import httpx

        resp = httpx.get(
            "https://api.example.com/search",
            params={"q": query, "limit": limit},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def post(self, data: str) -> dict:
        import httpx

        resp = httpx.post(
            "https://api.example.com/data",
            json={"data": data},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()


def dispatch(name: str, args: dict[str, Any]) -> Any:
    args.pop("anima_dir", None)
    client = MyAPIClient()
    if name == "myapi_query":
        return client.query(query=args["query"], limit=args.get("limit", 10))
    if name == "myapi_post":
        return client.post(data=args["data"])
    raise ValueError(f"Unknown tool: {name}")
```

**Per-Anima 인증** (Chatwork 등): `args.get("anima_dir")`에서 Anima 이름을 가져와 `CHATWORK_API_TOKEN__{anima_name}`와 같은 **Anima 전용 키**를 `resolve_env_style_credential(...)`로 해결하는 패턴이 있다 (`core/integrations/_chatwork_identity.py`의 `resolve_identity` 등. 미등록이면 폴백하지 않고 오류로 처리). 동일한 키 명명 방식을 커스텀 툴에서도 사용할 수 있다.

#### `cli_main`（animaworks-tool 용）

`animaworks-tool <tool_name> …`는 코어·공통·개인 모두 **모듈에 `cli_main`가 없으면 CLI 실행 불가**（`cli/tool_dispatch.py`의 `cli_dispatch`）。`argparse`에서 서브커맨드를 파싱하고, 내부에서 `dispatch(f"{tool}_{action}", args_dict)`를 호출하는 형태가 일반적. 스키마에서 사용법을 생성하려면 `core/integrations/_base.py`의 `auto_cli_guide`도 참조.

### Step 3: 파일 저장

개인 도구:

```
write_memory_file(path="tools/my_tool.py", content=<コード>)
```

`tool_creation.personal`이 허용되어 있을 것.

### Step 4: 도구 활성화(핫 리로드)

프로세스 시작 **후**에 `tools/*.py`이나 `common_tools/*.py`를 추가·변경한 경우만:

```
refresh_tools()
```

동일 세션 내의 `ExternalToolDispatcher` 파일 기반 맵이 재스캔되고, `use_tool`에서 새 모듈 이름이 해결된다(시작 전부터 존재하는 파일은 보통 불필요).

### Step 5: 공유(선택)

```
share_tool(tool_name="my_tool")
```

`~/.animaworks/common_tools/`에 복사된다. `tool_creation.shared`가 필요. 다른 Anima는 각자 `refresh_tools`(또는 재시작 시 자동 발견)이 필요.

## 필수 인터페이스

| 함수 / 상수 | 필수 | 설명 |
|-------------|------|------|
| `get_tool_schemas()` | 개인·공유에서는 **강력 권장** | 스키마 로드·가이드 생성용. 코어에도 빈 리스트 모듈이 있다(예: `web_search`는 `[]`). **중요**: `ExternalToolDispatcher.dispatch`(`tool_use`에서 스키마 이름을 직접 전달하는 경로)는 코어에 대해 **`get_tool_schemas()`의 `name` 목록에 포함된 스키마만** 모듈에 매치한다. 빈 모듈은 그 경로에서 코어 측에 히트하지 않는다. 한편 **`use_tool`**는 `TOOL_MODULES`에서 모듈을 직접 import하여 `dispatch`를 호출하므로, **스키마 목록이 비어 있어도 `dispatch`가 있으면 실행할 수 있다**. 커스텀 도구는 양쪽 경로를 의식하고, 보통은 스키마를 정의해 두는 것이 안전. |
| `dispatch(name, args)` | **권장** | `ExternalToolDispatcher._call_module`가 우선 이용. |
| 스키마 이름과 동일한 함수 | 대체 | `dispatch`가 없을 때 `getattr(mod, name)(**args)`. |
| `cli_main(argv)` | **CLI 이용 시 필수** | `animaworks-tool` 엔트리. |
| `EXECUTION_PROFILE` | 선택 | `expected_seconds`, `background_eligible`, 코어 도구에서는 **`gated: True`**로 전송 계열 등을 허용 목록 필수로 할 수 있다(`core/tooling/permissions.py`). |

## 호출 및 스키마 이름

- **`use_tool`**: `schema_name = f"{tool_name}_{action}"`에서 모듈의 `dispatch`(또는 같은 이름의 함수)에 전달된다. 허용 여부 판정은 **코어**: `tool_registry`(`get_permitted_tools`의 결과에 `tool_name`가 포함되어 있을 것), **파일 기반(공유·개인)**: 병합된 `_personal_tools`에 `tool_name`가 있을 것(`core/tooling/handler.py`의 `_handle_use_tool`).
- **`animaworks-tool`**: 첫 번째 토큰이 `submit`이면 백그라운드 작업으로 투입된다(아래 참조). **코어**는 `TOOL_MODULES`에서 import한 뒤 `cli_main`하고, **공유·개인**은 파일에서 로드한 뒤 `cli_main`한다. 알 수 없는 첫 번째 인수는 메인 CLI(`animaworks`)로 폴백할 수 있다(`cli/tool_dispatch.py`).
- **게이트가 적용되는 하위 명령(코어만)**: `EXECUTION_PROFILE`의 해당 작업에 `"gated": True`가 있으면, `permissions`의 허용 목록에 **`{tool_name}_{action}`**(예: `gmail_send`)이 포함되지 않은 경우 CLI와 디스패치 양쪽에서 차단된다. **파일 기반 개인·공유 도구**는 `TOOL_MODULES`에 없으므로 이 게이트 메커니즘의 대상이 아니다.

## 스키마 정규화

`core/tooling/schemas/loader.py`의 `_normalise_schema`가 `input_schema` / `parameters`를 받아, 내부 표현에서는 `parameters`로 통일한다.

## permissions(tool_creation·외부 도구)

- **로드**: `load_permissions(anima_dir)`(`core/config/schemas.py`). **`permissions.json`가 우선**. 없을 때만 `permissions.md`를 파싱하여 JSON 생성·마이그레이션(`migrate_permissions_md_to_json`).
- **도구 생성**(JSON 예):

```json
{
  "version": 1,
  "tool_creation": {
    "personal": true,
    "shared": false
  }
}
```

Markdown의 「도구 생성」 섹션(`個人ツール` / `共有ツール` 행)도 마이그레이션 시 같은 구조가 된다.

- **외부 도구(코어)**: `external_tools`는 `get_permitted_tools`에서 **코어 `TOOL_MODULES`의 모듈 이름**과, 게이트 해제용 **`{tool}_{action}`** 문자열(예: `gmail_send`)을 모은다. `use_tool`에서는 코어 도구는 이 집합에 들어간 이름이 `tool_registry` 측에서 사용되고, **개인·공유 도구**는 시작 시 병합 또는 `refresh_tools` 후의 **`_personal_tools`에 이름이 있으면** 코어 집합 밖이어도 실행된다(코어와 동일 이름 파일은 발견 시 스킵되므로 충돌하지 않는다).

## EXECUTION_PROFILE

- **`background_eligible: True`**: `animaworks-tool submit <tool> <subcommand> …`는 `task_type="command"`의 입력을 TaskStore에 등록하고, `PendingTaskExecutor`가 시도를 가져와 실행한다（`cli/tool_dispatch.py`의 `_handle_submit`）。실행 결과는 `state/background_tasks/{task_id}.json`에도 저장된다. 프로필 참조는 **import 가능한 코어 모듈**에 대해서만 실시（파일 도구는 submit 시 경고 대상이 되기 쉬움).
- **`gated: True`**: 코어 도구의 해당 액션에 대해 permissions에서 `tool_action`의 명시적 허가가 필요.

```python
EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "pipeline": {"expected_seconds": 1800, "background_eligible": True},
    "send": {"expected_seconds": 15, "background_eligible": False, "gated": True},
}
```

## 리포지토리에 코어 도구를 추가할 경우

1. `core/integrations/{name}.py`을 추가한다(`_`로 시작하는 항목은 스캔 대상에서 제외).
2. `TOOL_MODULES`은 `discover_core_tools()`에서 자동 등록되므로 `core/integrations/__init__.py`의 수동 목록은 필요 없다.
3. MCP의 엄선된 허용 목록은 `core/tooling/surface.py`의 `MCP_TOOL_NAMES`이다. `resolve_tool_surface`은 MCP 사용 모드(S / C / D / G / X)에서 트리거와 역할에 따라 필터링한다. 예: `search_memory`, `read_memory_file`, `write_memory_file`, `archive_memory_file`, `send_message`, `call_human`, `delegate_task`, `submit_tasks`, `update_task`, `create_skill`. Board 읽기 및 게시 작업은 Bash(`animaworks-tool board read/post`)를 통해 수행한다. **Slack / Gmail / `web_search` 등의 외부 서비스 계열 코어 도구는 MCP에 표시되지 않는다** — 일반적으로 **`use_tool` / Bash(`animaworks-tool`) / 스킬** 경로를 사용한다.
4. 테스트를 `tests/`에 추가한다. 스키마나 참조 문서를 자동 생성하는 경우에는 `scripts/generate_reference.py`의 대상도 확인한다.
5. 파괴적 작업은 `gated: True`와 permissions 쪽 설명의 업데이트를 검토한다.

## 검증 체크리스트

- [ ] 파일 이름: 스네이크 케이스, `.py`, 선두 `_` 없음(스캔 대상에 넣기 위해)
- [ ] `from __future__ import annotations`를 선두에 붙인다(프로젝트 규약)
- [ ] `get_tool_schemas()`가 올바른 스키마 이름을 반환한다(개인·공유)
- [ ] `dispatch` 또는 스키마 이름 함수로 모든 스키마를 처리
- [ ] `anima_dir`를 사용하지 않는다면 `args.pop("anima_dir", None)`로 부작용을 피한다
- [ ] `cli_main`를 구현하고 `animaworks-tool`로 동작 확인
- [ ] 외부 HTTP에는 `timeout=`를 붙인다
- [ ] 인증은 `get_credential`(또는 코어와 동형의 per-anima 해결)
- [ ] 로그는 `logging.getLogger(__name__)`를 권장

## 보안

1. 비밀 정보를 코드에 내장하지 않는다. `get_credential` / vault / config를 사용.
2. 다른 Anima의 디렉터리에 접근하지 않는다.
3. 코어에서 「쓰기·전송」 계열은 `gated`와 permissions를 세트로 설계한다.

## 참고 구현

- 얇은 엔트리 + `_client` / `_cli` 분할: `core/integrations/chatwork.py`, `slack.py`, `discord.py`
- 인증·API: `core/integrations/gmail.py`, `github.py`, `notion.py`, `google_calendar.py`, `google_tasks.py`
- 장시간·파이프라인: `core/integrations/image_gen.py`（파사드, `image/` 서브패키지 + `EXECUTION_PROFILE`）
- 검색·로컬 LLM: `core/integrations/web_search.py`（`get_tool_schemas`가 비어 있음 → `ExternalToolDispatcher.dispatch`의 코어 경로에서는 매치하지 않음. `use_tool`는 `dispatch`에서 가능）, `x_search.py`, `local_llm.py`
- 디스패처·CLI 엔트리: `core/tooling/dispatch.py`, `cli/tool_dispatch.py`（`cli_dispatch` / `_handle_submit`）

## 주의사항

- 도구는 실행 가능한 Python. 스킬(Markdown)과는 별개.
- **시작 후**에 추가한 도구만 **`refresh_tools`**가 필요（시작 전부터 존재하는 파일은 시작 시 스캔 완료）.
- 코어와 동일한 이름의 개인·공유 파일은 채택되지 않음.
