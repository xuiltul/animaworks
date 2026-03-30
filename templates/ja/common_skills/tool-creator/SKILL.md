---
name: tool-creator
description: >-
  AnimaWorks向けPython外部ツールモジュールを作成するメタスキル。core/tools連携・get_credential・permissionsを扱う。
  Use when: core/toolsへ新規モジュール追加、Web APIラッパー実装、animaworks-toolから呼ぶカスタムツール開発が必要なとき。
---

# tool-creator

## 概要

AnimaWorksのツールは3種類に分かれる:

| 種類 | 配置先 | 発見方法 |
|------|--------|----------|
| **コアツール** | `core/tools/*.py`（`_` 接頭辞のファイルは除外） | `discover_core_tools()` → `TOOL_MODULES`（パッケージ import） |
| **共有ツール** | `{data_dir}/common_tools/*.py` | `discover_common_tools()` |
| **個人ツール** | `{anima_dir}/tools/*.py` | `discover_personal_tools()` |

`{data_dir}` は通常 `~/.animaworks/`。

- **ディスパッチ**: `ExternalToolDispatcher` は `_DISPATCH_TABLE` を廃止し、各モジュールの `dispatch(name, args)`（またはスキーマ名と同名の関数）に統一している（`core/tooling/dispatch.py`）。
- **マージ**: `AgentCore` 起動時の `_discover_personal_tools()` と `refresh_tools` はいずれも **共通→個人** の順でマージし、**個人が同名を上書き** する（`{**common, **personal}`）。マージ結果は `ExternalToolDispatcher` の `_personal_tools` に保持される（名前は historical だが **共通ツールも含む**）。
- **コアとの衝突**: コア `TOOL_MODULES` と同名のファイルは、共通・個人の発見時に **スキップ** される（警告ログのみ）。
- **ツールファイルの書き込み**: `write_memory_file` で `tools/*.py` に書くときは `permissions`（**`permissions.json` 優先**）の **tool_creation.personal** を満たす必要がある（`core/tooling/handler_memory.py`）。

## 実行パス（LLM からどう呼ばれるか）

| モード | 典型経路 |
|--------|-----------|
| **A（LiteLLM 等）** | 統合ツール **`use_tool(tool_name, action, args)`** → モジュールの `dispatch`（`core/tooling/handler.py`）。詳細は **`read_memory_file`** で各ツールのスキルを読む設計（`core/tooling/schemas/skill.py` の `USE_TOOL`）。 |
| **S（Agent SDK）** | Claude Code 組み込み **Bash** で `animaworks-tool <ツール> …`、または MCP 経由（MCP に載るのは厳選サブセットのみ。下記「コアツールをリポジトリに追加する場合」参照）。 |
| **Anthropic フォールバック等** | `build_tool_list` で `include_use_tool=False` の構成があり得る → 外部は **Bash + animaworks-tool** やスキル前提。 |

起動時は上記マージ済みマップが `ToolHandler` に渡るため、**プロセス起動前に置いた** 共通・個人ツールは最初から `use_tool` / `ExternalToolDispatcher` で参照できる。**セッション中に新規追加した** `.py` だけ、`refresh_tools` で再スキャンしないと `use_tool` がツール名を認識しない（マップ未更新のため）。

## 手順

### Step 1: ツールの設計

1. ツール名（モジュール名）を決める（スネークケース、例: `my_api_tool`）。`animaworks-tool my_api_tool …` の第1引数になる。
2. **アクション**（サブコマンド）を決める。スキーマ名は原則 **`{tool_name}_{action}`**（例: `myapi_query`）。`use_tool` では `tool_name="myapi"`, `action="query"`。
3. パラメータを JSON Schema で定義する（`input_schema` または `parameters`）。

### Step 2: モジュールファイルの作成

#### 単一アクションの例

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

`animaworks-tool` から叩く場合は、このあと **`cli_main` を必ず実装**する（下記「`cli_main`」節）。

#### 複数アクション + 認証（API 連携）

`get_credential(credential_name, tool_name, key_name="api_key", env_var=...)` の解決順序は **`config.json` の `credentials.{credential_name}`**（`api_key` または `keys[key_name]`）→ **`vault.json` の `shared` セクション**（キー名は引数 `env_var` で渡した文字列）→ **`shared/credentials.json`（レガシー、キーは `env_var`）** → **環境変数 `env_var`**（`core/tools/_base.py`）。

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
        from core.tools._base import get_credential

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

**Per-Anima 認証**（Chatwork 等）: `args.get("anima_dir")` から Anima 名を取り、**まず** `_lookup_shared_credentials("CHATWORK_API_TOKEN_WRITE__{anima_name}")` のように **Anima 専用キー**を試し、無ければ `get_credential(...)` にフォールバックするパターンがある（`core/tools/chatwork.py` の `_resolve_write_token` 等）。同様のキー命名をカスタムツールでも使える。

#### `cli_main`（animaworks-tool 用）

`animaworks-tool <tool_name> …` はコア・共通・個人いずれも **モジュールに `cli_main` が無いと CLI 実行不可**（`core/tools/__init__.py` の `cli_dispatch`）。`argparse` でサブコマンドをパースし、内部で `dispatch(f"{tool}_{action}", args_dict)` を呼ぶ形が一般的。スキーマから用法を生成したい場合は `core/tools/_base.py` の `auto_cli_guide` も参照。

### Step 3: ファイルの保存

個人ツール:

```
write_memory_file(path="tools/my_tool.py", content=<コード>)
```

`tool_creation.personal` が許可されていること。

### Step 4: ツールの有効化（ホットリロード）

プロセス起動**後**に `tools/*.py` や `common_tools/*.py` を追加・変更した場合のみ:

```
refresh_tools()
```

同一セッション内の `ExternalToolDispatcher` のファイルベースマップが再スキャンされ、`use_tool` から新しいモジュール名が解決される（起動前から存在するファイルは通常不要）。

### Step 5: 共有（任意）

```
share_tool(tool_name="my_tool")
```

`~/.animaworks/common_tools/` にコピーされる。`tool_creation.shared` が必要。他 Anima は各自 `refresh_tools`（または再起動時の自動発見）が必要。

## 必須インターフェース

| 関数 / 定数 | 必須 | 説明 |
|-------------|------|------|
| `get_tool_schemas()` | 個人・共有では **強く推奨** | スキーマ読み込み・ガイド生成用。コアでも空リストのモジュールがある（例: `web_search` は `[]`）。**重要**: `ExternalToolDispatcher.dispatch`（`tool_use` でスキーマ名を直接渡す経路）は、コアについて **`get_tool_schemas()` の `name` 一覧に含まれるスキーマだけ**モジュールにマッチする。空のモジュールはその経路ではコア側にヒットしない。一方 **`use_tool`** は `TOOL_MODULES` からモジュールを直接 import して `dispatch` を呼ぶため、**スキーマ一覧が空でも `dispatch` があれば実行できる**。カスタムツールは両経路を意識し、通常はスキーマを定義しておくのが安全。 |
| `dispatch(name, args)` | **推奨** | `ExternalToolDispatcher._call_module` が優先利用。 |
| スキーマ名と同名の関数 | 代替 | `dispatch` が無い場合に `getattr(mod, name)(**args)`。 |
| `cli_main(argv)` | **CLI 利用時は必須** | `animaworks-tool` エントリ。 |
| `EXECUTION_PROFILE` | 任意 | `expected_seconds`, `background_eligible`、コアツールでは **`gated: True`** で送信系などを許可リスト必須にできる（`core/tooling/permissions.py`）。 |

## 呼び出しとスキーマ名

- **`use_tool`**: `schema_name = f"{tool_name}_{action}"` でモジュールの `dispatch`（または同名関数）に渡る。許可判定は **コア**: `tool_registry`（`get_permitted_tools` の結果に `tool_name` が含まれること）、**ファイルベース（共通・個人）**: マージ済み `_personal_tools` に `tool_name` があること（`core/tooling/handler.py` の `_handle_use_tool`）。拒否メッセージに `permissions.md` と出ることがあるが、実体は **`load_permissions`（JSON 優先）** の `external_tools`。
- **`animaworks-tool`**: 第1トークンが `submit` の場合はバックグラウンド投入（下記）。**コア**は `TOOL_MODULES` から import して `cli_main`、**共通・個人**はファイルからロードして `cli_main`。未知の第1引数はメイン CLI（`animaworks`）へフォールバックする場合あり（`core/tools/__init__.py` の `_MAIN_CLI_COMMANDS` / `_ANIMA_SUBCOMMANDS`）。
- **ゲート付きサブコマンド（コアのみ）**: `EXECUTION_PROFILE` の該当アクションに `"gated": True` があると、`permissions` の許可集合に **`{tool_name}_{action}`**（例: `gmail_send`）が含まれていないと CLI / ディスパッチの両方でブロックされる。**ファイルベースの個人・共有ツール**は `TOOL_MODULES` に無いため、このゲート機構の対象外。

## スキーマ正規化

`core/tooling/schemas/loader.py` の `_normalise_schema` が `input_schema` / `parameters` を受け取り、内部表現では `parameters` に統一する。

## permissions（tool_creation・外部ツール）

- **読み込み**: `load_permissions(anima_dir)`（`core/config/schemas.py`）。**`permissions.json` が優先**。無い場合のみ `permissions.md` をパースして JSON 生成・移行（`migrate_permissions_md_to_json`）。
- **ツール作成**（JSON の例）:

```json
{
  "version": 1,
  "tool_creation": {
    "personal": true,
    "shared": false
  }
}
```

Markdown の「ツール作成」セクション（`個人ツール` / `共有ツール` 行）も移行時に同じ構造になる。

- **外部ツール（コア）**: `external_tools` は `get_permitted_tools` で **コア `TOOL_MODULES` のモジュール名** と、ゲート解除用の **`{tool}_{action}`** 文字列（例: `gmail_send`）を集める。`use_tool` では、コアツールはこの集合に入った名前が `tool_registry` 側で使われ、**個人・共有ツール**は起動時マージまたは `refresh_tools` 後の **`_personal_tools` に名前があれば**コア集合外でも実行される（コアと同名ファイルは発見時にスキップされるため衝突しない）。

## EXECUTION_PROFILE

- **`background_eligible: True`**: `animaworks-tool submit <tool> <subcommand> …` で `state/background_tasks/pending/` に JSON が書かれ、`PendingTaskExecutor` が拾う（`core/tools/__init__.py` の `_handle_submit`）。プロファイル参照は **import 可能なコアモジュール**に対してのみ実施（ファイルツールは submit 時の警告対象外になりやすい）。
- **`gated: True`**: コアツールの該当アクションに対し、permissions で `tool_action` の明示許可が必要。

```python
EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "pipeline": {"expected_seconds": 1800, "background_eligible": True},
    "send": {"expected_seconds": 15, "background_eligible": False, "gated": True},
}
```

## コアツールをリポジトリに追加する場合

1. `core/tools/{name}.py` を追加（`_` 始まりはスキャン対象外）。
2. `TOOL_MODULES` は `discover_core_tools()` で自動登録。`core/tools/__init__.py` の手動リストは不要。
3. **Mode S（MCP）** に載せるのは `core/mcp/server.py` の `_EXPOSED_TOOL_NAMES` のみ（厳選）。2026-03 時点の例: `search_memory`, `read_memory_file`, `write_memory_file`, `archive_memory_file`, `send_message`, `post_channel`, `call_human`, `delegate_task`, `submit_tasks`, `update_task`, `create_skill`, `completion_gate`（最終回答前の自己検証）。**Slack / Gmail / `web_search` 等の外部サービス系コアツールは MCP に出ない** — 通常は **`use_tool` / Bash（`animaworks-tool`）/ スキル** 経路。
4. テストを `tests/` に追加。スキーマやリファレンス文書を自動生成している場合は `scripts/generate_reference.py` の対象も確認。
5. 破壊的操作は `gated: True` と permissions 側の説明更新を検討。

## バリデーションチェックリスト

- [ ] ファイル名: スネークケース、`.py`、先頭 `_` なし（スキャン対象に入れるため）
- [ ] `from __future__ import annotations` を先頭に付ける（プロジェクト規約）
- [ ] `get_tool_schemas()` が正しいスキーマ名を返す（個人・共有）
- [ ] `dispatch` またはスキーマ名関数で全スキーマを処理
- [ ] `anima_dir` を使わないなら `args.pop("anima_dir", None)` で副作用を避ける
- [ ] `cli_main` を実装し `animaworks-tool` で動作確認
- [ ] 外部 HTTP には `timeout=` を付ける
- [ ] 認証は `get_credential`（またはコアと同型の per-anima 解決）
- [ ] ログは `logging.getLogger(__name__)` を推奨

## セキュリティ

1. 秘密情報をコードに埋め込まない。`get_credential` / vault / config を使う。
2. 他 Anima のディレクトリに触れない。
3. コアで「書き込み・送信」系は `gated` と permissions をセットで設計する。

## 参考実装

- 薄いエントリ + `_client` / `_cli` 分割: `core/tools/chatwork.py`, `slack.py`, `discord.py`
- 認証・API: `core/tools/gmail.py`, `github.py`, `notion.py`, `google_calendar.py`, `google_tasks.py`
- CLI マシンガイド集約: `core/tools/machine.py`（`read_memory_file(path="common_skills/machine-tool/SKILL.md")` 等の参照元）
- 長時間・パイプライン: `core/tools/image_gen.py`（ファサード、`image/` サブパッケージ + `EXECUTION_PROFILE`）
- 検索・ローカル LLM: `core/tools/web_search.py`（`get_tool_schemas` が空 → `ExternalToolDispatcher.dispatch` のコア経路ではマッチしない。`use_tool` は `dispatch` で可）、`x_search.py`, `local_llm.py`
- ディスパッチャ・CLI エントリ: `core/tooling/dispatch.py`, `core/tools/__init__.py`（`cli_dispatch` / `_handle_submit`）

## 注意事項

- ツールは実行可能な Python。スキル（Markdown）とは別物。
- **起動後**に追加したツールだけ **`refresh_tools`** が必要（起動前から存在するファイルは起動時スキャン済み）。
- コアと同名の個人・共有ファイルは採用されない。
- `use_tool` のスキーマ説明（`core/tooling/schemas/skill.py`）に `permissions.md` とある箇所があるが、実体は **`load_permissions`（`permissions.json` 優先）**。
