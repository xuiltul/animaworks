---
name: tool-creator
description: >-
  AnimaWorks用のPythonツールモジュールを正しいインターフェースで作成するメタスキル。
  個人ツール（animas/{name}/tools/）・共有ツール（common_tools/）の作成手順、
  ExternalToolDispatcher連携、Bash + animaworks-tool 経由の呼び出し、get_credentialによるAPIキー管理、
  permissions.json許可設定、EXECUTION_PROFILEによる長時間実行対応を提供。
  Web API連携・外部サービス統合のカスタムツール開発時に使用。
  「ツールを作成」「ツール化」「新しいツール」「カスタムツール」「Python ツール」
---

# tool-creator

## 概要

AnimaWorksのツールは3種類に分かれる:

| 種類 | 配置先 | 発見方法 |
|------|--------|----------|
| **コアツール** | `core/tools/*.py` | `discover_core_tools()` → TOOL_MODULES（起動時固定） |
| **共有ツール** | `{data_dir}/common_tools/*.py` | `discover_common_tools()` |
| **個人ツール** | `{anima_dir}/tools/*.py` | `discover_personal_tools()` |

`{data_dir}` は通常 `~/.animaworks/`。個人ツール・共有ツールは `ExternalToolDispatcher` が `refresh_tools` で再スキャンし、ホットリロード可能。同一ツール名の場合、個人ツールが共有ツールを上書きする（`{**common, **personal}`）。ToolHandler は `write_memory_file` で `tools/*.py` への書き込み時に permissions.json の「ツール作成」セクションで許可をチェックする。

個人ツール・共有ツールは **Bash** 経由で `animaworks-tool <ツール> <サブコマンド> [引数]` を実行する。スキーマ名は `{tool_name}_{action}` 形式（例: `my_tool` + `query` → `my_tool_query`）。

## 手順

### Step 1: ツールの設計

1. ツール名を決める（スネークケース、例: `my_api_tool`）
2. 提供するスキーマ（操作）を決める
3. 必要なパラメータを定義する

### Step 2: モジュールファイルの作成

以下のテンプレートに沿ってPythonファイルを作成します。

#### 単一スキーマのツール（シンプル）

ファイル名 `my_tool.py` の場合、`animaworks-tool my_tool action [args]` で呼び出され、スキーマ名 `my_tool_action` として dispatch に渡される。

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("animaworks.tools")


def get_tool_schemas() -> list[dict]:
    """ツールスキーマを返す（必須）。"""
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
    args.pop("anima_dir", None)  # フレームワークから注入されるが、本ツールでは未使用
    if name == "my_tool_action":
        return _do_action(
            param1=args["param1"],
            param2=args.get("param2", 10),
        )
    raise ValueError(f"Unknown tool: {name}")


def _do_action(param1: str, param2: int = 10) -> dict[str, Any]:
    """実際の処理ロジック。"""
    # ここに実装を書く
    return {"result": f"Processed {param1} with {param2}"}
```

#### 複数スキーマのツール（API連携等）

スキーマ名は `{ツール名}_{アクション}` 形式にする。`animaworks-tool myapi query [args]` 呼び出し時に `myapi_query` として dispatch に渡される。ファイル名は `myapi.py`。

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("animaworks.tools")


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
    """API クライアント。"""

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
    args.pop("anima_dir", None)  # フレームワークから注入されるが、本ツールでは未使用
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

### Step 3: ファイルの保存

個人ツールとして保存（`write_memory_file` の path は anima_dir からの相対パス）:

```
write_memory_file(path="tools/my_tool.py", content=<コード>)
```

`tools/` への書き込みには permissions.json の「ツール作成」セクションで **個人ツール** の許可が必要。

### Step 4: ツールの有効化

保存後、`refresh_tools` を呼び出してホットリロード:

```
refresh_tools()
```

これにより、セッションを再起動せずに即座にツールが使えるようになる。個人ツール・共有ツールは permissions.json の「外部ツール」セクションに登録不要。`refresh_tools` で発見されれば Bash + animaworks-tool から呼び出し可能。

### Step 5: 共有（任意）

他のAnimaにも使ってほしい場合は共有ツールに:

```
share_tool(tool_name="my_tool")
```

これにより `~/.animaworks/common_tools/` にコピーされ、全Animaから利用可能になります。共有には permissions.json で **共有ツール** の許可が必要。

## 必須インターフェース

| 関数 | 必須 | 説明 |
|------|------|------|
| `get_tool_schemas()` | ✅ 必須 | ツールスキーマのリストを返す。`name`, `description`, `input_schema`（または `parameters`）を含む |
| `dispatch(name, args)` | 🔵 推奨 | スキーマ名に基づくディスパッチ。ExternalToolDispatcher が優先して呼ぶ。`args` に `anima_dir` が注入される場合は `args.pop("anima_dir", None)` で除去 |
| スキーマ名と同名の関数 | 🟡 代替 | `dispatch()` の代わりに使用可能（`_call_module` が `getattr(mod, name)(**args)` で呼ぶ） |
| `cli_main(argv)` | ⚪ 任意 | `animaworks-tool <tool_name>` での単体実行用 |
| `EXECUTION_PROFILE` | ⚪ 任意 | 長時間実行ツール向け。`animaworks-tool submit` で `state/background_tasks/pending/` に投入可能にする |

## 呼び出し方法

Anima は **Bash** 経由で `animaworks-tool <ツール> <サブコマンド> [引数]` を実行する。CLI の `cli_main(argv)` が呼ばれる。`schema_name = tool_name + "_" + action` として `dispatch(name, args)` に渡される（例: `animaworks-tool myapi query "検索語" --limit 10` → `name="myapi_query"`）。

## スキーマ定義の規約

`input_schema` と `parameters` の両方がサポートされ、`core.tooling.schemas._normalise_schema` で正規化される（内部では `parameters` に統一）。

```python
{
    "name": "tool_action_name",       # スネークケース。{tool_name}_{action} 形式
    "description": "1-2文の説明",      # LLMがツール選択に使う
    "input_schema": {                  # JSON Schema形式（parameters も可）
        "type": "object",
        "properties": { ... },
        "required": [ ... ],
    },
}
```

## 認証情報の取得（get_credential）

APIキー等は `get_credential()` 経由で取得する。ハードコードしない。

```python
from core.tools._base import get_credential

api_key = get_credential(
    credential_name="myapi",   # config.json の credentials.{credential_name} キー
    tool_name="myapi_tool",    # エラーメッセージ用
    key_name="api_key",        # デフォルト。keys 内の別キーも指定可
    env_var="MYAPI_KEY",       # vault.json / shared/credentials.json / 環境変数 のフォールバック用キー
)
```

**解決順序**: 1) config.json の `credentials.{credential_name}` → 2) vault.json の `shared` セクション（`env_var` をキーに検索）→ 3) shared/credentials.json（`env_var` をキーに検索）→ 4) 環境変数 `env_var`。いずれにもない場合は ToolConfigError。

## permissions.json のツール作成許可

ツール作成・共有には permissions.json に「ツール作成」セクションを追加し、以下を記述:

```markdown
## ツール作成
- 個人ツール: yes
- 共有ツール: yes
```

`yes` の代わりに `OK`, `enabled`, `true` も有効。`_check_tool_creation_permission` が行単位で `{kind}: yes` 形式を検証する。

- **個人ツール**: `write_memory_file` で `tools/*.py` に書き込む際に必要
- **共有ツール**: `share_tool` で common_tools にコピーする際に必要

## EXECUTION_PROFILE（長時間実行ツール向け）

`background_eligible: True` のサブコマンドは `animaworks-tool submit <tool_name> <subcommand> [args...]` でバックグラウンド投入可能。タスクは `state/background_tasks/pending/` に書き出され、PendingTaskExecutor が実行する。

```python
EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "pipeline": {"expected_seconds": 1800, "background_eligible": True},
    "query": {"expected_seconds": 10, "background_eligible": False},
}
```

## バリデーションチェックリスト

- [ ] ファイル名: スネークケース、`.py` 拡張子（例: `my_tool.py`）
- [ ] `from __future__ import annotations` がファイル先頭にあるか
- [ ] `get_tool_schemas()` が存在し、リストを返すか
- [ ] スキーマ名が `{tool_name}_{action}` 形式か（animaworks-tool 連携）
- [ ] スキーマに `name`, `description`, `input_schema`（または `parameters`）があるか
- [ ] `dispatch()` またはスキーマ名と同名の関数が存在するか
- [ ] `dispatch()` で `args.pop("anima_dir", None)` しているか（args を他関数に渡す場合）
- [ ] 全スキーマに対応するハンドラがあるか
- [ ] エラー時に適切な例外を発生させるか
- [ ] 外部APIにタイムアウトを設定しているか（推奨: 30秒）

## セキュリティガイドライン

1. **認証情報**: `get_credential()` 経由で取得する。ハードコードしない

2. **アクセス制限**: 他のAnimaのディレクトリにはアクセスしない

3. **タイムアウト**: 外部APIには必ずタイムアウトを設定する（推奨: 30秒）

4. **ロギング**: `logging.getLogger("animaworks.tools")` を使用する

5. **依存管理**: 外部ライブラリは関数内でインポートする（遅延インポート）

## 注意事項

- ツールはPythonコードなので、スキル（Markdown手順書）とは異なります
- ツール作成には permissions.json の「ツール作成」セクションで **個人ツール: yes** の許可が必要です
- 共有ツール化には **共有ツール: yes** の許可が必要です
- 作成したツールは `refresh_tools` 呼び出しで即座に発見されます（ホットリロード）
- 個人ツール・共有ツールは permissions.json の「外部ツール」セクションに登録不要。`refresh_tools` で発見されれば Bash + animaworks-tool から利用可能（コアツールのみ外部ツールで許可制御）
- スキーマ名は `{tool_name}_{action}` 形式。他ツールと衝突しないよう一意にすること
- コアツールと同名の個人・共有ツールはシャドウされるためスキップされます（`discover_common_tools` / `discover_personal_tools`）
- 参考実装: `core/tools/web_search.py`, `core/tools/chatwork.py`, `core/tools/image_gen.py`
