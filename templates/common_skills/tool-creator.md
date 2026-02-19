---
name: tool-creator
description: >-
  AnimaWorks用のPythonツールモジュール(core/tools/)を正しいインターフェースで作成するメタスキル。
  ExternalToolDispatcher連携、APIキー管理、ToolHandler登録、permissions.md許可設定の
  手順を提供する。Web API連携・外部サービス統合のカスタムツール開発時に使用。
  「ツールを作成」「ツール化」「新しいツール」「カスタムツール」「Python ツール」
---

# tool-creator

## 手順

### Step 1: ツールの設計

1. ツール名を決める（スネークケース、例: `my_api_tool`）
2. 提供するスキーマ（操作）を決める
3. 必要なパラメータを定義する

### Step 2: モジュールファイルの作成

以下のテンプレートに沿ってPythonファイルを作成します。

#### 単一スキーマのツール（シンプル）

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
        self._api_key = get_credential("myapi", "myapi_tool", env_var="MYAPI_KEY")

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

個人ツールとして保存:
```
write_memory_file("tools/my_tool.py", <コード>)
```

### Step 4: ツールの有効化

保存後、`refresh_tools` を呼び出してホットリロード:
```
refresh_tools()
```

これにより、セッションを再起動せずに即座にツールが使えるようになります。

### Step 5: 共有（任意）

他のAnimaにも使ってほしい場合は共有ツールに:
```
share_tool(tool_name="my_tool")
```

これにより `common_tools/` にコピーされ、全Animaから利用可能になります。

## 必須インターフェース

| 関数 | 必須 | 説明 |
|------|------|------|
| `get_tool_schemas()` | ✅ 必須 | ツールスキーマのリストを返す |
| `dispatch(name, args)` | 🔵 推奨 | スキーマ名に基づくディスパッチ |
| スキーマ名と同名の関数 | 🟡 代替 | `dispatch()`の代わりに使用可能 |
| `cli_main(argv)` | ⚪ 任意 | CLI単体実行用 |
| `get_cli_guide()` | ⚪ 任意 | エージェント向けCLIガイド |

## スキーマ定義の規約

```python
{
    "name": "tool_action_name",       # スネークケース、ツール名をプレフィックスに
    "description": "1-2文の説明",      # LLMがツール選択に使う
    "input_schema": {                  # JSON Schema形式
        "type": "object",
        "properties": { ... },
        "required": [ ... ],
    },
}
```

## バリデーションチェックリスト

- [ ] ファイル名: スネークケース、`.py` 拡張子（例: `my_tool.py`）
- [ ] `from __future__ import annotations` がファイル先頭にあるか
- [ ] `get_tool_schemas()` が存在し、リストを返すか
- [ ] スキーマに `name`, `description`, `input_schema` があるか
- [ ] `dispatch()` または同名関数が存在するか
- [ ] 全スキーマに対応するハンドラがあるか
- [ ] エラー時に適切な例外を発生させるか
- [ ] 外部APIにタイムアウトを設定しているか

## セキュリティガイドライン

1. **認証情報**: `get_credential()` 経由で取得する。ハードコードしない
   ```python
   from core.tools._base import get_credential
   api_key = get_credential("service_name", "tool_name", env_var="ENV_VAR_NAME")
   ```

2. **アクセス制限**: 他のAnimaのディレクトリにはアクセスしない

3. **タイムアウト**: 外部APIには必ずタイムアウトを設定する（推奨: 30秒）

4. **ロギング**: `logging.getLogger("animaworks.tools")` を使用する

5. **依存管理**: 外部ライブラリは関数内でインポートする（遅延インポート）

## 注意事項

- ツールはPythonコードなので、スキル（Markdown手順書）とは異なります
- ツール作成にはpermissions.mdで「ツール作成: yes」の許可が必要です
- 作成したツールは次の `refresh_tools` 呼び出しまたは次回セッション開始時に発見されます
- スキーマ名はグローバルに一意である必要があります（他ツールと衝突しないこと）
