# セキュリティ修正: common_knowledge パストラバーサル + create_anima パスバリデーション

## Overview

`read_memory_file` / `write_memory_file` の `common_knowledge/` プレフィックス処理、および `create_anima` の `character_sheet_path` 処理でパストラバーサルが可能。ファイルシステム全域への読み書きにつながる Critical 脆弱性。

## Problem / Background

### 脆弱性 1: common_knowledge パストラバーサル

`core/tooling/handler_memory.py` の `_handle_read_memory_file()` / `_handle_write_memory_file()` で、`rel` が `common_knowledge/` で始まる場合に suffix を切り出して `get_common_knowledge_dir() / suffix` を構築するが、`suffix` に `../` が含まれる場合のパス正規化チェックがない。

```python
# handler_memory.py（読み取り・書き込み両方）
if rel.startswith("common_knowledge/"):
    suffix = rel[len("common_knowledge/"):]
    path = get_common_knowledge_dir() / suffix  # ← ../../../etc/passwd が通る
```

攻撃例:
- `read_memory_file("common_knowledge/../../../etc/shadow")`
- `write_memory_file("common_knowledge/../../../tmp/malicious.sh", "...")`

プロンプトインジェクション経由で LLM が騙されて実行する可能性がある。

### 脆弱性 2: create_anima の character_sheet_path

`core/tooling/handler_org.py` の `_handle_create_anima()` で `character_sheet_path` が相対パスの場合に `self._anima_dir / md_path` で解決されるが、`_check_file_permission` が呼ばれておらず、`../../other_anima/IDENTITY.md` のような指定で他 Anima のファイルを読める。

```python
# handler_org.py
elif sheet_path_raw:
    md_path = Path(sheet_path_raw).expanduser()
    if not md_path.is_absolute():
        md_path = self._anima_dir / md_path  # ← ../../ が通る
```

### Impact

| コンポーネント | 影響 | 説明 |
|--------------|------|------|
| `core/tooling/handler_memory.py` | Direct | パス正規化チェック追加（読み取り・書き込み両方） |
| `core/tooling/handler_org.py` | Direct | `character_sheet_path` のパスバリデーション追加 |

## Decided Approach / 確定方針

### 修正 1: common_knowledge パス検証

`read_memory_file` / `write_memory_file` の `common_knowledge/` 分岐に、`path.resolve().is_relative_to(get_common_knowledge_dir().resolve())` チェックを追加。範囲外なら `PermissionError` を返す。

```python
if rel.startswith("common_knowledge/"):
    suffix = rel[len("common_knowledge/"):]
    ck_dir = get_common_knowledge_dir()
    path = (ck_dir / suffix).resolve()
    if not path.is_relative_to(ck_dir.resolve()):
        return "Error: path traversal detected — access denied."
```

読み取り（`_handle_read_memory_file`）と書き込み（`_handle_write_memory_file`）の両方に適用。

### 修正 2: create_anima パスバリデーション

`character_sheet_path` の解決後に `_check_file_permission(resolved_path, write=False)` を呼ぶか、最低限 `anima_dir` 内にあることを `is_relative_to` で検証する。

```python
md_path = md_path.resolve()
if not md_path.is_relative_to(self._anima_dir.resolve()):
    return "Error: character_sheet_path must be within anima directory."
```

### 修正 3: sender_name のパス検証（関連）

`core/memory/priming.py` の `_channel_a_sender_profile()` で `sender_name` を使ってパスを構築する箇所にも、`profile_path.resolve().is_relative_to(shared_users_dir.resolve())` チェックを追加。

## Test Plan

- [ ] `read_memory_file("common_knowledge/../../etc/passwd")` が拒否されること
- [ ] `write_memory_file("common_knowledge/../../../tmp/test", "x")` が拒否されること
- [ ] `read_memory_file("common_knowledge/valid_file.md")` が正常動作すること
- [ ] `write_memory_file("common_knowledge/valid_file.md", "x")` が正常動作すること
- [ ] `create_anima` で `character_sheet_path="../../other/IDENTITY.md"` が拒否されること
- [ ] 正規の `character_sheet_path` が正常動作すること
- [ ] `sender_name` に `../` を含む場合に sender_profile が空を返すこと
