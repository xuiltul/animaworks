---
name: workspace-manager
description: >-
  ワークスペース（作業ディレクトリ）の登録・一覧・削除・割り当てを行う。
  「ワークスペース」「作業ディレクトリ」「プロジェクト登録」
tags: [workspace, directory, project, management]
---

# ワークスペース管理

Animaが作業するプロジェクトディレクトリ（ワークスペース）を管理するスキル。

## 概念

Animaは普段「自分の家」（~/.animaworks/animas/{name}/）にいる。
プロジェクトの作業をするときは「仕事場」（ワークスペース）に出かけて作業する。

ワークスペースは組織共有のレジストリ（config.json の workspaces セクション）に登録し、エイリアス#ハッシュで参照する。

## エイリアスとハッシュ

- エイリアス: 人間が付ける短い名前（例: `aischreiber`）
- ハッシュ: パスのSHA-256先頭8桁が自動付与される（例: `3af4be6e`）
- 完全形: `aischreiber#3af4be6e` — 衝突の可能性がゼロ
- ツール引数にはエイリアスのみ、完全形、ハッシュのみ、絶対パスのいずれも使用可能

## 操作方法

### 登録

config.json の workspaces セクションに追加する:

1. `read_memory_file(path="config.json")` で現在の設定を確認
2. workspaces セクションにエイリアスとパスを追加
3. `write_memory_file` で保存

または、Bash で以下を実行:
```bash
python3 -c "
from core.workspace import register_workspace
result = register_workspace('エイリアス名', '/absolute/path/to/project')
print(result)
"
```

**注意**: ディレクトリが存在しない場合はエラーになる。

### 一覧

config.json の workspaces セクションを読む:
```
read_memory_file(path="config.json")
```

### 削除

config.json の workspaces セクションから該当エントリを削除する。

### 自分のデフォルトワークスペース変更

自分の `status.json` の `default_workspace` フィールドを更新する:
1. `read_memory_file(path="status.json")` で現在の内容を確認
2. `default_workspace` にエイリアス（例: `aischreiber`）または完全形（例: `aischreiber#3af4be6e`）を設定
3. `write_memory_file(path="status.json", content=...)` で保存

記載例:
```json
{
  "default_workspace": "aischreiber#3af4be6e"
}
```

### 部下への割り当て（スーパバイザー用）

1. ワークスペースを登録（上記）
2. 部下の `status.json` の `default_workspace` フィールドを更新する:
   - `read_memory_file(path="../{subordinate}/status.json")`
   - `default_workspace` にエイリアスを設定
   - `write_memory_file(path="../{subordinate}/status.json", content=...)`

## ツールでの使用

- **machine_run**: `working_directory` にエイリアスまたは完全形を指定
- **submit_tasks**: 各タスクの `workspace` フィールドにエイリアスを指定
- **delegate_task**: `workspace` フィールドにエイリアスを指定

## 注意事項

- ディレクトリは登録時・使用時の両方で存在確認される
- 存在しないディレクトリを登録しようとするとエラー
- エイリアスを上書きするとハッシュも変わるため、旧ハッシュ参照は解決失敗する
- 人間はハッシュを覚える必要なし — エイリアスだけでOK
