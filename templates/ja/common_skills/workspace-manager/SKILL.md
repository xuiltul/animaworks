---
name: workspace-manager
description: >-
  ワークスペース（作業ディレクトリ）の登録・一覧・削除・割り当てを設定する。
  Use when: プロジェクトパスをAnimaに紐付け、エイリアス管理、作業ディレクトリの切り替えが必要なとき。
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

人間から明示指示を受けたトップレベルAnimaは `grant_workspace_access` を使う:

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "make_default": true
}
```

このツールは組織共有レジストリへの登録、`permissions.json.file_roots` への書き込み権限追加、必要に応じた `status.json.default_workspace` 更新をまとめて行う。

**注意**: ディレクトリが存在しない場合はエラーになる。
**注意**: `read_memory_file(path="config.json")` は自分のAnimaディレクトリの `config.json` を読む。組織共有レジストリの登録には使わない。

### 一覧

組織共有レジストリの一覧は `core.workspace.list_workspaces()` で確認する。`read_memory_file(path="config.json")` は使わない。

### 削除

削除は管理者操作として扱う。通常の作業では既存エイリアスを上書きせず、新しいエイリアスを登録する。

### 自分のデフォルトワークスペース変更

トップレベルAnimaは `grant_workspace_access` に `make_default: true` を指定する。
非トップレベルAnimaは自分で権限追加できない。トップレベルAnimaに人間から指示してもらう。

### 部下への割り当て（スーパバイザー用）

人間から明示指示を受けたトップレベルAnimaは `target_anima` を指定して部下または子孫Animaに権限を付与できる:

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "target_anima": "ritsu",
  "make_default": true
}
```

## ツールでの使用

- **machine_run**: `working_directory` にエイリアスまたは完全形を指定
- **submit_tasks**: 各タスクの `workspace` フィールドにエイリアスを指定
- **delegate_task**: `workspace` フィールドにエイリアスを指定

## 注意事項

- ディレクトリは登録時・使用時の両方で存在確認される
- 存在しないディレクトリを登録しようとするとエラー
- エイリアスを上書きするとハッシュも変わるため、旧ハッシュ参照は解決失敗する
- 人間はハッシュを覚える必要なし — エイリアスだけでOK
