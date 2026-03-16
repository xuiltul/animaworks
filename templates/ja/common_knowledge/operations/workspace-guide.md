# ワークスペースガイド

Animaが作業するプロジェクトディレクトリ（ワークスペース）の概念と使い方。

## ワークスペースとは

### 家と仕事場の概念

Animaは普段「自分の家」（`~/.animaworks/animas/{name}/`）にいる。
ここには identity、記憶、設定など Anima 固有のデータが格納される。

プロジェクトの作業（コード変更、調査、ビルドなど）をするときは、
「仕事場」（ワークスペース）に出かけて作業する。
ワークスペースはプロジェクトのソースコードや成果物が置かれたディレクトリである。

### レジストリとエイリアス

ワークスペースは組織共有のレジストリ（`config.json` の `workspaces` セクション）に登録される。
人間が付ける短い名前（エイリアス）と、パスの SHA-256 先頭8桁（ハッシュ）で一意に参照できる。

| 形式 | 例 | 用途 |
|------|-----|------|
| エイリアスのみ | `aischreiber` | 通常の参照（衝突時はハッシュ付きを推奨） |
| 完全形 | `aischreiber#3af4be6e` | 衝突ゼロの厳密参照 |
| ハッシュのみ | `3af4be6e` | エイリアスが分からない場合 |
| 絶対パス | `/home/main/dev/AI-Schreiber` | 直接指定（レジストリ未登録でも可） |

## ツールでの使用

### machine_run（工作機械）

`working_directory` にエイリアス、完全形、ハッシュ、または絶対パスを指定できる。

```bash
animaworks-tool machine run "コードをリファクタして" -d aischreiber
animaworks-tool machine run "テストを実行して" -d aischreiber#3af4be6e
animaworks-tool machine run "ビルドして" -d /home/main/dev/AI-Schreiber
```

### submit_tasks

各タスクの `workspace` フィールドにエイリアスを指定すると、
TaskExec がそのワークスペースを作業ディレクトリとして使用する。

```
submit_tasks(batch_id="build", tasks=[
  {"task_id": "t1", "title": "コンパイル", "description": "...", "workspace": "aischreiber", "parallel": true}
])
```

### delegate_task

`workspace` フィールドにエイリアスを指定すると、
委譲先の部下がそのワークスペースで作業する。

```
delegate_task(name="aoi", instruction="API テストを実施して", deadline="2d", workspace="aischreiber")
```

## 登録と割り当て

### 登録手順

詳細は `common_skills/workspace-manager` スキルを参照すること。
要点:

1. `config.json` の `workspaces` セクションにエイリアスとパスを追加
2. または `core.workspace.register_workspace` を Python から呼び出す
3. ディレクトリは登録時に存在確認される（存在しないとエラー）

### 部下への割り当て

スーパーバイザーは部下の `status.json` の `default_workspace` フィールドを更新し、
主な作業ディレクトリを割り当てる。`workspace-manager` スキルを参照。

## よくある問題

### ディレクトリが存在しない

- **登録時**: 存在しないパスを登録しようとするとエラーになる
- **使用時**: 登録済みでも後からディレクトリが削除された場合、解決時にエラーになる
- **対処**: パスを確認し、正しい絶対パスで再登録する

### エイリアスが見つからない

- **原因**: エイリアスがレジストリに登録されていない、または typo
- **対処**: `read_memory_file(path="config.json")` で `workspaces` セクションを確認し、正しいエイリアスを使用する

### ハッシュが変わった

- **原因**: エイリアスを上書きしてパスを変更した場合、ハッシュも変わる
- **対処**: 完全形（`alias#hash`）を使っていた場合、新しいハッシュに更新する。エイリアスのみを使っていれば影響なし
