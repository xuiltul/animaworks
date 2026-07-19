# Company 管理ガイド

Company 機構は、複数の会社を一つの AnimaWorks ランタイムで分離して運用し、必要になったときは会社単位で移転できるようにする仕組みです。本書では、日常運用から分割・エクスポートまでを説明します。

## データモデル

Anima の所属先は `<data_dir>/animas/<anima>/status.json` の `company` フィールドが正本です。会社本体は次の構造を持ちます。

```text
<data_dir>/
├── animas/<anima>/
├── companies/<company>/
│   ├── company.json
│   ├── vision.md
│   ├── animas/          # 所属 Anima への相対 symlink ビュー
│   ├── knowledge/
│   ├── skills/
│   ├── shared/
│   └── credentials/
├── common_knowledge/
└── common_skills/
```

`company.json` は機械向けの `name`、表示用の `display_name`、`created_at` を保持します。`companies/<company>/animas/` は閲覧用のリンクであり、Anima の実体は常に `animas/<anima>/` にあります。

### 3 層の知識とスキル

参照範囲は次の 3 層です。

| 層 | 主なパス | 対象 |
|---|---|---|
| Anima 個別層 | `animas/<anima>/knowledge/`, `animas/<anima>/skills/` | その Anima 固有 |
| Company 層 | `companies/<company>/knowledge/`, `companies/<company>/skills/` | その会社に所属する Anima |
| 共通層 | `common_knowledge/`, `common_skills/` | ランタイム内の全 Anima |

会社固有の方針・手順は Company 層、全社横断で公開してよい内容だけを共通層に置きます。個別層は移転時に Anima ディレクトリと一緒に運ばれます。

### 境界ルールと無所属 Anima

- 異なる会社に所属する Anima 間では、内部メッセージ、委任、ミーティングなどの直接連携を境界チェックで拒否します。
- Company 層の knowledge/skills は、自分が所属する会社のものだけを参照できます。他社の `companies/<company>/...` を直接指定しても参照範囲には入りません。
- 共通層は全会社から見えるため、会社固有情報や秘密を置かないでください。
- `status.json` に `company` がない無所属 Anima は、既存環境との互換性のため company 間境界の対象外です。無所属は隔離状態ではありません。会社分離を必要とする Anima は必ず `assign` してください。
- 所属判定は実行時に `status.json` から読み直されるため、所属変更にサーバー再起動は不要です。ただし、共有 RAG インデックスは再構築して内容を同期してください。

## CLI リファレンス

以下の例の `<company>`、`<anima>`、`<path>`、`<dir>` は実際の値に置き換えます。Company の `name` は `[a-z0-9][a-z0-9_-]*` に一致する必要があります。

### `company create`

```bash
animaworks company create <company> [--display-name <label>]
```

`companies/<company>/` の骨格を作成します。既存の場合は不足ファイルと不足ディレクトリだけを補うため、再実行できます。既存の値は上書きしません。

### `company list`

```bash
animaworks company list
```

会社の name、display name、所属 Anima 数に加え、無所属 Anima を表示します。分割前後の確認に使います。

### `company assign`

```bash
animaworks company assign <anima> [<anima> ...] --to <company>
animaworks company assign <anima> [<anima> ...] --unassign
```

`--to` は Anima を会社へ所属させ、会社側の `animas/` に相対 symlink ビューを作ります。すでに別会社に所属している場合は旧ビューを除去して移籍させます。`--unassign` は所属フィールドと会社側ビューを除去します。存在しない会社または Anima はエラーです。

### `company adopt`

```bash
animaworks company adopt <path> [<path> ...] --to <company> \
  [--dest shared|knowledge|skills|credentials|.] [--no-symlink]
```

data directory 内の既存資産を会社ディレクトリへ移します。操作前のコピーは `backup/company-adopt-<UTC timestamp>/` に元の相対パスを保って保存されます。

`--dest` 省略時は、`shared/` 配下を `shared/`、`credentials/` 配下を `credentials/`、単体 Markdown ファイルを `knowledge/`、その他を `shared/` へ配置します。通常は旧パスに新しい実体への相対 symlink を残します。旧パスを廃止できると確認済みの場合だけ `--no-symlink` を使います。data directory 外のパス、symlink 自体、既存の移動先への上書きは拒否されます。

### `company split`

```bash
animaworks company split --manifest <file>
animaworks company split --manifest <file> --execute
```

マニフェストに従い `create` → `assign` → `adopt` を順に処理します。既定は dry-run で、ファイルシステムを変更せず予定を表示します。内容を確認してから `--execute` を付けます。完了済みの操作は `SKIP` となるため再実行できます。一部で失敗した場合はそこで停止するので、表示された完了済み操作とエラーを確認し、原因を直して再実行します。

### `company export`

```bash
animaworks company export <company> --out <dir>
```

会社を新環境へ移すための bundle を `<dir>` に生成します。出力先が存在して空でない場合は上書きせずエラーになります。bundle には次のものが含まれます。

- `animas/`: 所属 Anima の実体一式
- `companies/<company>/`: 会社ディレクトリ一式（`animas/` の symlink ビューを除く）
- `common_knowledge/` と `common_skills/`: 共通層
- `config.export.json`: 所属 Anima だけに絞った設定骨格。credential の値は `REDACTED`
- `README.md`: 新環境で必要な手作業と、スキップされた symlink の一覧
- `scan-report.md`: 他社の company name、display name、所属 Anima 名の混入候補

Anima ディレクトリ内部を指す安全な相対 symlink は維持されます。data directory 外または他社領域を指す symlink はコピーせず、`README.md` に記録します。コマンドの最後に member 数、skipped symlink 数、scan hit 数が表示されます。スキャン結果はエクスポートをブロックしないため、移転前に必ず `scan-report.md` をレビューしてください。

## Split マニフェスト

JSON と YAML を受理します。YAML を使うには PyYAML が必要です。トップレベルの `companies` は配列で、各要素は次のフィールドを持ちます。

| フィールド | 必須 | 型 | 説明 |
|---|---:|---|---|
| `name` | yes | string | Company name |
| `display_name` | no | string | 表示名。省略時は name |
| `members` | no | string[] | 所属させる Anima 名 |
| `adopt` | no | object[] | 移動する資産 |
| `adopt[].path` | yes | string | data directory 相対または絶対パス |
| `adopt[].dest` | no | string | `shared`, `knowledge`, `skills`, `credentials`, `.`。省略時は自動推定 |
| `adopt[].symlink` | no | boolean | 旧パスに symlink を残すか。既定 `true`。`false` なら残さない |

```yaml
companies:
  - name: <company-a>
    display_name: <display-label-a>
    members:
      - <anima-a1>
      - <anima-a2>
    adopt:
      - path: shared/<asset-a>
        dest: shared
      - path: credentials/<credential-set-a>
        dest: credentials
        symlink: false
  - name: <company-b>
    members:
      - <anima-b1>
```

山括弧を含む値は説明用プレースホルダーであり、そのままでは有効な company name ではありません。

## 標準運用フロー: split から export まで

1. バックアップを取得し、`company list` で現在の会社と無所属 Anima を記録します。
2. 移転対象、所属 Anima、Company 層へ移す資産を棚卸しし、マニフェストを作ります。共通層に会社固有情報がないかも確認します。
3. `company split --manifest <file>` を実行し、`DRY-RUN` の作成先・所属先・移動先をレビューします。
4. `company split --manifest <file> --execute` を実行します。もう一度実行し、全項目が `SKIP` になることを確認します。
5. `company list` で無所属と member 数を確認し、各 member に対して `animaworks index --anima <anima>` を実行します。
6. `company export <company> --out <dir>` を実行します。
7. `scan-report.md` と `README.md` をレビューします。scan hit がある場合は、その内容が移転可能か、削除・匿名化が必要かを人が判断します。
8. 新環境へ bundle を配置し、`config.export.json` を基に `config.json` を整えます。credential の実値と必要な vault key を安全な別経路で投入します。
9. Anima 名を `group_id` とする Neo4j データを対象 member ごとに移行し、各 member の vectordb を `animaworks index --anima <anima>` で再構築します。
10. systemd unit を新環境用に作成し、権限、起動、Company 層の参照、会社間境界、メッセージ、RAG 検索を確認します。

## トラブルシューティング

### 誤った会社へ assign した

正しい会社が決まっている場合は、直接付け替えます。

```bash
animaworks company assign <anima> --to <correct-company>
```

未決定なら一度無所属へ戻します。

```bash
animaworks company assign <anima> --unassign
```

`company list` で結果を確認し、会社共有の古い RAG 内容を残さないため `animaworks index --anima <anima>` を実行してください。無所属は境界で隔離されない点に注意します。

### adopt を取り消したい

`adopt` に自動 undo はありません。出力された `Backup:` の場所、または `backup/company-adopt-<UTC timestamp>/` から復元します。

1. サービスを停止し、対象資産への書き込みを止めます。
2. 会社側へ移動した実体と、旧パスに残った symlink の双方を確認します。
3. 必要なら現在の会社側実体を別途退避します。
4. 旧パスの symlink を除去し、backup 内の同じ相対パスから実体を旧位置へコピーします。
5. 会社側の不要な実体を除去し、参照元と権限を確認してからサービスを再開します。

削除・上書き対象を取り違えると復旧不能になるため、backup の存在と復元先を確認してから手動操作してください。マニフェストを再実行する場合は、復元後の状態に合わせて `adopt` エントリも修正します。

### split が途中で停止した

エラーより前の操作は完了しています。原因（存在しない member、移動先の重複、data directory 外のパス、予期しない symlink など）を直し、まず dry-run、次に `--execute` で再実行します。完了済みの項目は `SKIP` されます。

### export の出力先が拒否された

既存の非空ディレクトリは上書きしません。空の新規ディレクトリを指定してください。失敗した bundle の内容を再利用せず、原因を修正して別の空ディレクトリへ出力します。

### scan hit がある

`scan-report.md` のファイル・行を確認し、単なる説明か、他社の秘密・個人データ・依存関係かを分類します。レポートは候補検出であり、自動削除も export の停止も行いません。必要な修正または匿名化後、空の出力先へ再 export して hit 数を確認します。
