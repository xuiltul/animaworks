# 組織構造の仕組み

AnimaWorks における組織構造は、各 Anima の `status.json`（または `IDENTITY.md`）を Single Source of Truth（SSoT）として構築される。
`core/org/org_sync.py` がディスク上の **supervisor** を `config.json` に同期し、プロンプト構築時に利用される。
本ドキュメントでは、組織構造がどのように定義・解釈・表示されるかを説明する。

## データソースと優先順位

### supervisor（上司）

組織の上下関係は、各 Anima の `supervisor` で定義される。読み取り優先順位:

1. **status.json** — `"supervisor"` キー（推奨）
2. **IDENTITY.md** — 表形式 `| 上司 | name |` の行（日本語のみ。`core/config/models.py` の `read_anima_supervisor` が解析）

`supervisor` が未設定・空・「なし」「(なし)」「（なし）」「-」「---」の場合はトップレベル（最上位）となる。
`animaworks config set animas.<name>.supervisor <supervisor>` は root 所有 status/config をまとめて更新する。`config.json` だけを直接変更すると org_sync によってディスクから上書きされるため、CLI/API を使用する。

### speciality（専門）

専門領域は `core/prompt/builder.py` の `_scan_all_animas()` により以下の優先順位で解決される:

1. **status.json** — `"speciality"` キー（自由テキスト）
2. **config.json** — `animas.<name>.speciality`（status.json に `speciality` キーがない場合のフォールバック）
3. **status.json** — `"role"` キー（上記で解決されない場合の最終フォールバック。ロール名: engineer, researcher, manager, writer, ops, general）

**注意:** org_sync は **speciality を同期しない**。speciality はプロンプト構築時にディスクと config から都度解決される。
`animaworks anima create --from-md` で作成した Anima は `status.json` に `role` が入るが `speciality` は入らない。
カスタム表示（例: 「開発リード」）にしたい場合は、`animaworks config set animas.<name>.speciality "開発リード"` を使用する。root 所有 status/config の両方が更新される。

## org_sync による config.json 同期

`core/org/org_sync.py` の `sync_org_structure()` が以下を行う:

1. 各 Anima ディレクトリ（`IDENTITY.md` が存在するもののみ）から `status.json` / `IDENTITY.md` を読み、supervisor を抽出（`read_anima_supervisor`）
2. 循環参照を検出（検出された Anima は同期対象外）
3. `config.json` の `animas.<name>.supervisor` をディスクの値に合わせて更新（**supervisor のみ**）
4. ディスク上に存在しない Anima の config エントリを削除（prune）

**同期される項目:** supervisor のみ。speciality は org_sync では更新されない。

**実行タイミング:**

- サーバー起動時（`animaworks start` の Anima プロセス起動後）
- Anima が reconciliation で追加されたとき（`on_anima_added` コールバック）

## supervisor による階層定義

- `supervisor: null` または未設定 → その Anima はトップレベル（最上位）
- `supervisor: "alice"` → alice が上司

root 所有の status 値の例（表示用。直接編集せず CLI/API で設定）:

```json
{
  "enabled": true,
  "supervisor": null,
  "speciality": "経営戦略・全体統括"
}
```

```json
{
  "enabled": true,
  "supervisor": "alice",
  "speciality": "開発リード"
}
```

この設定で以下の階層が構築される:

```
alice（経営戦略・全体統括）
├── bob（開発リード）
│   └── dave（バックエンド開発）
└── carol（デザイン・UX）
```

重要な制約:
- supervisor に指定する名前は既知の Anima 名（英名）でなければならない
- 循環参照（alice → bob → alice）は検出され同期対象外となる
- 1人の Anima が持てる supervisor は1名のみ

## 組織コンテキストの構築プロセス

`core/prompt/builder.py` の `_build_org_context()` が、ディレクトリスキャンと config.json のマージ結果から以下の情報を算出する:

1. **上司（supervisor）**: 自分の supervisor の値。未設定なら「あなたがトップです」
2. **部下（subordinates）**: supervisor が自分の名前になっている全 Anima
3. **同僚（peers）**: 自分と同じ supervisor を持つ Anima（自分を除く）

算出結果はシステムプロンプトに「あなたの組織上の位置」として注入される:

```
## あなたの組織上の位置

あなたの専門: 開発リード

上司: alice (経営戦略・全体統括)
部下: dave (バックエンド開発)
同僚（同じ上司を持つメンバー）: carol (デザイン・UX)
```

## 自分の位置の読み取り方

システムプロンプトの「あなたの組織上の位置」セクションから、以下を確認できる:

| 項目 | 意味 | 行動への影響 |
|------|------|-------------|
| あなたの専門 | speciality の値 | この分野に関する質問や判断は自分が責任を持つ |
| 上司 | 報告先の Anima | 進捗報告・問題のエスカレーション先 |
| 部下 | 自分の配下の Anima | タスクの委任先・進捗確認の対象 |
| 同僚 | 同じ上司を持つ仲間 | 関連業務で直接連携する相手 |

### 確認すべきポイント

- 上司が「(なし — あなたがトップです)」なら、あなたは組織のトップとして全体責任を負う
- 部下が「(なし)」なら、あなたはタスク実行者として自分で手を動かす
- 同僚がいれば、関連する業務で直接調整ができる

## 組織変更時の挙動

組織構造の変更は以下の手順で反映される:

1. 組織設定は root 所有の CLI/API で変更する（例: `animaworks config set animas.<name>.supervisor <supervisor>` / `animaworks config set animas.<name>.speciality <speciality>`）。
2. CLI/API は `status.json` と `config.json` の両方を更新し、org_sync が階層を同期する。次回の reconciliation / prompt で新しい値が使われる。
3. **speciality の変更:** プロンプト構築時に読み取られるため、Anima の再起動は不要。次回のチャット/ハートビートで反映される。

注意点:
- Anima プロセスから `status.json` / `config.json` を直接編集しない。書き手は root とサーバー停止中の CLI のみ。
- 組織変更後は、影響を受ける Anima にメッセージで通知することを SHOULD（推奨）

## 組織構造のパターン例

以下は組織設定の例。root 所有 CLI/API で設定し、org_sync が `supervisor` を同期する。`speciality` はプロンプト構築時に解決される。

### パターン1: フラット組織

全員がトップレベル。上下関係なし。

各 Anima の status.json:
```json
{ "supervisor": null, "speciality": "企画" }
{ "supervisor": null, "speciality": "開発" }
{ "supervisor": null, "speciality": "デザイン" }
```

```
alice（企画）
bob（開発）
carol（デザイン）
```

特徴:
- 全員が対等な立場で直接やりとりできる
- 小規模チームや、各自が独立した業務を持つ場合に適する
- 全員の同僚は「(なし)」（同じ supervisor を共有していないため）

### パターン2: 階層型組織

明確な上下関係がある。最も一般的なパターン。

各階層フィールドは root 所有 CLI/API で設定する（例: `animaworks config set animas.dave.supervisor bob`）:

```
alice（CEO・全体統括）
├── bob（開発部長）
│   ├── dave（バックエンド）
│   └── eve（フロントエンド）
└── carol（営業部長）
    └── frank（顧客対応）
```

特徴:
- bob と carol は同僚（同じ supervisor = alice）
- dave と eve は同僚（同じ supervisor = bob）
- dave から frank への連絡は bob → alice → carol → frank の経路を辿る（他部署ルール）

### パターン3: 専門家＋マネージャー型

少数のマネージャーが多数の専門家を統括する。

```
manager（プロジェクト管理）
├── dev1（API開発）
├── dev2（DB設計）
├── dev3（インフラ）
└── qa（品質保証）
```

特徴:
- 全メンバーが同僚関係。直接連携が容易
- manager が全体のタスク配分と進捗管理を担当
- スタートアップやプロジェクトチームに適する

## speciality の活用

`speciality` は root 所有の `status.json` に自由テキストで保存される。`animaworks config set animas.<name>.speciality <value>` で設定する。未設定時は `role`（ロール名）がフォールバックとして表示される。

- 組織コンテキストで各 Anima の名前の横に表示される（例: `bob (開発リード)` または `bob (engineer)`）
- 他の Anima がタスクの相談先や委任先を判断する手がかりになる
- 未設定の場合は「(未設定)」と表示される

**Anima 作成時の挙動（`core/anima/factory.py`）:**
- `animaworks anima create --from-md PATH [--role ROLE] [--supervisor NAME] [--name NAME]` で作成すると、`status.json` に `supervisor` と `role` が書き込まれる
- **supervisor**: `--supervisor` オプションが指定されていればそれを優先。未指定の場合はキャラクターシートの基本情報テーブル（`| 上司 | name |`）から解析
- **speciality**: キャラクターシートの基本情報テーブルには含まれず、`_create_status_json` も speciality を書き込まないため、作成時に自動設定されない
- カスタム専門表示が必要な場合は、作成後に `animaworks config set animas.<name>.speciality "開発リード"` で設定する。設定ファイルを直接編集しない
- `create_from_template` / `create_blank` で作成した場合も同様に、speciality は status.json に自動設定されない（テンプレートに status.json が含まれる場合はその内容がコピーされる）

効果的な speciality の書き方:
- 具体的で短い: `バックエンド開発` `顧客サポート` `データ分析`
- 曖昧すぎない: `いろいろ` → `企画・調整・進行管理`
- 複数の専門がある場合は中黒で区切る: `UI設計・フロントエンド開発`
