# 組織構造の仕組み

AnimaWorks における組織構造は `config.json` の設定から動的に構築される。
本ドキュメントでは、組織構造がどのように定義・解釈・表示されるかを説明する。

## supervisorフィールドによる階層定義

組織の上下関係は、各 Anima の `supervisor` フィールドのみで定義される。

- `supervisor: null` または未設定 → その Anima はトップレベル（最上位）
- `supervisor: "alice"` → alice が上司

config.json での設定例:

```json
{
  "animas": {
    "alice": {
      "supervisor": null,
      "speciality": "経営戦略・全体統括"
    },
    "bob": {
      "supervisor": "alice",
      "speciality": "開発リード"
    },
    "carol": {
      "supervisor": "alice",
      "speciality": "デザイン・UX"
    },
    "dave": {
      "supervisor": "bob",
      "speciality": "バックエンド開発"
    }
  }
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
- supervisor に指定する名前は `animas` に存在する Anima 名でなければならない
- 循環参照（alice → bob → alice）は避けなければならない
- 1人の Anima が持てる supervisor は1名のみ

## 組織コンテキストの構築プロセス

`core/prompt/builder.py` の `_build_org_context()` が、config.json から以下の情報を動的に算出する:

1. **上司（supervisor）**: 自分の `supervisor` フィールドの値。未設定なら「あなたがトップです」
2. **部下（subordinates）**: `supervisor` が自分の名前になっている全 Anima
3. **同僚（peers）**: 自分と同じ `supervisor` を持つ Anima（自分を除く）

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
| あなたの専門 | `speciality` フィールドの値 | この分野に関する質問や判断は自分が責任を持つ |
| 上司 | 報告先の Anima | 進捗報告・問題のエスカレーション先 |
| 部下 | 自分の配下の Anima | タスクの委任先・進捗確認の対象 |
| 同僚 | 同じ上司を持つ仲間 | 関連業務で直接連携する相手 |

### 確認すべきポイント

- 上司が「(なし — あなたがトップです)」なら、あなたは組織のトップとして全体責任を負う
- 部下が「(なし)」なら、あなたはタスク実行者として自分で手を動かす
- 同僚がいれば、関連する業務で直接調整ができる

## 組織変更時の挙動

組織構造の変更は以下の手順で反映される:

1. `config.json` の `animas` セクションを編集（supervisor / speciality の変更）
2. サーバーを再起動（`animaworks start`）
3. 次回の起動時（メッセージ受信・ハートビート・cron）に新しい組織コンテキストが構築される

注意点:
- config.json を変更しただけではすぐには反映されない。Anima の次回起動まで旧コンテキストが使われる
- Anima の追加・削除も config.json の編集 + サーバー再起動で行う
- 組織変更後は、影響を受ける Anima にメッセージで通知することを SHOULD（推奨）

## 組織構造のパターン例

### パターン1: フラット組織

全員がトップレベル。上下関係なし。

```json
{
  "animas": {
    "alice": { "supervisor": null, "speciality": "企画" },
    "bob":   { "supervisor": null, "speciality": "開発" },
    "carol": { "supervisor": null, "speciality": "デザイン" }
  }
}
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

```json
{
  "animas": {
    "alice": { "supervisor": null,    "speciality": "CEO・全体統括" },
    "bob":   { "supervisor": "alice", "speciality": "開発部長" },
    "carol": { "supervisor": "alice", "speciality": "営業部長" },
    "dave":  { "supervisor": "bob",   "speciality": "バックエンド" },
    "eve":   { "supervisor": "bob",   "speciality": "フロントエンド" },
    "frank": { "supervisor": "carol", "speciality": "顧客対応" }
  }
}
```

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

```json
{
  "animas": {
    "manager": { "supervisor": null,      "speciality": "プロジェクト管理" },
    "dev1":    { "supervisor": "manager", "speciality": "API開発" },
    "dev2":    { "supervisor": "manager", "speciality": "DB設計" },
    "dev3":    { "supervisor": "manager", "speciality": "インフラ" },
    "qa":      { "supervisor": "manager", "speciality": "品質保証" }
  }
}
```

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

## specialityフィールドの活用

`speciality` は自由テキストで、Anima の専門領域を記述する。

- 組織コンテキストで各 Anima の名前の横に表示される（例: `bob (開発リード)`）
- 他の Anima がタスクの相談先や委任先を判断する手がかりになる
- 未設定の場合は「(未設定)」と表示される

効果的な speciality の書き方:
- 具体的で短い: `バックエンド開発` `顧客サポート` `データ分析`
- 曖昧すぎない: `いろいろ` → `企画・調整・進行管理`
- 複数の専門がある場合は中黒で区切る: `UI設計・フロントエンド開発`
