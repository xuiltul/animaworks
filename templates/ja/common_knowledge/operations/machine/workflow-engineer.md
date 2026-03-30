# machine ワークフロー — Engineer（エンジニア）

## ロール定義

Engineer は **plan.md を受け取り、具体化と実装を machine に投げ、2回の検証チェックポイントを担う** ロールである。

- 実装詳細計画（impl.plan.md）の具体化 → machine に委託し、Engineer が検証
- コードの実装 → machine に委託し、Engineer が検証
- 検証は2回: impl.plan.md の承認時と、実装出力の確認時

> 前提: `operations/machine/tool-usage.md` のメタパターンと共通原則を理解していること。

## Phase 1: 具体化（plan.md → impl.plan.md）

### Step 1: plan.md を読む

PdM から受け取った plan.md を読み、以下を確認する:

- [ ] `status: approved` であること
- [ ] ゴールと完了条件が明確であること
- [ ] 実装方針に技術的な矛盾がないこと
- [ ] 不明点があれば PdM に確認する（次工程に進む前に解決）

### Step 2: machine に具体化を投げる

plan.md を入力として、実装詳細計画書（impl.plan.md）の生成を machine に依頼する。

```bash
animaworks-tool machine run \
  "以下の plan.md を元に impl.plan.md を作成せよ。
  対象ファイルごとの具体的な変更内容、依存関係、実装順序を詳細化すること。

  $(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{plan.md})" \
  -d /path/to/worktree
```

結果を `state/plans/{date}_{概要}.impl-plan.md` として保存する（`status: draft`）。

### Step 3: impl.plan.md を検証する（チェックポイント 1）

Engineer が impl.plan.md を読み、以下を確認する:

- [ ] plan.md のゴール・方針と整合しているか
- [ ] 対象ファイルのパスが正確か（実際に存在するか）
- [ ] 変更内容が技術的に妥当か
- [ ] 依存関係と実装順序が正しいか
- [ ] 既存コードとの整合性が取れているか
- [ ] ロールバック可能な設計になっているか

問題があれば Engineer 自身が修正し、`status: approved` に変更する。

## Phase 2: 実装

### Step 4: machine に実装を投げる

`status: approved` の impl.plan.md を machine に渡して実装させる。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{impl-plan.md})" \
  -d /path/to/worktree
```

### Step 5: 実装出力を検証する（チェックポイント 2）

machine の出力を impl.plan.md と突合して検証する:

- [ ] `git diff` で変更内容が impl.plan.md と一致するか
- [ ] 意図しないファイルが変更されていないか
- [ ] テストが実行可能で全件パスするか
- [ ] plan.md の制約条件に違反していないか
- [ ] plan.md の完了条件を満たしているか

### Step 6: 結果の処理

- **合格**: Reviewer / Tester にレビュー・テストを依頼する
- **問題あり**: impl.plan.md を修正して machine に再委託するか、Engineer 自身が修正する

## 実装詳細計画書テンプレート（impl.plan.md）

```markdown
# 実装詳細計画書: {タスク名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: impl-plan
source: state/plans/{元のplan.md}

## ゴール（plan.md から継承）

{plan.md のゴールをそのまま転記}

## 対象ファイル

| ファイル | アクション | 変更概要 |
|---------|-----------|---------|
| {path/to/file1} | Create | {新規作成の目的} |
| {path/to/file2} | Modify | {変更内容の概要} |
| {path/to/file3} | Delete | {削除の理由} |

## 実装ステップ

### Phase 1: {準備}

- [ ] **Step 1.1**: {具体的な変更内容}
  - ファイル: `{path}`
  - 詳細: {追加コンテキスト}

- [ ] **Step 1.2**: {具体的な変更内容}
  - ファイル: `{path}`
  - 詳細: {追加コンテキスト}

### Phase 2: {コア実装}

- [ ] **Step 2.1**: {具体的な変更内容}
  - ファイル: `{path}`
  - 詳細: {追加コンテキスト}

### Phase 3: {テスト・クリーンアップ}

- [ ] **Step 3.1**: {テスト作成}
  - ファイル: `{path}`

## 依存関係

| 依存元 | 依存先 | 理由 |
|--------|--------|------|
| Step 2.1 | Step 1.1 | {理由} |

## 制約条件（plan.md から継承）

{plan.md の制約条件をそのまま転記}

## 完了条件（plan.md から継承）

{plan.md の完了条件をそのまま転記}

## ロールバック計画

1. {問題発生時の復元手順}
2. {復元後の確認手順}
```

## 制約事項

- `status: approved` でない plan.md を元に作業を開始してはならない（NEVER）
- impl.plan.md の `status: approved` なしに machine に実装を投げてはならない（NEVER）
- machine の実装出力を検証せずにコミット・プッシュしてはならない（NEVER）
- 検証で発見した問題は、修正してから次工程に進む（MUST）
