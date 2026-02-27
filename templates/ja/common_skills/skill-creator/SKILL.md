---
name: skill-creator
description: >-
  Markdownスキルファイル(.md)を正しいYAMLフロントマター形式で作成するためのメタスキル。
  個人スキル(skills/)・共通スキル(common_skills/)のdescription記述ルール、
  「」キーワード設計、Progressive Disclosure構造、write_memory_fileでの保存手順を提供する。
  「スキル作成」「スキルを作りたい」「新しいスキル」「手順書を作成」「スキルファイル」
---

# skill-creator

## スキルファイルの構造

スキルファイルはYAMLフロントマターとMarkdown本文で構成される。
フロントマターには `name` と `description` の2フィールドのみを記載する。

```yaml
---
name: スキル名
description: >-
  スキルの説明。
  「キーワード1」「キーワード2」
---
```

`description` はスキルの発動を決定する最重要フィールドである。
ユーザーのメッセージと description の内容がマッチした場合にのみ、本文がシステムプロンプトに注入される。
つまり description が「主要なトリガー機構」として機能する。

**descriptionの詳細な書き方**（具体化チェック、キーワード設計など）は `references/description_guide.md` を参照。

## Progressive Disclosure（段階的開示）

スキルの情報は3段階で開示される。

| Level | 内容 | 表示タイミング |
|-------|------|----------------|
| Level 1 | description | 常にスキルテーブルに表示。スキル選択の判断材料 |
| Level 2 | body（本文） | descriptionがマッチした時にシステムプロンプトに注入 |
| Level 3 | 外部リソース | 本文中の指示に従い、必要時にファイルを読み込む |

Level 1 は常にコンテキストを消費するため、descriptionは簡潔に保つ。
Level 2 の本文には具体的な手順を記載する。
Level 3 は長大な参照資料やコード例を外部ファイルに分離する場合に使う。

## 作成手順

### Step 1: ヒアリング

ユーザーの要求を理解する。以下を確認する:

- 何を自動化・手順化したいか
- 対象は個人スキルか共通スキルか
- どのような言葉で発動させたいか（トリガーキーワード）

### Step 2: 設計

以下を決定する:

- **name**: スキル名（ケバブケース、例: `my-skill`）
- **description**: トリガーとなる説明文とキーワード（`references/description_guide.md` を参照）
- **body**: 手順の構成（セクション分け）

### Step 3: 作成

スキルファイルを保存する:

```
write_memory_file("skills/{name}.md", content)
```

共通スキルの場合:

```
write_memory_file("common_skills/{name}.md", content)
```

テンプレートは `templates/skill_template.md` を参照。

### Step 4: 確認

保存後に読み直して内容を検証する:

```
read_memory_file("skills/{name}.md")
```

## チェックリスト

保存前に以下を確認する:

- [ ] `---` で始まり `---` で閉じるYAMLフロントマターがある
- [ ] `name` フィールドがある
- [ ] `description` フィールドがある
- [ ] descriptionに `「」` キーワードが1つ以上ある
- [ ] **descriptionがドメイン固有で具体的**（「管理を行う」「確認する」のような汎用表現を避け、ツール名・操作名・対象を明記）
- [ ] bodyに具体的な手順が記載されている
- [ ] `## 概要` / `## 発動条件` の旧形式を使っていない

## テンプレート

`templates/skill_template.md` をコピーして使用する。または以下を参照:

```markdown
---
name: {スキル名}
description: >-
  {具体的な対象}の{具体的な操作}スキル。
  {使用するツール名/API名}で{具体的な手順の概要}を実行する。
  「{ドメイン固有キーワード1}」「{ドメイン固有キーワード2}」「{ドメイン固有キーワード3}」
---

# {スキル名}

## 手順

1. ...
2. ...

## 注意事項

- ...
```

## 注意事項

- スキルはMarkdown手順書であり、Pythonコード（ツール）とは異なる
- フロントマターに `name` と `description` 以外のフィールドを入れない
- bodyが長くなりすぎるとコンテキストを圧迫するため、150行以内を目安にする
- 外部リソース参照（Level 3）を活用して本文を簡潔に保つ
