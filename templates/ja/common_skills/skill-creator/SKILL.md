---
name: skill-creator
description: >-
  Markdownスキルを作成するメタスキル。SKILL.mdのfrontmatterと本文、Progressive Disclosureとcreate_skillの手順を扱う。
  Use when: 新規スキル追加、read_memory_file 向けの記述ルール確認、referencesやtemplates付きスキル生成が必要なとき。
---

# skill-creator

## 実装との対応

| 役割 | モジュール |
|------|------------|
| `read_memory_file` ツール（スキル／手続きの相対パスを指定して本文を読む） | `ToolHandler` 経由（記憶ツリー内ファイル） |
| システムプロンプト内のスキルカタログ（パス一覧・バジェット付き） | プロンプト構築（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`） |
| `create_skill` ツール（ディレクトリ生成） | `core/tooling/skill_creator.py` |
| スキーマ（パラメータ定義） | `core/tooling/schemas/skill.py` |
| フロントマター解析（行ベース・本文側の `---` で誤分割しない） | `core/memory/frontmatter.py` の `parse_frontmatter()` |
| メタデータ型・抽出（`SkillMeta`、手続きパス推定） | `core/schemas.py`、`core/memory/skill_metadata.py` の `SkillMetadataService.extract_skill_meta()` |
| 説明文ベースのスキルメタ抽出（カタログ・検索補助） | `core/memory/skill_metadata.py` の `SkillMetadataService` 等 |
| `*-tool` 本文のゲート行除去 | `core/tooling/guide.py` の `filter_gated_from_guide()` |
| 許可ツール集合（permissions） | `core/config/models.load_permissions()` + `core/tooling/permissions.get_permitted_tools()` |

## スキルの種類とパス

スキルと手続きは**別レイアウト**で管理される。

| 種別 | パス | 備考 |
|------|------|------|
| 個人スキル | `skills/{name}/SKILL.md` | ディレクトリ + `SKILL.md` |
| 共通スキル | `common_skills/{name}/SKILL.md` | ランタイムでは `~/.animaworks/common_skills/` 等 |
| 手続き（procedure） | `procedures/{name}.md` | **フラット1ファイル**。ディレクトリではない |

`create_skill` が生成するのは上表の**スキル**（個人または共通）のみ。手続きは `write_memory_file` 等で `procedures/*.md` として別途作成する。

## read_memory_file でのスキル読み込み

スキル・手続きの本文は **`read_memory_file(path="...")`** で記憶ツリー相対パスを指定して読む。システムプロンプトのスキルカタログに、利用可能なパス（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`）が示される。

- **個人スキル**: `skills/{name}/SKILL.md`
- **共通スキル**: `common_skills/{name}/SKILL.md`
- **手続き**: `procedures/{name}.md`

`path` は Anima ディレクトリ基準の相対パス（共有ツリーは `common_skills/` 等のプレフィックス）で渡す。

### フロントマターと本文

SKILL.md 先頭の YAML は `core/memory/frontmatter.parse_frontmatter()` で除去して読む。**区切り線 `---` は行単位のみ**認識されるため、YAML 値や本文中に `---` が含まれても誤分割しにくい。

### 本文のプレースホルダ（`*-tool` ガイド）

スキル本文では `{{now_local}}`, `{{anima_name}}`, `{{anima_dir}}` 等のプレースホルダが使われる場合がある。外部ツール向けスキル（名前が `*-tool` で終わる）は、許可設定に応じて `animaworks-tool` 行のゲート処理が行われる（`core/tooling/guide.py` の `filter_gated_from_guide()` 等）。

### カタログと description

スキルカタログの **Level 1** では `name` + `description` がバジェット内に載る。スキル数が多いほど省略されやすいので、**`description` は短く具体的**に保つ。全文が必要なときはカタログのパスを `read_memory_file` で開く。

**レガシー互換**: `description` が空のとき、`extract_skill_meta()` は本文の **`## 概要` セクションの最初の非空行**をフォールバックとして使う。新規スキルではフロントマターを正とする。

### レイアウト上の注意

個人 `skills/` 直下の **`*.md`（フラット単一ファイル）**は、カタログやメタ抽出の対象になりうるが、**推奨パスは `skills/{name}/SKILL.md`**。運用上は `create_skill` によるディレクトリ形式を正とする。

## スキルファイルの構造

SKILL.md はYAMLフロントマターとMarkdown本文で構成される。
フロントマターには `name` と `description` を**必須**とする。

**`create_skill` が書き出す任意キーは `allowed_tools` のみ**（`name` / `description` に加えて）。それ以外の YAML キーを手編集で足しても、`SkillMetadataService.extract_skill_meta()` が読むのは **`name`・`description`・`allowed_tools`**（不正型の `allowed_tools` は空リスト扱い）。

```yaml
---
name: skill-name
description: >-
  スキルが行うことの簡潔な説明（三人称）。
  Use when: このスキルを使う具体的な場面をカンマ区切りで列挙する。
allowed_tools:
  - read_memory_file
  - web_search
---
```

### `description` の役割

**新規スキルの記述**は Agent Skills 標準に沿い、**`Use when:`** で利用シーンを書く（詳細は `references/description_guide.md`）。作成後は **`python scripts/lint_skill.py`** で形式を検証できる。

- **`read_memory_file` でパスを指定して読んだ場合**: ファイルが存在すれば **description のマッチに関係なく** 本文が得られる（フロントマター処理・`*-tool` 関連の扱いは読み込み経路に依存）。
- **システムプロンプトのスキルカタログ**: Level 1 として `name` + `description` がバジェット付きで載る。全文は載らないため、手順が必要なら **`read_memory_file(path="skills/.../SKILL.md")` 等**で開く。

**レガシー互換**: `description` が空のとき、`extract_skill_meta()` は本文先頭付近の **`## 概要` セクションの最初の非空行**を description のフォールバックとして使う。新規スキルではフロントマターを正とし、`## 概要` 依存は避ける。

**description の書き方**（**Use when:** パターン・lint）は `references/description_guide.md` を参照。

## Progressive Disclosure（段階的開示）

スキルの情報は概ね次の段階で開示される。

| Level | 内容 | 表示タイミング |
|-------|------|----------------|
| Level 1 | `name` + `description` | システムプロンプトのスキルカタログ（バジェット内）の材料 |
| Level 2 | body（本文） | エージェントが `read_memory_file(path="skills/.../SKILL.md")` 等で読み込んだとき |
| Level 3 | 外部リソース | 本文の指示に従い、必要時に `read_memory_file` 等で `references/` や `templates/` を読む |

Level 1 はカタログでコンテキストを消費しやすいため **description は簡潔**に。Level 2 は手順の中核。Level 3 で長大な参照を分離する。

※ Priming のスキルマッチチャネル（旧 Channel D）は廃止。本文が必要なら **`read_memory_file`** でスキルパスを開く。

## 作成手順

### Step 1: ヒアリング

ユーザーの要求を理解する。以下を確認する:

- 何を自動化・手順化したいか
- 対象は個人スキルか共通スキルか（手続きなら `procedures/` への別設計）
- **Use when:** に書く利用シーン（いつこのスキルを選ぶか）

### Step 2: 設計

以下を決定する:

- **name**: スキル名（ケバブケース、例: `my-skill`）。外部ツールガイドなら `*-tool` 規約を検討
- **description**: 三人称の要約 + **`Use when:`** 行（`references/description_guide.md` を参照）
- **body**: 手順の構成（セクション分け）。必要なら `{{now_local}}` 等のプレースホルダを利用
- **references** / **templates**: 必要なら外部ファイルの設計
- **allowed_tools**: 推奨ツールを絞りたい場合のみ

### Step 3: 作成

`create_skill` ツールでスキルをディレクトリ構造として作成する。

**基本（個人スキル）**:

```
create_skill(skill_name="{name}", description="{description}", body="{body}")
```

**共通スキル**:

```
create_skill(skill_name="{name}", description="{description}", body="{body}", location="common")
```

**references と templates を含める場合**:

```
create_skill(
  skill_name="{name}",
  description="{description}",
  body="{body}",
  location="personal",
  references=[
    {"filename": "description_guide.md", "content": "..."},
  ],
  templates=[
    {"filename": "skill_template.md", "content": "..."},
  ],
  allowed_tools=["read_memory_file", "write_memory_file"]
)
```

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| skill_name | ✓ | スキル名（ケバブケース）。`/`, `\`, `..` 不可 |
| description | ✓ | frontmatter description（**Use when:** 推奨。`references/description_guide.md`） |
| body | ✓ | SKILL.md本文（Markdown）。ビルトイン置換対象 |
| location | | `personal`（デフォルト）または `common` |
| references | | `references/` に配置するファイル群。`[{filename, content}, ...]` |
| templates | | `templates/` に配置するファイル群。`[{filename, content}, ...]` |
| allowed_tools | | frontmatter の `allowed_tools`（任意） |

`references` / `templates` の `filename` にパス成分は含めない。`_validate_filename()` で親ディレクトリ外に解決されないことを確認し、不正なら**黙ってスキップ**（そのファイルは作られない）。

※ 新規スキルには必ず `create_skill` を使うこと。`write_memory_file` で `skills/foo.md` のような**直下の単一ファイル**だけを作ると、**`read_memory_file(path="skills/foo/SKILL.md")` では開けない**（推奨は `skills/foo/SKILL.md` のディレクトリ形式）。

※ 手続きは `procedures/{name}.md`（フラット1ファイル）。フロントマターはスキルと同様に `name` / `description`（＋任意 `allowed_tools`）を推奨。

### Step 4: 確認

- **個人スキル**: `read_memory_file(path="skills/{name}/SKILL.md")` で内容確認
- **共通スキル**: `read_memory_file(path="common_skills/{name}/SKILL.md")` 等、カタログに示されたパスで確認
- **手続き**: `read_memory_file(path="procedures/{name}.md")` で解決できることを確認
- **`python scripts/lint_skill.py`** で `SKILL.md` の frontmatter / description を検証（任意だが推奨）

## チェックリスト

保存前に以下を確認する:

- [ ] `---` で始まり `---` で閉じるYAMLフロントマターがある
- [ ] `name` フィールドがある
- [ ] `description` フィールドがある
- [ ] description に **`Use when:`** があり、かつドメイン固有の具体語を含める（旧 `「」` 列挙は使わない）
- [ ] **descriptionがドメイン固有で具体的**（「管理を行う」「確認する」のような汎用表現を避け、ツール名・操作名・対象を明記）
- [ ] bodyに具体的な手順が記載されている
- [ ] 新規ではフロントマターに `description` を置き、**`## 概要` だけに説明を依存しない**（`## 発動条件` 等の旧テンプレート形式も避ける）
- [ ] スキルは `create_skill` で `{name}/SKILL.md` を作成している（手続きは `procedures/*.md` で意図通りか）

## テンプレート

本スキル同梱の `templates/skill_template.md` を参照する。または以下をコピーして使用:

```markdown
---
name: {スキル名}
description: >-
  {具体的な対象}の{具体的な操作}スキル（三人称の短い要約）。
  Use when: {利用シーンをカンマ区切り}
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
- フロントマターの必須フィールドは `name` と `description`
- `create_skill` が設定できる任意フィールドは **`allowed_tools` のみ**（`tags` 等の他キーは手編集可だが、メタ抽出で使うのは基本 `name` / `description` / `allowed_tools`）
- bodyが長くなりすぎるとコンテキストを圧迫するため、150行以内を目安にする
- 外部リソース参照（Level 3）は `references/` を活用して本文を簡潔に保つ
