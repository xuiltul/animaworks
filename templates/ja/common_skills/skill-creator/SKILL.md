---
name: skill-creator
description: >-
  Markdownスキルを作成するメタスキル。SKILL.mdのfrontmatterと本文、Progressive Disclosureとcreate_skillの手順を扱う。
  Use when: 新規スキル追加、skillツール向けの記述ルール確認、referencesやtemplates付きスキル生成が必要なとき。
---

# skill-creator

## 実装との対応

| 役割 | モジュール |
|------|------------|
| `skill` ツール（解決・読み込み・ビルトイン置換・`*-tool` ゲート・`allowed_tools` 付与） | `core/tooling/skill_tool.py` |
| ツール説明内 `<available_skills>` の組み立て（バジェット付き） | 同上 `build_skill_tool_description()` |
| `create_skill` ツール（ディレクトリ生成） | `core/tooling/skill_creator.py` |
| スキーマ（パラメータ定義） | `core/tooling/schemas/skill.py` |
| フロントマター解析（行ベース・本文側の `---` で誤分割しない） | `core/memory/frontmatter.py` の `parse_frontmatter()` |
| メタデータ型・抽出（`SkillMeta`、手続きパス推定） | `core/schemas.py`、`core/memory/skill_metadata.py` の `SkillMetadataService.extract_skill_meta()` |
| 説明文ベースのスキル候補（Priming 等・最大5件・3層マッチ） | `core/memory/skill_metadata.py` の `match_skills_by_description()` |
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

## skill ツールでの解決と読み込み

`skill(skill_name=..., context=...)` の内部（`load_and_render_skill()`）では次の順で名前を解決する。

1. **個人スキル** `{anima}/skills/{name}/SKILL.md`
2. **共通スキル** `get_common_skills_dir()/{name}/SKILL.md`（通常はデータディレクトリ配下の `common_skills/`）
3. **手続き** `{anima}/procedures/{name}.md`

**同名がある場合は個人が共通より優先**される。`skill_name` に `/`, `\`, `..` を含めると解決されない（パストラバーサル防止のため `_resolve_skill_path` が拒否）。

### 見つからない場合

解決できないときは i18n メッセージ（`t("skill.not_found", ...)`）を返し、**利用可能な名前**を `_list_available_names()` で列挙して併記する（個人・共通の `*/SKILL.md` と手続きの `*.md` の stem）。

### パラメータ

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `skill_name` | ✓ | 読み込むスキル／手続き名（上記解決のみ対象） |
| `context` | | 任意。本文の後に `t("skill.context_header")` 付きで追記される |

### フロントマターと本文の取り出し

先頭の YAML は `core/memory/frontmatter.parse_frontmatter()` で除去する。**区切り線 `---` は行単位のみ**認識されるため、YAML 値や本文中に `---` 文字列が含まれても、旧来の単純 split より誤分割しにくい。置換・ゲート・`allowed_tools` 処理の対象は**本文（フロントマター除去後）**。

### 本文へのビルトイン置換

本文に対して `_resolve_builtins(anima_dir)` の値で `{{key}}` を**文字列置換**（`str.replace`）する。

| プレースホルダ | 置換内容（キー） |
|----------------|------------------|
| `{{now_local}}` | `now_local().isoformat()` |
| `{{anima_name}}` | `anima_dir.name` |
| `{{anima_dir}}` | `str(anima_dir)` |

### frontmatter `allowed_tools`

`load_and_render_skill` はフロントマター辞書の `allowed_tools` を読み、リストであれば返却末尾に `t("skill.tool_constraint_header")` / `t("skill.tool_constraint_desc")` と箇条書きを付与する。**ソフト制約**（強制ではない）。

### 外部ツール用スキル名（`*-tool`）

スキル名が **`*-tool`** で終わる場合（例: `gmail-tool`）、`_skill_name_to_tool_name()` でツールモジュール名に写す: 末尾 `-tool` を除き、ハイフンをアンダースコアに（`gmail-tool` → `gmail`、`image-gen-tool` → `image_gen`）。空ベース（`-tool` だけ等）はマッピングしない。

そのツール名について `load_permissions(anima_dir)` と `get_permitted_tools()` で許可集合を得て（例外時は空集合）、`filter_gated_from_guide(content, tool_name, permitted)` を適用する。各外部ツールの `EXECUTION_PROFILE` で `gated: True` のアクションは、`{tool_name}_{action}` が許可集合に無いとき、**`animaworks-tool <tool> <action>` にマッチする行を本文から除去**する。ゲート定義が無いツールではフィルタは実質 no-op。

### 一覧表示のバジェット

`build_skill_tool_description()` は固定バジェット `_DESCRIPTION_BUDGET = 8000`（各エントリ文字列の `len` で累積）まで、個人スキル → 共通スキル（ラベル付き）→ 手続き（ラベル付き）の順で1行ずつ追記し、超過時は `t("skill.truncated")` を挿入して打ち切る。スキル数が多いほど省略されやすいので、**Level 1 の `description` は短く具体的**に保つ。

### Priming 側のメタ収集との差（重要）

Priming のスキルマッチ用収集（例: `channel_d_skill_match`）では、個人 `skills/` 直下の **`*.md`（フラット）も**メタ対象に含められる。一方 **`load_and_render_skill` は常に `skills/{name}/SKILL.md` のみ**を個人スキルとして解決する。したがって **`skills/foo.md` だけでは `skill(skill_name="foo")` は失敗しうる**。運用上は `create_skill` による `{name}/SKILL.md` 形式を正とする。

## スキルファイルの構造

SKILL.md はYAMLフロントマターとMarkdown本文で構成される。
フロントマターには `name` と `description` を**必須**とする。

**`create_skill` が書き出す任意キーは `allowed_tools` のみ**（`name` / `description` に加えて）。それ以外の YAML キーを手編集で足しても、`SkillMetadataService.extract_skill_meta()` が読むのは **`name`・`description`・`allowed_tools`**（不正型の `allowed_tools` は空リスト扱い）。`load_and_render_skill` も `allowed_tools` 以外のフロントマターキーは参照しない。

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

### `description` の役割（明示呼び出し vs 自動候補）

**新規スキルの記述**は Agent Skills 標準に沿い、**`Use when:`** で利用シーンを書く（詳細は `references/description_guide.md`）。作成後は **`python scripts/lint_skill.py`** で形式を検証できる。

- **`skill` ツールを名前指定で呼んだ場合**: 上記解決パスにファイルがあれば **description のマッチに関係なく** 本文が読み込まれ、ビルトイン置換・`*-tool` フィルタ・`allowed_tools` 追記まで行われる。
- **Priming（Channel D 等）でスキル名だけが候補として載る場合**: `match_skills_by_description(message, metas, retriever=..., anima_name=...)` の **3層マッチ**を通過したスキル／手続きが最大 **5 件**（`_MAX_SKILL_MATCHES`）まで選ばれる。本文は載らず **名前のみ**。
  - **Tier 1**: `description` を正規化し、**全角括弧 `「…」` 内キーワード**のいずれかがメッセージに含まれる、または括弧が無い場合は読点・カンマ等で区切った短いフレーズのいずれかが部分一致。
  - **Tier 2**: 説明文から抽出した語（ASCII は単語境界、ストップワード除外）がメッセージ中に **2 語以上**一致。
  - **Tier 3**: RAG で `memory_type="skills"` 検索。スコア閾値は設定 `rag.skill_match_min_score`（既定 **0.75**）。Retriever 不整合時は Tier 3 をスキップしうる。

**レガシー互換**: `description` が空のとき、`extract_skill_meta()` は本文先頭付近の **`## 概要` セクションの最初の非空行**を description のフォールバックとして使う。新規スキルではフロントマターを正とし、`## 概要` 依存は避ける。

**description の書き方**（**Use when:** パターン・lint）は `references/description_guide.md` を参照。

## Progressive Disclosure（段階的開示）

スキルの情報は概ね次の段階で開示される。

| Level | 内容 | 表示タイミング |
|-------|------|----------------|
| Level 1 | `name` + `description` | ツール定義内の `<available_skills>`（バジェット内）や、Priming での候補提示（名前のみ）の材料 |
| Level 2 | body（本文） | エージェントが `skill(skill_name=...)` を実行したときのツール結果として注入 |
| Level 3 | 外部リソース | 本文の指示に従い、必要時に `read_memory_file` 等で `references/` や `templates/` を読む |

Level 1 は一覧・マッチングでコンテキストを消費しやすいため **description は簡潔**に。Level 2 は手順の中核。Level 3 で長大な参照を分離する。

※ Priming は **本文を自動注入しない**（候補名の提示まで）。本文が必要なら `skill` 呼び出しが前提。

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

※ 新規スキルには必ず `create_skill` を使うこと。`write_memory_file` で `skills/foo.md` のような**直下の単一ファイル**だけを作ると、Priming の一覧には載る可能性があっても **`skill(skill_name="foo")` は `skills/foo/SKILL.md` を要求するため参照できない**（失敗しうる）。

※ 手続きは `procedures/{name}.md`（フラット1ファイル）。フロントマターはスキルと同様に `name` / `description`（＋任意 `allowed_tools`）を推奨。

### Step 4: 確認

- **個人スキル**: `read_memory_file(path="skills/{name}/SKILL.md")` で内容確認、または `skill` ツールで `skill_name="{name}"` を実行
- **共通スキル**: 同上（パスは組織の `common_skills/` 基準）
- **手続き**: `skill(skill_name="{name}")` で `procedures/{name}.md` が解決されることを確認
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
- `create_skill` が設定できる任意フィールドは **`allowed_tools` のみ**（`tags` 等の他キーは手編集可だが、メタ抽出・`skill` 返却で使うのは基本 `name` / `description` / `allowed_tools`）
- bodyが長くなりすぎるとコンテキストを圧迫するため、150行以内を目安にする
- 外部リソース参照（Level 3）は `references/` を活用して本文を簡潔に保つ
