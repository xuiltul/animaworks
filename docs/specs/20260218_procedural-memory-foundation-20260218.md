# 手続き記憶基盤整備 — frontmatter導入・自動想起・RAGインデックス・成功/失敗追跡

## Overview

手続き記憶（procedures/, skills/）のライフサイクル管理基盤を整備する。proceduresにYAML frontmatter + descriptionベース3-tierマッチングを導入し、write_memory_file後のRAGインデックス自動更新を追加し、成功/失敗追跡メカニズム（フレームワーク自動追跡 + エージェント明示報告）を実装する。後続のIssue（自動蒸留・再固定化・効用ベース忘却）の前提となる基盤Issue。

## Problem / Background

### Current State

- **proceduresは想起されにくい**: skills/は3-tierマッチング（キーワード→語彙→ベクトル）で自動注入されるが、procedures/は`search_memory`による手動キーワード検索のみ — `core/prompt/builder.py:434-458`
- **proceduresにフォーマット規約がない**: skills/にはYAML frontmatter + descriptionのソフトバリデーションがあるが、procedures/にはフォーマットがない — `core/tooling/handler.py:481-484`（skillsのみバリデーション）
- **write_memory_file後のRAGインデックス未更新**: スキルや手順を作成・更新してもRAGが古いまま。Tier 3ベクトル検索が不正確 — `core/tooling/handler.py:425-486`（インデックス更新呼び出しなし）
- **成功/失敗の追跡がない**: 手順に従って実行した結果のフィードバックが手順ファイルに反映されない
- **メタデータの欠如**: 作成日時、使用回数、信頼度などの構造化メタデータがない

### Root Cause

1. **procedures/とskills/の設計格差**: skills/は3-tierマッチング + frontmatter + バリデーションが整備されているが、procedures/にはこれらが一切ない — `core/memory/manager.py:102-164`（skills用match_skills_by_description）
2. **RAGインデックス更新の漏れ**: `_handle_write_memory_file()` にインデックス更新がない。`append_episode()` と `write_knowledge()` にはある — `core/tooling/handler.py:425-486` vs `core/memory/manager.py:741-745, 796-800`
3. **手続き記憶のライフサイクル管理が設計されていない**: エピソード→意味記憶のパイプラインは整備済みだが、手続き記憶には作成以降の管理がない

### Impact

| Component | Impact | Description |
|-----------|--------|-------------|
| `core/memory/manager.py` | Direct | procedures frontmatter読み書き、match_procedures_by_description追加 |
| `core/prompt/builder.py` | Direct | proceduresの3-tierマッチング注入、スキル注入追跡記録 |
| `core/tooling/handler.py` | Direct | write_memory_file後のRAGインデックス更新、report_procedure_outcome追加 |
| `core/tooling/schemas.py` | Direct | report_procedure_outcomeツールスキーマ追加 |
| `core/memory/rag/indexer.py` | Direct | proceduresのfrontmatterストリップ |
| `core/prompt/builder.py`, `core/skills/index.py` | Indirect | 現行の SkillIndex / Skill Router / catalog 経路で procedures も候補化 |

## Decided Approach / 確定方針

### Design Decision

確定: **procedures/をskills/と同等の想起・管理基盤に引き上げる**。YAML frontmatter（description, tags, success_count, failure_count, confidence, version）を導入し、builder.pyの3-tierマッチングでシステムプロンプトに自動注入する。write_memory_file後のRAGインデックス自動更新を追加する。成功/失敗追跡はフレームワーク自動追跡とエージェント明示報告の2経路で実装し、明示報告を優先する。

### Rejected Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| 現状維持（手動search_memoryのみ） | 変更なし | proceduresがほぼ死蔵される | **Rejected**: 手順の存在をエージェントが忘れると使われない |
| procedures/をskills/に統合 | ディレクトリ構造の簡素化 | 手順書とスキルの性質が異なる。手順書はより長く詳細 | **Rejected**: 概念的な分離を維持する方が管理しやすい |
| **procedures/にskills/同等の基盤を導入 (Adopted)** | 自動想起可能、メタデータ管理 | 既存procedures/のマイグレーション必要 | **Adopted**: 手続き記憶ライフサイクル管理の前提条件 |

### Key Decisions from Discussion

1. **frontmatterフィールド**: description, tags, success_count, failure_count, last_used, confidence, version, created_at — フレームワークが管理。LLMは本文のみ生成
2. **3-tierマッチング**: skills/と同じ`match_skills_by_description()`ロジックを共有。builder.pyでprocedures/もマッチング対象に追加
3. **RAGインデックス自動更新**: `_handle_write_memory_file()` でパスが`skills/`, `procedures/`, `common_skills/`配下の場合にインデックス更新を呼ぶ
4. **成功/失敗追跡の2経路**:
   - フレームワーク自動追跡: builder.pyでスキル/手順注入時に記録 → セッション境界時の差分要約で成否判定 → メタデータ更新
   - エージェント明示報告: `report_procedure_outcome(path, success, notes)` ツール。明示報告を優先
5. **既存proceduresのマイグレーション**: frontmatter未付与ファイルを検出し、ファイル名から推測したdescriptionで自動付与。confidence=0.5

### Changes by Module

| Module | Change Type | Description |
|--------|------------|-------------|
| `core/memory/manager.py` | Modify | procedures frontmatter読み書きメソッド追加、`match_procedures_by_description()`追加（既存`match_skills_by_description()`を拡張して共通化） |
| `core/prompt/builder.py` | Modify | procedures/の3-tierマッチング注入追加。注入したスキル/手順のパスを`_injected_procedures`として記録 |
| `core/tooling/handler.py` | Modify | write_memory_file後のRAGインデックス更新追加、`_handle_report_procedure_outcome()`追加、proceduresのソフトバリデーション追加 |
| `core/tooling/schemas.py` | Modify | `report_procedure_outcome`ツールスキーマ追加 |
| `core/memory/rag/indexer.py` | Modify | procedures/のfrontmatterストリップ処理追加 |
| `core/prompt/builder.py`, `core/skills/index.py` | Modify | 現行の SkillIndex / Skill Router / catalog 経路で procedures/ も提示 |
| `core/memory/conversation.py` | Modify | セッション境界時に注入された手順の成否を追跡・記録 |

#### Change 1: procedures frontmatter

```yaml
---
description: "Chatworkモニタリングの手順。新着メッセージの確認と振り分け"
tags: [chatwork, monitoring]
success_count: 0
failure_count: 0
last_used: null
confidence: 0.5
version: 1
created_at: "2026-02-18T00:00:00"
---

# 手順書: Chatworkモニタリングフロー

手順の本文...
```

#### Change 2: 3-tierマッチング共通化

**Target**: `core/memory/manager.py`

```python
# Before: skills専用
def match_skills_by_description(message, skill_metas, ...) -> list[SkillMeta]:

# After: skills + proceduresの統合マッチング
def match_memory_by_description(
    message: str,
    metas: list[SkillMeta],  # SkillMetaをProcedureMetaと共通化
    retriever: RAGRetriever | None = None,
    anima_name: str = "",
) -> list[SkillMeta]:
    # 既存の3-tierロジックをそのまま適用
```

#### Change 3: write_memory_file後のRAGインデックス更新

**Target**: `core/tooling/handler.py:425-486`

```python
# After file write (追加):
if rel.startswith(("skills/", "procedures/", "common_skills/")) and rel.endswith(".md"):
    indexer = self._memory._get_indexer()
    if indexer:
        await indexer.index_file(full_path, force=True)
```

#### Change 4: report_procedure_outcome ツール

**Target**: `core/tooling/schemas.py` + `core/tooling/handler.py`

```python
# schemas.py
{
    "name": "report_procedure_outcome",
    "description": "Report the outcome of following a procedure or skill.",
    "parameters": {
        "path": {"type": "string", "description": "procedures/xxx.md or skills/xxx.md"},
        "success": {"type": "boolean"},
        "notes": {"type": "string", "description": "Optional notes on what worked/failed"},
    },
}

# handler.py
async def _handle_report_procedure_outcome(self, args):
    path = self._anima_dir / args["path"]
    meta = self._memory.read_procedure_metadata(path)
    if args["success"]:
        meta["success_count"] = meta.get("success_count", 0) + 1
    else:
        meta["failure_count"] = meta.get("failure_count", 0) + 1
    meta["last_used"] = datetime.now(UTC).isoformat()
    # confidence再計算
    total = meta["success_count"] + meta["failure_count"]
    if total > 0:
        meta["confidence"] = round(meta["success_count"] / total, 2)
    self._memory.write_procedure_with_meta(path, self._memory.read_procedure_content(path), meta)
```

### Edge Cases

| Case | Handling |
|------|----------|
| 既存proceduresにfrontmatterがない | マイグレーション: ファイル名からdescription推測、confidence=0.5、version=1で付与 |
| write_memory_fileでproceduresを作成するがfrontmatterなし | ソフトバリデーション（警告）+ 次回のマイグレーションで補完 |
| procedures/とskills/に同名ファイルがある場合 | それぞれ独立にマッチング。注入時はskills/を優先（既存動作維持） |
| report_procedure_outcome で存在しないパスが指定される | エラーメッセージを返す |
| procedures/のfrontmatterが壊れている場合 | `read_procedure_metadata()` が空dictを返し、デフォルト値でフォールバック |
| RAGインデクサーが利用不可の場合 | インデックス更新をスキップし、warningログ |

## Implementation Plan

### Phase 1: frontmatter基盤

| # | Task | Target |
|---|------|--------|
| 1-1 | procedures用frontmatter読み書きメソッド追加（`write_procedure_with_meta`, `read_procedure_content`, `read_procedure_metadata`） | `core/memory/manager.py` |
| 1-2 | procedures用ソフトバリデーション追加（description必須チェック） | `core/tooling/handler.py` |
| 1-3 | procedures/のfrontmatterストリップ処理追加（チャンキング時） | `core/memory/rag/indexer.py` |
| 1-4 | 既存proceduresのマイグレーション実装 | `core/memory/manager.py` |
| 1-5 | Phase 1のユニットテスト | `tests/` |

**Completion condition**: proceduresファイルのfrontmatter読み書きが動作し、既存ファイルが自動マイグレーションされる

### Phase 2: 3-tierマッチング + 自動注入

| # | Task | Target |
|---|------|--------|
| 2-1 | `match_skills_by_description()` をprocedures対応に拡張（共通化） | `core/memory/manager.py` |
| 2-2 | builder.pyでprocedures/もマッチング対象に追加、注入追跡記録 | `core/prompt/builder.py` |
| 2-3 | 現行の SkillIndex / Skill Router / catalog 経路で procedures/ を提示 | `core/prompt/builder.py`, `core/skills/index.py` |
| 2-4 | Phase 2のユニットテスト | `tests/` |

**Completion condition**: procedures/がメッセージに基づいて自動マッチングされ、システムプロンプトに注入される

### Phase 3: RAGインデックス更新 + 成功/失敗追跡

| # | Task | Target |
|---|------|--------|
| 3-1 | write_memory_file後のRAGインデックス自動更新追加 | `core/tooling/handler.py` |
| 3-2 | `report_procedure_outcome` ツールスキーマ + ハンドラ実装 | `core/tooling/schemas.py`, `core/tooling/handler.py` |
| 3-3 | builder.pyの注入追跡 → セッション境界時の自動成否判定 | `core/prompt/builder.py`, `core/memory/conversation.py` |
| 3-4 | Phase 3の統合テスト | `tests/` |

**Completion condition**: write_memory_file後にRAGが更新され、手順の成功/失敗が2経路で追跡される

## Scope

### In Scope

- procedures/へのYAML frontmatter導入
- procedures/の3-tierマッチング + システムプロンプト自動注入
- write_memory_file後のRAGインデックス自動更新（procedures/, skills/, common_skills/）
- `report_procedure_outcome` ツール追加
- フレームワーク自動成否追跡（注入記録 + セッション境界判定）
- 既存proceduresのマイグレーション

### Out of Scope

- エピソード→手続きの自動蒸留 — 理由: 別Issue（`20260218_procedural-memory-auto-distillation.md`）で対応
- 予測誤差ベースの再固定化 — 理由: 別Issue（`20260218_procedural-memory-reconsolidation.md`）で対応
- 効用ベースの忘却 — 理由: 別Issue（`20260218_procedural-memory-utility-forgetting.md`）で対応
- skills/のfrontmatter変更 — 理由: skills/には既にfrontmatterがあり変更不要。成功/失敗フィールドの追加のみ

## Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| proceduresのマッチングノイズ（不要な手順の過剰注入） | プロンプトの肥大化 | SKILL_INJECTION_BUDGETの既存制約で制御。procedures用の別バジェット枠を設ける |
| 自動成否判定の精度が低い | 誤った成功/失敗カウント | エージェント明示報告を優先。自動追跡は補助的な位置づけ |
| 既存proceduresのマイグレーションで内容が壊れる | 手順が使えなくなる | archive/に退避後にマイグレーション。後方互換の読み出し |

## Acceptance Criteria

- [ ] proceduresファイルにYAML frontmatter（description, tags, success_count, failure_count, confidence, version, created_at）が付与されている
- [ ] proceduresがdescriptionベース3-tierマッチングでシステムプロンプトに自動注入される
- [ ] write_memory_fileでskills/procedures/を更新した後にRAGインデックスが自動更新される
- [ ] `report_procedure_outcome` ツールで成功/失敗を報告でき、メタデータが更新される
- [ ] フレームワークの自動成否追跡が動作する（注入記録→セッション境界判定）
- [ ] 既存proceduresが初回実行時に自動マイグレーションされる
- [ ] frontmatter未付与ファイルの読み出しが後方互換で動作する
- [ ] テストカバレッジ80%以上

## References

- `core/memory/manager.py:102-164` — `match_skills_by_description()` 既存3-tierマッチング
- `core/memory/manager.py:513-514` — `list_procedure_files()` 現行のprocedures列挙
- `core/memory/manager.py:520-567` — `_extract_skill_meta()` frontmatter解析
- `core/prompt/builder.py:434-458` — スキル全文注入ロジック
- `core/prompt/builder.py:25-30` — SKILL_INJECTION_BUDGET
- `core/tooling/handler.py:425-486` — `_handle_write_memory_file()` RAGインデックス更新なし
- `core/tooling/handler.py:110-168` — `_validate_skill_format()` skillsバリデーション
- `core/memory/rag/indexer.py:232-253` — procedures/のチャンキング（whole_file方式）
- `core/prompt/builder.py` — Skill catalog / active skill context / procedure pointers
- `core/memory/manager.py` — `search_procedures()` and procedure metadata facade
- [VOYAGER](https://arxiv.org/abs/2305.16291) — スキルライブラリの検証パイプライン
- [ExpeL](https://arxiv.org/abs/2308.10144) — 洞察の投票システム（UPVOTE/DOWNVOTE）
