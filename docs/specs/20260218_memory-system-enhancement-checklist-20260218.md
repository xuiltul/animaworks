# 記憶システム強化 統合チェックリスト — 固定化品質保証・矛盾解決・手続き記憶ライフサイクル

## Overview

memory.md レビューで特定された6つの改善Issueの統合実装チェックリスト。依存関係に基づく実装順序を定義し、各Issueのフェーズ単位で進捗を追跡する。

## 背景

記憶システム設計仕様書（`docs/memory.md`）のレビューで以下の課題が特定された:

1. **LLM固定化の品質保証の欠如** — ハルシネーションが knowledge/ に永続化するリスク
2. **矛盾する記憶の解決メカニズムの不在** — 古い知識と新しい知識の矛盾が放置される
3. **手続き記憶のライフサイクル管理の不在** — procedures/ に自動化された作成・更新・忘却がない

これらを6つのIssueに分解し、依存関係を考慮した実装順序で解決する。

## 依存関係

```
Wave 1（並列可能）
├── Issue 1: 固定化バリデーション
└── Issue 3: 手続き記憶基盤整備

Wave 2（Wave 1完了後、内部は並列可能）
├── Issue 2: 矛盾検出・解決        ← Issue 1
├── Issue 4: 自動蒸留              ← Issue 3
└── Issue 5: 再固定化              ← Issue 3

Wave 3
└── Issue 6: 効用ベース忘却        ← Issue 3 + Issue 5
```

## 2026-06-11 現行ステータス参照

このチェックリストは当初の 2026-02-18 設計メモを、2026-06 時点の実装に合わせて注釈したもの。主な反映元:

| Ref | 内容 |
|-----|------|
| #207 / `1917fe14` | knowledge self-correction post-processing を production path に接続 |
| #208 / `48ad24fa` | 日次 consolidation の episode note 保全と日付ずれ修正 |
| #209 / `0b5cdb02` | neurogenesis merge の whole-file overwrite 防止 |
| #212 / `56b58781` | consolidation timeout 後も post-processing / RAG rebuild を保証 |
| #215 / `11bcbd82` | 月次忘却・retention・bad-date cleanup を有効化 |
| #216 / `5c27b587` | 長期記憶 BM25 と access-count feedback loop 修正 |
| #217 | 本ファイル、memory docs、distributed templates の現行化 |

## 共有リソース

以下のコンポーネントは複数Issueで変更される。実装順序を守ることでコンフリクトを回避する。

| ファイル | 変更するIssue | 注意点 |
|---------|--------------|--------|
| `core/memory/consolidation.py` | 1, 2, 4, 5 | 日次固定化フローへのステージ追加順序を厳守 |
| `core/memory/manager.py` | 1, 3, 5 | frontmatter読み書きはIssue 1で基盤構築、Issue 3で拡張 |
| `core/memory/forgetting.py` | 6 | Issue 6のみ。他Issueとの競合なし |
| `core/memory/validation.py` | 1, 2 | Issue 1で新設、Issue 2で拡張 |
| `core/memory/rag/indexer.py` | 1, 2, 3 | frontmatterストリップ + メタデータ連携 |
| `core/memory/rag/retriever.py` | 2 | supersededフィルタ追加 |
| `core/prompt/builder.py` | 3 | procedures注入追加 |
| `core/tooling/handler.py` | 3 | RAGインデックス更新 + report_procedure_outcome |
| NLIモデル（mDeBERTa） | 1, 2 | Issue 1でロード基盤構築、Issue 2で共有 |

---

## Wave 1: 基盤構築（並列実装可能）

### Issue 1: 記憶固定化バリデーションパイプライン

> `docs/issues/20260218_consolidation-validation-pipeline.md`

- [x] **Phase 1: YAMLフロントマター基盤 + マイグレーション** — 済: `core/memory/frontmatter.py`, `core/memory/manager.py`, `core/memory/consolidation.py`, `core/memory/rag/indexer.py`
  - [x] 1-1: manager.py にフロントマター読み書きメソッド追加 — `write_knowledge_with_meta`, `read_knowledge_content`, `read_knowledge_metadata`
  - [x] 1-2: indexer.py にフロントマターストリップ処理追加 — `_strip_frontmatter`, `_parse_frontmatter`
  - [x] 1-3: レガシーknowledgeファイルの自動マイグレーション実装 — `_migrate_legacy_knowledge`
  - [x] 1-4: ユニットテスト（フロントマター読み書き、マイグレーション、後方互換性） — `tests/unit/core/memory/test_knowledge_frontmatter.py`
- [x] **Phase 2: バリデーションモジュール** — 済: `core/memory/validation.py`
  - [x] 2-1: KnowledgeValidator クラス実装（NLIモデルロード・推論） — `KnowledgeValidator`
  - [x] 2-2: NLIフォールバック（GPU→CPU、ロード失敗→LLMのみ） — `_load_nli_model`
  - [x] 2-3: LLMセルフレビューメソッド実装 — `_llm_review`
  - [x] 2-4: ユニットテスト（NLI判定、LLMレビュー、フォールバック） — `tests/unit/core/memory/test_knowledge_validation.py`
- [x] **Phase 3: パイプライン統合 + 既存問題修正** — 済/更新: 旧 `daily_consolidate()` 実装は Anima 主導の `run_consolidation()` と `core/lifecycle/knowledge_correction.py` に移行
  - [x] 3-1: 固定化後のバリデーション/自己修正ステージ組み込み — 2026-06-10 #207 で production path に配線
  - [x] 3-2: 既存knowledgeコンテキスト確認 — Anima 主導の統合指示と knowledge self-correction 経路で継続
  - [x] 3-3: コードフェンスサニタイズ — `ConsolidationEngine._sanitize_llm_output`
  - [x] 3-4: フロントマター付き書き込み — `FrontmatterService.write_knowledge_with_meta`
  - [x] 3-5: フォーマット検証リトライ — Anima tool-loop と自己修正経路で扱う
  - [x] 3-6: 週次統合のアーカイブ方式変更 — 旧週次統合は削除済み、矛盾/再固定化アーカイブは `archive/superseded` / `archive/versions` に集約
  - [x] 3-7: 統合テスト（日次固定化E2E、週次統合E2E） — 旧メソッド削除は `tests/unit/core/memory/test_consolidation_refactored.py`、現行経路は lifecycle / memory tests

### Issue 3: 手続き記憶基盤整備

> `docs/issues/20260218_procedural-memory-foundation.md`

- [x] **Phase 1: frontmatter基盤** — 済: `core/memory/frontmatter.py`
  - [x] 1-1: procedures用frontmatter読み書きメソッド追加 — `write_procedure_with_meta`, `read_procedure_content`, `read_procedure_metadata`
  - [x] 1-2: procedures用ソフトバリデーション追加 — `core/tooling/handler_base.py`
  - [x] 1-3: indexer.py のfrontmatterストリップ処理追加 — `MemoryIndexer._strip_frontmatter`
  - [x] 1-4: 既存proceduresのマイグレーション実装 — `ensure_procedure_frontmatter`
  - [x] 1-5: ユニットテスト — `tests/unit/core/memory/test_procedure_frontmatter.py`
- [x] **Phase 2: 3-tierマッチング + 自動注入** — 更新: main priming はスキル/手順本文を直接注入しない設計へ変更
  - [x] 2-1: `match_skills_by_description()` をprocedures対応に拡張 — 互換 matcher と `SkillMetadataService` は維持
  - [x] 2-2: builder.py でprocedures/ 参照を提示 — active skill context / Skill Router / `read_memory_file` 経路へ置換
  - [x] 2-3: 旧プライミング手続き注入 — 廃止済み。手順本文は意図的想起で読む
  - [x] 2-4: ユニットテスト — `tests/unit/core/memory/test_procedure_matching.py`, `tests/unit/core/test_role_templates.py`
- [x] **Phase 3: RAGインデックス更新 + 成功/失敗追跡** — 済
  - [x] 3-1: write_memory_file 後のRAGインデックス自動更新 — `tests/unit/core/memory/test_procedure_integration.py`
  - [x] 3-2: report_procedure_outcome ツール実装 — `core/tooling/handler_skills.py`, `core/tooling/prompt_db.py`
  - [x] 3-3: フレームワーク自動成否追跡（注入記録→セッション境界判定） — completion gate / outcome tracking 経路で実装
  - [x] 3-4: 統合テスト — `tests/unit/core/memory/test_procedure_integration.py`

**Wave 1 完了条件**: Issue 1, 3 の全フェーズが完了し、テストが通ること

---

## Wave 2: 検出・蒸留・更新（Wave 1完了後、内部は並列可能）

### Issue 2: 知識矛盾検出・解決メカニズム

> `docs/issues/20260218_knowledge-contradiction-detection-resolution.md`
> 前提: Issue 1 完了

- [x] **Phase 1: 矛盾検出ロジック** — 済: `core/memory/contradiction.py`
  - [x] 1-1: ContradictionDetector クラス実装（NLIペア判定） — `ContradictionDetector`
  - [x] 1-2: LLM矛盾解決プロンプト（supersede/merge/coexist判定） — `_resolve_contradictions`
  - [x] 1-3: ユニットテスト — `tests/unit/core/memory/test_contradiction.py`
- [x] **Phase 2: 解決実行ロジック** — 済
  - [x] 2-1: supersede_knowledge 相当処理 — `_apply_supersede`
  - [x] 2-2: merge実行ロジック — `_apply_merge`
  - [x] 2-3: coexist実行ロジック — `_apply_coexist`
  - [x] 2-4: retriever.py にsupersededフィルタ追加 — `valid_until` filter
  - [x] 2-5: indexer.py にsupersedes関連メタデータ連携 — `valid_until`, `superseded_at` migration
  - [x] 2-6: ユニットテスト — `tests/unit/core/memory/test_contradiction.py`
- [x] **Phase 3: パイプライン統合 + 初回スキャン** — 済/更新: #207 で nightly knowledge self-correction path に統合
  - [x] 3-1: 固定化後の矛盾検出・解決ステージ組み込み — `core/lifecycle/knowledge_correction.py`
  - [x] 3-2: 初回/継続スキャン — `run_post_consolidation_knowledge_correction`
  - [x] 3-3: activity_log への矛盾解決イベント記録 — `ContradictionDetector`
  - [x] 3-4: 統合テスト — `tests/unit/core/lifecycle/test_knowledge_correction.py`

### Issue 4: エピソード→手続きの自動蒸留

> `docs/issues/20260218_procedural-memory-auto-distillation.md`
> 前提: Issue 3 完了

- [x] **Phase 1: 日次固定化の振り分け** — 済/更新: LLM tool-loop と `ProceduralDistiller.classify_and_distill` で実装
  - [x] 1-1: プロンプト拡張（knowledge/procedures振り分け） — `core/memory/distillation.py`
  - [x] 1-2: procedures 書き込み実装 — `ProceduralDistiller.save_procedure`
  - [x] 1-3: 重複チェック（RAG類似度検索）実装 — `_check_rag_duplicate`
  - [x] 1-4: ユニットテスト — `tests/unit/core/memory/test_distillation.py`
- [x] **Phase 2: 週次パターン検出・蒸留** — 済/更新
  - [x] 2-1: activity_log からのタスクパターンクラスタリング — `_cluster_activities`
  - [x] 2-2: LLMによる共通手順蒸留プロンプト — `weekly_pattern_distill`
  - [x] 2-3: 週次統合への組み込み — 旧 `weekly_integrate()` は削除済み。現行では batch / knowledge-correction path から利用
  - [x] 2-4: 統合テスト — `tests/unit/core/memory/test_distillation.py`

### Issue 5: 予測誤差ベースの手続き再固定化

> `docs/issues/20260218_procedural-memory-reconsolidation.md`
> 前提: Issue 3 完了

- [x] **Phase 1: 再固定化トリガー + LLM Reflection** — 済: `core/memory/reconsolidation.py`
  - [x] 1-1: 再固定化対象検出メソッド実装 — `find_reconsolidation_targets`
  - [x] 1-2: LLM Reflectionプロンプト + レスポンスパーサー — `_revise_procedure`
  - [x] 1-3: ユニットテスト — `tests/unit/core/memory/test_reconsolidation.py`
- [x] **Phase 2: バージョン管理 + パイプライン統合** — 済/更新
  - [x] 2-1: バージョン退避/更新実装 — `_archive_version`, `apply_reconsolidation`
  - [x] 2-2: 自動再固定化ステージ組み込み — `core/lifecycle/knowledge_correction.py`
  - [x] 2-3: activity_log への再固定化イベント記録 — `procedure_reconsolidated`
  - [x] 2-4: 統合テスト — `tests/unit/core/lifecycle/test_knowledge_correction.py`, `tests/unit/core/memory/test_reconsolidation.py`

**Wave 2 完了条件**: Issue 2, 4, 5 の全フェーズが完了し、テストが通ること

---

## Wave 3: ライフサイクル完成

### Issue 6: 効用ベースの手続き記憶忘却

> `docs/issues/20260218_procedural-memory-utility-forgetting.md`
> 前提: Issue 3 + Issue 5 完了

- [x] **Phase 1: PROTECTED解除 + 閾値設定** — 済: `core/memory/forgetting.py`
  - [x] 1-1: PROTECTED_MEMORY_TYPES から procedures 除外 — `PROTECTED_MEMORY_TYPES = {"skills", "shared_users"}`
  - [x] 1-2: _is_protected_procedure() 実装
  - [x] 1-3: _should_downscale_procedure() 実装
  - [x] 1-4: ユニットテスト — `tests/unit/core/memory/test_forgetting_procedures.py`
- [x] **Phase 2: 統合 + クリーンアップ** — 済
  - [x] 2-1: Stage 1-3 にprocedures専用パス追加 — downscaling/reorganization/complete forgetting scans include procedures
  - [x] 2-2: archive/procedure_versions/ のクリーンアップ — `cleanup_procedure_archives`
  - [x] 2-3: 統合テスト（日次→週次→月次の全ステージ） — `tests/unit/core/memory/test_forgetting.py`, `tests/unit/core/memory/test_forgetting_procedures.py`

**Wave 3 完了条件**: Issue 6 の全フェーズが完了し、テストが通ること

---

## 最終検証

全Wave完了後の統合検証:

- [x] 日次固定化の現行パイプライン検証 — 旧 `daily_consolidate()` は削除済み。Anima 主導 consolidation と post-consolidation correction を単体/統合テストで検証
- [x] 週次統合の現行検証 — 旧 `weekly_integrate()` は削除済み。蒸留/矛盾解決/merge candidates は個別テストで検証
- [x] 月次忘却のフルパイプライン検証 — `monthly_forget`, procedure archive cleanup, episode retention tests
- [x] docs/memory.md の更新（新機能の反映） — 現行 priming / forgetting / skill-router 記述へ更新
- [ ] 実Anima環境での動作確認（最低1サイクルの日次→週次→月次を通す） — ローカル自動テスト外。運用確認項目として残す

## 現行固定化パイプライン（最終形）

```
run_consolidation()
│
├── [現行] 旧形式メモリの frontmatter / procedure metadata 互換処理
├── [現行] episode / activity / tool-result 文脈を Anima tool loop に提示
├── [現行] Anima が `consolidation_instruction` に従って knowledge/procedures を読む・書く
├── [現行] post-consolidation correction (#207): validation / contradiction / reconsolidation
├── [現行] timeout-safe post-processing (#212): RAG rebuild / repair / bookkeeping
├── [現行] forgetting (#215): monthly_forget / procedure archive cleanup / bad-date cleanup
└── [廃止] 旧 `daily_consolidate()` / `weekly_integrate()` entrypoint は現行コードに存在しない
```

## Issue一覧

| # | Issue | ファイル | Wave |
|---|-------|---------|------|
| 1 | 固定化バリデーションパイプライン | `20260218_consolidation-validation-pipeline.md` | 1 |
| 2 | 知識矛盾検出・解決メカニズム | `20260218_knowledge-contradiction-detection-resolution.md` | 2 |
| 3 | 手続き記憶基盤整備 | `20260218_procedural-memory-foundation.md` | 1 |
| 4 | エピソード→手続き自動蒸留 | `20260218_procedural-memory-auto-distillation.md` | 2 |
| 5 | 予測誤差ベースの再固定化 | `20260218_procedural-memory-reconsolidation.md` | 2 |
| 6 | 効用ベースの忘却 | `20260218_procedural-memory-utility-forgetting.md` | 3 |
