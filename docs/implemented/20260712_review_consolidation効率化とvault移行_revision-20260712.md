# Code Review: consolidation効率化とvault移行 - Revision Required

**Review Date**: 2026-07-12
**Original Plan**: `docs/plans/20260712_consolidation効率化とvault移行.md`
**Worktree**: `/home/main/dev/animaworks-bak-issue-20260712-154221`
**Status**: ❌ REVISION REQUIRED

## Summary

Iteration 1は指定単体テストと関連E2E smoke testを通過したが、独立レビューでデータ保全・credential更新・LLM応答検証の修正必須事項が見つかった。Iteration 2で下記を修正し、再レビューする。

## Priority Changes

### Critical

1. 一時的なvector service全体障害でも任意のmemory sourceを物理移動し得る。原本を移動しないskip-list方式に変更し、基盤障害による大量隔離を防ぐ。

### High

1. `save_config()` が既存vault参照を無条件復元するため、設定APIからcredentialを更新・解除できない。秘密値の平文再保存防止と明示更新を両立する。
2. バッチLLM応答の型検証が不十分で、文字列 `"false"` を真と解釈し得る。必須キー・型・resolution enumを厳密検証し、不正時は単一ペア判定へフォールバックする。

### Medium

1. Upsert成功後のstale chunk削除に失敗すると、Upsert失敗カウンターがリセットされない。Upsert成功直後にリセットする。
2. `anima_defaults.consolidation_enabled` が実際のskip判定に反映されない。status.json overrideを含む解決済み設定を判定に使う。

## Automated Checks

- 計画書指定選択テスト: 537 passed
- 関連E2E smoke + migration apply: 13 passed
- Ruff/import/diff check: passed
- Cursor Agent review: failed（launcher outputが空）
- Codex independent review: completed、上記5件を報告
- coverage helper: pytest-covが環境に未導入のため測定不能
- file-size helper: リポジトリ既存の多数ファイルが500行基準超過。今回の新規実装ファイルには新たな500行超過なし

## Next Steps

1. Critical/High/Medium項目をすべて修正する。
2. 対象テスト、計画書指定テスト、関連E2E smokeを再実行する。
3. 独立レビューを再実施し、APPROVEDを確認する。
