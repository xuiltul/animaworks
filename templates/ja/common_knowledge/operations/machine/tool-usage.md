# machine ツール利用ガイド — 共通原則

## 概要

`animaworks-tool machine run` は外部エージェント（cursor-agent / claude / codex / gemini）に
コード変更・調査・分析・レビュー・テスト等の重い作業を委託するツールである。

machine は AnimaWorks インフラにアクセスできない独立した実行環境で動作する。
記憶・メッセージ・組織情報は利用できないため、必要な情報はすべて計画書に含めること。

## メタパターン — 全ロール共通

全フェーズ・全ロールで以下の4ステップを徹底する:

```
① Anima が指示書（計画書）を作成する
② machine に計画書を渡して実行させる
③ machine の出力はドラフトとして扱う
④ Anima が出力を検証し、承認 or 修正する
```

**原則:**
- 最初の指示書（machine に何をさせるかの計画）は **Anima 自身が書く**（MUST）
- 中間成果物の具体化（plan → impl.plan 等）は machine に投げてよい
- machine 出力は常にドラフト扱い — **Anima の承認なしに次工程に渡してはならない**（NEVER）
- 問題があれば計画書を修正して machine に再委託するか、Anima 自身が修正する

## ステータス管理

計画書・成果物の状態はドキュメント自身にメタデータとして持たせる。
フレームワークはステータスを自動管理しない — Anima が自律的に遷移を判断する。

各ドキュメントの冒頭に以下のメタデータブロックを配置する:

```markdown
status: draft | reviewed | approved
author: {anima名}
date: {YYYY-MM-DD}
type: investigation | plan | impl-plan | review | test-plan | test-report
```

遷移の判断基準:
- `draft` → `reviewed`: Anima が出力を読み、内容を確認した
- `reviewed` → `approved`: Anima が品質を承認し、次工程に渡して良いと判断した
- 承認は上位ロール（上司）が行う場合もある — 組織の運用ルールに従うこと

## 計画書の保存場所

**全ての計画書は `state/plans/` に保存する。** `/tmp/` への保存は禁止（再起動で消失するため）。

ファイル命名規則: `{YYYY-MM-DD}_{タスク概要}.{type}.md`

| type | 用途 | 例 |
|------|------|-----|
| `investigation` | 調査報告書 | `2026-03-27_login-bug.investigation.md` |
| `plan` | 実装計画書 | `2026-03-27_fix-email-validation.plan.md` |
| `impl-plan` | 実装詳細計画書 | `2026-03-27_fix-email-validation.impl-plan.md` |
| `review` | レビュー報告書 | `2026-03-27_review-PR-42.review.md` |
| `test-plan` | テスト計画書 | `2026-03-27_e2e-login-flow.test-plan.md` |
| `test-report` | テスト結果報告書 | `2026-03-27_e2e-login-flow.test-report.md` |

`state/plans/` を使う理由:
- **永続性**: OS 再起動で消えない
- **追跡可能**: 後から「何を指示したか」を検証できる
- **上司が閲覧可能**: スーパーバイザーが `read_memory_file` で部下の計画書を確認できる

```
read_memory_file(path="../{部下名}/state/plans/2026-03-27_fix-email-validation.plan.md")
```

## machine 実行コマンド

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{計画書ファイル名})" \
  -d /path/to/worktree
```

エンジンを明示する場合:

```bash
animaworks-tool machine run -e cursor-agent \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{計画書ファイル名})" \
  -d /path/to/worktree
```

バックグラウンド実行（重い作業向け）:

```bash
animaworks-tool machine run --background \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{計画書ファイル名})" \
  -d /path/to/worktree
```

## レート制限

- セッション（chat）: 5 回 / セッション
- Heartbeat: 2 回 / heartbeat
- `--background` の結果は `state/background_notifications/` で確認

## 禁止事項

- **計画書なしの `machine run` 実行は禁止**（インラインの短い指示文字列のみでの実行は不可）
- 計画書に「ゴール」と「完了条件」の両方が欠けている状態での実行は禁止
- **machine の出力を検証せずに次工程に渡すことは禁止**（NEVER）
- machine の出力を検証せずにコミット・プッシュすることは禁止

## machine の制約

- machine は AnimaWorks インフラにアクセスできない（記憶・メッセージ・組織情報なし）
- GitHub API 操作（diff 取得・コメント投稿等）は Anima 側で行い、結果を計画書に含める
- 長時間かかる場合は `--background` を使用する

## チーム設計テンプレートを使用している場合

`injection.md` で `team-design/development/{role}/machine.md` が指定されている場合、
**そちらが本ファイルに優先する**。各ロールの machine.md は基本ルール・プロンプトの書き方を含め自己完結している。

本ファイルは team-design テンプレートを使用しない単独 Anima 向けの共通ガイドである。
