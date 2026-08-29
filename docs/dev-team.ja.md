# 開発チーム テンプレート

**[English version](dev-team.md)**

AnimaWorks には、GitHub のプルリクエストワークフローを支える社内開発チーム向けの
anima テンプレートが同梱されています。既存の `animaworks create <名前> --template <テンプレート名>`
機構からそのまま注入できるため、フレームワーク本体のコード変更は不要です。

> これらのテンプレートは**出発点**です。記憶や知識、手順は実運用の中で育つものです。
> 誰かの運用履歴の写しではなく、土台として扱ってください。

## 推奨するチーム構成

現実的な既定は **lead 1 + engineer 2〜3 + researcher 1** です。

| テンプレート | role | 役割 |
|--------------|------|------|
| `dev-lead` | manager | プロダクトマネージャー。engineer / researcher へ委任し、Open PR を巡回し、品質ゲートとエスカレーションを担う |
| `dev-engineer` | engineer | worktree での隔離実装、PR 作成、自分の PR の CI 監視と自律修正、完了報告 |
| `dev-researcher` | researcher | 読み取り専用の調査と、根拠のあるレポート（結論 → 根拠 → 未確認） |

全ロールは**任意**です。blank から手作業で育てたい場合は、従来どおり既定の `_blank`
テンプレートから始めて役割を段階的に育てる選択肢が残されています。

## チームの作成

```bash
animaworks create alice --template dev-lead
animaworks create bob   --template dev-engineer
animaworks create carol --template dev-engineer
animaworks create dave  --template dev-researcher
```

各コマンドは `identity.md` / `heartbeat.md` / `cron.md` / `injection.md` /
`permissions.json` / `status.json` を持つ anima を作成します。個体名は作成時に与えられます。
テンプレート自体はロールベースかつエンジン非依存です（モデル名・エンジン名はハードコードしません）。

## GitHub パイプラインとの接続

開発チームの anima はプルリクエストのパイプラインの前面に置くことを前提としています。主な接続ポイント:

- **Webhook ゲートウェイ** — 受信した GitHub イベント（PR オープン、レビューコメント、CI 結果）は
  webhook エンドポイントから流入し、チームのタスクになります。エンドポイントの詳細は
  [API リファレンス](api-reference.ja.md) を参照してください。
- **`gh-ci` / `gh-review` / `gh-comment` タスク** — ゲートウェイが受信イベントをこれらのタスク型として
  起票し、PR 単位で直列化します（同じ PR を複数の anima が同時に触らない）。lead の cron 巡回は、
  イベント経路から漏れたものを拾う安全網です。

パイプライン自体の詳細は既存のリファレンスドキュメントに委ね、ここでは重複させません。

## 補足

- テンプレートが定義するのは規律（委任ファースト、worktree 分離、読み取り専用調査）であり、
  ドメイン知識ではありません。チームごとに学ぶ内容は、運用の中で育つ各チームの `knowledge/` と
  `episodes/` に蓄積されます。
- `read_memory_file(...)` で参照する役割ガイドラインは、テンプレートに同梱の
  `common_knowledge/` を指しています。
