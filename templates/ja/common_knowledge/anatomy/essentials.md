# AnimaWorks エッセンシャルガイド

[IMPORTANT] AnimaWorks の全体像を1枚で把握するための統合ガイド。
Heartbeat / Cron / チーム設計 / 記憶 / コスト最適化の要点を網羅。
初めて読む場合や、概念の関係を整理したいときに最初に参照すること。
各トピックの詳細は末尾のリンク先を参照。

---

## AnimaWorks とは

AIエージェントを「ツール」ではなく **自律的な人格** として運用するフレームワーク。

- 各 Anima は固有の人格・記憶・判断基準を持つ
- 人間の指示がなくても定期的に自ら行動する（Heartbeat / Cron）
- 組織の中で役割を担い、他の Anima や人間と協働する
- 経験から学び、記憶を蓄積し、成長する

---

## 5つの実行パス — 「いつ・どう動くか」

Anima は以下の5つのパスで稼働する。Chat 以外はすべて自動で起動する。

| パス | いつ動く | 何をする | 誰が使う |
|------|---------|---------|---------|
| **Chat** | 人間がメッセージを送ったとき | 対話応答 | 人間 → Anima |
| **Inbox** | 他 Anima から DM が来たとき | 組織内メッセージへの即時応答 | Anima → Anima |
| **Heartbeat** | 定期自動起動（デフォルト30分） | 観察 → 計画 → 振り返り。**実行はしない** | 自動 |
| **Cron** | cron.md のスケジュール（例: 毎朝9:00） | 決まった時間の確定タスク実行 | 自動 |
| **TaskExec** | 正規タスクが実行可能で依存先が完了したとき | 保存済み入力を取得済みのLLM試行で実行 | submit_tasks または delegate_task で登録 |

Chat と Heartbeat は**別ロック**で動くため、巡回中でも人間の会話に即座に応答できる。

→ 詳細: `anatomy/what-is-anima.md`

---

## Heartbeat vs Cron — 2つの自律行動

どちらも「人間の指示なしに動く」仕組みだが、目的が根本的に異なる。

| 観点 | Heartbeat（定期巡回） | Cron（定時タスク） |
|------|---------------------|------------------|
| **たとえ** | 定期的にオフィスを見回る警備員 | 毎朝9時に届く新聞配達 |
| **目的** | 状況確認・計画立案・振り返り | 決められた業務の実行 |
| **実行するか** | **しない**。タスクを発見したら `submit_tasks` か `delegate_task` で投入するだけ | **する**。LLM型なら思考と実行、Command型なら即座に実行 |
| **間隔** | 固定間隔（デフォルト30分、Activity Level で変動） | cron式で柔軟に指定（毎日9:00、毎週金曜17:00 等） |
| **設定ファイル** | `heartbeat.md`（チェックリスト） | `cron.md`（タスク定義） |
| **典型例** | 未読メッセージ確認、ブロッカー検出、進捗振り返り | 朝の業務計画、週次レポート、バックアップ実行 |

**判断に迷ったら**: 「確認だけして判断する」→ Heartbeat のチェックリストに追加。「決まった時間に何かをやる」→ Cron タスクとして定義。

→ 詳細: `operations/heartbeat-cron-guide.md`

---

## Cron の2タイプ — LLM型 vs Command型

Cron タスクには、思考が必要かどうかで2つのタイプがある。

| 観点 | LLM型（`type: llm`） | Command型（`type: command`） |
|------|---------------------|---------------------------|
| **たとえ** | 「今日何を優先すべきか考えて」 | 「毎朝このボタンを押して」 |
| **判断** | あり（状況に応じて出力が変わる） | なし（毎回同じことを実行） |
| **APIコスト** | あり（LLM呼び出し） | コマンド自体はなし。ただし **follow-up LLM が起動する場合あり**（下記参照） |
| **出力** | 不定形（タスクごとに異なる） | 確定的（コマンドの stdout） |
| **適したタスク** | 計画立案、振り返り、文章作成、記憶整理 | バックアップ、通知送信、データ取得、ヘルスチェック |

### Command型の follow-up LLM（重要）

Command型はコマンド自体は機械的に実行するが、**stdout に返り値がある場合、デフォルトで LLM が起動して結果を分析する**（follow-up）。つまり完全にコストゼロではない場合がある。

```
コマンド実行 → stdout あり？
  → なし → 終了（LLM 不要）
  → あり → skip_pattern にマッチする？
      → マッチ → 終了（LLM スキップ）
      → マッチしない → LLM が起動して結果を解釈・対処判断
```

この follow-up を制御するオプション:
- **`trigger_heartbeat: false`** — follow-up LLM を常にスキップ（結果の分析が不要な場合）
- **`skip_pattern: <正規表現>`** — stdout がマッチしたときだけスキップ（正常時のみ無視、異常時はLLMに判断させる）

### 使い分けの判断基準

```
「毎回同じことをするだけ？」
  → はい、結果も見なくてよい → Command型 + trigger_heartbeat: false
  → はい、ただし異常時だけ判断が必要 → Command型 + skip_pattern（正常パターン）
  → いいえ（状況に応じて判断が変わる） → LLM型
  → コマンド実行 + 毎回結果の解釈が必要 → LLM型（description にコマンド実行を指示）
```

### 記述例

```markdown
## 毎朝の業務計画（LLM型）
schedule: 0 9 * * *
type: llm
episodes/ から昨日の進捗を確認し、今日のタスクを計画する。

## バックアップ実行（Command型・follow-up不要）
schedule: 0 2 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/backup.sh

## 監視チェック（Command型・正常時はスキップ、異常時はLLMが判断）
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

→ 詳細: `operations/heartbeat-cron-guide.md`

---


## タスクの流し方 — submit_tasks vs delegate_task

タスクを実行に移す方法は2つある。

| 観点 | `submit_tasks` | `delegate_task` |
|------|---------------|----------------|
| **誰が実行するか** | **自分自身**の TaskExec パス | **直属の部下** |
| **使う場面** | 自分でやるべきタスクを非同期実行したい | 部下に委任したい |
| **DAG/並列** | `parallel: true` で並列、`depends_on` で依存 | 1件ずつ委任 |
| **進捗追跡** | 正規タスクストアを参照する `list_tasks` / TaskBoard | `task_tracker` で追跡 |
| **典型例** | Heartbeat で発見したタスクを自分で実行 | 上司が部下に作業を委任 |

**判断フロー**:
```
「このタスクは部下がやるべき？」
  → はい、直属の部下がいる → delegate_task
  → いいえ、自分でやる → submit_tasks
  → 部下がいない → submit_tasks
```

→ 詳細: `operations/task-management.md`, `anatomy/task-architecture.md`

---

## チーム設計 — ソロから始めてスケールする

### なぜロールを分けるのか

| 理由 | 説明 |
|------|------|
| コンテキスト汚染の防止 | 全工程を1人に担わせるとコンテキストが肥大化し判断精度が落ちる |
| 品質の構造的担保 | 実行する者と検証する者を分ける（セルフレビューの盲点を排除） |
| 並行実行 | 独立したロールは同時実行でスループット向上 |
| 専門性の深化 | ロール固有のチェックリスト・記憶で汎用エージェントより高品質 |

### スケーリングの段階

| 規模 | 構成 | いつ使う |
|------|------|---------|
| **ソロ** | 1 Anima が全ロール兼務 | 小タスク、プロトタイプ、始めたばかり |
| **ペア** | PdM + Engineer | 中規模の定型タスク |
| **フルチーム** | PdM + Engineer + Reviewer + Tester | 本格的なプロジェクト |
| **スケールド** | PdM + 複数 Engineer + 複数 Reviewer + Tester | 大規模・複数モジュール |

### 判断の目安

- 失敗コストが高い → ロール分離を増やす
- 「実装した本人がレビュー」になっている → Reviewer を分離
- 並行作業可能なモジュールが多い → Engineer を増やす

---

## 記憶 — 5種類を使い分ける

| 記憶の種類 | ディレクトリ | 一言で | 例 |
|-----------|------------|--------|-----|
| **エピソード記憶** | `episodes/` | いつ何をしたか | 「3/15にSlack API調査をした」 |
| **意味記憶** | `knowledge/` | 学んだこと | 「Slack APIのレート制限は100回/分」 |
| **手続き記憶** | `procedures/` | どうやるか | 「Gmail認証のセットアップ手順」 |
| **スキル** | `skills/` | 実行可能な手順書 | 画像生成スキル、調査スキル |
| **短期記憶** | `shortterm/` | 直近の文脈 | 今の会話の流れ |

**Priming（自動想起）** が会話や巡回のたびに関連する記憶を自動で想起し、必要な文脈をシステムプロンプトに注入する。加えて `search_memory` で能動的に検索もできる。

**Consolidation（記憶統合）** は新しい活動チャンクだけをエピソード化し、変化のない再実行では生成しない。知識の自動変更・週次月次整理・スキル自動学習は既定で無効。記憶保存と必要時の検索は残る。

→ 詳細: `anatomy/memory-system.md`

---

## コスト最適化 — background_model と Activity Level

### background_model

Heartbeat / Cron は明示設定された background_model を使える。Inbox はメインモデルを使い、タスク固有のモデル指定はそのタスクで優先される。安価という理由だけで自動選択しない。

| 区分 | 使用モデル | 対象 |
|------|-----------|------|
| foreground | メインモデル、または明示されたタスク固有モデル | Chat、Inbox、TaskExec |
| background | 明示設定の background_model、未設定ならメインモデル | Heartbeat、Cron |

設定: `animaworks anima set-background-model {名前} claude-sonnet-4-6`

### Activity Level

全体の活動頻度を 10%〜400% で調整できる。Heartbeat 間隔に直接影響する。

| Activity Level | Heartbeat間隔（ベース30分の場合） | 用途 |
|---------------|-------------------------------|------|
| 200% | 15分 | 繁忙期・アクティブ開発 |
| 100%（デフォルト） | 30分 | 通常運用 |
| 50% | 60分 | 低負荷・コスト節約 |
| 30% | 100分 | 夜間・休日 |

**Activity Schedule** で時間帯別に自動切替もできる（例: 9:00-22:00 は 100%、22:00-6:00 は 30%）。

→ 詳細: `reference/operations/model-guide.md`, `operations/heartbeat-cron-guide.md`

---

## 組織の基本 — 上司・部下・同僚

Anima は `status.json` の `supervisor` フィールドで階層が決まる。

| 関係 | 定義 | コミュニケーション |
|------|------|------------------|
| **上司** | `supervisor` に指定された Anima | 進捗報告（MUST）、問題エスカレーション |
| **部下** | `supervisor` が自分の Anima | `delegate_task` で委任、`org_dashboard` で監視 |
| **同僚** | 同じ `supervisor` を持つ Anima | 直接連絡OK |
| **他部署** | 上記いずれでもない | 自分の上司経由（直接連絡は原則禁止） |
| **人間** | `supervisor: null`（トップレベル）の場合 | `call_human` で通知 |

→ 詳細: `organization/hierarchy-rules.md`, `organization/roles.md`

---

## 困ったときの最初の一手

| 状況 | やること |
|------|---------|
| 操作方法がわからない | `search_memory(query="キーワード", scope="common_knowledge")` |
| タスクがブロックされた | `troubleshooting/escalation-flowchart.md` を参照 |
| ツールが動かない | `troubleshooting/common-issues.md` を参照 |
| 何をすべかわからない | current_state.md と `list_tasks` を確認。必要なら設定済みのHeartbeatチェックリストを参照 |
| 判断に迷う | 上司に `send_message(intent="question")` で相談 |

→ 全ドキュメント目次: `common_knowledge/00_index.md`
