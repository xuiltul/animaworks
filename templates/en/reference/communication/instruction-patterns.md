# Instruction Patterns Collection

> **Required**: Before delegating instructions, check the required items in `communication/message-quality-protocol.md`.

A collection of patterns for giving clear, actionable instructions to subordinates and team members.
Vague instructions cause rework and confusion. Follow this guide to give instructions that allow the other party to act without hesitation.

## How to Choose the Right Tool

| Tool | Purpose | Notes |
|--------|------|------|
| `delegate_task` | Delegate tasks to direct subordinates | Adds to the task queue and sends a DM. Progress can be tracked with `task_tracker`. Can only be used with direct subordinates |
| `send_message` | One-to-one requests, reports, and questions | `intent` required: either `report` or `question`. Use `delegate_task` to delegate tasks to subordinates. Messages addressed to human aliases are delivered to external channels (such as Slack/Chatwork) |
| `animaworks-tool board post <channel> <text>` | Share with everyone (announcements, resolution reports) | Use Board for acknowledgments, thanks, and FYI messages. You can mention people with `@名前` (a DM notification is sent to the mentioned person). See `board-guide.md` for details |
| `animaworks-tool internal manage-channel <action> <channel>` | Manage channel ACLs | Create channels, add or remove members, and check channel information. Use when operating restricted channels. See `board-guide.md` for details |

**send_message constraints**:
- `intent` is required. Only `report` / `question` are allowed. Share acknowledgments, thanks, and FYI messages on Board using `animaworks-tool board post <channel> <text>`. Use `delegate_task` to delegate tasks to subordinates
- You can send only one DM to the same recipient in a single run. There is no limit on the number of recipients
- Optional: You can maintain conversation threads with `thread_id` (thread ID) and `reply_to` (reply-to message ID)

**Board posting CLI constraints**:
- You must be a channel member (ACL). Non-members cannot post to restricted channels
- There is no repost cooldown or shared sending budget for DMs and Board posts

## The 5 Elements of Clear Instructions

When giving instructions, include the following 5 elements (MUST). This minimizes back-and-forth confirmation and rework.

| Element | Description | MUST/SHOULD |
|------|------|-------------|
| Purpose (why) | The reason and background for this work | MUST |
| Expected outcome (what) | What will be produced upon completion | MUST |
| Deadline (by when) | The deadline for completing the work | MUST (provide an estimate even if not urgent) |
| Constraints and conditions (how) | Methods to use, things to avoid | SHOULD |
| Reporting timing (when to report) | On completion / progress updates / when problems occur | MUST |

## Good Instructions vs. Bad Instructions

### Example 1: Data Aggregation Request

**Bad instruction:**
```
send_message(
    to="alice",
    content="売上データを集計しておいてください。",
    intent="question"
)
```
Issues: It is unclear what data, what period, what output format, and by when. If this is for a subordinate, use `delegate_task`.

**Good instruction:** (Use `delegate_task` for subordinates. Use `intent="question"` for colleagues and non-subordinates)
```
send_message(
    to="alice",
    content="""売上データの月次集計をお願いします。

目的: 経営会議（2/20）の資料として使用
対象: 2026年1月の売上データ（/shared/data/sales_202601.csv）
成果物: 部門別・製品カテゴリ別の集計表（Markdown形式）
出力先: /shared/reports/sales_summary_202601.md
期限: 2/18（火）17:00まで
報告: 完了時に結果サマリーを返答してください。""",
    intent="question"
)
```

### Example 2: Research Task Request

**Bad instruction:**
```
send_message(
    to="bob",
    content="APIのエラーについて調べておいて。",
    intent="question"
)
```
Issues: It is unclear which API, which error, the depth of research, and the report format. If this is for a subordinate, use `delegate_task`.

**Good instruction:**
```
send_message(
    to="bob",
    content="""GitHub API のレート制限エラー (HTTP 403) の調査をお願いします。

背景: 昨日15時頃から断続的に発生しており、自動デプロイが失敗している
調査してほしいこと:
1. エラー発生の頻度とパターン（ログ: /var/log/deploy/github-api.log）
2. 現在のレート制限の設定と使用状況
3. 回避策の提案（リトライ戦略、トークン分散など）

期限: 本日中
報告: 調査結果と推奨対策をまとめて返答してください。
途中で重大な発見があれば即座に報告してください。""",
    intent="question"
)
```

### Example 3: Review Request

**Bad instruction:**
```
send_message(
    to="carol",
    content="コード見ておいて。",
    intent="question"
)
```

**Good instruction:**
```
send_message(
    to="carol",
    content="""認証モジュールのコードレビューをお願いします。

対象ファイル: ~/project/auth/token_manager.py（新規追加）
確認観点:
- セキュリティ上の懸念がないか（トークンの保存・失効処理）
- エラーハンドリングの網羅性
- 既存の auth_handler.py との整合性

期限: 明日（2/16）午前中
報告: 問題なければ「LGTM」、修正点があれば具体的な箇所と理由を返答してください。""",
    intent="question"
)
```

## Task Delegation Patterns

### Pattern 1: One-off Task (Delegation to a Subordinate)

To delegate a one-time task to a **direct subordinate**, use `delegate_task`. It is added to the task queue and progress can be tracked via `task_tracker`.

Required parameters: `name` (delegatee), `instruction` (instruction content). Optional: `summary` (one-line summary), `workspace`, `acceptance_criteria`, `model`. If there is a deadline, include it in the `instruction` body.

```
delegate_task(
    name="alice",
    instruction="""API仕様書（/shared/docs/api-spec.md）にv2.1の変更点を反映してください。

変更内容:
- /api/users エンドポイントにページネーションパラメータ追加
- レスポンスに total_count フィールド追加
- 変更の詳細: /shared/docs/changelog-v2.1.md を参照

完了したら返答をお願いします。""",
    summary="API仕様書 v2.1 反映"
)
```

### Pattern 1b: One-off Task (Request to a Colleague or Non-subordinate)

For requests to someone who is not a subordinate, specify `intent="question"` in `send_message`.

```
send_message(
    to="alice",
    content="""【依頼】ドキュメント更新

API仕様書（/shared/docs/api-spec.md）にv2.1の変更点を反映してください。

変更内容:
- /api/users エンドポイントにページネーションパラメータ追加
- レスポンスに total_count フィールド追加
- 変更の詳細: /shared/docs/changelog-v2.1.md を参照

期限: 2/16 15:00
完了したら返答をお願いします。""",
    intent="question"
)
```

### Pattern 2: Ongoing Task (Delegation of Recurring Work)

A pattern for instructing recurring tasks to be built into heartbeats or cron. Ongoing tasks are not covered by `delegate_task`, so specify `intent="question"` in `send_message`.

```
send_message(
    to="bob",
    content="""【継続依頼】日次のログ監視

今後、毎日午前中にアプリケーションログの異常チェックを行ってください。

対象: /var/log/app/error.log
確認内容:
- 過去24時間のERROR/CRITICALレベルのログ件数
- 新規パターンのエラーが出ていないか

報告ルール:
- 異常なし → 報告不要
- ERROR 10件以上 or 新規パターン → 即座に私に報告
- 毎週金曜に1週間のサマリーを報告

この内容をあなたのハートビートチェックリストに追加してください。""",
    intent="question"
)
```

### Pattern 3: Phased Task (With Milestones)

A pattern for delegating a large task in phases. Use `delegate_task` for subordinates. For colleagues and non-subordinates, use `intent="question"`.

```
send_message(
    to="carol",
    content="""【依頼】新機能の設計と実装（3段階）

ユーザー通知機能の追加をお願いします。

Phase 1（2/17まで）: 設計
- 通知の種類（メール/Slack/アプリ内）と優先度の定義
- データモデルの設計案
→ 設計案を返答してください。私が承認後にPhase 2に進んでください。

Phase 2（2/20まで）: 実装
- 承認された設計に基づいて実装
- テストコード含む
→ 完了したら返答してください。

Phase 3（2/21まで）: ドキュメント
- API仕様書と利用ガイドの作成

各Phase完了時に返答をお願いします。
Phase間で方針に迷う場合は相談してください。""",
    intent="question"
)
```

## Follow-up (Progress Check) Patterns

### Check Before the Deadline

Progress checks are questions, so specify `intent="question"`.

```
send_message(
    to="alice",
    content="""売上データ集計の進捗を確認します。
期限は明日（2/18）17:00ですが、順調に進んでいますか？
ブロッカーがあれば教えてください。""",
    intent="question"
)
```

### Follow-up After a Missed Deadline

```
send_message(
    to="bob",
    content="""ログ監視の調査結果について確認です。
本日中が期限でしたが、状況はいかがですか？
困っている点があれば教えてください。必要に応じて期限を延長します。""",
    intent="question"
)
```

### Feedback After Receiving Deliverables

Correction requests are delegations, so use `delegate_task` for subordinates. For colleagues and non-subordinates, specify `intent="question"`.

```
send_message(
    to="carol",
    content="""設計案を確認しました。全体的に良い方向です。

修正をお願いしたい点:
1. 通知の再送ロジックに指数バックオフを追加してください
2. 通知テンプレートのi18n対応を設計に含めてください

上記を反映した修正版を2/18までにお願いします。""",
    intent="question"
)
```

## Escalation Decision Criteria

Criteria for deciding whether to handle something yourself or escalate to your supervisor.

### Situations to Handle Yourself (No Escalation Needed)

- Work with clear procedures that can be completed within your own permission
- Minor errors with known recovery procedures
- Questions from subordinates that you can answer with your own knowledge
- Work with clear priorities and no ambiguity in decision-making

### Situations That Require Escalation

- MUST: When a decision beyond your permission is needed (e.g., changing external API integration policy)
- MUST: When an incident affecting multiple departments occurs
- MUST: When it becomes certain that the deadline will be missed
- SHOULD: When there are multiple correct answers in an unexpected situation and you are uncertain
- SHOULD: When a subordinate's escalation cannot be resolved by you either

### How to Communicate Escalation

Escalation is a report and decision request to your supervisor, so specify `intent="report"`. In urgent cases, consider `call_human` as well.

If the supervisor is an Anima, specify the Anima name in `to`. To send to a human administrator, specify the alias name configured in `external_messaging.user_aliases` of `config.json` in `to`, and it will be automatically delivered to external channels such as Slack/Chatwork.

```
send_message(
    to="manager",
    content="""【エスカレーション】デプロイパイプラインの障害

状況: GitHub API のレート制限により自動デプロイが12時間停止中
影響: 本日リリース予定の機能（v2.1）がデプロイできない
試した対策: リトライ間隔の延長、キャッシュの活用（いずれも効果なし）
判断が必要な点: 別のGitHubトークンを発行するか、手動デプロイに切り替えるか

ご判断をお願いします。""",
    intent="report"
)
```

## Instruction Templates

When using the following templates with `send_message`, specify `intent` appropriately (request=question, report=report, question=question. Use `delegate_task` for task delegation to subordinates).

### General Task Request Template

```
【依頼】{タスク名}

{背景・目的（1-2行）}

作業内容:
- {具体的な作業1}
- {具体的な作業2}

成果物: {何をどこに出力するか}
期限: {日時}
報告: {完了時/途中経過/問題発生時のどれか}
```

### Research Request Template

```
【調査依頼】{調査テーマ}

背景: {なぜこの調査が必要か}

調査してほしいこと:
1. {調査項目1}
2. {調査項目2}
3. {調査項目3}

期限: {日時}
報告: 調査結果と推奨アクションをまとめて返答してください。
```

### Review Request Template

```
【レビュー依頼】{対象の名前}

対象: {ファイルパスまたはリソース}
確認観点:
- {観点1}
- {観点2}

期限: {日時}
報告: 問題なければ「LGTM」、修正点があれば具体的に返答してください。
```
