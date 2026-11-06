# Reporting and Escalation Methods

> **Required**: Before reporting or escalating, check the required items in `communication/message-quality-protocol.md`.

Reporting to your supervisor is the lifeline of organizational operations.
Reporting at the right time and in the right format supports early problem detection and rapid decision-making.

## Tool Usage

Use the `send_message` tool for reports. Observe the following constraints.

| Tool | Purpose | Notes |
|--------|------|------|
| `send_message` | One-to-one reports or questions to a supervisor or colleague | Use `intent="report"` for reports, progress updates, and requests for decisions; use `intent="question"` for questions. Anima names, aliases, `slack:`/`chatwork:`, etc. (see below) |
| `animaworks-tool board post <channel> <text>` | Announcements to the whole team (Board) | Use Board for acknowledgments, thanks, and FYI messages |
| `call_human` | Emergency notifications to humans | Service shutdowns, security incidents, etc. |

**send_message constraints**:
- `intent` is required, and its value must be **only `report` or `question`** (`report` for reports and progress updates; `question` for questions). **`delegation` cannot be used with `send_message`** (the tool will reject it). Use `delegate_task` to delegate tasks
- You may send only one DM to the same recipient in a given run. There is no limit on the number of recipients
- Use `animaworks-tool board post <channel> <text>` for additional communications
- **Recipients**: See “Resolving recipients” below (Anima names, human aliases, directly specifying `slack:` / `chatwork:`, etc.)
- **Note**: During a chat session, you cannot use send_message to message humans. Reply directly in text instead.

### Recipient Resolution and External Delivery (`core/messaging/outbound.py`)

The `to` of `send_message` is resolved to either the internal inbox or an external service (Slack / Chatwork) using the following **priority order**. An empty string cannot be resolved and results in an error.

1. **Exact match with a known Anima name** (case-sensitive) → internal inbox. Known names are the directory names directly under `~/.animaworks/animas/`
2. **Match with a key in `external_messaging.user_aliases` of `config.json`** (aliases are **case-insensitive**) → external delivery. When `slack` is `external_messaging.preferred_channel` (`slack` / `chatwork`), if `slack_user_id` exists, use Slack; otherwise, if `chatwork_room_id` exists, use Chatwork. When `preferred_channel` is `chatwork`, prefer `chatwork_room_id`; if absent, use Slack if `slack_user_id` exists. **Aliases with neither ID** result in an error (set contact information in `external_messaging.user_aliases`)
3. **`slack:USERID`** (starts with `slack:`, followed by a Slack user ID) → direct Slack DM (USERID is **normalized to uppercase** in the implementation)
4. **`chatwork:ROOMID`** (starts with `chatwork:`, ROOMID part has only leading and trailing whitespace removed) → direct Chatwork room
5. **A bare Slack user ID** (`U` + alphanumeric **8 or more characters**, case is tolerated during matching. Example `U0123456789`) → direct Slack DM
6. **Case-insensitive match with a known Anima name** → internal (normalized to the canonical name on disk)
7. **No match** → unknown recipient error (the tool layer may switch the hint text depending on whether a chat is running)

**External configuration** (`external_messaging` of `config.json`):

| Field | Role |
|-----------|------|
| `preferred_channel` | Default channel for alias recipients (`slack` / `chatwork`) |
| `user_aliases` | Alias name → `{ "slack_user_id": "...", "chatwork_room_id": "..." }` (**at least one** is required) |

**External send behavior** (`send_external`):

- The send order is **resolved channel first**, then if that fails, **the other channel** is tried (the second option only exists when both Slack and Chatwork destination IDs are available, such as with aliases. `slack:` / `chatwork:` direct specification usually targets only that channel)
- If sending fails because the external channel is not configured, the tool result may return `NoChannelConfigured` or `DeliveryFailed` as JSON
- The body is converted from **Markdown to Slack mrkdwn / Chatwork format**
- **Slack**: If a `SLACK_BOT_TOKEN__{anima名}` (from Vault or shared credentials) linked to the Anima name exists, send with the Bot token, attaching the display name (Anima name) and **icon URL** (derived from Anima assets). **If no Bot token exists**, prepend `[送信者Anima名] ` to the body before sending
- **Chatwork**: Uses the sending Anima's own identity token (`CHATWORK_API_TOKEN__<Anima名>`). When sent via `send_message`, a `[Anima名] ` prefix is added to the body

**Send behavior**:

- There are no time- or daily-based send budgets, and no send rejection based on conversation depth.
- Depth between internal Animas may be recorded for diagnostic purposes, but messages are delivered as-is.

## Reporting Timing

### Situations Requiring Immediate Reporting (MUST)

In the following situations, report to your supervisor immediately upon recognition:

| Situation | Reason | Example |
|------|------|-----|
| Task completion | So the supervisor can decide on next actions | Deployment complete, report creation complete |
| Error or failure occurrence | For early response | API outage, detection of data inconsistency |
| When a decision is needed | To seek decisions outside your authority | Proposing policy changes, requesting additional resources |
| When a deadline will definitely be missed | So the supervisor can adjust the schedule | Work halted due to a blocker |
| Security concerns | Because immediate action is required | Signs of unauthorized access, suspected credential leakage |

### Situations Requiring Regular Reporting (SHOULD)

| Situation | Frequency | Content |
|------|------|------|
| Daily summary | Every day (at end of work) | Day's achievements and next day's plans |
| Weekly review | Every week (Friday) | Week's achievements, challenges, and next week's plans |
| Progress updates on long-term tasks | As appropriate (guideline: every 2-3 days) | Progress rate, remaining work, presence of blockers |

### Situations Where Reporting Is Not Needed

- Memory writes and organization (internal work)
- Routine work with no anomalies (equivalent to `HEARTBEAT_OK` via heartbeat)
- Work that the supervisor has explicitly instructed does not require reporting

## Basic Report Format

### SCANA Format

Reports consist of the following five items. Not all items need to be included; select as appropriate for the situation.

| Item | English | Description | MUST/MAY |
|------|------|------|----------|
| Situation | Situation | What is currently happening | MUST |
| Cause | Cause | Why it happened (if known) | MAY |
| Action taken | Action taken | What actions you have taken | SHOULD |
| Next steps | Next steps | What you will do next / what decision you need | MUST |
| Appendix | Appendix | Related logs, file paths, numerical values | MAY |

### Thread Continuation When Replying

When replying to a message from your supervisor, specify `reply_to` (the reply target message ID) and `thread_id` (the thread ID) to maintain conversation context. The ID format is `YYYYMMDD_HHMMSS_ffffff` (example: `20260215_093000_123456`).

```
send_message(
    to="manager",
    content="承知しました。15時までに対応します。",
    intent="report",
    reply_to="20260215_093000_123456",
    thread_id="20260215_090000_000000"
)
```

### Format Usage Example

```
send_message(
    to="manager",
    content="""【完了報告】売上データ月次集計

状況: 2026年1月の売上データ集計が完了しました。
対処: 部門別・カテゴリ別に集計し、前月比も算出しました。
次のステップ: 特にアクション不要です。レポートをご確認ください。
付加情報: /shared/reports/sales_summary_202601.md""",
    intent="report"
)
```

## Templates by Report Type

### Completion Report

A report when a task has completed successfully.

```
send_message(
    to="manager",
    content="""【完了報告】{タスク名}

状況: {タスク名}が完了しました。
成果物: {ファイルパスまたは結果の要約}
所要時間: {かかった時間}
備考: {特記事項があれば}""",
    intent="report"
)
```

**Concrete example:**

```
send_message(
    to="manager",
    content="""【完了報告】API仕様書のv2.1更新

状況: API仕様書にv2.1の変更点を反映しました。
成果物: /shared/docs/api-spec.md（差分: セクション3.2, 4.1を更新）
所要時間: 約45分
備考: ページネーションの例を3パターン追加しました。""",
    intent="report"
)
```

### Error Report

A report when a failure or error has occurred.

```
send_message(
    to="manager",
    content="""【エラー報告】{何が起きたか}

状況: {現在の状態の説明}
原因: {判明している範囲での原因}
影響範囲: {何が止まっているか、誰に影響しているか}
対処: {自分が試した対策とその結果}
次のステップ: {今後の対応案 / 判断が必要なこと}""",
    intent="report"
)
```

**Concrete example:**

```
send_message(
    to="manager",
    content="""【エラー報告】定期バッチ処理の失敗

状況: 毎朝9:00のデータ同期バッチが3回連続で失敗しています。
原因: 外部APIのレスポンスタイムアウト（30秒超過）。API側の障害の可能性が高い。
影響範囲: 今朝以降のデータが未同期。ダッシュボードの数値が昨日時点で止まっている。
対処: タイムアウトを60秒に延長して再実行 → 同様に失敗。API提供元のステータスページを確認 → メンテナンス情報なし。
次のステップ: API提供元への問い合わせが必要です。ご判断をお願いします。""",
    intent="report"
)
```

### Decision Request (Escalation of Decision-Making)

A report when a decision cannot be made within your own authority.

```
send_message(
    to="manager",
    content="""【判断依頼】{何について判断が必要か}

状況: {現在の状況}
選択肢:
A. {選択肢Aの内容} — メリット: {利点} / デメリット: {欠点}
B. {選択肢Bの内容} — メリット: {利点} / デメリット: {欠点}
私の推奨: {A or B}（理由: {なぜ}）

ご判断をお願いします。""",
    intent="report"
)
```

**Concrete example:**

```
send_message(
    to="manager",
    content="""【判断依頼】ログ保存期間の変更

状況: ディスク使用量が85%に達しており、1週間以内に90%を超える見込みです。
選択肢:
A. ログ保存期間を90日→30日に短縮 — メリット: 即座に50GB削減 / デメリット: 古いログでの調査が不可能に
B. ストレージを追加（500GB） — メリット: ログを維持できる / デメリット: 月額コスト増（約$50/月）
C. 古いログをアーカイブストレージに移動 — メリット: ログ維持+コスト抑制 / デメリット: 実装に2日必要
私の推奨: C（コストとデータ保全のバランスが良い）

ご判断をお願いします。""",
    intent="report"
)
```

### Progress Report

An interim report on a long-term task.

```
send_message(
    to="manager",
    content="""【途中経過】{タスク名}

進捗: {完了した作業 / 全体に対する進捗率}
残作業: {まだ終わっていないこと}
ブロッカー: {あれば記載。なければ「なし」}
見込み: {期限に間に合うか}""",
    intent="report"
)
```

**Concrete example:**

```
send_message(
    to="manager",
    content="""【途中経過】ユーザー通知機能の実装

進捗: Phase 1（設計）完了、Phase 2（実装）の70%が完了
  - メール通知: 完了
  - Slack通知: 完了
  - アプリ内通知: 実装中（残り1日程度）
残作業: アプリ内通知の実装 + テストコード作成
ブロッカー: なし
見込み: 期限（2/20）に間に合います。""",
    intent="report"
)
```

## Daily Summary Format

The daily summary is a review of the day sent to your supervisor at the end of work.

### Template

```
send_message(
    to="manager",
    content="""【日次サマリー】2026-02-15

■ 完了したこと
- {完了タスク1}
- {完了タスク2}

■ 進行中
- {進行中タスク1}（進捗: {XX%}、見込み: {期限}）

■ 課題・懸念
- {あれば記載}

■ 明日の予定
- {予定1}
- {予定2}""",
    intent="report"
)
```

### Concrete Example

```
send_message(
    to="manager",
    content="""【日次サマリー】2026-02-15

■ 完了したこと
- API仕様書のv2.1更新（/shared/docs/api-spec.md）
- ログ監視スクリプトのバグ修正（タイムゾーン処理）

■ 進行中
- ユーザー通知機能の実装（進捗: 70%、見込み: 2/20完了予定）

■ 課題・懸念
- 外部APIのレスポンスが遅くなっている兆候あり（現時点では影響なし、引き続き監視）

■ 明日の予定
- アプリ内通知の実装完了
- テストコード作成着手""",
    intent="report"
)
```

## Determining Urgent vs. Normal Reports

### Urgent Report (Immediate, MUST)

If any of the following apply, report immediately even if it means interrupting other work:

- **Service outage**: A failure affecting users
- **Data loss or corruption**: Potential for an unrecoverable state
- **Security incident**: Suspected unauthorized access or information leakage
- **Confirmed deadline miss**: Certain that an already committed deadline will not be met

Urgent report messages MUST: prefix with "【緊急】" at the beginning.
If human immediate response is needed (service outage, data loss, security incident, etc.), also use `call_human`.

```
send_message(
    to="manager",
    content="""【緊急】本番データベースの応答遅延

状況: 本番DBの応答時間が通常の10倍（平均300ms→3000ms）に悪化。
影響: Web UIのページ読み込みが10秒以上かかる状態。
対処中: スロークエリの特定を開始。10分以内に続報します。""",
    intent="report"
)
```

### Normal Report (At the Next Heartbeat or Daily Summary)

The following are low urgency and can be reported together:

- Progress on tasks proceeding as planned
- Minor issues already resolved by yourself
- Improvement suggestions or ideas

## Escalation Decision Flowchart

Procedure for deciding whether a report should be escalated:

```
1. 自分の権限内で解決可能か？
   → Yes: 自分で対処し、完了後に結果を報告
   → No: ステップ2へ

2. 緊急性はあるか？（サービス影響・データ損失リスク）
   → Yes: 【緊急】として即座にエスカレーション
   → No: ステップ3へ

3. 判断の選択肢を整理できるか？
   → Yes: 選択肢と推奨を添えて【判断依頼】として報告
   → No: 状況をありのまま報告し、上司の指示を仰ぐ
```

## Reporting Precautions

### Things to Do (MUST/SHOULD）

- MUST: Specify **`"report"` (report, progress) or `"question"` (question)** for `intent` of `send_message`. `"delegation"` is not allowed (use `delegate_task` for delegation)
- MUST: Distinguish between facts and speculation. Add "likely" or "probably" for speculation
- MUST: Clearly state the scope of impact. What is affected and who is affected
- SHOULD: Include the countermeasures you tried and their results. This avoids the supervisor proposing the same countermeasures
- SHOULD: Propose next actions yourself. Do not just wait for instructions

### Things to Avoid

- Writing a long background before getting to the conclusion (put the conclusion first)
- Reporting "a problem occurred" without any specific information
- Combining multiple different topics into one message (separate by topic)
- Hiding a problem or making it seem minor (convey the accurate situation)
- Sending multiple `send_message` messages to the same recipient in the same run (only one send per recipient; there is no limit on the number of recipients)
