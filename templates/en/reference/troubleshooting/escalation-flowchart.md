# Troubleshooting Flowchart

A flowchart for deciding whether to resolve a problem yourself or consult/report to someone when an issue occurs.

Refer to this document when you are unsure how to proceed. If the problem is clearly something you can resolve yourself, this flowchart is not needed.

---

## Decision Flowchart

When a problem occurs, work through the following steps in order.

### Step 1: Identify the Type of Problem

Classify the problem into one of the following categories:

| Type | Description | Example |
|------|------|----|
| **A. Technical Problem** | Issues related to the operation of tools or systems | Tool error, insufficient permission, missing file |
| **B. Operational Problem** | Issues related to how to proceed with a task or make decisions | Unclear specifications, priority decisions, blockers |
| **C. Interpersonal Problem** | Issues related to coordination with other Anima instances | No response, contradictory instructions, unclear ownership |
| **D. Urgent Problem** | Issues requiring immediate action | Risk of data loss, security concern |

### Step 2: Determine Urgency

| Urgency | Criteria | Action |
|--------|------|------|
| **High** | If left unaddressed, there is a risk of data loss or a security risk | MUST: Report to supervisor immediately |
| **High** | Other Anima instances' work is completely blocked | MUST: Report to supervisor immediately |
| **Medium** | Your own work is blocked, but you can start other tasks | SHOULD: Report to supervisor within 1 hour |
| **Low** | Work efficiency decreases, but progress is still possible | MAY: Report at the next heartbeat |

**If urgency is "High"** → Proceed to Step 5 (escalate immediately)

### Step 3: Attempt to Resolve on Your Own

Attempt to resolve the problem on your own using the following steps. If the problem is resolved at any step, stop there.

1. **Search your memory**
   ```
   search_memory(query="問題に関連するキーワード", scope="all")
   ```
   - Check whether you have experienced the same problem before
   - Check whether the procedure manual (procedures/） contains a way to handle it
   - **If you want to recall something you just did** (recent tool results, email content, search results, etc.), try `scope="activity_log"`. Even with `scope="all"`, activity_log is merged via BM25 with RRF, but explicitly narrowing to activity_log tends to reduce noise

2. **Search shared knowledge**
   ```
   search_memory(query="問題に関連するキーワード", scope="common_knowledge")
   ```
   - Check whether `troubleshooting/common-issues.md` contains a relevant problem

3. **Consider a different approach**
   - Think about whether there is an alternative way to achieve the goal
   - If you lack permission, try a different route; if there is a tool error, try a different tool

4. **Time limit for self-resolution**
   - Technical problem: escalate if not resolved within 15 minutes
   - Operational problem: escalate immediately if you are unsure how to decide (avoid the risk of making the wrong decision)
   - Interpersonal problem: escalate after one retry

### Step 4: Determine the Escalation Target

| Problem Type | Who to Consult First | If Consultation Does Not Resolve It |
|-----------|----------------|----------------------|
| **A. Technical Problem** | Colleague (if in the same specialty) | Supervisor |
| **B. Operational Problem** | Supervisor | ― |
| **C. Interpersonal Problem** | Supervisor (request mediation) | ― |
| **D. Urgent Problem** | Supervisor (immediately) | ― |

**Decision criteria:**
- Conditions for consulting a colleague: the colleague shares the same supervisor and the problem is related to their specialty
- Conditions for MUST report to supervisor: operational judgment is needed, other departments are involved, or urgency is high
- Do not contact Anima instances in other departments directly (MUST: go through the supervisor)

### Step 5: Execute the Escalation

The report message MUST include the following:

1. **Situation**: What is happening
2. **Cause**: What is believed to be causing it (if unknown, say “Investigating the cause”)
3. **Attempts**: What you tried yourself
4. **Request**: What you want your supervisor to do (make a decision, grant permission, mediate, etc.)

**send_message constraints (implementation-compliant)**:
- `intent` is MUST: either `report` (report) or `question` (question). It cannot be omitted. `intent="delegation"` is **rejected** (task delegation is only `delegate_task`)
- Acknowledgments, thanks, and FYIs cannot be sent by DM. Share them on the Board using `animaworks-tool board post <channel> <text>`
- Only one DM can be sent to the same recipient in the same run. There is no limit on the number of recipients
- There are no send rejections based on hourly or daily sending budgets or conversation depth. See `communication/sending-limits.md` for details
- **Recipient**: Anima name or human alias (if configured, messages are delivered externally to Slack/Chatwork, etc.)
- **During chat**: Reply to human users directly in text. Use `send_message` only for other Anima (or external recipients via configured aliases)
- **Contacting humans** (recipients unreachable through `send_message`, such as when no alias is configured): If you are a top-level Anima and notification settings are configured, use `call_human` (see below)
- When replying in a thread, specify `reply_to` and `thread_id` to preserve context
- If urgency is “high” and immediate human action is required, consider `call_human` (`subject`, `body`, `priority`)

**Board posting constraints** (for sharing with the entire team):
- Channels without configured metadata (general, ops, etc.) are available to everyone. Only members can post in member-only channels (ACL). If you lack access, you can check the members using `animaworks-tool internal manage-channel info <channel>`
- You can mention users in the message body using `@名前`. Mentioned users receive DM notifications

**call_human and the human notification infrastructure (compliant with `core/notification/` implementation)**:

- **Conditions for tool availability**: `config.json`’s `human_notification.enabled` must be true, and `HumanNotifier.from_config` must be able to construct **at least one actual delivery channel** (of `channels[]`, only `enabled: true` and registered `type` are eligible. `enabled: false` is skipped, and unregistered `type` is skipped with a warning logged)
- **Top-level only (supervisor gate)**: If `config.animas` contains an entry for **that Anima name** and `supervisor` is non-null, `HumanNotifier` is not granted and `call_human` cannot be used (subordinates escalate to their supervisor using `send_message`). **Anima not registered in `animas` do not pass through this gate**, so in theory, they may be granted `call_human` solely through notification channels. For safe operation, explicitly list all Anima in `animas` and ensure that only `supervisor: null` has human notifications enabled
- **Delivery method**: Send **in parallel** to each channel enabled in `HumanNotifier.notify` (`asyncio.gather(..., return_exceptions=True)`). Each channel returns either a success string or a failure string containing `ERROR`, and **an exception is caught for one channel while the others continue**
- **Supported channel types** (`human_notification.channels[].type`): `slack`, `chatwork`, `line`, `telegram`, `ntfy` (supports `@register_channel` in `core/notification/channels/*.py`). Multiple channels can be configured in parallel
- **Parameters**: `subject` and `body` are required. `priority` is optional. Allowed values are `low` / `normal` / `high` / `urgent` (defaults to `normal` if omitted). **Strings outside `PRIORITY_LEVELS` are normalized to `normal` within `HumanNotifier.notify`**
- **How priority appears**:
  - **Slack / Chatwork / LINE / Telegram**: For `high` / `urgent`, **`[HIGH]` / `[URGENT]`** are prepended (`priority.upper()`). They are not added for `low` / `normal`
  - **ntfy**: Set `low=2`, `normal=3`, `high=4`, and `urgent=5` in the HTTP `Priority` header. The body is the request body (up to approximately 4096 characters); the `Title` header contains the subject plus `(from Anima名)` if needed
- **Slack** (`channels/slack.py`):
  - **Bot Token + `channel`** (`chat.postMessage`) or **Incoming Webhook**. The body is formatted for Slack using `md_to_slack_mrkdwn`
  - **When using a Bot and `anima_name` is present**: The Anima name is passed to the API’s `username`, so **`(from Anima名)` is not added to the body** (in Webhook mode, `(from Anima名)` is added to the body). `icon_url` can also be added if the configuration and assets are in place
  - **Thread reply routing** (`reply_routing.py`): Saved to `notification_map.json` only when posted by a Bot, `anima_name` is non-empty, and the API response contains `ts`. The path is `{data_dir}/run/notification_map.json` (typically `~/.animaworks/run/`). Entries are discarded **no later than 7 days after creation**. Webhooks cannot be mapped because `ts` cannot be obtained
  - When routing, retrieve a thread summary through the Slack API if possible; if that fails, fall back to a summary of the saved notification text. External messages to the Inbox are `intent="question"`
- **Chatwork**: `room_id` accepts **numeric values only**. The body is converted using `md_to_chatwork` and formatted as `[info][title]…[/title]…[/info]`
- **LINE**: Push API. Text is truncated to a maximum of 5000 characters
- **Telegram**: `parse_mode=HTML`. The subject is `<b>…</b>`; the total is adjusted to fit within 4096 characters (truncated after escaping)
- **Credentials**: The base `NotificationChannel._resolve_credential_with_vault` is resolved in this order: the setting key’s env → `{キー}__{anima_name}` (vault/shared）→ plain key). Slack Bot also has a fallback to `get_credential("slack", "notification", …)` (see each `channels/*.py`)
- **Chat UI**: A streaming response sends the **`notification_sent`** event (via `core/anima/messaging.py`; this uses a separate route from external channels)
- **Logging**: When `call_human` runs, the unified activity log records **`human_notify`** (`via` is fixed in the implementation as `configured_channels`). `tool_result` is also retained. Priming’s “Pending Human Notifications” aggregates `human_notify` from the past 24 hours, up to 10 entries (`core/memory/priming/outbound.py`)
- **Other uses of HumanNotifier**: The framework has routes for notifying humans using the **same `HumanNotifier`**, such as when background tools complete (other than the `call_human` tool; the top-level Anima restriction still applies)
- **Mode S (CLI)**: The same type of notification can also be sent through `animaworks-tool call_human "件名" "本文" [--priority …]`

**call_human parameters (summary)**:
- `subject` and `body` are required. `priority` is optional (`low` / `normal` / `high` / `urgent`; defaults to `normal`. Invalid values are treated as `normal`)

---

## Escalation Message Templates

### Template 1: Block Report

```
send_message(
    to="上司の名前",
    content="""【ブロック報告】

■ 状況
タスク「月次レポート作成」がブロックされています。

■ 原因
売上データが格納されている /data/sales/ ディレクトリへの読み取り権限がありません。

■ 試行済み
- permissions.json を確認 → /data/sales/ は未許可
- 代替データソースを検索 → 該当なし

■ 依頼
/data/sales/ への読み取り権限の追加をお願いします。""",
    intent="report"
)
```

### Template 2: Decision Request

```
send_message(
    to="上司の名前",
    content="""【判断依頼】

■ 状況
タスク「顧客対応フロー改善」で2つの方針が考えられます。

■ 選択肢
A案: 既存フローを段階的に修正（工数: 小、リスク: 低、効果: 中）
B案: フローを全面刷新（工数: 大、リスク: 中、効果: 高）

■ 私の見解
A案を推奨します。理由: 現行フローの問題点は限定的であり、段階的修正で十分対応可能なため。

■ 依頼
方針の決定をお願いします。""",
    intent="question"
)
```

### Template 3: Technical Consultation with a Colleague

```
send_message(
    to="同僚の名前",
    content="""【技術相談】

Slack APIの rate limit に引っかかっています。

■ 状況
- 100件以上のメッセージを一括送信しようとしている
- 50件目あたりで 429 Too Many Requests が返される

■ 質問
Slack API の rate limit 回避策について知見はありますか？
バッチ処理の間隔を空ける方法を検討していますが、適切な間隔がわかりません。""",
    intent="question"
)
```

### Template 4: Urgent Report

If urgency is "High" and immediate human response is needed, also use `call_human` (the tool is only available to **top-level Anima** instances when `human_notification` is valid. Subordinate Anima instances should only use `send_message` to their supervisor).

```
send_message(
    to="上司の名前",
    content="""【緊急報告】

■ 状況
外部API（XXXサービス）から認証エラーが継続的に発生しています。

■ 影響
- YYYタスクが完全に停止
- ZZZタスクも同じAPIを使用しており影響の可能性あり

■ 試行済み
- リトライ3回実施 → すべて失敗
- APIキーの有効性は自分では確認できない

■ 依頼
APIキーの確認と、影響範囲の調査をお願いします。""",
    intent="report"
)
```

If immediate human notification is needed:
```
call_human(
    subject="【緊急】外部API認証エラー継続発生",
    body="XXXサービスから認証エラーが継続しています。YYYタスク停止中。APIキー確認をお願いします。",
    priority="urgent"
)
```

---

## Common Escalation Scenarios

### Scenario 1: Unclear Instructions

**Situation**: A supervisor instructed you to "create a report," but the target period, format, and submission destination are unclear.

**Correct approach**:
1. Organize what you can reasonably infer on your own
2. Ask specific questions about the unclear points

```
send_message(
    to="上司の名前",
    content="""レポート作成の件、以下を確認させてください。

1. 対象期間: 今月分でよろしいでしょうか？
2. フォーマット: 前回と同じMarkdown形式でよろしいでしょうか？
3. 提出先: knowledge/ に保存でよろしいでしょうか？

上記で問題なければ着手します。""",
    intent="question"
)
```

**What not to do**:
- Proceed with your own interpretation without confirming
- Reply only with "the instructions are unclear" (without specific questions)
- Omit `intent` in `send_message` (either `report` or `question` is required. Omitting it will cause an error)

### Scenario 2: Conflicting instructions from multiple supervisors

**Situation**: Your direct supervisor tells you to "prioritize A," but another Anima asks you to "do B first."

**Correct response**:
1. Prioritize the direct supervisor's instruction (MUST)
2. Report the situation to your direct supervisor

```
send_message(
    to="直属の上司の名前",
    content="""【優先順位の確認】

現在タスクAに着手中ですが、XXXさんからタスクBの優先依頼がありました。
指示通りタスクAを優先して進めますが、問題ないでしょうか？

タスクBの依頼内容: YYYの対応（XXXさんからの依頼）""",
    intent="question"
)
```

### Scenario 3: Repeated errors during work

**Situation**: Message sending to the Chatwork API has failed 3 times in a row.

**Correct response**:
1. Record the error details
2. If 3 retries do not resolve the issue, escalate
3. Proceed with other tasks that are not blocked

```
# エラーを記録
write_memory_file(
    path="state/current_state.md",
    content="## ブロック中\n\nChatwork API連続エラー\n- 1回目: 10:00 - 500 Internal Server Error\n- 2回目: 10:05 - 500 Internal Server Error\n- 3回目: 10:10 - 500 Internal Server Error\n\n上司に報告済み。他タスクに着手中。",
    mode="overwrite"
)

# 上司に報告
send_message(
    to="上司の名前",
    content="【ブロック報告】Chatwork APIが3回連続で500エラーを返しています。外部障害の可能性があります。復旧を待ちつつ、他のタスクに着手します。",
    intent="report"
)
```

### Scenario 4: Discovering an issue outside your scope of responsibility

**Situation**: During your work, you discover an inconsistency in data managed by another Anima.

**Correct response**:
1. Record the finding
2. Report to your direct supervisor (do not contact the Anima in another department directly)

```
send_message(
    to="上司の名前",
    content="""【情報共有】

作業中に以下の不整合を発見しました。私の担当外ですが共有します。

■ 発見内容
/data/reports/monthly.md の売上合計と /data/sales/summary.md の値が一致しません。
- monthly.md: 1,234,567円
- summary.md: 1,234,000円

■ 発見経緯
月次レポート作成中にデータ参照した際に気づきました。

対応の要否はお任せします。""",
    intent="report"
)
```

---

## Checklist when uncertain about a decision

Escalate with MUST if any of the following apply:

- [ ] A wrong decision would have irreversible consequences
- [ ] An operation beyond your permission scope is required
- [ ] It affects Anima in other departments
- [ ] No solution has been found after 15 minutes
- [ ] The same issue has occurred more than twice
- [ ] It involves security or data safety

You may attempt to resolve it yourself with MAY if any of the following apply:

- [ ] You have experience solving similar problems in the past
- [ ] The procedure manual (procedures/）) describes how to handle it
- [ ] The shared knowledge (common_knowledge/）) contains a solution
- [ ] It can be completed within your permission scope
- [ ] The impact of failure is limited
