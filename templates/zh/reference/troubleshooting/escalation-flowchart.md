# 遇到困难时的流程图

当问题发生时，用于判断“应该自己解决”还是“应该找谁商量/报告”的流程图。

本文档在判断犹豫时参照。对于明显可以自己解决的问题，不需要使用此流程图。

---

## 判断流程图

问题发生后，按以下步骤依次判断。

### Step 1: 确定问题的类型

将问题归类为以下之一:

| 类型 | 说明 | 示例 |
|------|------|----|
| **A. 技术问题** | 与工具或系统运行相关的问题 | 工具错误、权限不足、文件不存在 |
| **B. 业务问题** | 与任务推进方式或判断相关的问题 | 规格不明、优先级判断、阻塞 |
| **C. 人际问题** | 与其他Anima协作相关的问题 | 无回复、指令矛盾、负责人不明 |
| **D. 紧急问题** | 需要立即处理的问题 | 数据丢失风险、安全担忧 |

### Step 2: 判定紧急程度

| 紧急程度 | 标准 | 应对 |
|--------|------|------|
| **高** | 若搁置会有数据丢失・安全风险 | MUST: 立即向上司报告 |
| **高** | 其他Anima的工作完全停止 | MUST: 立即向上司报告 |
| **中** | 自己的工作被阻塞，但可以着手其他任务 | SHOULD: 1小时内向上司报告 |
| **低** | 工作效率下降但可以推进 | MAY: 在下次心跳时报告 |

**紧急程度为“高”时** → 进入 Step 5（立即升级）

### Step 3: 尝试自行解决

按以下步骤尝试自行解决。每一步若已解决则在此结束。

1. **搜索记忆**
   ```
   search_memory(query="問題に関連するキーワード", scope="all")
   ```
   - 确认过去是否经历过同样的问题
   - 确认操作手册（procedures/）中是否有应对方法
   - 如果想**回忆刚才做过的事**（最近的工具结果・邮件正文・搜索结果等），尝试 `scope="activity_log"`。`scope="all"` 中 activity_log 也会经由 BM25 进行 RRF 合并，但显式限定到 activity_log 更容易减少噪声

2. **搜索共享知识**
   ```
   search_memory(query="問題に関連するキーワード", scope="common_knowledge")
   ```
   - 确认 `troubleshooting/common-issues.md` 中是否有对应的问题

3. **考虑其他方法**
   - 思考是否有达成目的的替代手段
   - 若权限不足则走其他路径，若工具错误则换用其他工具

4. **自行解决的时限**
   - 技术问题: 15分钟内未解决则升级
   - 业务问题: 判断犹豫时立即升级（避免错误判断的风险）
   - 人际问题: 重试1次后升级

### Step 4: 确定升级对象

| 问题类型 | 首先商量的对象 | 商量后仍未解决时 |
|-----------|----------------|----------------------|
| **A. 技术问题** | 同僚（同一专业领域时） | 上司 |
| **B. 业务问题** | 上司 | ― |
| **C. 人际问题** | 上司（请求居中协调） | ― |
| **D. 紧急问题** | 上司（立即） | ― |

**判断标准:**
- 可以找同僚商量的条件: 拥有同一上司的同僚，且问题与对方专业领域相关时
- 必须向上司报告的条件: 需要业务判断、涉及其他部门、紧急程度高
- 不直接联系其他部门的Anima（MUST: 经由上司）

### Step 5: 执行升级

报告消息 MUST 包含以下要素：

1. **情况**：发生了什么
2. **原因**：认为可能是什么原因（如果不清楚，则写“正在调查原因”）
3. **尝试**：自己尝试过什么
4. **请求**：希望上司做什么（判断、授予权限、协调等）

**send_message 的限制（符合实现）**：
- `intent` MUST 为 `report`（报告）或 `question`（提问）。不可省略。`intent="delegation"` 会被**拒绝**（任务委派仅可使用 `delegate_task`）
- acknowledgment（确认回复）、感谢和 FYI 不可通过 DM 发送。请通过 `animaworks-tool board post <channel> <text>` 分享到 Board
- 同一 run 中，只能向同一收件人发送 1 条 DM。收件人数不限
- 不会因按小时或按天计算的发送额度，或对话深度而拒绝发送。详情请参阅 `communication/sending-limits.md`
- **收件人**：Anima 名称或人员别名（如果已在 config 中配置，则会向 Slack/Chatwork 等外部发送）
- **聊天期间**：直接以文本回复人类用户。`send_message` 仅用于发送给其他 Anima（或通过已配置的别名发送到外部）
- **联系人员**（例如未设置别名，导致 `send_message` 无法送达的收件人）：如果是顶层 Anima 且已配置通知，请使用 `call_human`（见下文）
- 回复线程时，指定 `reply_to` 和 `thread_id` 以保留上下文
- 紧急程度为“高”且需要人员立即处理时，请考虑使用 `call_human`（`subject`、`body`、`priority`）

**Board 帖子的限制**（用于向整个团队分享）：
- 未设置元数据的频道（general、ops 等）所有人都可以使用。成员制频道仅允许成员发帖（ACL）。如果没有访问权限，可通过 `animaworks-tool internal manage-channel info <channel>` 查看成员
- 正文中可以使用 `@名前` 提及用户。被提及者会收到 DM 通知

**call_human 与人员通知基础设施（符合 `core/notification/` 实现）**：

- **工具的启用条件**：`config.json` 的 `human_notification.enabled` 为 true，且 `HumanNotifier.from_config` 确实成功构建了至少一个发送频道（仅适用于 `channels[]` 中 `enabled: true` 且已注册的 `type`。`enabled: false` 会被跳过；未注册的 `type` 会记录警告日志后跳过）
- **仅限顶层（supervisor 门控）**：当 `config.animas` 中存在该 Anima 名称对应的条目，且 `supervisor` 非 null 时，不会授予 `HumanNotifier`，也无法使用 `call_human`（部下应通过 `send_message` 向上司升级）。**未登记在 `animas` 中的 Anima 不受此门控限制**，因此理论上可能仅凭通知频道就获得 `call_human`。为确保安全，运维时应将所有 Anima 明确列入 `animas`，并确保只有 `supervisor: null` 具有人员通知功能
- **发送方式**：向每个已启用的 `HumanNotifier.notify` **并行发送**（`asyncio.gather(..., return_exceptions=True)`）。每个频道都会返回成功文本，或包含 `ERROR` 的失败文本；**某个频道发生异常时会被捕获并忽略，其他频道继续处理**
- **支持的频道类型**（`human_notification.channels[].type`）：`slack`、`chatwork`、`line`、`telegram`、`ntfy`（对应 `core/notification/channels/*.py` 的 `@register_channel`）。可以并行定义多个频道
- **参数**：`subject`、`body` 为必填项。`priority` 为可选项。枚举值为 `low` / `normal` / `high` / `urgent`（省略时为 `normal`）。**`PRIORITY_LEVELS` 之外的字符串会在 `HumanNotifier.notify` 内规范化为 `normal`**
- **优先级的显示方式**：
  - **Slack / Chatwork / LINE / Telegram**：当为 `high` / `urgent` 时，会在开头添加 **`[HIGH]` / `[URGENT]`**（`priority.upper()`）。`low` / `normal` 时不添加
  - **ntfy**：在 HTTP 标头 `Priority` 中设置 `low=2`、`normal=3`、`high=4`、`urgent=5`。正文位于请求体中（最多约 4096 个字符）；在 `Title` 标头中设置主题，并在需要时设置 `(from Anima名)`
- **Slack**（`channels/slack.py`）：
  - **Bot Token + `channel`**（`chat.postMessage`）或 **Incoming Webhook**。正文会通过 `md_to_slack_mrkdwn` 格式化为适用于 Slack 的内容
  - **使用 Bot 且存在 `anima_name` 时**：由于会将 Anima 名称传给 API 的 `username`，因此**正文中不会添加 `(from Anima名)`**（Webhook 模式下会在正文中添加 `(from Anima名)`）。配置和资源齐全时，也可以添加 `icon_url`
  - **线程回复路由**（`reply_routing.py`）：仅当使用 Bot 发帖、`anima_name` 不为空，且 API 响应中包含 `ts` 时，才会保存到 `notification_map.json`。路径为 `{data_dir}/run/notification_map.json`（通常为 `~/.animaworks/run/`）。条目会在**创建后最多 7 天**被丢弃。Webhook 无法获取 `ts`，因此无法建立映射
  - 路由时，如果可能，会通过 Slack API 获取线程摘要；失败时则使用已保存通知文本的摘要作为后备。发往 Inbox 的外部消息为 `intent="question"`
- **Chatwork**：`room_id` 仅允许使用**数字**。正文会转换为 `md_to_chatwork`，并采用 `[info][title]…[/title]…[/info]` 格式
- **LINE**：Push API。文本会截断至最多 5000 个字符
- **Telegram**：`parse_mode=HTML`。主题为 `<b>…</b>`，并会调整为总长不超过 4096 个字符（转义后截断）
- **凭据**：基础 `NotificationChannel._resolve_credential_with_vault` 的获取顺序为**配置键对应的 env → `{キー}__{anima_name}`（vault/shared）→ 裸键）**。Slack Bot 还会回退到 `get_credential("slack", "notification", …)`（参阅各 `channels/*.py`）
- **聊天 UI**：流式响应中会发送 **`notification_sent`** 事件（通过 `core/anima/messaging.py`；与外部频道走不同路径）
- **记录**：执行 `call_human` 时，会在统一活动日志中记录 **`human_notify`**（`via` 在实现中固定为 `configured_channels`）。同时也会保留 `tool_result`。Priming 中的“Pending Human Notifications”会汇总**过去 24 小时内最多 10 条** `human_notify`（`core/memory/priming/outbound.py`）
- **HumanNotifier 的其他用途**：后台工具完成等情况下，框架也会通过**同一 `HumanNotifier`** 向人员发送消息（不使用 `call_human` 工具；同样仅限顶层 Anima）
- **Mode S（CLI）**：通过 `animaworks-tool call_human "件名" "本文" [--priority …]` 也可以发送同类通知

**call_human 参数（摘要）**：
- `subject`、`body` 为必填项。`priority` 为可选项（`low` / `normal` / `high` / `urgent`，默认值为 `normal`。无效值按 `normal` 处理）

---

## 升级消息的模板

### 模板1: 阻塞报告

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

### 模板2: 请求判断

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

### 模板3: 向同僚的技术咨询

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

### 模板4: 紧急报告

紧急程度为“高”且需要人类立即处理时，也并用 `call_human`（**仅当为顶层 Anima** 且 `human_notification` 有效时工具可用。部下 Anima 仅限向上司的 `send_message`）。

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

需要向人类立即通知时:
```
call_human(
    subject="【緊急】外部API認証エラー継続発生",
    body="XXXサービスから認証エラーが継続しています。YYYタスク停止中。APIキー確認をお願いします。",
    priority="urgent"
)
```

---

## 常见的升级场景

### 场景1: 指令内容不明确

**状况**: 上司指示“做个报告”，但对象期间・格式・提交对象不明。

**正确应对**:
1. 整理自己能够推测的范围
2. 具体提问不明之处

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

**不可取的应对**:
- 不确认就按自己的解释推进工作
- 只回复“指令不明确”（没有具体提问）
- 在 `send_message` 中省略 `intent`（`report` / `question` 之一为必填。省略会出错）

### 场景2: 多个上司给出矛盾指令

**状况**: 直属上司说“优先A”，但另一个Anima请求“先做B”。

**正确应对**:
1. 优先直属上司的指令（MUST）
2. 向直属上司报告状况

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

### 场景3：作业中连续发生错误

**情况**：向Chatwork API发送消息连续失败3次。

**正确应对**：
1. 记录错误内容
2. 重试3次仍无法解决则上报升级
3. 着手处理未被阻塞的其他任务

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

### 场景4：发现自己责任范围之外的问题

**情况**：在自己的作业过程中，发现其他Anima管理的数据存在不一致。

**正确应对**：
1. 记录发现的事实
2. 向直属上司报告（不要直接联系其他部门的Anima）

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

## 判断迷茫时的检查清单

符合以下任一情况时，必须（MUST）上报升级：

- [ ] 如果此判断出错，将产生不可挽回的影响
- [ ] 需要超出自己权限范围的操作
- [ ] 会影响其他部门的Anima
- [ ] 15分钟以上仍未找到解决方案
- [ ] 同一问题已发生2次以上
- [ ] 涉及安全性或数据的安全性

符合以下情况时，可以（MAY）尝试自行解决：

- [ ] 过去有解决类似问题的经验
- [ ] 操作手册（procedures/）中记载了应对方法
- [ ] 共享知识（common_knowledge/）中有解决方法
- [ ] 在自己的权限范围内即可完成
- [ ] 即使失败影响也有限
