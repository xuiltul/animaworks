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

报告消息中 MUST 包含以下要素:

1. **状况**: 发生了什么
2. **原因**: 认为是什么原因（不明则写“原因调查中”）
3. **尝试**: 自己尝试了什么
4. **请求**: 希望上司做什么（判断、授予权限、居中协调 等）

**send_message 的约束（实现依据）**:
- `intent` 是 MUST: `report`（报告）、`question`（提问）之一。不可省略。`intent="delegation"` 会被**拒绝**（任务委派仅限 `delegate_task`）
- acknowledgment（确认应答）・感谢・FYI 不可用 DM。使用 Board（post_channel）
- 同一 run 中向同一收件人只能发送1封 DM。收件人数量无上限
- 没有按时间・按天计算的发送预算或基于对话深度的发送拒绝。详情参照 `communication/sending-limits.md`
- **收件人**: Anima名，或人类别名（若已在 config 中配置，则通过 Slack/Chatwork 等外部投递）
- **聊天中**: 对人类的回复直接以文本进行。`send_message` 仅用于发往其他Anima（或经由已配置别名的外部）
- **联系人类**（别名未配置等 `send_message` 无法送达的收件人）: 若为顶层 Anima 且已配置通知，则使用 `call_human`（见下文）
- 线程回复时指定 `reply_to` 和 `thread_id` 以维持上下文
- 紧急程度为“高”且需要人类立即处理时，考虑 `call_human`（`subject`, `body`, `priority`）

**post_channel（Board）的约束**（用于向整个团队共享）:
- 未设置元数据的频道（general, ops 等）所有人可用。成员制频道仅成员可投稿（ACL）。若无访问权限，可通过 `manage_channel(action="info", channel="チャネル名")` 确认成员
- 同一频道每个 run 只能投稿1次。run 之间没有投稿 cooldown
- 正文中可用 `@名前` 提及。被提及者会收到 DM 通知

**call_human 与人类通知基础设施（`core/notification/` 实现依据）**:

- **工具启用条件**: `config.json` 的 `human_notification.enabled` 为 true，且 `HumanNotifier.from_config` **实际构建了1个以上的发送频道**（`channels[]` 中仅 `enabled: true` 且已注册的 `type` 为对象。`enabled: false` 跳过，未注册的 `type` 记录警告日志后跳过）
- **仅限顶层（supervisor 门控）**: 若 `config.animas` 中**有该 Anima 名的条目**，且 `supervisor` 非 null，则 `HumanNotifier` 不被授予，`call_human` 不可用（部下通过 `send_message` 向上司升级）。**未在 `animas` 中注册的 Anima 不经过此门控**，因此理论上仅凭通知频道就可能获得 `call_human`。运维中安全做法是在 `animas` 中显式列出所有 Anima，仅让 `supervisor: null` 拥有人类通知
- **发送方式**: 向 `HumanNotifier.notify` 有效的各频道**并行发送**（`asyncio.gather(..., return_exceptions=True)`）。每个频道返回成功字符串或包含 `ERROR` 的失败字符串，**异常在一个频道被吞掉，其他继续**
- **支持的频道类型**（`human_notification.channels[].type`）: `slack`, `chatwork`, `line`, `telegram`, `ntfy`（对应 `core/notification/channels/*.py` 的 `@register_channel`）。可并行定义多个频道
- **参数**: `subject`, `body` 为必填。`priority` 为可选。枚举为 `low` / `normal` / `high` / `urgent`（省略时 `normal`）。**`PRIORITY_LEVELS` 之外的字符串在 `HumanNotifier.notify` 内被规范化为 `normal`**
- **优先级的呈现方式**:
  - **Slack / Chatwork / LINE / Telegram**: `high` / `urgent` 时在开头加上 **`[HIGH]` / `[URGENT]`**（`priority.upper()`）。`low` / `normal` 时不加
  - **ntfy**: 在 HTTP 头 `Priority` 中设置 `low=2`, `normal=3`, `high=4`, `urgent=5`。正文为请求体（最大约4096字符），`Title` 头中为标题＋必要时 `(from Anima名)`
- **Slack**（`channels/slack.py`）:
  - **Bot Token + `channel`**（`chat.postMessage`）或 **Incoming Webhook**。正文由 `md_to_slack_mrkdwn` 针对 Slack 格式化
  - **Bot 且存在 `anima_name` 时**: 由于向 API 的 `username` 传入 Anima 名，**正文侧不加 `(from Anima名)`**（Webhook 模式下在正文中加 `(from Anima名)`）。若配置与资源齐备，也可加 `icon_url`
  - **线程回复路由**（`reply_routing.py`）: 仅当以 Bot 投稿、且 `anima_name` 非空、且 API 响应中有 `ts` 时，保存到 `notification_map.json`。路径为 `{data_dir}/run/notification_map.json`（通常为 `~/.animaworks/run/`）。条目**自创建起最长7天**后丢弃。Webhook 无法获取 `ts`，无法映射
  - 路由时尽可能通过 Slack API 获取线程摘要，失败时回退到已保存通知文的摘要。发往 Inbox 的外部消息为 `intent="question"`
- **Chatwork**: `room_id` 仅允许**数字**。正文经 `md_to_chatwork` 转换后为 `[info][title]…[/title]…[/info]` 格式
- **LINE**: Push API。文本截断至最大 5000 字符
- **Telegram**: `parse_mode=HTML`。标题为 `<b>…</b>`，整体调整至 4096 字符以内（转义后截断）
- **凭据**: 基础 `NotificationChannel._resolve_credential_with_vault` 按 **配置键的 env → `{キー}__{anima_name}`（vault/shared）→ 原始键**的顺序。Slack Bot 在此基础上还有 `get_credential("slack", "notification", …)` 的回退（参照各 `channels/*.py`）
- **聊天 UI**: 流式响应中发送 **`notification_sent`** 事件（经由 `core/anima/messaging.py`。与外部频道是不同路径）
- **记录**: `call_human` 执行时，在统一活动日志中记录 **`human_notify`**（`via` 在实现上固定为 `configured_channels`）。同时也会留下 `tool_result`。Priming 的“Pending Human Notifications”汇总**过去24小时・最多10件**的 `human_notify`（`core/memory/priming/outbound.py`）
- **其他 HumanNotifier 用途**: 存在后台工具完成等由框架通过**同一 `HumanNotifier`** 向人类发送的路径（`call_human` 工具以外。仅限顶层 Anima 这一点相同）
- **Mode S（CLI）**: `animaworks-tool call_human "件名" "本文" [--priority …]` 中也可发送同系统的通知

**call_human 的参数（摘要）**:
- `subject`, `body` 为必填。`priority` 为可选（`low` / `normal` / `high` / `urgent`，默认 `normal`。非法值按 `normal` 处理）

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
