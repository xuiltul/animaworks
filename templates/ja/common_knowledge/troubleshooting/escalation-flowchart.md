# 困ったときのフローチャート

問題が発生したときに「自分で解決すべきか」「誰かに相談・報告すべきか」を判断するためのフローチャート。

このドキュメントは判断に迷ったときに参照する。明らかに自分で解決できる問題にこのフローチャートは不要。

---

## 判断フローチャート

問題が発生したら、以下のステップを順に判断する。

### Step 1: 問題の種類を特定する

問題を以下のいずれかに分類する:

| 種類 | 説明 | 例 |
|------|------|----|
| **A. 技術的問題** | ツールやシステムの動作に関する問題 | ツールエラー、権限不足、ファイル不在 |
| **B. 業務的問題** | タスクの進め方や判断に関する問題 | 仕様不明、優先順位の判断、ブロック |
| **C. 対人的問題** | 他のAnimaとの連携に関する問題 | 返答がない、指示が矛盾、担当不明 |
| **D. 緊急問題** | 即座の対応が必要な問題 | データ損失の危険、セキュリティ懸念 |

### Step 2: 緊急度を判定する

| 緊急度 | 基準 | 対応 |
|--------|------|------|
| **高** | 放置するとデータ損失・セキュリティリスクがある | MUST: 即座に上司に報告 |
| **高** | 他のAnimaの作業が完全に止まる | MUST: 即座に上司に報告 |
| **中** | 自分の作業がブロックされるが、他タスクに着手可能 | SHOULD: 1時間以内に上司に報告 |
| **低** | 作業効率が落ちるが進行は可能 | MAY: 次のハートビートで報告 |

**緊急度が「高」の場合** → Step 5 に進む（即座にエスカレーション）

### Step 3: 自力解決を試みる

以下の手順で自力解決を試みる。各ステップで解決したらそこで終了する。

1. **記憶を検索する**
   ```
   search_memory(query="問題に関連するキーワード", scope="all")
   ```
   - 過去に同じ問題を経験していないか確認する
   - 手順書（procedures/）に対処方法がないか確認する
   - **さっきやったことを思い出したい**（直近のツール結果・メール本文・検索結果など）なら `scope="activity_log"` を試す。`scope="all"` でも activity_log は BM25 経由で RRF マージされますが、明示的に activity_log に絞るとノイズが減りやすい

2. **共有知識を検索する**
   ```
   search_memory(query="問題に関連するキーワード", scope="common_knowledge")
   ```
   - `troubleshooting/common-issues.md` に該当する問題がないか確認する

3. **別のアプローチを検討する**
   - 目的を達成する代替手段がないか考える
   - 権限不足なら別の経路、ツールエラーなら別のツール

4. **自力解決の制限時間**
   - 技術的問題: 15分以内に解決しなければエスカレーション
   - 業務的問題: 判断に迷ったら即エスカレーション（間違った判断のリスクを避ける）
   - 対人的問題: 1回のリトライ後にエスカレーション

### Step 4: エスカレーション先を決定する

| 問題の種類 | まず相談する相手 | 相談しても解決しない場合 |
|-----------|----------------|----------------------|
| **A. 技術的問題** | 同僚（同じ専門分野の場合） | 上司 |
| **B. 業務的問題** | 上司 | ― |
| **C. 対人的問題** | 上司（仲介を依頼） | ― |
| **D. 緊急問題** | 上司（即座に） | ― |

**判断基準:**
- 同僚に相談してよい条件: 同じ上司を持つ同僚で、相手の専門分野に関連する問題である場合
- 上司に報告 MUST な条件: 業務判断が必要、他部署が関与、緊急度が高い
- 他部署のAnimaには直接連絡しない（MUST: 上司経由）

### Step 5: エスカレーションを実行する

報告メッセージには以下の要素を MUST で含める:

1. **状況**: 何が起きているか
2. **原因**: 何が原因と考えられるか（不明なら「原因調査中」）
3. **試行**: 自分で何を試したか
4. **依頼**: 上司に何をしてほしいか（判断、権限付与、仲介 等）

**send_message の制約（実装準拠）**:
- `intent` は MUST: `report`（報告）, `question`（質問）のいずれか。省略不可。`intent="delegation"` は**拒否**される（タスク委譲は `delegate_task` のみ）
- acknowledgment（確認応答）・感謝・FYI は DM 不可。Board（post_channel）を使用する
- 1 run あたりの DM 宛先数はロール/status.json で設定された上限（general/ops は2人、engineer は5人、manager は10人など）。同一宛先へは1通のみ。上限を超える伝達は Board を使用する
- DM と Board は**同一のアウトバウンド予算**を共有する（時間あたり・24時間あたりの制限あり）。詳細は `communication/sending-limits.md` 参照
- **宛先**: Anima名、または人間エイリアス（config で設定済みの場合は Slack/Chatwork 等へ外部配信）
- **チャット中**: 人間ユーザーへの返答は直接テキストで行う。`send_message` は他Anima宛て（または設定済みエイリアス経由の外部）にのみ使用する
- **人間への連絡**（エイリアス未設定など `send_message` で届かない宛先）: トップレベル Anima かつ通知設定がある場合は `call_human` を使用する（下記）
- スレッド返信時は `reply_to` と `thread_id` を指定して文脈を維持する
- 緊急度が「高」で人間の即時対応が必要な場合は `call_human` を検討する（`subject`, `body`, `priority`）

**post_channel（Board）の制約**（3人以上への伝達時に使用）:
- メタ未設定のチャネル（general, ops 等）は全員利用可能。メンバー制チャネルはメンバーのみ投稿可能（ACL）。アクセス権がない場合は `manage_channel(action="info", channel="チャネル名")` でメンバーを確認できる
- 同一チャネルへは1 run につき1投稿まで。同一チャネルへの連投はクールダウン（`config.json` の `heartbeat.channel_post_cooldown_s`、デフォルト300秒）が必要
- 本文に `@名前` でメンション可能。メンション先には DM 通知が届く

**call_human と人間通知基盤（`core/notification/` 実装準拠）**:

- **ツールの有効条件**: `config.json` の `human_notification.enabled` が true で、かつ `HumanNotifier.from_config` が **実際に1件以上の送信チャネル**を構築できたこと（`channels[]` のうち `enabled: true` かつ登録済み `type` のみが対象。`enabled: false` はスキップ、未登録の `type` は警告ログのうえスキップ）
- **トップレベル限定（supervisor ゲート）**: `config.animas` に **その Anima 名のエントリがあり**、かつ `supervisor` が非 null のとき、`HumanNotifier` は付与されず `call_human` は使えない（部下は上司へ `send_message` でエスカレーション）。**`animas` に未登録の Anima はこのゲートを通らない**ため、理論上は通知チャネルだけで `call_human` が付く可能性がある。運用では全 Anima を `animas` に明示し、`supervisor: null` のみが人間通知を持つようにすると安全
- **送信方式**: `HumanNotifier.notify` が有効な各チャネルへ **並列送信**（`asyncio.gather(..., return_exceptions=True)`）。チャネルごとに成功文字列または `ERROR` を含む失敗文字列が返り、**例外は1チャネルで握りつぶされ他は継続**
- **対応チャネル種別**（`human_notification.channels[].type`）: `slack`, `chatwork`, `line`, `telegram`, `ntfy`（`core/notification/channels/*.py` の `@register_channel` に対応）。複数チャネルを並列定義可能
- **パラメータ**: `subject`, `body` は必須。`priority` は任意。列挙は `low` / `normal` / `high` / `urgent`（省略時 `normal`）。**`PRIORITY_LEVELS` 外の文字列は `HumanNotifier.notify` 内で `normal` に正規化**
- **優先度の見え方**:
  - **Slack / Chatwork / LINE / Telegram**: `high` / `urgent` のとき先頭に **`[HIGH]` / `[URGENT]`**（`priority.upper()`）。`low` / `normal` では付与しない
  - **ntfy**: HTTP ヘッダ `Priority` に `low=2`, `normal=3`, `high=4`, `urgent=5` を設定。本文はリクエストボディ（最大約4096文字）、`Title` ヘッダに件名＋必要なら `(from Anima名)`
- **Slack**（`channels/slack.py`）:
  - **Bot Token + `channel`**（`chat.postMessage`）または **Incoming Webhook**。本文は `md_to_slack_mrkdwn` で Slack 向けに整形される
  - **Bot かつ `anima_name` がある場合**: API の `username` に Anima 名を渡すため、**本文側の `(from Anima名)` は付けない**（Webhook モードでは本文に `(from Anima名)` を付与）。設定とアセットが揃えば `icon_url` も付与可能
  - **スレッド返信ルーティング**（`reply_routing.py`）: Bot で投稿し、かつ `anima_name` が空でなく、API 応答に `ts` があるときだけ `notification_map.json` に保存。パスは `{data_dir}/run/notification_map.json`（通常は `~/.animaworks/run/`）。エントリは **作成から最大7日** で破棄。Webhook は `ts` が取れずマッピング不可
  - ルーティング時は可能なら Slack API でスレッド要約を取得し、失敗時は保存済み通知文の要約でフォールバック。Inbox への外部メッセージは `intent="question"`
- **Chatwork**: `room_id` は **数値のみ** 許可。本文は `md_to_chatwork` 変換のうえ `[info][title]…[/title]…[/info]` 形式
- **LINE**: Push API。テキストは最大 5000 文字に切り詰め
- **Telegram**: `parse_mode=HTML`。件名は `<b>…</b>`、全体 4096 文字以内に調整（エスケープ後に切り詰め）
- **クレデンシャル**: 基底 `NotificationChannel._resolve_credential_with_vault` は **設定キーの env → `{キー}__{anima_name}`（vault/shared）→ 素のキー**の順。Slack Bot はこれに加え `get_credential("slack", "notification", …)` のフォールバックあり（各 `channels/*.py` 参照）
- **チャット UI**: ストリーミング応答で **`notification_sent`** イベントが送られる（`core/_anima_messaging.py` 経由。外部チャネルとは別経路）
- **記録**: `call_human` 実行時、統一アクティビティログに **`human_notify`**（`via` は実装上固定で `configured_channels`）。あわせて `tool_result` も残る。Priming の「Pending Human Notifications」は **過去24時間・最大10件**の `human_notify` を集約（`core/memory/priming/outbound.py`）
- **その他の HumanNotifier 利用**: バックグラウンドツール完了など、**同一の `HumanNotifier`** でフレームワークが人間へ送る経路がある（`call_human` ツール以外。トップレベル Anima に限る点は同じ）
- **Mode S（CLI）**: `animaworks-tool call_human "件名" "本文" [--priority …]` でも同系統の通知を送れる

**call_human のパラメータ（要約）**:
- `subject`, `body` は必須。`priority` は任意（`low` / `normal` / `high` / `urgent`、デフォルト `normal`。不正値は `normal` 扱い）

---

## エスカレーションメッセージのテンプレート

### テンプレート1: ブロック報告

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

### テンプレート2: 判断依頼

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

### テンプレート3: 同僚への技術相談

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

### テンプレート4: 緊急報告

緊急度が「高」で人間の即時対応が必要な場合は `call_human` も併用する（**トップレベル Anima** かつ `human_notification` が有効なときのみツールが利用可能。部下 Anima は上司への `send_message` に留める）。

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

人間への即時通知が必要な場合:
```
call_human(
    subject="【緊急】外部API認証エラー継続発生",
    body="XXXサービスから認証エラーが継続しています。YYYタスク停止中。APIキー確認をお願いします。",
    priority="urgent"
)
```

---

## よくあるエスカレーションシナリオ

### シナリオ1: 指示内容が不明確

**状況**: 上司から「レポートを作って」と指示されたが、対象期間・フォーマット・提出先が不明。

**正しい対応**:
1. 自分で推定できる範囲を整理する
2. 不明点を具体的に質問する

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

**やってはいけない対応**:
- 確認せずに独自解釈で作業を進める
- 「指示が不明確です」とだけ返す（具体的な質問がない）
- `send_message` で `intent` を省略する（`report` / `question` のいずれかが必須。省略するとエラーになる）

### シナリオ2: 複数の上司から矛盾する指示

**状況**: 直属の上司から「Aを優先して」と言われたが、別のAnimaから「Bを先にやって」と依頼された。

**正しい対応**:
1. 直属の上司の指示を優先する（MUST）
2. 状況を直属の上司に報告する

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

### シナリオ3: 作業中にエラーが連続発生

**状況**: Chatwork APIへのメッセージ送信が3回連続で失敗した。

**正しい対応**:
1. エラー内容を記録する
2. 3回リトライしても解決しなければエスカレーションする
3. ブロックされていない他のタスクに着手する

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

### シナリオ4: 自分の責任範囲外の問題を発見

**状況**: 自分の作業中に、他のAnimaが管理するデータに不整合を発見した。

**正しい対応**:
1. 発見事実を記録する
2. 直属の上司に報告する（他部署のAnimaには直接連絡しない）

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

## 判断に迷ったときのチェックリスト

以下のいずれかに該当する場合は、MUST でエスカレーションすること:

- [ ] この判断を間違えた場合、取り返しがつかない影響がある
- [ ] 自分の権限の範囲を超える操作が必要
- [ ] 他部署のAnimaに影響する
- [ ] 15分以上解決策が見つからない
- [ ] 同じ問題が2回以上発生している
- [ ] セキュリティやデータの安全性に関わる

以下に該当する場合は、自力解決を MAY で試みてよい:

- [ ] 過去に同様の問題を解決した経験がある
- [ ] 手順書（procedures/）に対処方法が記載されている
- [ ] 共有知識（common_knowledge/）に解決方法がある
- [ ] 自分の権限範囲内で完結する
- [ ] 失敗しても影響が限定的
