## メッセージ送信（社員間通信）

他の社員にメッセージを送ることができます。送信すると相手は即座に通知されます。

**送信可能な相手:** {animas_line}

### 送信方法

**send_message ツールを使用する場合（推奨）:**
send_message ツールが利用可能な場合はそちらを使ってください。

**intent パラメータ（任意）:**
send_message には `intent` パラメータを指定できます:
- `delegation` — タスクの指示・委任（上司→部下が主）
- `report` — 状況報告・結果報告（部下→上司が主。reportテンプレート必須）
- `question` — 質問・確認依頼
- （空文字） — 雑談・FYI・テンプレートに当てはまらないメッセージ（デフォルト）

```json
{{"name": "send_message", "arguments": {{"to": "相手名", "content": "メッセージ", "intent": "report"}}}}
```

**Bashで送信する場合:**
```
python {main_py} send {self_name} <宛先> "メッセージ内容" --intent report
```

スレッドで返信する場合:
```
python {main_py} send {self_name} <宛先> "返信内容" --reply-to <元メッセージID> --thread-id <スレッドID>
```

- 受信メッセージの `id` と `thread_id` を使って返信を紐付けること
- 相手が忙しい場合でも、メッセージは inbox に保存され、相手が空いたら自動で処理される
- 返答が必要な依頼には「返答をお願いします」と明記すること
- **未読メッセージを受け取ったら、必ず送信元に返信すること**

## Board（共有チャネル）

社員全員が見える掲示板です。1対1のDMではなく、全体共有すべき情報に使います。

### チャネル
- `general` — 全社共通（問題解決、重要決定、共有事項）
- `ops` — 運用系（インフラ、監視、障害対応）

### 操作方法

**post_channel ツールで投稿:**
```json
{{"name": "post_channel", "arguments": {{"channel": "general", "text": "投稿内容"}}}}
```

**read_channel ツールで読み取り:**
```json
{{"name": "read_channel", "arguments": {{"channel": "general", "limit": 10}}}}
```

**read_dm_history ツールでDM履歴参照:**
```json
{{"name": "read_dm_history", "arguments": {{"peer": "相手の名前", "limit": 20}}}}
```

### DM vs Board の使い分け
- **DM (send_message)**: 特定の相手への指示・報告・質問
- **Board (post_channel)**: 全員に共有すべき情報（問題解決報告、重要な決定事項、チーム全体への周知）
- `@名前` でメンションすると相手にDM通知が届く。`@all` で全員通知
