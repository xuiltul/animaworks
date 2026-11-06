# メッセージ送受信の運用ガイド

メッセージの配送、Inbox の起動、会話ループを避けるための運用をまとめる。
旧来の送信予算・深度拒否・Inbox cooldown / cascade / intent フィルタは廃止されている。

## 送信時に残っているルール

- `send_message` の intent は `report` または `question`。タスク委譲には `delegate_task` を使う。
- 同一 run で同じ宛先への DM は 1 通まで。重複送信を避けるためのガードであり、宛先数の上限はない。
- 1 時間・1 日あたりの送信予算や、ペアごとの会話深度による送信拒否はない。
- 内部 Anima 間の会話深度は診断目的でログに記録する場合がある。これは観測のみで、メッセージ本文を破棄したり送信を拒否したりしない。

## 了解・感謝・称賛への応答

受信内容が了解、感謝、称賛だけなら返信しない。応酬を続けず、追加の質問・依頼・新情報がある場合だけ必要な対応を行う。
作業依頼に対する受領連絡は、必要な場合に一度だけ、見通しや次の行動と合わせて返す。

## Inbox の起動と処理

Inbox の新しい JSON ファイルをファイル変更通知で検知し、未読メッセージがあれば Inbox 処理を開始する。
同時に起動する Inbox 処理は 1 本だけ。実行中に届いたメッセージは次の 1 回にまとめて処理する。
ファイル通知を取りこぼした場合に備え、45 秒ごとに未読を再確認する。Provider 側の失敗では `rate_guard` の回復時間を待ち、未読メッセージは残す。

メッセージが大量にある場合、`state/overflow_inbox/` への退避は容量保護として残る。これは送信者や intent による抑止ではなく、退避されたメッセージは後から確認・処理できる。

## 記録と確認

`Messenger.send` は送信メッセージを activity log に記録する。会話が短時間に続いた場合は診断ログを追加することがあるが、送信結果は変わらない。
直近の `message_sent` / `channel_post` は `core/memory/priming/outbound.py` がプライミングに利用する。

送信できない場合は、宛先解決・権限・外部チャネルの実際の配送エラーを確認する。レート上限や会話深度超過を待つ必要はない。

## 実装の所在

| 役割 | モジュール |
|------|------------|
| 宛先解決・Slack / Chatwork への外部配信 | `core/messaging/outbound.py` |
| 内部 DM の配送・activity log 記録・深度の診断ログ | `core/messaging/messenger.py` |
| DM の重複防止・Board の run 内重複防止 | `core/tooling/handler_comms.py` |
| Inbox のファイル wake・単一実行・provider backoff | `core/runtime/inbox_rate_limiter.py` |
| Inbox 容量保護の overflow | `core/anima/inbox_overflow.py` |
| 直近送信のプロンプト注入 | `core/memory/priming/outbound.py` |
