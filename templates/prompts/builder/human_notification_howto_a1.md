Bashツールから以下のコマンドで人間の管理者に連絡してください:

```
animaworks-tool call_human "件名" "本文" [--priority high]
```

優先度: `low` / `normal`（デフォルト）/ `high` / `urgent`
例: `animaworks-tool call_human "障害発生" "本番サーバーがダウンしています" --priority urgent`

連絡内容は外部通知チャネル（Slack等）に届きます。