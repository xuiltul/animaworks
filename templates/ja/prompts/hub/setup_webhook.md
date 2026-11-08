Webhook と毎日1回のスケジュールの2つのトリガーを作ります。Webhook で投稿が届いたらすぐ起き、スケジュールで定期的に新着を確認します。

Webhook の URL と key は、本人が vault（shared）に `{url_ref}` と `{key_ref}` の名前で入れます。値は隠し入力で聞かれます。

```bash
animaworks vault store --shared {url_ref}
animaworks vault store --shared {key_ref}
```

routine の指示（Webhook とスケジュールの両方に貼る）:

```text
{routine}
```
