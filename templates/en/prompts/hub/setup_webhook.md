Create two triggers: a Webhook and a once-daily schedule. The Webhook fires as soon as a post arrives, and the schedule periodically checks for new items.

The person themselves puts the Webhook URL and key into the vault (shared) under the names `{url_ref}` and `{key_ref}`. The values are requested via hidden input.

```bash
animaworks vault store --shared {url_ref}
animaworks vault store --shared {key_ref}
```

Routine instruction (paste into both the Webhook and the schedule):

```text
{routine}
```
