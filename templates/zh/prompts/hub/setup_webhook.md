创建两个触发器：Webhook 和每天一次的日程。Webhook 在帖子到达时立即触发，日程则定期检查新内容。

Webhook 的 URL 和 key 由本人以 `{url_ref}` 和 `{key_ref}` 的名称放入 vault（shared）。值通过隐藏输入询问。

```bash
animaworks vault store --shared {url_ref}
animaworks vault store --shared {key_ref}
```

routine 的指令（粘贴到 Webhook 和日程两者中）：

```text
{routine}
```
