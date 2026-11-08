Webhook과 매일 1회 스케줄의 두 가지 트리거를 만듭니다. Webhook으로 게시물이 도착하면 즉시 실행되고, 스케줄로 정기적으로 새 글을 확인합니다.

Webhook의 URL과 key는 본인이 vault(shared)에 `{url_ref}`와 `{key_ref}`라는 이름으로 넣습니다. 값은 숨김 입력으로 물어봅니다.

```bash
animaworks vault store --shared {url_ref}
animaworks vault store --shared {key_ref}
```

routine의 지침(Webhook과 스케줄 양쪽에 붙여넣기):

```text
{routine}
```
