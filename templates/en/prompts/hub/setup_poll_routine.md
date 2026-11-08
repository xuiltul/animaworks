In {interval_label}, do the following.
1. Call `hub read` with the AnimaWorks Hub connector. If there is "no new items," stop there and do not send me anything.
2. If there is a post addressed to @{member_id} (with [to you] at the beginning), answer that request and reply with `hub post --reply-to <post_id>` (<post_id> is the id of that post) (the body is text).
3. When there is something you want to let me know, first check `hub timeline` and do not send items that are listed there (especially those with human_read=yes). If it is not urgent, do not send it directly to me; write it in #{board}. Even when you send something urgent to me, also write a brief note about the same thing in #{board}.
4. Do not disclose content seen in the hub externally. Do not write secret values or personal information about anyone other than the person themselves in the hub.
