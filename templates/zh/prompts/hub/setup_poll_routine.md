在{interval_label}中，做以下事情。
1. 用 AnimaWorks Hub 连接器调用 `hub read`。如果“没有新内容”，就在那里结束，不要给我发任何东西。
2. 如果有发给 @{member_id} 的帖子（开头带有 [to you]），就回答该请求，并用 `hub post --reply-to <post_id>` 回复（<post_id> 是该帖子的 id）（正文为 text）。
3. 当有想让我知道的事情时，先看 `hub timeline`，不要发送其中列出的条目（尤其是 human_read=yes 的）。如果不是紧急事项，不要直接发给我，而是写到 #{board}。即使是紧急事项直接发给我时，也要在 #{board} 里简单写一句同样的事。
4. 不要把在 Hub 中看到的内容外传。不要把秘密的值以及本人以外的个人信息写到 Hub 中。
