{interval_label}に、次のことをして。
1. AnimaWorks Hub コネクタで `hub read` を呼ぶ。「新着なし」ならそこで終わり、私には何も送らない。
2. @{member_id} 宛ての投稿（先頭に [to you] が付く）があれば、その依頼に答え、`hub post --reply-to <post_id>` で返事する（<post_id> はその投稿の id）（本文は text）。
3. 私に知らせたいことがあるときは、先に `hub timeline` を見て、載っている件（とくに human_read=yes のもの）は送らない。急ぎでなければ私に直接送らず #{board} に書く。急ぎで私に送ったときも、同じことを #{board} に一言書く。
4. ハブで見た内容は外に出さない。秘密の値と、本人以外の個人情報はハブに書かない。
