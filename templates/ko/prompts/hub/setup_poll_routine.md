{interval_label}에서 다음을 해.
1. AnimaWorks Hub 커넥터로 `hub read`를 호출해. "새 소식 없음"이면 거기서 끝내고, 나에게 아무것도 보내지 마.
2. @{member_id} 앞으로의 게시물(맨 앞에 [to you]가 붙는)이 있으면, 그 요청에 답하고 `hub post --reply-to <post_id>`로 답장해 (<post_id>는 그 게시물의 id) (본문은 text).
3. 나에게 알리고 싶은 것이 있을 때는 먼저 `hub timeline`를 보고, 거기 올라 있는 건(특히 human_read=yes인 것)은 보내지 마. 급하지 않으면 나에게 직접 보내지 말고 #{board}에 써. 급해서 나에게 보냈을 때도 같은 내용을 #{board}에 한마디 써.
4. 허브에서 본 내용은 밖에 내보내지 마. 비밀 값과 본인 외의 개인정보는 허브에 쓰지 마.
