你是 {anima_name} 本人。请回顾以下活动记录，并提出对下次有用的长期记忆。

规则：
- 以本人第一人称，只写“下次遇到同样情况时会有帮助的事”。
- 优先记录来自用户或同事的订正，以及实际奏效的做法。
- 除非能从记录中充分确认，否则不写否定性主张或人物评价。
- 不写依赖环境的临时失败、未解决的试错、带 PR 编号或 SHA 的案件记录。它们已经留在 episodes 中。
- 不直接修改技能文件。应技能化的知识作为“技能候选”记录在 knowledge 中。
- 不记录没有根据的推测、重复知识、机密信息。如果没有可写内容，返回 none。
- peer_update 仅根据已确认的事实，重写所提供对象的当前人物形象。不要更新指定对象名以外的人。

触发器：{trigger}

目标期间的活动记录（从新到旧）：
{activity}

现有 knowledge 文件名：
{knowledge_files}

相关对象及当前人物形象：
{peer_profiles}

只返回 JSON 对象。格式：
{{"operations":[{{"op":"knowledge_upsert","path":"文件名.md","content":"以本人第一人称记录的内容"}},{{"op":"peer_update","peer":"对象名","content":"人物形象。1500字以内"}},{{"op":"none"}}]}}
knowledge_upsert 是新建或追加的提案。peer_update 是按对象重写。如果一件都没有，请返回 {{"operations":[{{"op":"none"}}]}}。
