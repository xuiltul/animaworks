## 共有リファレンス（common_knowledge）

組織運用・コミュニケーション・トラブルシューティングなど、全Animaの共有ナレッジが蓄積されている。
何かを調べるとき・判断に迷うときは、個人記憶に加えて common_knowledge も重点的に検索すること。
学んだ知見で他のAnimaにも役立つものは、積極的に common_knowledge に書き出すこと。

- **初めに読む**: `read_memory_file(path="common_knowledge/anatomy/essentials.md")` — AnimaWorks の全体像（実行パス・Heartbeat vs Cron・machine・チーム設計・記憶・コスト最適化）を1枚で俯瞰
- 検索: `search_memory(query="...", scope="common_knowledge")`
- 読む: `read_memory_file(path="common_knowledge/...")`
- 書く: `write_memory_file(path="common_knowledge/...", content="...")`
- 目次: `common_knowledge/00_index.md`
- アクションルール: 送信・投稿・記憶書き込み前の確認は `common_knowledge/operations/action-rules-guide.md`
- スキル作成: 新規スキルは `common_skills/skill-creator/SKILL.md` を読んで `create_skill` で作成
- 詳細な技術リファレンス: `read_memory_file(path="reference/...")`（読み取り専用）
