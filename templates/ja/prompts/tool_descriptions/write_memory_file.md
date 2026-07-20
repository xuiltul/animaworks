自分の記憶ディレクトリ内のファイルに書き込みまたは追記する。
以下の場面で記録すべき:
- 問題を解決した → knowledge/ に原因と解決策を記録
- 正しいパラメータ・設定値を発見した → knowledge/ に記録
- 作業手順を確立・改善した → procedures/ に手順書を作成
- 新しい再利用可能な能力を習得した → common_skills/skill-creator/SKILL.md を読み、create_skill で skills/{name}/SKILL.md を作成
- 送信・投稿・通知・記憶書き込み前の確認ルールが必要 → knowledge/action-rule-*.md に [ACTION-RULE] と trigger_tools を記録
- heartbeat.md や cron.md の更新
mode='overwrite' で全体置換、mode='append' で末尾追記。
自動統合（日次consolidation）を待たず、重要な発見は即座に書き込むこと。
