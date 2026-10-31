自分の記憶ディレクトリ内のファイルに書き込みまたは追記する。記録すべき時: 問題解決→knowledge/に原因と解決策、正しい設定値→knowledge/、手順確立→procedures/、新能力→create_skillで skills/{name}/SKILL.md を作成、送信・投稿前の確認ルール→knowledge/action-rule-*.md に [ACTION-RULE] と trigger_tools、HEARTBEAT.md / cron.md の更新。mode='overwrite' で全体置換、mode='append' で末尾追記。重要な発見はconsolidationを待たず即座に書き込む。
記録先の注意:
- 根拠を確認していない「Xは使えない／やってはいけない」等の否定的主張を knowledge/・skills/ にしない。
- 環境依存の一時的な失敗や未解決の試行錯誤は knowledge/ にせず episodes/ に残す。
- PR番号・SHA・複数の日付に紐づく案件記録は episodes/ に書く。
