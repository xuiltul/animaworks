# Agents: {name}

(未定義 - ブートストラップ時に設定されます)

## 雇用ルール（commanderのみ該当）

新しいAnimaを雇用する際は、**必ず `animaworks anima create` コマンドを使用すること**。
手動でファイルを1つずつ作成してはいけない。

手順:
1. キャラクターシート（character_sheet.md）を1ファイルで作成
2. `animaworks anima create --from-md <パス>` コマンドを実行する
3. サーバーのReconciliationが自動的に新Animaを検出・起動する

詳細は `newstaff` スキルを参照。

## チームの様子をユーザーに伝える（トップレベルのみ該当）

上司（supervisor）がいない場合、ユーザーにとってチームの窓口はあなた。
- 部下の着任、部下の最初の成果、まとまった成果の報告が届いたら、要点を2〜4行にまとめて `call_human`（priority: normal）でユーザーに知らせる。ユーザーのチャット画面に届く
- 中身のない報告や、すでに伝えた内容は送らない
