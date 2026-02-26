## 雇用ルール

新しいAnimaを雇用する際は、以下の手順に従ってください。
手動で identity.md 等のファイルを個別に作成してはいけません。

1. キャラクターシートを1ファイルのMarkdownとして作成する
   - 必須セクション: `## 基本情報`, `## 人格`, `## 役割・行動方針`
2. Bashで以下のコマンドを実行する:
   ```
   animaworks create-anima --from-md <キャラクターシートのパス> --supervisor $(basename $ANIMAWORKS_ANIMA_DIR)
   ```
3. サーバーのReconciliationが自動的に新Animaを検出・起動します