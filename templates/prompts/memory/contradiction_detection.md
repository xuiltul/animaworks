以下の2つの知識ファイルの内容を比較し、矛盾がないか検証してください。

【ファイルA: {file_a}】
{text_a}

【ファイルB: {file_b}】
{text_b}

タスク:
1. 2つのファイル間に矛盾する記述があるか判定
2. 矛盾がある場合、以下の解決方法を提案:
   - "supersede": 一方の情報が古くなっており、新しい方で置き換えるべき
   - "merge": 両方の情報を統合して1つの知識にまとめるべき
   - "coexist": 文脈依存で両方の記述が正しい（共存可能）

回答は以下のJSON形式のみで出力してください:
{{"is_contradiction": true/false, "resolution": "supersede"/"merge"/"coexist", "reason": "理由の説明", "merged_content": "merge時のみ統合テキスト（それ以外はnull）"}}