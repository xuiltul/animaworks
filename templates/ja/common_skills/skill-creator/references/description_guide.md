# description 記述ガイド（Agent Skills 標準準拠）

## 基本ルール

descriptionはスキルの発見と選択に使われる最重要フィールド。
LLMがdescriptionを読んで「このスキルが今の会話に関連するか」を判断する。

### フォーマット

```yaml
description: >-
  [1行目: 何をするスキルかの簡潔な説明（三人称）]
  Use when: [利用シーンをカンマ区切りで列挙]
```

### ルール

1. **250文字以内** — カタログ表示で250文字を超える部分は打ち切られる
2. **三人称で記述** — 「〜ができる」「〜を行う」形式。「私は」「あなたは」は不可
3. **Use when: を必ず含める** — LLMが利用判断の手がかりにする
4. **具体的な動詞・名詞を使う** — 「ログイン」「スクリーンショット」「メール送信」など
5. **XMLタグ不可** — `<` `>` はセキュリティ上禁止
6. **`「」` キーワード列挙は使用しない** — 旧方式。LLM推論に任せる

### 良い例

```yaml
description: >-
  ヘッドレスブラウザ操作CLI。Webページを開いて閲覧・操作・ログイン・スクリーンショット撮影ができる。
  Use when: ブラウザでサイトを開く、Webアプリの操作・確認、ログイン操作、スクショ撮影、画面のUI確認が必要なとき。
```

```yaml
description: >-
  Gmail operations via CLI. Send, receive, search emails, and manage labels.
  Use when: sending emails, checking inbox, searching mail, managing Gmail labels or filters.
```

### 悪い例

```yaml
# ❌ 「」キーワード列挙（旧方式）
description: >-
  ブラウザ操作CLI。
  「ブラウザで確認」「スクショ撮って」「ブラウザ操作」

# ❌ 曖昧すぎる
description: Helps with documents

# ❌ 一人称
description: I can help you process PDF files

# ❌ 250文字超
description: >-
  （非常に長い説明文...）
```

### チェックリスト

- [ ] 250文字以内か
- [ ] 三人称で書かれているか
- [ ] `Use when:` を含んでいるか
- [ ] 具体的な動詞・名詞があるか
- [ ] `「」` キーワード列挙がないか
- [ ] XMLタグがないか

### linter で検証

スキル作成後は linter で検証できる:

```bash
python scripts/lint_skill.py /path/to/SKILL.md
```
