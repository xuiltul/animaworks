# description 编写指南（符合 Agent Skills 标准）

## 基本规则

description 是用于技能发现和选择的最重要字段。
LLM 会阅读 description 来判断“该技能是否与当前对话相关”。

### 格式

```yaml
description: >-
  [1行目: 何をするスキルかの簡潔な説明（三人称）]
  Use when: [利用シーンをカンマ区切りで列挙]
```

### 规则

1. **250 字符以内** — 在目录显示中，超过 250 字符的部分会被截断
2. **使用第三人称描述** — 采用“能够～”“执行～”的形式。不可使用“我”“你”
3. **必须包含 Use when:** — LLM 会将其作为判断是否使用的线索
4. **使用具体的动词和名词** — 如“登录”“截图”“发送邮件”等
5. **不可使用 XML 标签** — `<` `>` 出于安全原因被禁止
6. **不要使用 `「」` 关键词列举** — 旧方式。交由 LLM 推理

### 好例子

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

### 坏例子

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

### 检查清单

- [ ] 是否在 250 字符以内
- [ ] 是否使用第三人称书写
- [ ] 是否包含 `Use when:`
- [ ] 是否有具体的动词和名词
- [ ] 是否没有 `「」` 关键词列举
- [ ] 是否没有 XML 标签

### 用 linter 验证

创建技能后，可以用 linter 验证：

```bash
python scripts/lint_skill.py /path/to/SKILL.md
```
