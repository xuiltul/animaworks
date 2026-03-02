---
name: image-posting
description: >-
  チャット応答に画像を添付・表示するスキル。
  ツール結果(web_search等)に含まれる画像URLの自動検出・プロキシ経由表示の仕組み、
  応答テキスト内でのMarkdown画像構文による埋め込み方法、
  自分のassets画像の表示方法を提供する。
  「画像を貼る」「画像を見せて」「イラスト表示」「画像添付」「写真を貼って」「検索画像を表示」
---

# image-posting — チャット応答への画像表示

## 概要

チャット応答に画像を含める仕組みは2系統ある:

1. **ツール結果からの自動抽出** — ツール結果に画像URLやパスが含まれると、フレームワークが自動検出してチャットバブルに表示する
2. **Markdown画像構文** — 応答テキスト内に `![alt](url)` を書くとフロントエンドがレンダリングする

## 方法1: ツール結果からの自動表示

ツール（web_search、image_gen等）を呼び出した結果に画像情報が含まれていれば、フレームワークが自動でチャットバブルに画像を表示する。Anima側で特別な操作は不要。

### 自動検出される条件

ツール結果のJSON内で以下が検出されると画像として扱われる:

- **パス検出**: `assets/` または `attachments/` で始まるパス → `source: generated`（信頼済み）
- **URL検出**: `https://` で始まり `.png` `.jpg` `.jpeg` `.gif` `.webp` で終わるURL → `source: searched`（プロキシ経由）
- **キー名検出**: `image_url`, `thumbnail`, `src`, `url` キーに画像URLがある場合も検出

1応答あたり最大5枚まで。

### searched画像のプロキシ制限

外部URL画像はセキュリティのためプロキシ経由で配信される。ドメイン固定の許可リストではなく、以下の安全検査を通過した画像のみ表示される:

- `https://` のみ（`http://` は拒否）
- private/local/loopback/link-local 宛先は拒否
- リダイレクト先も同じ検査を再適用
- 画像実体（magic bytes）検証に失敗した場合は拒否
- SVG (`image/svg+xml`) は拒否
- サイズ上限・レート制限超過時はブロック

## 方法2: Markdown画像構文

応答テキスト内にMarkdown画像構文を直接書いて画像を表示する。

### 短縮パス（推奨）

フロントエンドが自動的に自分のAnima名でAPIパスを補完する。ファイル名だけ書けばOK:

```
![説明](attachments/ファイル名)
![説明](assets/ファイル名)
```

例:

```
スクショ撮りました！
![ANAトップページ](attachments/ana_top.png)
```

### フルパス

明示的にAPIパスを書くこともできる:

```
![説明](/api/animas/{自分の名前}/assets/{ファイル名})
![説明](/api/animas/{自分の名前}/attachments/{ファイル名})
```

## スクリーンショットの保存先

agent-browser等でスクリーンショットを撮る場合、**自分のattachmentsディレクトリに直接保存する**のが確実:

```bash
agent-browser screenshot ~/.animaworks/animas/{自分の名前}/attachments/screenshot.png
```

例（aoiの場合）:

```bash
agent-browser screenshot ~/.animaworks/animas/aoi/attachments/page_screenshot.png
```

保存後、応答に以下を書けば表示される:

```
![ページのスクショ](attachments/page_screenshot.png)
```

`~/.animaworks/tmp/attachments/` に保存した場合もフォールバックで表示されるが、一時ディレクトリなので永続性は保証されない。

## 注意事項

- 他のAnimaのアセットパスは直接参照できない（権限外）
- 外部URLの直リンクは非推奨。安全検査でブロックされる場合がある
- 画像生成ツール（generate_fullbody等）の結果は自動表示されるため、Markdown構文は不要
- 1応答あたりの自動表示は最大5枚
