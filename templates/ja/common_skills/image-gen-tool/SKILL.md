---
name: image-gen-tool
description: >-
  画像・3Dモデル生成ツール。NovelAI・Flux・Meshyで立ち絵・バストアップ・ちび・3Dモデルを生成する。
  Use when: イラスト生成、キャラクター画像作成、3Dモデル・Meshy出力、画像パイプライン実行が必要なとき。
tags: [image, 3d, generation, external]
---

# Image Gen ツール

キャラクター画像・3Dモデルを生成する外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool image_gen <サブコマンド> [引数]` で実行。長時間処理は `animaworks-tool submit image_gen pipeline ...` でバックグラウンド実行。

## アクション一覧

### character_assets — パイプライン一括生成
```bash
animaworks-tool image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

### fullbody — 全身立ち絵
```bash
animaworks-tool image_gen fullbody "1girl, standing, ..."
```

### bustup — バストアップ
```bash
animaworks-tool image_gen bustup reference.png
```

### chibi — ちびキャラ
```bash
animaworks-tool image_gen chibi reference.png
```

### 3d_model — 3Dモデル生成
```bash
animaworks-tool image_gen 3d image.png
```

## CLI使用法

```bash
animaworks-tool image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
animaworks-tool image_gen fullbody "1girl, standing, ..."
animaworks-tool image_gen bustup reference.png
animaworks-tool image_gen chibi reference.png
animaworks-tool image_gen 3d image.png
```

## 注意事項

- 長時間処理のため `animaworks-tool submit image_gen pipeline ...` でバックグラウンド実行推奨
- NovelAI APIキーまたはfal.ai APIキーが必要
- 3D生成にはMeshy APIキーが必要
