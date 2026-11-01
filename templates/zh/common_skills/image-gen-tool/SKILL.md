---
name: image-gen-tool
description: >-
  图像・3D模型生成工具。使用NovelAI・Flux・Meshy生成立绘・半身像・Q版・3D模型。
  Use when: 需要插画生成、角色图像制作、3D模型・Meshy输出、图像流水线执行时。
tags: [image, 3d, generation, external]
---


# Image Gen 工具

生成角色图像・3D模型的外部工具。

## 调用方法

**Bash**: 使用 `animaworks-tool image_gen <サブコマンド> [引数]` 执行。长时间处理使用 `animaworks-tool submit image_gen pipeline ...` 在后台执行。

## 动作列表

### character_assets — 流水线批量生成
```bash
animaworks-tool image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

### fullbody — 全身立绘
```bash
animaworks-tool image_gen fullbody "1girl, standing, ..."
```

### bustup — 半身像
```bash
animaworks-tool image_gen bustup reference.png
```

### chibi — Q版角色
```bash
animaworks-tool image_gen chibi reference.png
```

### 3d_model — 3D模型生成
```bash
animaworks-tool image_gen 3d image.png
```

## CLI用法

```bash
animaworks-tool image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
animaworks-tool image_gen fullbody "1girl, standing, ..."
animaworks-tool image_gen bustup reference.png
animaworks-tool image_gen chibi reference.png
animaworks-tool image_gen 3d image.png
```

## 注意事项

- 由于是长时间处理，推荐使用 `animaworks-tool submit image_gen pipeline ...` 在后台执行
- 需要NovelAI API密钥或fal.ai API密钥
- 3D生成需要Meshy API密钥
