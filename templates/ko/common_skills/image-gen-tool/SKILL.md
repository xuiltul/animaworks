---
name: image-gen-tool
description: >-
  이미지·3D 모델 생성 도구. NovelAI·Flux·Meshy로 전신·바스트업·치비·3D를 생성한다.
  Use when: 일러스트 생성, 캐릭터 이미지, 3D·Meshy 출력, 이미지 파이프라인 실행이 필요할 때.
tags: [image, 3d, generation, external]
---

# Image Gen 도구

캐릭터 이미지 및 3D 모델을 생성하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool image_gen <서브커맨드> [인수]`로 실행합니다. 장시간 처리는 `animaworks-tool submit image_gen pipeline ...`으로 백그라운드 실행하세요.

## 액션 목록

### character_assets — 파이프라인 일괄 생성
```bash
animaworks-tool image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

### fullbody — 전신 일러스트
```bash
animaworks-tool image_gen fullbody "1girl, standing, ..."
```

### bustup — 바스트업
```bash
animaworks-tool image_gen bustup reference.png
```

### chibi — 치비 캐릭터
```bash
animaworks-tool image_gen chibi reference.png
```

### 3d_model — 3D 모델 생성
```bash
animaworks-tool image_gen 3d image.png
```

## CLI 사용법

```bash
animaworks-tool image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
animaworks-tool image_gen fullbody "1girl, standing, ..."
animaworks-tool image_gen bustup reference.png
animaworks-tool image_gen chibi reference.png
animaworks-tool image_gen 3d image.png
```

## 주의사항

- 장시간 처리이므로 `animaworks-tool submit image_gen pipeline ...`으로 백그라운드 실행을 권장합니다
- NovelAI API 키 또는 fal.ai API 키가 필요합니다
- 3D 생성에는 Meshy API 키가 필요합니다
