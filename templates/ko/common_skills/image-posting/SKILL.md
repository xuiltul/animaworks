---
name: image-posting
description: >-
  채팅 응답에 이미지를 넣어 표시하는 스킬. 도구 결과 URL 감지, Markdown 이미지, assets 경로를 다룬다.
  Use when: 검색·생성 도구 이미지를 답변에 포함, Markdown으로 이미지 삽입, 첨부 표시가 필요할 때.
---

# image-posting — 채팅 응답에 이미지 표시하기

## 개요

채팅 응답에 이미지를 포함하는 방법은 2가지가 있습니다:

1. **도구 결과에서 자동 추출** — 도구 결과에 이미지 URL이나 경로가 포함되면 프레임워크가 자동으로 감지하여 채팅 버블에 표시합니다
2. **Markdown 이미지 구문** — 응답 텍스트에 `![alt](url)`을 작성하면 프론트엔드가 렌더링합니다

## 방법 1: 도구 결과에서 자동 표시

도구(web_search, image_gen 등)를 호출한 결과에 이미지 정보가 포함되어 있으면 프레임워크가 자동으로 채팅 버블에 이미지를 표시합니다. Anima 측에서 특별한 조작은 필요 없습니다.

### 자동 감지 조건

도구 결과 JSON 내에서 다음이 감지되면 이미지로 처리됩니다:

- **경로 감지**: `assets/` 또는 `attachments/`로 시작하는 경로 → `source: generated` (신뢰됨)
- **URL 감지**: `https://`로 시작하고 `.png` `.jpg` `.jpeg` `.gif` `.webp`로 끝나는 URL → `source: searched` (프록시 경유)
- **키 감지**: `image_url`, `thumbnail`, `src`, `url` 키의 값에 이미지 URL이 있는 경우

응답당 최대 5장까지 표시됩니다.

### 검색 이미지의 프록시 제한

외부 URL 이미지는 보안을 위해 프록시를 통해 제공됩니다. 허용 도메인:

- `cdn.search.brave.com`
- `images.unsplash.com`
- `images.pexels.com`
- `upload.wikimedia.org`

위 이외의 도메인 URL은 프록시에서 차단됩니다.

## 방법 2: Markdown 이미지 구문

응답 텍스트에 Markdown 이미지 구문을 직접 작성하여 이미지를 표시합니다.

### 단축 경로 (권장)

프론트엔드가 자동으로 자신의 Anima 이름으로 API 경로를 보완합니다. 파일명만 작성하면 됩니다:

```
![설명](attachments/파일명)
![설명](assets/파일명)
```

예시:

```
스크린샷을 찍었습니다!
![ANA 톱 페이지](attachments/ana_top.png)
```

### 전체 경로

명시적으로 API 경로를 작성할 수도 있습니다:

```
![설명](/api/animas/{자신의_이름}/assets/{파일명})
![설명](/api/animas/{자신의_이름}/attachments/{파일명})
```

## 스크린샷 저장 위치

agent-browser 등으로 스크린샷을 촬영할 때는 **자신의 attachments 디렉토리에 직접 저장**하는 것이 확실합니다:

```bash
agent-browser screenshot ~/.animaworks/animas/{자신의_이름}/attachments/screenshot.png
```

예시 (aoi의 경우):

```bash
agent-browser screenshot ~/.animaworks/animas/aoi/attachments/page_screenshot.png
```

저장 후 응답에 다음을 작성하면 표시됩니다:

```
![페이지 스크린샷](attachments/page_screenshot.png)
```

`~/.animaworks/tmp/attachments/`에 저장한 경우에도 폴백으로 표시되지만, 임시 디렉토리이므로 영속성이 보장되지 않습니다.

## 주의사항

- 다른 Anima의 에셋 경로는 직접 참조할 수 없습니다 (권한 외)
- 외부 URL 직접 링크는 비권장. 허용 도메인 외에는 자동 표시되지 않습니다
- 이미지 생성 도구(generate_fullbody 등)의 결과는 자동 표시되므로 Markdown 구문이 불필요합니다
- 응답당 자동 표시는 최대 5장입니다
