---
name: image-posting
description: >-
  Skill for displaying images in chat responses.
  Covers auto-detection of image URLs in tool results (web_search etc.) with proxy-based display,
  embedding images via Markdown syntax in response text,
  and showing own assets images.
  "post image", "show image", "display picture", "attach image", "show search results", "show found image"
---

# image-posting — Displaying Images in Chat Responses

## Overview

There are two ways to include images in chat responses:

1. **Auto-extraction from tool results** — When tool results contain image URLs or paths, the framework auto-detects and displays them in the chat bubble
2. **Markdown image syntax** — Write `![alt](url)` in response text for frontend rendering

## Method 1: Auto-Display from Tool Results

When a tool (web_search, image_gen, etc.) returns results containing image information, the framework automatically displays images in the chat bubble. No special action needed.

### Auto-Detection Rules

The following patterns in tool result JSON are detected as images:

- **Path detection**: Paths starting with `assets/` or `attachments/` → `source: generated` (trusted)
- **URL detection**: URLs starting with `https://` ending in `.png` `.jpg` `.jpeg` `.gif` `.webp` → `source: searched` (proxied)
- **Key detection**: Values under `image_url`, `thumbnail`, `src`, `url` keys with image URLs

Up to 5 images per response.

### Proxy Restrictions for Searched Images

External URL images are served through a security proxy. Allowed domains:

- `cdn.search.brave.com`
- `images.unsplash.com`
- `images.pexels.com`
- `upload.wikimedia.org`

Image URLs from other domains are blocked by the proxy.

## Method 2: Markdown Image Syntax

Write Markdown image syntax directly in response text to display images.

### Short Paths (Recommended)

The frontend automatically prepends the API path with your Anima name. Just write the filename:

```
![description](attachments/filename)
![description](assets/filename)
```

Example:

```
Here's a screenshot!
![ANA Top Page](attachments/ana_top.png)
```

### Full Paths

You can also write the full API path explicitly:

```
![description](/api/animas/{your_name}/assets/{filename})
![description](/api/animas/{your_name}/attachments/{filename})
```

## Saving Screenshots

When taking screenshots with agent-browser, **save directly to your own attachments directory**:

```bash
agent-browser screenshot ~/.animaworks/animas/{your_name}/attachments/screenshot.png
```

Example (for aoi):

```bash
agent-browser screenshot ~/.animaworks/animas/aoi/attachments/page_screenshot.png
```

Then include in your response:

```
![Page screenshot](attachments/page_screenshot.png)
```

Files saved to `~/.animaworks/tmp/attachments/` will also work via fallback, but the temp directory does not guarantee persistence.

## Notes

- Cannot directly reference other Animas' asset paths (no permission)
- Direct external URL linking is discouraged; only proxy-allowed domains are displayed
- Image generation tool results (generate_fullbody, etc.) are auto-displayed — no Markdown syntax needed
- Auto-display is capped at 5 images per response
