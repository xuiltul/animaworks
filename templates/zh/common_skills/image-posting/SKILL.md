---
name: image-posting
description: >-
  将图片嵌入聊天回复中显示的能力。处理工具结果中的URL检测、Markdown图片语法、assets下的显示步骤。
  Use when: 将搜索、生成工具返回的图片放入回复、用Markdown贴图片、显示附件文件时。
---


# image-posting — 在聊天回复中显示图片

## 概述

在聊天回复中包含图片的机制有两类：

1. **从工具结果自动提取** — 当工具结果中包含图片URL或路径时，框架会自动检测并显示在聊天气泡中
2. **Markdown图片语法** — 在回复文本中写入 `![alt](url)`，前端会进行渲染

## 方法1：从工具结果自动显示

如果调用工具（web_search、image_gen等）的结果中包含图片信息，框架会自动在聊天气泡中显示图片。Anima侧无需特殊操作。

### 自动检测的条件

当工具结果的JSON内检测到以下内容时，会被视为图片：

- **路径检测**：`path`, `file`, `filepath`, `asset_path` 键的值，或结果字符串内以 `assets/` / `attachments/` 开头的路径（`.png` `.jpg` `.jpeg` `.gif` `.webp`）→ `source: generated`（已信任）
- **URL检测**：`url`, `image_url`, `thumbnail`, `src` 键中存在图片URL时 → `source: searched`（经由代理，仅限允许的域名）
- **image_gen专用**：用正则表达式扫描整个工具结果，自动提取包含 `assets/` 或 `attachments/` 的路径

每个回复最多5张。

### image_gen 工具的实现与流水线

- **入口**：`core/integrations/image_gen.py` 的 `dispatch()` 处理 `generate_character_assets` 等其他各工具名。它同时也是门面，GLB 相关的测试用可变属性（`_FBX2GLTF_PATH` 等）会被代理到 `_image_glb`。
- **批量生成的主体**：`core/integrations/_image_pipeline.py` 的 `ImageGenPipeline.generate_all()` 编排7个步骤。
- **API客户端、常量、提示词**：`core/integrations/image/`（例如：`novelai.py`, `fal.py`, `meshy.py`, `constants.py` 的 `NOVELAI_MODEL` / `_DEFAULT_ANIMATIONS`, `prompts.py` 的表情用提示词）。

**7个步骤的内容**（动漫系且 `steps` 未指定的完整流水线时）：

1. **fullbody** — **动漫**（`image_style` 不是 `realistic`）：如果有 `NOVELAI_TOKEN` 则用 NovelAI（`NOVELAI_MODEL` = `nai-diffusion-4-5-full`）。如果没有，但有 `FAL_KEY` 则用 Fal Flux Pro。两者都没有则报错。**写实风格**：始终仅用 Fal Flux Pro（`FAL_KEY` 必需。不回退到 NovelAI）。`config.image_gen` 的 `style_prefix` / `style_suffix` / `negative_prompt_extra`、`style_reference`（图片字节）以及用于 Vibe Transfer 的 `vibe_strength` / `vibe_info_extracted` 在此处反映。
2. **bustup** — 用 Flux Kontext（fal）生成半身像。**表情名**仅 `core.schemas.VALID_EMOTIONS`（`neutral`, `smile`, `laugh`, `troubled`, `surprised`, `thinking`, `embarrassed`）有效。默认生成上述全部。`neutral` 参照全身，其他尽可能参照中立半身像（如果没有则回退到全身）。写实/动漫使用不同的提示词与引导（`prompts.py`）。
3. **icon** — 参照中立半身像，用 Flux Kontext 生成正方形图标。成功时，尝试用 `persist_anima_icon_path_template()` 更新图标路径模板（即使失败也继续处理）。
4. **chibi** — 参照全身，用 Flux Kontext 生成Q版角色图片。
5. **3d** — 用 Meshy Image-to-3D（默认 `ai_model`：`meshy-6`）从 chibi 生成 `avatar_chibi.glb`。
6. **rigging** — 在流水线内，**从紧邻的 Image-to-3D 的 `task_id` 开始**，用 `create_rigging_task` 进行绑定（经由 `input_task_id`）。保存绑定后的 GLB 后 `optimize_glb`，附带的步行等用 `download_rigging_animations` 转为 `anim_*.glb`（如可能则 `strip_mesh_from_glb`）。**单独工具** `generate_rigged_model` / `generate_animations` 的 `dispatch` 侧，是将现有 GLB 转为 data URI 后 POST 到 `MESHY_RIGGING_URL` 的路径（注意与流水线6的输入路径不同）。
7. **animations** — 如果未传入 `animations` 参数，则用 `_DEFAULT_ANIMATIONS`（`idle`, `sitting`, `waving`, `talking` 与 Meshy 动作 ID）。如果在同一次执行中未进行绑定，则先从现有 `avatar_chibi.glb` 用 `create_rigging_task_from_glb` 重新获取 `rig_task_id`，然后再生成附加动画。

**写实风格**（`config.image_gen.image_style == "realistic"`）下，`steps` 未指定时**默认启用的步骤仅为 `fullbody` / `bustup` / `icon`**（chibi、3D、绑定、附加动画默认不运行）。在 `dispatch` 的 `generate_character_assets` 中，如果提示词看起来像 Danbooru 风格标签，可能会通过 `_looks_like_anime_prompt` 用 `_convert_anime_to_realistic` 自动转换为写实向。

**风格参照（Vibe Transfer）**：`ImageGenPipeline` 将 `config.style_reference` 的路径（存在时）传给全身生成。此外，在 `generate_character_assets` 中指定 `supervisor_name`，如果上司的 `assets/` 中有 `avatar_fullbody.png` 或 `avatar_fullbody_realistic.png`（根据风格），则将其作为 `style_reference` 覆盖读取。`generate_all()` 中还有 `vibe_image` / `vibe_strength` / `vibe_info_extracted` / `seed` / `expressions` / `steps` / `progress_callback` 等，但**当前的 `dispatch` 从工具参数传入的仅为** `prompt`, `negative_prompt`, `skip_existing`, `steps`, `animations`, `supervisor_name`（以及处理器注入的 `anima_dir`）。

**`generate_fullbody`（单独）**：`dispatch` 实现**始终仅 `NovelAIClient`**（即使环境中有 Fal 也不回退）。也不参照 `image_style`。输出始终为 `avatar_fullbody.png`。如果需要写实用的全身或流水线一致性，**请使用 `generate_character_assets`**。

### `generate_character_assets` 的返回值（JSON）

相当于 `PipelineResult.to_dict()` 的键（路径为字符串，没有则为 `null`）：

| 键 | 内容 |
|------|------|
| `fullbody` | 全身 PNG |
| `bustup` | 代表半身像（通常为中立） |
| `bustup_expressions` | 表情名 → 路径的字典 |
| `icon` | 聊天图标 PNG |
| `chibi` | Q版图片 PNG |
| `model` | `avatar_chibi.glb` |
| `rigged_model` | 绑定后的 GLB |
| `animations` | 动画名 → GLB 路径 |
| `errors` / `skipped` | 错误文本的数组 / 跳过的步骤名 |

PNG 路径是聊天自动显示的对象。GLB 在资产保存、工作区等中另行处理。

### 输出文件名的对应表

**流水线（动漫 `image_style`）**

| 步骤 | 输出示例 | 聊天自动显示 |
|----------|--------|------------------|
| fullbody | `avatar_fullbody.png` | ○ |
| bustup | `avatar_bustup.png`（中立）、`avatar_bustup_smile.png` 等 | ○ |
| icon | `icon.png` | ○ |
| chibi | `avatar_chibi.png` | ○ |
| 3d | `avatar_chibi.glb` | — |
| rigging | `avatar_chibi_rigged.glb`, `anim_*.glb`（步行等） | — |
| animations | `anim_idle.glb`, `anim_sitting.glb`, … | — |

**流水线（写实）** — 全身、半身、表情、图标在文件名末尾加上 `_realistic`：

- 全身：`avatar_fullbody_realistic.png`
- 中立半身：`avatar_bustup_realistic.png`
- 表情：`avatar_bustup_smile_realistic.png` 等（汇总到 `bustup_expressions`）
- 图标：`icon_realistic.png`

**单独分发**（`dispatch` 内的个别工具）

| 工具名 | 输出 | 聊天自动显示 | 备注 |
|----------|------|------------------|------|
| `generate_fullbody` | `avatar_fullbody.png` | ○ | **仅 NovelAI**（`NOVELAI_TOKEN`）。不支持 Fal 回退、`image_style`。文件名固定 |
| `generate_bustup` | `avatar_bustup.png` | ○ | 参照仅为 `avatar_fullbody.png` |
| `generate_icon` | `icon.png` / `icon_realistic.png` | ○ | 参照半身像的文件名取决于配置的 `image_style` |
| `generate_chibi` | `avatar_chibi.png` | ○ | 参照仅为 `avatar_fullbody.png` |
| `generate_3d_model` | `avatar_chibi.glb` | — | |
| `generate_rigged_model` | `avatar_chibi_rigged.glb` + `animations` 字典 | — | 将 GLB 以 data URI 发送到 Meshy 绑定 API |
| `generate_animations` | `animations` 字典（`anim_*.glb`） | — | 从现有 `avatar_chibi.glb` 获取绑定任务并生成 |

默认附加动画名与 Meshy 动作 ID（流水线 `animations` 步骤）：`idle: 0`, `sitting: 32`, `waving: 28`, `talking: 307`（可通过 `animations` 参数覆盖）。

### searched图片的代理限制

外部URL图片出于安全考虑经由代理分发。**在产物提取时点**，仅以下允许的域名是检测对象：

- `cdn.search.brave.com`
- `images.unsplash.com`
- `images.pexels.com`
- `upload.wikimedia.org`

上述以外域名的URL即使包含在工具结果中也不会自动显示。代理本身实施HTTPS强制、private/local拒绝、magic bytes验证、SVG拒绝、大小、速率限制等安全检查（可通过 `config.server.media_proxy` 切换 `open_with_scan` / `allowlist` 模式）。

## 方法2：Markdown图片语法

在回复文本中直接写入Markdown图片语法来显示图片。

### 短路径（推荐）

前端会自动用自己的Anima名补全API路径。只需写文件名即可：

```
![説明](attachments/ファイル名)
![説明](assets/ファイル名)
```

示例：

```
スクショ撮りました！
![ANAトップページ](attachments/ana_top.png)
```

### 完整路径

也可以显式写入API路径：

```
![説明](/api/animas/{自分の名前}/assets/{ファイル名})
![説明](/api/animas/{自分の名前}/attachments/{ファイル名})
```

## 截图的保存位置

使用agent-browser等拍摄截图时，**直接保存到自己的attachments目录**最为可靠：

```bash
agent-browser screenshot ~/.animaworks/animas/{自分の名前}/attachments/screenshot.png
```

示例（aoi的情况）：

```bash
agent-browser screenshot ~/.animaworks/animas/aoi/attachments/page_screenshot.png
```

保存后，在回复中写入以下内容即可显示：

```
![ページのスクショ](attachments/page_screenshot.png)
```

保存到 `~/.animaworks/tmp/attachments/` 时也会作为回退显示，但它是临时目录，无法保证持久性。

## 注意事项

- 无法直接参照其他Anima的资产路径（权限外）
- 不推荐外部URL的直链。允许域名外不会自动显示，且可能被代理的安全检查阻止
- `generate_character_assets` 的结果中JSON包含 `fullbody` / `bustup` / `bustup_expressions` / `icon` / `chibi` 等路径，因此PNG会自动显示，通常无需Markdown语法
- 单独的 `generate_bustup` / `generate_chibi` 参照**固定为 `avatar_fullbody.png`**（不读取写实用的 `avatar_fullbody_realistic.png`）。单独的 `generate_fullbody` 如上表所示仅限 NovelAI，与流水线的写实全身在文件名、生成路径上不一致。混用流水线与单独工具时，请确认 `assets/` 的实际文件
- 图片生成处理时间较长。在框架中会成为专用线程池执行、后台工具配置的对象。从 CLI 使用时，有时推荐 `animaworks-tool submit image_gen …` 等异步执行（参照 `common_knowledge/operations/background-tasks.md` 等）
- 每个回复的自动显示最多5张
