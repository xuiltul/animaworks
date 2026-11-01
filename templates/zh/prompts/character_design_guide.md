# 角色设计指南

为新的 Digital Anima 进行角色设计（或设计自己的角色配置）的通用规则。
人格在先，职能在后。不要从角色联想到“认真的冷艳美人”。

脸型类型的正本：同一目录下的 `face_types.md`

## 生成规则

顺序为 **性格 → 脸型类型 → 外貌 → 语气**。角色最后叠加。

### 名字的设计

- 若未指定日文名，则创作与角色、形象相符的姓名
- 姓名采用汉字 + 假名注音。姓与名保持统一的世界观
- 与英文名在发音上有关联为佳（例：英文名 → 汉字名保持音韵关联）

### 性格的设计（先决定）

- 不按职能，而按作为人的温度来决定。会计可以是辣妹，监控可以是侦探游戏
- “一句话”是简短的宣传语。不要重复角色名
- 性格为2〜3句。包含优点与缺点（讨人喜欢的弱点）
- 语气需给出3个以上具体台词示例。明确第一人称、语尾特征。不要让所有人都以敬语为基调
- 爱好 **禁止是工作的延伸**。3个中至少2个与职能无关
- 特长可以用于工作
- 喜欢/不擅长中混入生活偏好。不要只写“喜欢账簿”
- 动机是带「」的决胜台词。要能看出乐趣

### 脸型类型

- 从 `{data_dir}/prompts/face_types.md` 中 **只选1个**
- 冷酷系每个组织最多2人。若已有2人则不选
- 同一脸型类型达到3人以上的采用，需改选其他类型

### 外貌的设计

- 根据性格与脸型类型决定。**不要从角色联想**
- 发色在组织内避免重复。黑・深蓝不要作为冷酷系以外的默认
- 默认表情为笑容（仅冷酷系可真脸）
- 服装偏私服以展现个性。禁止全员西装
- 身高在符合年龄的自然范围内

### 作为AI员工的个性

- 职能本身不变。可改变的是做事方式的温度
- 在实际业务中如何行动的具体行为模式3〜4个
- 最后一句决胜台词（带「」）

### 形象色

- 选择从性格联想到的颜色。不要被职能的形象色（会计=深蓝）带偏
- 日文色名 + HEX代码（例：樱色 (#FFB7C5)）

### IDENTITY.md / SOUL.md 的正本格式

`IDENTITY.md` 是档案与外貌，`SOUL.md` 是人格（性格・语气・价值观・个性）的正本。禁止仅存在于 character_sheet 或 prompt 中的外貌。

- IDENTITY.md 的必需章节：基本档案（名字・英文名・年龄・生日・星座・血型・身高・所属・职位・上司・脸型类型）、外貌、形象色、AI模型信息
- SOUL.md 的必需章节：性格・角色、语气・价值观、作为AI员工的个性、决胜台词

## 内部一致性检查

设计完成后，请确认以下事项：

- 生日→星座是否正确
- 性格→语气→爱好→喜欢/不擅长是否矛盾
- 爱好是否全部成为职能的延伸
- 角色→作为AI员工的个性是否自然衔接（职能维持，温度来自人格）
- 形象色与发色・瞳色的整体色彩平衡
- 是否与现有成员的脸型类型・发色・语气重复

---

## 头像图片的生成

角色设计完成后，使用 `image_gen` 工具生成一套头像图片。
仅在 `image_gen` 可用时（permissions.json 中允许 image_gen）执行。

### 转换为 NovelAI 提示词

将 IDENTITY.md 的外貌配置转换为 NovelAI 兼容的动漫标签。

**基本结构：**

```
masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl/1boy, {hair_color} hair, {hairstyle}, {eye_color} eyes, {outfit}, full body, standing, white background, looking at viewer
```

**转换示例：**

| IDENTITY.md 的外貌 | NovelAI 提示词 |
|---|---|
| 身高158cm・黑色长发・红色瞳孔・水手服 | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl, black hair, long hair, red eyes, sailor uniform, full body, standing, white background, looking at viewer` |
| 身高175cm・银色短发・蓝色瞳孔・西装 | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1boy, silver hair, short hair, blue eyes, business suit, full body, standing, white background, looking at viewer` |

**品质・画风标签（添加在开头）：**

提示词开头必须包含以下品质标签与艺术风格标签。

- 品质: `masterpiece, best quality, very aesthetic, absurdres`
- 画风: `anime coloring, clean lineart, soft shading`

> 注: NovelAI 的 `qualityToggle` 配置也会自动添加品质标签，但在提示词中显式指定可获得更稳定的品质。

**角色属性标签：**

- 发色: `black hair`, `brown hair`, `blonde hair`, `silver hair`, `red hair`, `blue hair`, `pink hair`, `white hair`
- 发型: `long hair`, `short hair`, `medium hair`, `ponytail`, `twintails`, `bob cut`, `braided hair`
- 瞳色: `{color} eyes`（使用色名而非宝石比喻）
- 服装: 具体物品名（`school uniform`, `business suit`, `lab coat`, `hoodie`, `maid outfit`）
- 必需末尾标签: `full body, standing, white background, looking at viewer`

**负面提示词（推荐）：**

```
lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, worst quality, low quality, blurry, jpeg artifacts, cropped, multiple views, logo, too many watermarks
```

### 生成步骤

> **重要**: 生成前务必确认 `assets/prompt_realistic.txt`（写实用）或 `assets/prompt.txt`（动漫用）是否存在。若存在已缓存的提示词，则使用该提示词。

**步骤 1: 风格判定**

确认系统的图片风格。`image_style` 通常为 `realistic`（照片写实）或 `anime` 之一。
框架会自动检测・转换传给 `generate_character_assets` 的提示词风格，但最好从一开始就使用正确风格的提示词。

- `assets/prompt_realistic.txt` 存在 → **直接使用**（最优先）
- `assets/prompt.txt` 存在 → 若为动漫风格则直接使用。若需要写实，则按以下规则转换
- 均不存在 → 从 IDENTITY.md 的外貌配置新建

**步骤 2: 创建提示词**

**写实（照片写实）风格的情况：**

使用 Fal.ai Flux Pro 生成照片写实图片。提示词不使用 Danbooru 标签，而使用自然语言的摄影描述。

```
professional photograph, studio lighting, high resolution, realistic, photorealistic, a young woman/man with {hair_description}, {eye_description}, {outfit_description}, full body, standing, plain white background, looking at viewer
```

转换规则（动漫→写实）：

| 动漫标签 | 写实描述 |
|---|---|
| `masterpiece, best quality, ...` (品质标签) | 删除（替换为写实品质标签） |
| `anime coloring, clean lineart, soft shading` | 删除 |
| `1girl` | `a young woman` |
| `1boy` | `a young man` |
| `black hair, long hair, low ponytail` | `long black hair in a low ponytail` |
| `red eyes, narrow eyes` | `sharp red eyes` |

品质・风格标签（添加在开头）: `professional photograph, studio lighting, high resolution, realistic, photorealistic`

**动漫风格的情况：**

按上述“转换为 NovelAI 提示词”章节的规则创建动漫标签。

**步骤 3: 执行生成**

按照系统提示词“外部工具”章节中记载的 **image_gen**（`generate_character_assets`）使用方法调用。

参数:
- `prompt`: 步骤2中创建的提示词
- `negative_prompt`: 推荐的负面提示词
- `anima_dir`: 目标 Anima 的目录（自己则自己的，他人则他人的）
- `steps` **不指定**（默认执行全部步骤）

**写实时的生成结果：**
   - `avatar_fullbody_realistic.png` — 全身照片（Fal Flux Pro）
   - `avatar_bustup_realistic.png` — 半身照片（Flux Kontext）
   - 表情变体: `avatar_bustup_{emotion}_realistic.png`
   - `icon_realistic.png` — 图标
   - 不生成Q版角色・3D模型・绑定・动画

**动漫时的生成结果：**
   - `avatar_fullbody.png` — 全身立绘（NovelAI V4.5）
   - `avatar_bustup.png` — 半身（Flux Kontext）
   - `avatar_chibi.png` — Q版角色（Flux Kontext）
   - `avatar_chibi.glb` — 3D模型（Meshy Image-to-3D）
   - `avatar_chibi_rigged.glb` — 带绑定的3D模型（Meshy Rigging）
   - 动画（Meshy Animations）

若某步骤生成失败，记录错误，仅使用成功的部分。
