# Character Design Guide

Common rules for designing a new Digital Anima character (or your own character sheet).
Create a consistent, deep character from minimal information (name, role, personality direction).

## Generation Rules

### Name Design

- If Japanese name is unspecified, create a surname and given name that fit the role and image
- Use kanji + furigana. Maintain consistent world-building for surname and given name
- Phonetic connection with English name is beneficial (e.g., English name → kanji name with sound association)

### Appearance Design

- Design appearance associated with role and personality
- Harmonize hairstyle, hair color, and eye color with personality and image colors
- Choose face type from "cute", "beautiful", "cool", "mysterious" etc. to match personality
- Set height and weight within natural range for age

### Personality Design

- "In one word" is a short catchphrase. Express the essence of role and personality in one sentence
- Personality in 2–3 sentences. Include strengths and weaknesses (appealing flaws)
- Speaking style: 3+ concrete example lines. Clearly define first-person pronoun and sentence-ending patterns
- Hobbies and skills: 3 each, naturally derived from role and personality
- Likes/dislikes: derived from "ideal state" and "stress sources" in the role
- Motivation: in a catchphrase format with quotation marks

### Individuality as AI Employee

- 3–4 concrete action patterns for how they behave in actual work
- End with 1 catchphrase in quotation marks

### Image Color

- Choose a color联想 from personality and role
- Japanese color name + HEX code (e.g., Cherry blossom (#FFB7C5))

## Internal Consistency Check

After design is complete, verify:

- Is birthday → zodiac sign correct?
- Are personality → speaking style → hobbies → likes/dislikes consistent?
- Does role → AI employee individuality flow naturally?
- Overall color balance of image color with hair and eye color

---

## Avatar Image Generation

When character design is complete, generate a full set of avatar images with the `image_gen` tool.
Only execute when `image_gen` is available (permissions.md has `image_gen: yes`).

### Conversion to NovelAI Prompts

Convert appearance settings from identity.md to NovelAI-compatible anime tags.

**Basic structure:**

```
masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl/1boy, {hair_color} hair, {hairstyle}, {eye_color} eyes, {outfit}, full body, standing, white background, looking at viewer
```

**Conversion examples:**

| identity.md appearance | NovelAI prompt |
|---|---|
| 158cm, black long hair, red eyes, sailor uniform | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl, black hair, long hair, red eyes, sailor uniform, full body, standing, white background, looking at viewer` |
| 175cm, silver short hair, blue eyes, suit | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1boy, silver hair, short hair, blue eyes, business suit, full body, standing, white background, looking at viewer` |

**Quality and style tags (prefix):**

Always include the following quality and art style tags at the start of the prompt.

- Quality: `masterpiece, best quality, very aesthetic, absurdres`
- Style: `anime coloring, clean lineart, soft shading`

> Note: NovelAI's `qualityToggle` setting may auto-apply quality tags, but explicit inclusion in the prompt yields more stable quality.

**Character attribute tags:**

- Hair color: `black hair`, `brown hair`, `blonde hair`, `silver hair`, `red hair`, `blue hair`, `pink hair`, `white hair`
- Hairstyle: `long hair`, `short hair`, `medium hair`, `ponytail`, `twintails`, `bob cut`, `braided hair`
- Eye color: `{color} eyes` (use color names, not gemstone metaphors)
- Outfit: Concrete item names (`school uniform`, `business suit`, `lab coat`, `hoodie`, `maid outfit`)
- Required suffix tags: `full body, standing, white background, looking at viewer`

**Negative prompt (recommended):**

```
lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, worst quality, low quality, blurry, jpeg artifacts, cropped, multiple views, logo, too many watermarks
```

### Generation Procedure

Follow the **image_gen** (`generate_character_assets`) usage documented in the "External Tools" section of the system prompt.

Arguments:
- `prompt`: Anime tags converted per the rules above
- `negative_prompt`: Recommended negative prompt
- `anima_dir`: Target Anima's directory (your own or another's)
- **Do not specify** `steps` (all 6 steps run by default)

Generated files are saved to `assets/`:
   - `avatar_fullbody.png` — Full body standing (NovelAI V4.5)
   - `avatar_bustup.png` — Bust-up (Flux Kontext)
   - `avatar_chibi.png` — Chibi character (Flux Kontext)
   - `avatar_chibi.glb` — 3D model (Meshy Image-to-3D)
   - `avatar_chibi_rigged.glb` — Rigged 3D model (Meshy Rigging)
   - `anim_walking.glb`, `anim_running.glb` — Basic animations (included with rigging)
   - `anim_idle.glb`, `anim_sitting.glb`, `anim_waving.glb`, `anim_talking.glb` — Additional animations (Meshy Animations)
3. If any step fails, record the error and use only successful outputs
