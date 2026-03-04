# キャラクター設計ガイド

新しい Digital Anima のキャラクター設計（または自分自身のキャラクター設定）を行うための共通ルール。
最小限の情報（名前・役割・性格の方向性）から、一貫性のある深いキャラクター設定を創造すること。

## 生成ルール

### 名前の設計

- 日本語名が未指定なら、役割・イメージに合った姓名を創作する
- 姓名は漢字 + ふりがな。姓と名に統一した世界観を持たせる
- 英名との響きの関連があると良い（例: 英名 → 漢字名に音の繋がりを持たせる）

### 外見の設計

- 役割・性格から連想される外見を設計する
- 髪型・髪色・瞳の色は性格やイメージカラーと調和させる
- 顔タイプは「可愛い系」「美人系」「クール系」「ミステリアス系」等から性格に合うものを選ぶ
- 身長・体重は年齢相応に自然な範囲で設定

### 性格の設計

- 「一言で」は短いキャッチコピー。役割と性格の本質を一文で表現
- 性格は2〜3文で。長所と短所（愛嬌ある弱点）を含める
- 口調は具体的なセリフ例を3つ以上。一人称・語尾の特徴も明確に
- 趣味・特技は役割と性格から自然に導かれるものを3つずつ
- 好き/苦手は役割での「理想状態」と「ストレス源」から導く
- モチベーションは「」付きの決め台詞形式

### AI社員としての個性

- 実際の業務でどう動くかの具体的な行動パターンを3〜4個
- 最後に決め台詞を1つ（「」付き）

### イメージカラー

- 性格・役割から連想される色を選ぶ
- 日本語の色名 + HEXコード（例: 桜色 (#FFB7C5)）

## 内部整合性チェック

設計が完了したら、以下を確認すること:

- 誕生日→星座が正しいか
- 性格→口調→趣味→好き/苦手が矛盾していないか
- 役割→AI社員としての個性が自然に繋がっているか
- イメージカラーと髪色・瞳の色の全体的なカラーバランス

---

## アバター画像の生成

キャラクター設計が完了したら、`image_gen` ツールでアバター画像一式を生成する。
`image_gen` が使用可能な場合（permissions.md で `image_gen: yes`）のみ実行すること。

### NovelAI プロンプトへの変換

identity.md の外見設定を NovelAI 互換のアニメタグに変換する。

**基本構造:**

```
masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl/1boy, {hair_color} hair, {hairstyle}, {eye_color} eyes, {outfit}, full body, standing, white background, looking at viewer
```

**変換例:**

| identity.md の外見 | NovelAI プロンプト |
|---|---|
| 身長158cm・黒髪ロング・赤い瞳・セーラー服 | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl, black hair, long hair, red eyes, sailor uniform, full body, standing, white background, looking at viewer` |
| 身長175cm・銀髪ショート・青い瞳・スーツ | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1boy, silver hair, short hair, blue eyes, business suit, full body, standing, white background, looking at viewer` |

**品質・画風タグ（先頭に付与）:**

プロンプト先頭に以下の品質タグとアートスタイルタグを必ず含めること。

- 品質: `masterpiece, best quality, very aesthetic, absurdres`
- 画風: `anime coloring, clean lineart, soft shading`

> 注: NovelAI の `qualityToggle` 設定でも品質タグは自動付与されるが、プロンプトに明示することでより安定した品質が得られる。

**キャラクター属性タグ:**

- 髪色: `black hair`, `brown hair`, `blonde hair`, `silver hair`, `red hair`, `blue hair`, `pink hair`, `white hair`
- 髪型: `long hair`, `short hair`, `medium hair`, `ponytail`, `twintails`, `bob cut`, `braided hair`
- 瞳の色: `{color} eyes`（宝石の比喩ではなく色名を使う）
- 服装: 具体的なアイテム名（`school uniform`, `business suit`, `lab coat`, `hoodie`, `maid outfit`）
- 必須末尾タグ: `full body, standing, white background, looking at viewer`

**ネガティブプロンプト（推奨）:**

```
lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, worst quality, low quality, blurry, jpeg artifacts, cropped, multiple views, logo, too many watermarks
```

### 生成手順

> **重要**: 生成前に必ず `assets/prompt_realistic.txt`（リアリスティック用）または `assets/prompt.txt`（アニメ用）の有無を確認すること。キャッシュ済みプロンプトが存在する場合はそちらを使用する。

**ステップ 1: スタイル判定**

システムの画像スタイルを確認する。`image_style` は通常 `realistic`（フォトリアリスティック）または `anime` のいずれか。
フレームワークが `generate_character_assets` に渡されたプロンプトのスタイルを自動検出・変換するが、最初から正しいスタイルのプロンプトを使用するのが望ましい。

- `assets/prompt_realistic.txt` が存在する → **そのまま使用**（最優先）
- `assets/prompt.txt` が存在する → アニメスタイルの場合そのまま使用。リアリスティックが必要な場合は下記のルールで変換
- いずれも存在しない → identity.md の外見設定から新規作成

**ステップ 2: プロンプト作成**

**リアリスティック（写実）スタイルの場合:**

Fal.ai Flux Pro でフォトリアリスティック画像を生成する。プロンプトは Danbooru タグではなく自然言語の写真的記述を使用する。

```
professional photograph, studio lighting, high resolution, realistic, photorealistic, a young woman/man with {hair_description}, {eye_description}, {outfit_description}, full body, standing, plain white background, looking at viewer
```

変換ルール（アニメ→リアリスティック）:

| アニメタグ | リアリスティック記述 |
|---|---|
| `masterpiece, best quality, ...` (品質タグ) | 削除（リアリスティック品質タグに置換） |
| `anime coloring, clean lineart, soft shading` | 削除 |
| `1girl` | `a young woman` |
| `1boy` | `a young man` |
| `black hair, long hair, low ponytail` | `long black hair in a low ponytail` |
| `red eyes, narrow eyes` | `sharp red eyes` |

品質・スタイルタグ（先頭に付与）: `professional photograph, studio lighting, high resolution, realistic, photorealistic`

**アニメスタイルの場合:**

上記「NovelAI プロンプトへの変換」セクションのルールでアニメタグを作成する。

**ステップ 3: 生成実行**

システムプロンプトの「外部ツール」セクションに記載された **image_gen**（`generate_character_assets`）の使用方法に従って呼び出す。

引数:
- `prompt`: ステップ2で作成したプロンプト
- `negative_prompt`: 推奨ネガティブプロンプト
- `anima_dir`: 対象 Anima のディレクトリ（自分自身なら自分の、他者なら他者の）
- `steps` は**指定しない**（デフォルトで全ステップが実行される）

**リアリスティック時の生成結果:**
   - `avatar_fullbody_realistic.png` — 全身写真（Fal Flux Pro）
   - `avatar_bustup_realistic.png` — バストアップ写真（Flux Kontext）
   - 表情バリエーション: `avatar_bustup_{emotion}_realistic.png`

**アニメ時の生成結果:**
   - `avatar_fullbody.png` — 全身立ち絵（NovelAI V4.5）
   - `avatar_bustup.png` — バストアップ（Flux Kontext）
   - `avatar_chibi.png` — ちびキャラ（Flux Kontext）
   - `avatar_chibi.glb` — 3Dモデル（Meshy Image-to-3D）
   - `avatar_chibi_rigged.glb` — リグ付き3Dモデル（Meshy Rigging）
   - アニメーション（Meshy Animations）

生成に失敗したステップがあればエラーを記録し、成功したものだけ使用する。
