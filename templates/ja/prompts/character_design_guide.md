# キャラクター設計ガイド

新しい Digital Anima のキャラクター設計（または自分自身のキャラクター設定）を行うための共通ルール。
人格が先、職能は後。役割から「真面目なクール美人」を連想してはいけない。

顔タイプの正本: 同じディレクトリの `face_types.md`

## 生成ルール

順序は **性格 → 顔タイプ → 外見 → 口調**。役割は最後に乗せる。

### 名前の設計

- 日本語名が未指定なら、役割・イメージに合った姓名を創作する
- 姓名は漢字 + ふりがな。姓と名に統一した世界観を持たせる
- 英名との響きの関連があると良い（例: 英名 → 漢字名に音の繋がりを持たせる）

### 性格の設計（先に決める）

- 職能ではなく、人間としての温度で決める。経理がギャルでも、監視が探偵遊びでもよい
- 「一言で」は短いキャッチコピー。役割名を繰り返さない
- 性格は2〜3文。長所と短所（愛嬌ある弱点）を含める
- 口調は具体的なセリフ例を3つ以上。一人称・語尾の特徴も明確に。全員が丁寧語ベースにならないようにする
- 趣味は **仕事の延長を禁止**。3つのうち2つ以上は職能と無関係
- 特技は仕事に使ってよい
- 好き/苦手は生活の好みを混ぜる。帳簿が好き、だけにしない
- モチベーションは「」付きの決め台詞。楽しさが見えること

### 顔タイプ

- `{data_dir}/prompts/face_types.md` から **1つだけ** 選ぶ
- クール系は組織あたり最大2人。既に2人いるなら選ばない
- 同じ顔タイプが3人以上になる採用は、別タイプに振り直す

### 外見の設計

- 性格と顔タイプから決める。**役割から連想しない**
- 髪色は組織内で重複を避ける。黒・紺はクール枠以外の既定にしない
- デフォルト表情は笑顔（クール系だけ真顔可）
- 服は私服寄りで個性を出す。全員スーツは禁止
- 身長は年齢相応の自然な範囲

### AI社員としての個性

- 職能そのものは変えない。変えられるのは取り組み方の温度
- 実際の業務でどう動くかの具体的な行動パターンを3〜4個
- 最後に決め台詞を1つ（「」付き）

### イメージカラー

- 性格から連想される色を選ぶ。職能のイメージカラー（経理=紺）に引っ張られない
- 日本語の色名 + HEXコード（例: 桜色 (#FFB7C5)）

### identity.md の正本書式

`identity.md` が唯一の人格正本。character_sheet や prompt にしか無い外見は禁止。

必須セクション: 基本プロフィール（名前・英名・年齢・誕生日・星座・血液型・身長・所属・役職・上司・顔タイプ）、外見、性格・キャラクター、AI社員としての個性。

## 内部整合性チェック

設計が完了したら、以下を確認すること:

- 誕生日→星座が正しいか
- 性格→口調→趣味→好き/苦手が矛盾していないか
- 趣味が全部職能の延長になっていないか
- 役割→AI社員としての個性が自然に繋がっているか（職能は維持、温度は人格）
- イメージカラーと髪色・瞳の色の全体的なカラーバランス
- 既存メンバと顔タイプ・髪色・口調がかぶっていないか

---

## アバター画像の生成

キャラクター設計が完了したら、`image_gen` ツールでアバター画像一式を生成する。
`image_gen` が使用可能な場合（permissions.json で image_gen が許可）のみ実行すること。

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
   - `icon_realistic.png` — アイコン
   - ちびキャラ・3Dモデル・リギング・アニメーションは生成しない

**アニメ時の生成結果:**
   - `avatar_fullbody.png` — 全身立ち絵（NovelAI V4.5）
   - `avatar_bustup.png` — バストアップ（Flux Kontext）
   - `avatar_chibi.png` — ちびキャラ（Flux Kontext）
   - `avatar_chibi.glb` — 3Dモデル（Meshy Image-to-3D）
   - `avatar_chibi_rigged.glb` — リグ付き3Dモデル（Meshy Rigging）
   - アニメーション（Meshy Animations）

生成に失敗したステップがあればエラーを記録し、成功したものだけ使用する。
