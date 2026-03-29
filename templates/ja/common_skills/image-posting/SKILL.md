---
name: image-posting
description: >-
  チャット応答へ画像を埋め込み表示するスキル。ツール結果のURL検出、Markdown画像構文、assets配下の表示手順を扱う。
  Use when: 検索・生成ツールの画像を返信に載せる、Markdownで画像を貼る、添付ファイルを表示するとき。
---

# image-posting — チャット応答への画像表示

## 概要

チャット応答に画像を含める仕組みは2系統ある:

1. **ツール結果からの自動抽出** — ツール結果に画像URLやパスが含まれると、フレームワークが自動検出してチャットバブルに表示する
2. **Markdown画像構文** — 応答テキスト内に `![alt](url)` を書くとフロントエンドがレンダリングする

## 方法1: ツール結果からの自動表示

ツール（web_search、image_gen等）を呼び出した結果に画像情報が含まれていれば、フレームワークが自動でチャットバブルに画像を表示する。Anima側で特別な操作は不要。

### 自動検出される条件

ツール結果のJSON内で以下が検出されると画像として扱われる:

- **パス検出**: `path`, `file`, `filepath`, `asset_path` キーの値、または結果文字列内に `assets/` / `attachments/` で始まるパス（`.png` `.jpg` `.jpeg` `.gif` `.webp`）→ `source: generated`（信頼済み）
- **URL検出**: `url`, `image_url`, `thumbnail`, `src` キーに画像URLがある場合 → `source: searched`（プロキシ経由、許可ドメインのみ）
- **image_gen専用**: ツール結果全体を正規表現で走査し、`assets/` または `attachments/` を含むパスを自動抽出

1応答あたり最大5枚まで。

### image_gen ツールの実装とパイプライン

- **エントリ**: `core/tools/image_gen.py` の `dispatch()` が `generate_character_assets` ほか各ツール名を処理する。GLB 周りのテスト用可変属性（`_FBX2GLTF_PATH` 等）は `_image_glb` へプロキシされるファサードでもある。
- **一括生成の本体**: `core/tools/_image_pipeline.py` の `ImageGenPipeline.generate_all()` が7ステップをオーケストする。
- **APIクライアント・定数・プロンプト**: `core/tools/image/`（例: `novelai.py`, `fal.py`, `meshy.py`, `constants.py` の `NOVELAI_MODEL` / `_DEFAULT_ANIMATIONS`, `prompts.py` の表情用プロンプト）。

**7ステップの内容**（アニメ系で `steps` 未指定のフルパイプライン時）:

1. **fullbody** — **アニメ**（`image_style` が `realistic` 以外）: `NOVELAI_TOKEN` があれば NovelAI（`NOVELAI_MODEL` = `nai-diffusion-4-5-full`）。無ければ `FAL_KEY` があれば Fal Flux Pro。どちらも無ければエラー。**リアリスティック**: 常に Fal Flux Pro のみ（`FAL_KEY` 必須。NovelAI にはフォールバックしない）。`config.image_gen` の `style_prefix` / `style_suffix` / `negative_prompt_extra`、`style_reference`（画像バイト）および Vibe Transfer 用の `vibe_strength` / `vibe_info_extracted` がここで反映される。
2. **bustup** — Flux Kontext（fal）でバストアップ。**表情名**は `core.schemas.VALID_EMOTIONS`（`neutral`, `smile`, `laugh`, `troubled`, `surprised`, `thinking`, `embarrassed`）のみ有効。デフォルトでは上記すべてを生成。`neutral` は全身参照から、その他は可能なら中立バストを参照（無ければ全身にフォールバック）。リアル/アニメで別プロンプト・ガイダンス（`prompts.py`）。
3. **icon** — 中立バストを参照に Flux Kontext で正方形アイコン。成功時、`persist_anima_icon_path_template()` でアイコンパステンプレを更新しようとする（失敗しても処理は継続）。
4. **chibi** — 全身を参照に Flux Kontext でちびキャラ画像。
5. **3d** — Meshy Image-to-3D（デフォルト `ai_model`: `meshy-6`）で chibi から `avatar_chibi.glb`。
6. **rigging** — パイプライン内では **直前の Image-to-3D の `task_id` から** `create_rigging_task` でリギング（`input_task_id` 経由）。リグ済み GLB 保存後 `optimize_glb`、付属の歩行などは `download_rigging_animations` で `anim_*.glb` 化（可能なら `strip_mesh_from_glb`）。**単体ツール** `generate_rigged_model` / `generate_animations` の `dispatch` 側は、既存 GLB を data URI にして `MESHY_RIGGING_URL` へ POST する経路（パイプライン6と入力経路が異なる点に注意）。
7. **animations** — `animations` 引数が渡されなければ `_DEFAULT_ANIMATIONS`（`idle`, `sitting`, `waving`, `talking` と Meshy アクション ID）。同一実行でリギングしていない場合、既存の `avatar_chibi.glb` から `create_rigging_task_from_glb` で `rig_task_id` を取り直してから追加アニメを生成する。

**リアリスティックスタイル**（`config.image_gen.image_style == "realistic"`）では、`steps` 未指定時の**デフォルト有効ステップは `fullbody` / `bustup` / `icon` のみ**（chibi・3D・リギング・追加アニメは既定では走らない）。`dispatch` の `generate_character_assets` では、プロンプトが Danbooru 風タグに見える場合 `_looks_like_anime_prompt` により `_convert_anime_to_realistic` でリアル向けに自動変換されることがある。

**スタイル参照（Vibe Transfer）**: `ImageGenPipeline` は `config.style_reference` のパス（存在時）を全身生成に渡す。加えて `generate_character_assets` で `supervisor_name` が指定され、上司の `assets/` に `avatar_fullbody.png` または `avatar_fullbody_realistic.png`（スタイルに応じて）があれば、それを `style_reference` として上書き読み込みする。`generate_all()` には `vibe_image` / `vibe_strength` / `vibe_info_extracted` / `seed` / `expressions` / `steps` / `progress_callback` などもあるが、**現在の `dispatch` がツール引数から渡すのは** `prompt`, `negative_prompt`, `skip_existing`, `steps`, `animations`, `supervisor_name`（およびハンドラ注入の `anima_dir`）に限られる。

**`generate_fullbody`（単体）**: `dispatch` 実装は **常に `NovelAIClient` のみ**（環境に Fal があってもフォールバックしない）。`image_style` も参照しない。出力は常に `avatar_fullbody.png`。リアリスティック用の全身やパイプライン整合が必要なら **`generate_character_assets` を使う**。

### `generate_character_assets` の戻り値（JSON）

`PipelineResult.to_dict()` 相当のキー（パスは文字列、無ければ `null`）:

| キー | 内容 |
|------|------|
| `fullbody` | 全身 PNG |
| `bustup` | 代表バストアップ（通常は中立） |
| `bustup_expressions` | 表情名 → パスの辞書 |
| `icon` | チャットアイコン PNG |
| `chibi` | ちび画像 PNG |
| `model` | `avatar_chibi.glb` |
| `rigged_model` | リギング済み GLB |
| `animations` | アニメ名 → GLB パス |
| `errors` / `skipped` | エラー文の配列 / スキップしたステップ名 |

PNG のパスはチャット自動表示の対象。GLB はアセット保存・ワークスペース等で別扱い。

### 出力ファイル名の対応表

**パイプライン（アニメ `image_style`）**

| ステップ | 出力例 | チャット自動表示 |
|----------|--------|------------------|
| fullbody | `avatar_fullbody.png` | ○ |
| bustup | `avatar_bustup.png`（中立）、`avatar_bustup_smile.png` 等 | ○ |
| icon | `icon.png` | ○ |
| chibi | `avatar_chibi.png` | ○ |
| 3d | `avatar_chibi.glb` | — |
| rigging | `avatar_chibi_rigged.glb`, `anim_*.glb`（歩行等） | — |
| animations | `anim_idle.glb`, `anim_sitting.glb`, … | — |

**パイプライン（リアリスティック）** — 全身・バスト・表情・アイコンはファイル名末尾に `_realistic` が付く:

- 全身: `avatar_fullbody_realistic.png`
- 中立バスト: `avatar_bustup_realistic.png`
- 表情: `avatar_bustup_smile_realistic.png` など（`bustup_expressions` に集約）
- アイコン: `icon_realistic.png`

**単体ディスパッチ**（`dispatch` 内の個別ツール）

| ツール名 | 出力 | チャット自動表示 | 備考 |
|----------|------|------------------|------|
| `generate_fullbody` | `avatar_fullbody.png` | ○ | **NovelAI のみ**（`NOVELAI_TOKEN`）。Fal フォールバック・`image_style` 非対応。ファイル名固定 |
| `generate_bustup` | `avatar_bustup.png` | ○ | 参照は `avatar_fullbody.png` のみ |
| `generate_icon` | `icon.png` / `icon_realistic.png` | ○ | 参照バストは設定の `image_style` に応じたファイル名 |
| `generate_chibi` | `avatar_chibi.png` | ○ | 参照は `avatar_fullbody.png` のみ |
| `generate_3d_model` | `avatar_chibi.glb` | — | |
| `generate_rigged_model` | `avatar_chibi_rigged.glb` + `animations` 辞書 | — | GLB を data URI で Meshy リギング API に送信 |
| `generate_animations` | `animations` 辞書（`anim_*.glb`） | — | 既存 `avatar_chibi.glb` からリグタスクを取得して生成 |

デフォルトの追加アニメーション名と Meshy アクション ID（パイプライン `animations` ステップ）: `idle: 0`, `sitting: 32`, `waving: 28`, `talking: 307`（`animations` 引数で上書き可）。

### searched画像のプロキシ制限

外部URL画像はセキュリティのためプロキシ経由で配信される。**アーティファクト抽出時点**で以下の許可ドメインのみが検出対象となる:

- `cdn.search.brave.com`
- `images.unsplash.com`
- `images.pexels.com`
- `upload.wikimedia.org`

上記以外のドメインのURLはツール結果に含まれていても自動表示されない。プロキシ自体はHTTPS強制・private/local拒否・magic bytes検証・SVG拒否・サイズ・レート制限などの安全検査を実施する（`config.server.media_proxy` で `open_with_scan` / `allowlist` モードを切り替え可能）。

## 方法2: Markdown画像構文

応答テキスト内にMarkdown画像構文を直接書いて画像を表示する。

### 短縮パス（推奨）

フロントエンドが自動的に自分のAnima名でAPIパスを補完する。ファイル名だけ書けばOK:

```
![説明](attachments/ファイル名)
![説明](assets/ファイル名)
```

例:

```
スクショ撮りました！
![ANAトップページ](attachments/ana_top.png)
```

### フルパス

明示的にAPIパスを書くこともできる:

```
![説明](/api/animas/{自分の名前}/assets/{ファイル名})
![説明](/api/animas/{自分の名前}/attachments/{ファイル名})
```

## スクリーンショットの保存先

agent-browser等でスクリーンショットを撮る場合、**自分のattachmentsディレクトリに直接保存する**のが確実:

```bash
agent-browser screenshot ~/.animaworks/animas/{自分の名前}/attachments/screenshot.png
```

例（aoiの場合）:

```bash
agent-browser screenshot ~/.animaworks/animas/aoi/attachments/page_screenshot.png
```

保存後、応答に以下を書けば表示される:

```
![ページのスクショ](attachments/page_screenshot.png)
```

`~/.animaworks/tmp/attachments/` に保存した場合もフォールバックで表示されるが、一時ディレクトリなので永続性は保証されない。

## 注意事項

- 他のAnimaのアセットパスは直接参照できない（権限外）
- 外部URLの直リンクは非推奨。許可ドメイン外は自動表示されず、プロキシの安全検査でブロックされる場合がある
- `generate_character_assets` の結果は `fullbody` / `bustup` / `bustup_expressions` / `icon` / `chibi` などのパスがJSONに含まれるため、PNGは自動表示され、Markdown構文は不要なことが多い
- 単体の `generate_bustup` / `generate_chibi` は参照が **`avatar_fullbody.png` 固定**（リアリスティック用の `avatar_fullbody_realistic.png` は読まない）。単体 `generate_fullbody` は上表のとおり NovelAI 専用で、パイプラインのリアリスティック全身とファイル名・生成経路が揃わない。パイプラインと単体ツールを混用する場合は `assets/` の実ファイルを確認すること
- 画像生成は処理時間が長い。フレームワークでは専用スレッドプール実行・バックグラウンドツール設定の対象になる。CLI からは `animaworks-tool submit image_gen …` 等の非同期実行が推奨される場合がある（`common_knowledge/operations/background-tasks.md` 等を参照）
- 1応答あたりの自動表示は最大5枚
