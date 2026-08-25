---
name: bustup-frame-gen
description: >-
  音声ポップアップの疑似Live2Dアニメーション用フレームセット（表情×5フレーム）を、
  grokのimage_editで既存bustupから生成しPIL合成で揃える手順。Animaのボイスチャットで
  口パク・瞬きを動かしたい、新しいanimaにアニメーションフレームを追加したい、
  「bustupをLive2D風に動かして」「口パクフレームを作って」と言われたときに使う。
  リギング・Live2D SDKは使わない（AIアバター用途は別途ライセンス契約対象になり得るため不採用）。
---

# Bustupアニメーションフレーム生成 — 疑似Live2D方式

音声ポップアップ（animaタブ長押し）のbustupを、静止画フレーム切替+CSS/JSで動かすための
アセット生成手順。2026-08-20にmeiで確立。

方式: リギングなし。口3段階×瞬きのフレーム切替 + CSS呼吸/揺れ + lerp視線追従 +
TTS音量(RMS)口パク。AI VTuberのデファクト（感情タグ→プリセット切替、Neuro-samaも同方式）。

## 成果物の契約（フロントエンドが期待する形）

配置先 `<data_dir>/animas/<name>/assets/`（通常 `~/.animaworks/animas/<name>/assets/`）:

```
avatar_bustup_<expr>_frame_base.png       # 目開き+口閉じ（その表情の口）
avatar_bustup_<expr>_frame_blink.png      # 目閉じ
avatar_bustup_<expr>_frame_half.png       # 口半開き
avatar_bustup_<expr>_frame_open.png       # 口開き
avatar_bustup_<expr>_frame_blinkhalf.png  # 目閉じ+口半開き
```

- `<expr>` は voice-popup.js の `VALID_EXPRESSIONS` と同じ7種:
  neutral / smile / laugh / troubled / surprised / thinking / embarrassed。
  **neutralも明示的に `_neutral_` を含む**（既存 `avatar_bustup.png` の命名特例は適用しない）
- 5枚×7表情=35枚。フロントは `_frame_base.png` へのHEAD 1発で存在判定するので
  **表情単位で5枚揃えて置く**（中途半端に置かない）
- フレームが無いanima/表情は自動で静的bustupへフォールバック（mei以外は当面フレーム無しでよい）
- 描画側の実装は `server/static/pages/chat/bustup-animator.js`（変更不要）。
  仕様の正本は `docs/issues/20260820_mei-pseudo-live2d-popup.md`

## 手順

### 1. 作業ディレクトリ準備

scratchpadに表情ごとのサブディレクトリを作り、既存bustupを `src.png` としてコピーする。

```bash
W=<scratchpad>/bustup-frames; A=~/.animaworks/animas/<name>/assets
for e in neutral smile laugh troubled surprised thinking embarrassed; do
  mkdir -p $W/$e
  cp $A/avatar_bustup_${e}.png $W/$e/src.png   # neutralは avatar_bustup.png の場合あり
done
```

### 2. grok image_editで差分3枚を生成（表情ごと）

`scripts/gen_one.sh <expr_dir>` を直列実行する（grokのheadlessに `image_edit` ツールがある。
プロンプト全文はスクリプト内。要点: src.png と完全同一を保ち、指定の1変化だけ加える）。

```bash
for e in neutral smile laugh troubled surprised thinking embarrassed; do
  bash .claude/skills/bustup-frame-gen/scripts/gen_one.sh $W/$e
done
```

各dirに `eyes_closed.png` / `mouth_half.png` / `mouth_open.png` ができる。
1表情10分程度（timeout 600）。

### 3. PIL合成で残り2枚を作る — ここが最大の罠

**grok image_editは元画像と解像度・フレーミングが変わる**（実測: 1024×1365→864×1152）。
ただし**同一呼び出しで生成された差分3枚同士はピクセル一致**する。したがって:

- `src.png`（元bustup）とはいかなる合成もしない。**frame_base すら src からは作れない**
- base（目開き+口閉じ）と blinkhalf（目閉じ+口半開き）は差分同士の口領域pasteで合成する

```bash
python3 .claude/skills/bustup-frame-gen/scripts/compose_frames.py $W/neutral $W/smile ...
```

口領域は `mouth_half` と `eyes_closed` の差分の行分布から自動検出（最下クラスタ=口、
マージン18px）。スクリプトがサイズ一致assertと検出失敗assertを持つ。

**smile/laughの注意**: 元絵が笑い目+開口の表情では「rest状態=元表情の見た目」になるよう
合成結果を目視確認する（口閉じフレームが不自然に真顔化していないか）。

### 4. 目視検証

配置前に必ず各表情5枚を実際に開いて確認する:

- 差分が目・口以外に漏れていないか（髪・服・背景が変わっていたらその表情を再生成）
- 5枚の間でフレーミングが揃っているか（1枚だけズレていると切替時にガタつく）

### 5. 配置

```bash
A=~/.animaworks/animas/<name>/assets
for e in neutral smile laugh troubled surprised thinking embarrassed; do
  cp $W/$e/frame_base.png      $A/avatar_bustup_${e}_frame_base.png
  cp $W/$e/eyes_closed.png     $A/avatar_bustup_${e}_frame_blink.png
  cp $W/$e/mouth_half.png      $A/avatar_bustup_${e}_frame_half.png
  cp $W/$e/mouth_open.png      $A/avatar_bustup_${e}_frame_open.png
  cp $W/$e/frame_blinkhalf.png $A/avatar_bustup_${e}_frame_blinkhalf.png
done
ls $A/*_frame_*.png | wc -l   # 35 になること
```

静的JSもアセットもサーバー再起動不要で即反映される。

### 6. 実機E2E確認

**nginx経由は認証で503になる**ので localhost:18500 直結 + agent-browser を使う:

```bash
curl -s "http://127.0.0.1:18500/api/animas/<name>/assets/avatar_bustup_neutral_frame_base.png" \
  -o /dev/null -w "%{http_code}\n"   # 200

# agent-browserでチャットページを開き、ポップアップを直接呼ぶ
agent-browser eval "openVoicePopup('<name>')"
agent-browser eval "document.querySelector('.bustup-animator-set.active') ? 'active-set' : 'no-active'"
```

確認項目: 常時の呼吸・揺れ・ランダム瞬き / TTS再生中の音量連動口パク /
感情タグでの表情クロスフェード / フレーム無しanimaが静的bustupのままであること。

## 罠まとめ

| 罠 | 対処 |
| --- | --- |
| image_editの解像度・フレーミング変化 | 差分同士だけで合成。src.pngとは絶対に混ぜない |
| Live2D Cubism SDK | AIアバター用途は規模問わず別途ライセンス契約対象になり得る。使わない |
| nginx経由のE2Eが503 | localhost:18500直結 + `agent-browser eval` |
| 表情フレームの部分配置 | `_frame_base.png` の存在=5枚全部あるとみなされる。5枚単位で置く |
| 笑い目表情のbase合成 | rest状態が元表情の見た目になっているか目視必須 |
