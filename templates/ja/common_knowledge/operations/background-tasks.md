# バックグラウンドタスク実行ガイド

## 概要

一部の外部ツール（画像生成、3Dモデル生成、ローカルLLM推論、音声文字起こし等）は
実行に数分〜数十分かかる。これらを直接実行すると、実行中ずっとロックが保持され、
メッセージの受信やheartbeatが停止してしまう。

`animaworks-tool submit` を使うことで、タスクをバックグラウンドで実行し、
自分自身はすぐに次の作業に移ることができる。

## いつ submit を使うか

### 必ず submit を使うべきツール

ツールガイド（システムプロンプト）に ⚠ マークが付いているサブコマンド:

- `image_gen pipeline` / `fullbody` / `bustup` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe`

### submit 不要のツール

実行時間が短い（30秒未満）ツール:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail`（通常の操作）
- `github`, `aws_collector`

### 判断基準

- ⚠ マークあり → 必ず submit
- ⚠ マークなし → 直接実行

## 使い方

### 基本構文

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### 実行例

```bash
# 3Dモデル生成（Meshy API、最大10分）
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# キャラクター画像一括生成（全ステップ、最大30分）
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# ローカルLLM推論（Ollama、最大5分）
animaworks-tool submit local_llm generate "要約してください: ..."

# 音声文字起こし（Whisper + Ollama整形、最大2分）
animaworks-tool submit transcribe "/path/to/audio.wav" --language ja
```

### 戻り値

submit は即座に以下のJSONを返して終了する:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "バックグラウンドタスクを投入しました。完了時にinboxに通知されます。"
}
```

## 結果の受け取り

1. submit 後、タスクはバックグラウンドで実行される
2. 完了すると `state/background_notifications/{task_id}.md` に結果が書かれる
3. 次回の heartbeat で自動的にこの通知を確認できる
4. 通知には成功/失敗のステータスと結果サマリが含まれる

## 失敗時の対応

- 通知に「失敗」と記載されている場合:
  1. エラー内容を確認する
  2. 原因を特定する（APIキー未設定、タイムアウト、引数ミス等）
  3. 修正して再度 submit する
  4. 解決できない場合は上司に報告する

## よくある間違い

### 直接実行してしまう

```bash
# 悪い例: 直接実行 → 10分間ロックされる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

直接実行してしまった場合、タスクが完了するまで待つしかない。
次回から必ず submit を使うこと。

### submit 後に結果を待ち続ける

submit したらすぐに次の作業に移ること。
結果は自動的に通知されるので、ポーリングや待機は不要。

## 技術的な仕組み（参考）

1. `animaworks-tool submit` が `state/background_tasks/pending/` にタスクJSONを書く
2. Runner の pending watcher が3秒間隔でこのディレクトリを監視
3. タスクを検出すると BackgroundTaskManager に投入（Anima の _lock 外で実行）
4. BackgroundTaskManager が asyncio.create_task() でスレッド実行
5. 完了時に `state/background_notifications/` に通知ファイルを書く
6. 次回 heartbeat で通知が処理される
