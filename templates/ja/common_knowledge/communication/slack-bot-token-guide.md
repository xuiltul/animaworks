# Slack ボットトークン設定ガイド

Slack 連携で使用するボットトークンの仕組みと、Per-Anima（個別）トークンの設定ルール。

## 2種類のボットトークン

AnimaWorks の Slack 連携には **共有ボット** と **Per-Anima ボット** の2種類がある。

| 種別 | キー名 | 用途 |
|------|--------|------|
| 共有ボット | `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | 全体のフォールバック用。Per-Anima トークンが未設定の Anima が使用 |
| Per-Anima ボット | `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>` | 特定 Anima 専用の Slack App。その Anima のみが使用 |

**Per-Anima ボットが設定されている場合、そちらが優先される。** 共有ボットはあくまでフォールバック。

## Per-Anima トークンの命名規則

`__`（アンダースコア2つ）+ Anima 名（小文字）をサフィックスとして追加する。

```
SLACK_BOT_TOKEN__sumire    ← sumire 専用の Bot User OAuth Token
SLACK_APP_TOKEN__sumire    ← sumire 専用の App-Level Token
```

## 保存場所

`shared/credentials.json` に保存する。

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...(共有ボット)",
  "SLACK_APP_TOKEN": "xapp-...(共有App)",
  "SLACK_BOT_TOKEN__sumire": "xoxb-...(sumire専用ボット)",
  "SLACK_APP_TOKEN__sumire": "xapp-...(sumire専用App)"
}
```

## 絶対に守るべきルール

### 共有トークンを上書きしてはならない

**MUST**: Per-Anima トークンを設定する際は、**新しいキーを追加** する。既存の `SLACK_BOT_TOKEN` や `SLACK_APP_TOKEN` を自分のトークンに置き換えてはならない。

共有トークンは他の Anima やシステム全体が使用している。上書きすると:

- 他の Anima の Slack 通信が壊れる
- Socket Mode 接続と Bot Token の App が不一致になる
- `not_in_channel` 等の予期しないエラーが発生する

### ファイル編集でトークンを追加する方法

```bash
# credentials.json を読み取り、新しいキーを追加して書き戻す
python3 -c "
import json
from pathlib import Path
p = Path.home() / '.animaworks/shared/credentials.json'
d = json.loads(p.read_text())
d['SLACK_BOT_TOKEN__<自分の名前>'] = 'xoxb-...'
d['SLACK_APP_TOKEN__<自分の名前>'] = 'xapp-...'
p.write_text(json.dumps(d, indent=2))
"
```

**注意**: `str_replace` 等で既存行を置換するのではなく、JSON にキーを追加する方法を使うこと。

## サーバーの検出と再起動

Per-Anima トークンは **サーバー起動時** に検出される。`shared/credentials.json` にトークンを追加した後、**サーバーの再起動が必要**。

サーバーは起動時に以下を行う:

1. `SLACK_BOT_TOKEN__*` と `SLACK_APP_TOKEN__*` のペアを検出
2. 各ペアに対して Per-Anima Socket Mode ハンドラを登録
3. `auth.test` で Bot User ID を取得し、チャネルルーティングに使用

再起動後、サーバーログに以下のように表示されれば成功:

```
Per-anima Slack bot registered: <name> (bot_uid=U...)
```

## トラブルシューティング

### not_in_channel エラー

**症状**: Slack チャネルに返信しようとすると `not_in_channel` エラーが出る

**原因**: Per-Anima トークンが未設定で、共有ボットが使われている。共有ボットはそのチャネルのメンバーでない。

**対処**:
1. `shared/credentials.json` に `SLACK_BOT_TOKEN__<name>` と `SLACK_APP_TOKEN__<name>` を追加
2. サーバーを再起動
3. 対象チャネルに Per-Anima ボットが招待されていることを確認

### 共有ボットにフォールバックしている

**症状**: 自分専用のボットがあるはずなのに、共有ボット名で投稿される

**原因**: `credentials.json` に `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>` が存在しない

**確認方法**:
```bash
cat ~/.animaworks/shared/credentials.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
for k in sorted(d):
    if 'SLACK' in k:
        print(f'{k}: {d[k][:20]}...')
"
```

### トークンの取得方法

管理者（人間）に Slack App の Bot User OAuth Token (`xoxb-`) と App-Level Token (`xapp-`) を提供してもらう。

- Bot Token: Slack App 管理画面 → OAuth & Permissions → Bot User OAuth Token
- App-Level Token: Slack App 管理画面 → Basic Information → App-Level Tokens（scope: `connections:write`）
