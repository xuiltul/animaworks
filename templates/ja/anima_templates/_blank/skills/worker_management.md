---
name: worker-management
description: >-
  AnimaWorksサーバープロセスの運用管理スキル。
  コード更新後のホットリロード(server reload)、Animaプロセスの再起動、
  サーバーステータス確認(起動中Anima一覧・メモリ使用量)を実行する。
  「リロードして」「更新を反映して」「再読み込みして」「システム状態」「サーバー再起動」「プロセス確認」
---

# スキル: システム管理

## CLIコマンド（推奨）

`animaworks` CLIで個別Anima管理が可能。**API直叩きよりCLIを優先すること。**

```bash
# 個別Animaのリスタート（設定変更の反映等）
animaworks anima restart <name>

# ステータス確認（全体 or 個別）
animaworks anima status
animaworks anima status <name>

# モデル変更（status.jsonを更新 + 自動リスタート）
animaworks anima set-model <name> <model>

# ロール変更
animaworks anima set-role <name> <role>

# Anima一覧
animaworks anima list

# 無効化 / 有効化
animaworks anima disable <name>
animaworks anima enable <name>

# 削除（--archive でバックアップ可）
animaworks anima delete <name>
```

### よくある使い方

```bash
# config.json変更後に特定Animaだけリスタート
animaworks anima restart tsumugi

# モデルを変更して自動リスタート
animaworks anima set-model tsumugi claude-sonnet-4-6
```

## API リファレンス（CLIが使えない場合）

ベース URL: `http://localhost:18500`

| エンドポイント | メソッド | 用途 |
|--------------|---------|------|
| `/api/system/status` | GET | システム状態確認 |
| `/api/system/reload` | POST | **全 anima のホットリロード** |
| `/api/animas` | GET | anima 一覧 |
| `/api/animas/{name}` | GET | anima 詳細 |
| `/api/animas/{name}/restart` | POST | 個別リスタート |
| `/api/animas/{name}/stop` | POST | 個別停止 |
| `/api/animas/{name}/start` | POST | 停止中animaの起動 |
| `/api/animas/{name}/chat` | POST | メッセージ送信 |
| `/api/animas/{name}/trigger` | POST | ハートビート即時実行 |

## リロード手順（プログラム更新後）

```bash
curl -s -X POST http://localhost:18500/api/system/reload | python3 -m json.tool
```

- `added`: 新たに検出された anima
- `refreshed`: 再読み込みされた anima（ファイル変更が反映される）
- `removed`: ディスクから削除された anima
- **サーバー再起動は不要。このエンドポイントで設定・プロンプト変更が即座に反映される**

## 注意事項

- ワーカーを停止しても anima のデータ（記憶・設定）は残る
- **自分自身を停止する操作は行わないこと**
- 個別操作はCLI → API の優先順位で使う
