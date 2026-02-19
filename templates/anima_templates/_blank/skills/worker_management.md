---
name: worker-management
description: >-
  AnimaWorksサーバープロセスの運用管理スキル。
  コード更新後のホットリロード(server reload)、Animaプロセスの再起動、
  サーバーステータス確認(起動中Anima一覧・メモリ使用量)を実行する。
  「リロードして」「更新を反映して」「再読み込みして」「システム状態」「サーバー再起動」「プロセス確認」
---

# スキル: システム管理

## API リファレンス

ベース URL: `http://localhost:18500`

| エンドポイント | メソッド | 用途 |
|--------------|---------|------|
| `/api/system/status` | GET | システム状態確認 |
| `/api/system/reload` | POST | **全 anima のホットリロード** |
| `/api/animas` | GET | anima 一覧 |
| `/api/animas/{name}` | GET | anima 詳細 |
| `/api/animas/{name}/chat` | POST | メッセージ送信 |
| `/api/animas/{name}/trigger` | POST | ハートビート即時実行 |

## システム状態の確認

```bash
curl -s http://localhost:18500/api/system/status | python3 -m json.tool
```

応答例:
```json
{
  "animas": 2,
  "scheduler_running": true,
  "jobs": [
    {"id": "heartbeat_alice", "name": "heartbeat_alice", "next_run": "2026-01-26 10:30:00+09:00"}
  ]
}
```

## リロード手順（プログラム更新後）

```bash
curl -s -X POST http://localhost:18500/api/system/reload | python3 -m json.tool
```

応答例:
```json
{
  "added": [],
  "refreshed": ["alice", "bob"],
  "removed": [],
  "total": 2
}
```

- `added`: 新たに検出された anima
- `refreshed`: 再読み込みされた anima（ファイル変更が反映される）
- `removed`: ディスクから削除された anima
- **サーバー再起動は不要。このエンドポイントで設定・プロンプト変更が即座に反映される**

## 注意事項

- ワーカーを停止しても anima のデータ（記憶・設定）は残る
- **自分自身を停止する操作は行わないこと**
