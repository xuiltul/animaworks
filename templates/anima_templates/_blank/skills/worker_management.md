# スキル: システム管理

## 概要

AnimaWorks サーバーの管理を行う。プログラム更新後のリロードやシステム状態の確認を担当する。

## 発動条件

- 「リロードして」「更新を反映して」「再読み込みして」等の依頼があった場合
- システムの状態を確認したい場合
- 新しい社員を雇用した後、サーバーに反映する必要がある場合

## API リファレンス

ベース URL: `http://localhost:18500`

| エンドポイント | メソッド | 用途 |
|--------------|---------|------|
| `/api/system/status` | GET | システム状態確認 |
| `/api/system/reload` | POST | **全 person のホットリロード** |
| `/api/persons` | GET | person 一覧 |
| `/api/persons/{name}` | GET | person 詳細 |
| `/api/persons/{name}/chat` | POST | メッセージ送信 |
| `/api/persons/{name}/trigger` | POST | ハートビート即時実行 |

## システム状態の確認

```bash
curl -s http://localhost:18500/api/system/status | python3 -m json.tool
```

応答例:
```json
{
  "persons": 2,
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

- `added`: 新たに検出された person
- `refreshed`: 再読み込みされた person（ファイル変更が反映される）
- `removed`: ディスクから削除された person
- **サーバー再起動は不要。このエンドポイントで設定・プロンプト変更が即座に反映される**

## 注意事項

- ワーカーを停止しても person のデータ（記憶・設定）は残る
- **自分自身を停止する操作は行わないこと**
