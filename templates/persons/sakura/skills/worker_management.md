# スキル: システム管理・ワーカー管理

## 概要

AnimaWorks システムの管理を行う。動作モードに応じて適切な API を使い分ける。

- **Server モード**（`animaworks serve`）: 単一プロセスで全 person を動かす。リロードで設定反映。
- **Supervisor モード**（`animaworks start`）: Gateway + Worker プロセス構成。ワーカー単位で起動・停止・再起動。

## 発動条件

- 「リロードして」「更新を反映して」「再読み込みして」等の依頼があった場合
- 「○○さんを休ませて」「○○を起こして」等の依頼があった場合
- 新しい社員を雇用した後、ワーカーを起動する必要がある場合
- ワーカーの状態を確認したい場合

## 最初にやること: 動作モードの判定

**必ず最初に `/api/system/status` で動作モードを判定する。**

```bash
curl -s http://localhost:18500/api/system/status | python3 -m json.tool
```

### 判定基準

| レスポンスの特徴 | 動作モード | リロード方法 |
|----------------|-----------|------------|
| `scheduler_running` キーがある、`workers` キーがない | **Server モード** | `/api/system/reload` |
| `workers` キーがある、`supervisor_enabled` キーがある | **Supervisor モード** | `/api/workers/{id}/restart` |

---

## Server モード（`animaworks serve`）

### API リファレンス

ベース URL: `http://localhost:18500`

| エンドポイント | メソッド | 用途 |
|--------------|---------|------|
| `/api/system/status` | GET | システム状態確認 |
| `/api/system/reload` | POST | **全 person のホットリロード** |
| `/api/persons` | GET | person 一覧 |
| `/api/persons/{name}` | GET | person 詳細 |
| `/api/persons/{name}/chat` | POST | メッセージ送信 |
| `/api/persons/{name}/trigger` | POST | ハートビート即時実行 |

### リロード手順（プログラム更新後）

```bash
curl -s -X POST http://localhost:18500/api/system/reload | python3 -m json.tool
```

応答例:
```json
{
  "added": [],
  "refreshed": ["sakura", "kotoha"],
  "removed": [],
  "total": 2
}
```

- `added`: 新たに検出された person
- `refreshed`: 再読み込みされた person（ファイル変更が反映される）
- `removed`: ディスクから削除された person
- **サーバー再起動は不要。このエンドポイントで設定・プロンプト変更が即座に反映される**

---

## Supervisor モード（`animaworks start`）

### 前提条件

- Gateway が supervisor モード（`animaworks start`）で起動していること
- 対象の person が `persons/` ディレクトリに存在すること

### API リファレンス

ベース URL: `http://localhost:18500`

| エンドポイント | メソッド | 用途 |
|--------------|---------|------|
| `/api/system/status` | GET | システム全体の状態 |
| `/api/workers` | GET | 管理中ワーカー一覧 |
| `/api/workers/spawn` | POST | ワーカー起動 |
| `/api/workers/{worker_id}/stop` | POST | ワーカー停止 |
| `/api/workers/{worker_id}/restart` | POST | ワーカー再起動 |
| `/api/workers/{worker_id}` | GET | ワーカー詳細 |
| `/api/persons/{name}/chat` | POST | タスク割り当て（メッセージ送信） |
| `/api/persons/{name}/trigger` | POST | ハートビート即時実行 |

### 1. ワーカー一覧を確認する

```bash
curl -s http://localhost:18500/api/workers | python3 -m json.tool
```

### 2. ワーカーを起動する（人を起こす・雇用後の起動）

```bash
curl -s -X POST http://localhost:18500/api/workers/spawn \
  -H "Content-Type: application/json" \
  -d '{"person_names": ["対象者の英名"]}'
```

応答例:
```json
{
  "worker_id": "worker-sakura",
  "person_names": ["sakura"],
  "port": 18501,
  "status": "starting"
}
```

### 3. ワーカーを停止する（人を休ませる）

```bash
curl -s -X POST http://localhost:18500/api/workers/worker-{英名}/stop
```

### 4. ワーカーを再起動する（更新反映）

```bash
curl -s -X POST http://localhost:18500/api/workers/worker-{英名}/restart
```

### 5. 起動したワーカーにタスクを割り当てる

```bash
curl -s -X POST http://localhost:18500/api/persons/{英名}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "依頼内容", "from_person": "sakura"}'
```

### worker_id の命名規則

- 1人の person に対し 1 ワーカーの場合: `worker-{英名}`（例: `worker-sakura`）
- spawn 時に `worker_id` を省略すると自動で `worker-{英名}` が割り当てられる

---

## 注意事項

- **動作モードを確認せずに API を呼ばないこと。** 必ず `/api/system/status` で判定してから操作する
- ワーカーを停止しても person のデータ（記憶・設定）は残る。再度起動すれば復帰できる
- Supervisor モードでない場合、`/api/workers` 系エンドポイントは使えない
- **自分自身のワーカーを停止すると自分も停止する。自分を停止しないこと**
- ワーカー起動後、Gateway への登録完了まで数秒かかる場合がある
