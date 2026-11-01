---
name: worker-management
description: >-
  AnimaWorks 服务器进程的运维管理技能。
  代码更新后的热重载（server reload）、Anima 进程的重启、
  执行服务器状态确认（启动中的 Anima 列表、内存使用量）。
  “重载一下”“把更新反映上去”“重新加载”“系统状态”“服务器重启”“进程确认”
---


# 技能：系统管理

## CLI 命令（推荐）

`animaworks` 可通过 CLI 对单个 Anima 进行管理。**应优先使用 CLI，而不是直接调用 API。**

```bash
# 個別Animaのリスタート（設定変更の反映等）
animaworks anima restart <name>

# ステータス確認（全体 or 個別）
animaworks anima status
animaworks anima status <name>

# モデル変更（root APIでstatusを更新 + 起動中ならreload）
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

### 常见用法

```bash
# config.json変更後に特定Animaだけリスタート
animaworks anima restart aoi

# モデルを変更して起動中プロセスに自動reload
animaworks anima set-model aoi claude-sonnet-5-5
```

## API 参考（无法使用 CLI 时）

基础 URL：`$ANIMAWORKS_SERVER_URL`（未设置时为 `http://localhost:18500`）

| 端点 | 方法 | 用途 |
|--------------|---------|------|
| `/api/system/status` | GET | 系统状态确认 |
| `/api/system/reload` | POST | **所有 anima 的热重载** |
| `/api/animas` | GET | anima 列表 |
| `/api/animas/{name}` | GET | anima 详情 |
| `/api/animas/{name}/restart` | POST | 单个重启 |
| `/api/animas/{name}/stop` | POST | 单个停止 |
| `/api/animas/{name}/start` | POST | 启动已停止的 anima |
| `/api/animas/{name}/chat` | POST | 发送消息 |
| `/api/animas/{name}/trigger` | POST | 立即执行心跳 |

## 重载步骤（程序更新后）

```bash
curl -s -X POST "${ANIMAWORKS_SERVER_URL:-http://localhost:18500}"/api/system/reload | python3 -m json.tool
```

- `added`：新检测到的 anima
- `refreshed`：被重新加载的 anima（文件变更会被反映）
- `removed`：已从磁盘删除的 anima
- **无需重启服务器。通过此端点，配置与提示词的变更会立即生效**

## 注意事项

- 即使停止 worker，anima 的数据（记忆、配置）仍会保留
- **不要执行停止自身的操作**
- 单个操作按 CLI → API 的优先顺序使用
