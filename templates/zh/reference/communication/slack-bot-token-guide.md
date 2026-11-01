# Slack 机器人令牌配置指南

Slack 集成中使用的机器人令牌机制，以及 Per-Anima（个别）令牌的配置规则。

## 两种机器人令牌

AnimaWorks 的 Slack 集成有 **共享机器人** 和 **Per-Anima 机器人** 两种。

| 类型 | 键名 | 用途 |
|------|--------|------|
| 共享机器人 | `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | 用于整体回退。供未设置 Per-Anima 令牌的 Anima 使用 |
| Per-Anima 机器人 | `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>` | 特定 Anima 专用的 Slack App。仅该 Anima 使用 |

**如果设置了 Per-Anima 机器人，则优先使用该机器人。** 共享机器人仅作为回退。

## Per-Anima 令牌的命名规则

在 `__`（两个下划线）+ Anima 名（小写）作为后缀追加。

```
SLACK_BOT_TOKEN__sumire    ← sumire 専用の Bot User OAuth Token
SLACK_APP_TOKEN__sumire    ← sumire 専用の App-Level Token
```

## 保存位置

保存到 `shared/credentials.json`。

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...(共有ボット)",
  "SLACK_APP_TOKEN": "xapp-...(共有App)",
  "SLACK_BOT_TOKEN__sumire": "xoxb-...(sumire専用ボット)",
  "SLACK_APP_TOKEN__sumire": "xapp-...(sumire専用App)"
}
```

## 必须绝对遵守的规则

### 不得覆盖共享令牌

**MUST**：设置 Per-Anima 令牌时，**添加新键**。不得将现有的 `SLACK_BOT_TOKEN` 或 `SLACK_APP_TOKEN` 替换为自己的令牌。

共享令牌被其他 Anima 或整个系统使用。如果覆盖，将会：

- 破坏其他 Anima 的 Slack 通信
- Socket Mode 连接与 Bot Token 的 App 不一致
- 产生 `not_in_channel` 等意外错误

### 通过编辑文件添加令牌的方法

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

**注意**：不要用 `str_replace` 等替换现有行，而应使用向 JSON 添加键的方法。

## 服务器检测与重启

Per-Anima 令牌在 **服务器启动时** 被检测。向 `shared/credentials.json` 添加令牌后，**需要重启服务器**。

服务器在启动时执行以下操作：

1. 检测 `SLACK_BOT_TOKEN__*` 与 `SLACK_APP_TOKEN__*` 的配对
2. 为每个配对注册 Per-Anima Socket Mode 处理器
3. 通过 `auth.test` 获取 Bot User ID，用于频道路由

重启后，如果服务器日志中显示如下内容，则表示成功：

```
Per-anima Slack bot registered: <name> (bot_uid=U...)
```

## 故障排查

### not_in_channel 错误

**症状**：尝试回复 Slack 频道时出现 `not_in_channel` 错误

**原因**：Per-Anima 令牌未设置，正在使用共享机器人。共享机器人不是该频道的成员。

**处理方法**：
1. 向 `shared/credentials.json` 添加 `SLACK_BOT_TOKEN__<name>` 和 `SLACK_APP_TOKEN__<name>`
2. 重启服务器
3. 确认 Per-Anima 机器人已被邀请到目标频道

### 回退到了共享机器人

**症状**：明明应该有自己专用的机器人，却以共享机器人名称发布

**原因**：`credentials.json` 中不存在 `SLACK_BOT_TOKEN__<name>` / `SLACK_APP_TOKEN__<name>`

**确认方法**：
```bash
cat ~/.animaworks/shared/credentials.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
for k in sorted(d):
    if 'SLACK' in k:
        print(f'{k}: {d[k][:20]}...')
"
```

### 令牌的获取方法

请管理员（人类）提供 Slack App 的 Bot User OAuth Token (`xoxb-`) 和 App-Level Token (`xapp-`)。

- Bot Token：Slack App 管理界面 → OAuth & Permissions → Bot User OAuth Token
- App-Level Token：Slack App 管理界面 → Basic Information → App-Level Tokens（scope: `connections:write`）
