from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""call_human CLI — send human escalation notification via configured channels.

Available as ``animaworks-tool call_human`` so that A1-mode animas can escalate
to humans from Bash without needing ToolHandler (A2/B only).

Usage:
    animaworks-tool call_human "Subject" "Body text" [--priority high]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _load_config() -> dict:
    from core.paths import get_data_dir
    cfg_path = get_data_dir() / "config.json"
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _get_bot_token(channel_cfg: dict) -> str:
    """Resolve bot token from channel config (direct or env-var reference)."""
    import os

    token = channel_cfg.get("bot_token", "")
    if token:
        return token

    env_key = channel_cfg.get("bot_token_env", "")
    if env_key:
        token = os.environ.get(env_key, "")
        if token:
            return token

    # Fall back to credentials.json
    from core.tools._base import get_credential
    try:
        return get_credential("slack", "call_human", env_var="SLACK_BOT_TOKEN")
    except Exception:
        return ""


async def _send_slack(channel: str, token: str, text: str) -> str:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {token}"},
                json={"channel": channel, "text": text},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                return f"ERROR: {data.get('error', 'unknown')}"
            return "OK"
    except Exception as e:
        return f"ERROR: {e}"


def get_cli_guide() -> str:
    """Return CLI guide text for the tool guide injection."""
    return """\
### call_human — 人間への緊急通知

重要な問題や判断が必要な事項を人間の管理者に通知します。

```
animaworks-tool call_human "件名" "本文" [--priority PRIORITY]
```

**優先度 (--priority):** `low` / `normal`（デフォルト）/ `high` / `urgent`

**例:**
```bash
# 通常通知
animaworks-tool call_human "タスク完了" "エアコン価格調査が完了しました"

# 緊急通知
animaworks-tool call_human "障害発生" "本番APIが503を返しています" --priority urgent
```

**いつ使うか:**
- 障害・エラーの検出（自力解決不可）
- 部下からのエスカレーション受領後、事実確認して重要と判断した場合
- 人間の判断が必要な意思決定
- 重要タスクの完了報告

**使わない場合:** 定常巡回で問題なし、軽微な自動修復完了"""


def cli_main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="animaworks-tool call_human",
        description="Send human escalation notification",
    )
    parser.add_argument("subject", help="Notification subject")
    parser.add_argument("body", help="Notification body")
    parser.add_argument(
        "--priority", choices=["low", "normal", "high", "urgent"],
        default="normal",
    )
    ns = parser.parse_args(args)

    cfg = _load_config()
    hn_cfg = cfg.get("human_notification", {})

    if not hn_cfg.get("enabled", False):
        print("ERROR: human_notification is disabled in config.json", file=sys.stderr)
        sys.exit(1)

    channels = hn_cfg.get("channels", [])
    if not channels:
        print("ERROR: no channels configured in human_notification", file=sys.stderr)
        sys.exit(1)

    prefix = f"[{ns.priority.upper()}] " if ns.priority in ("high", "urgent") else ""
    text = f"{prefix}*{ns.subject}*\n{ns.body}"

    results = []
    for ch in channels:
        if not ch.get("enabled", True):
            continue
        ch_type = ch.get("type", "")
        ch_cfg = ch.get("config", {})

        if ch_type == "slack":
            token = _get_bot_token(ch_cfg)
            if not token:
                results.append("slack: ERROR - no bot token found")
                continue
            channel_id = ch_cfg.get("channel", "")
            if not channel_id:
                results.append("slack: ERROR - no channel configured")
                continue
            result = asyncio.run(_send_slack(channel_id, token, text))
            results.append(f"slack: {result}")
        else:
            results.append(f"{ch_type}: not supported in CLI mode")

    for r in results:
        print(r)

    if any("ERROR" in r for r in results):
        sys.exit(1)
