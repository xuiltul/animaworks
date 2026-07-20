from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""call_human CLI — send human escalation notification via configured channels.

Available as ``animaworks-tool call_human`` so that Mode S animas can escalate
to humans from Bash without needing ToolHandler (Mode A/B only).

Usage:
    animaworks-tool call_human "Subject" "Body text" [--priority high]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from core.i18n import t
from core.notification.interactive import InteractionRequest

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    from core.paths import get_data_dir

    cfg_path = get_data_dir() / "config.json"
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _get_bot_token(channel_cfg: dict) -> str:
    """Resolve bot token from channel config (direct, env, vault/shared, credentials)."""
    token = channel_cfg.get("bot_token", "")
    if token:
        return token

    env_key = channel_cfg.get("bot_token_env", "")
    if env_key:
        token = os.environ.get(env_key, "")
        if token:
            return token

    # Per-anima vault/shared (Mode S subprocess sets ANIMAWORKS_ANIMA_DIR)
    anima_dir = os.environ.get("ANIMAWORKS_ANIMA_DIR")
    if anima_dir:
        from core.tools._base import _lookup_shared_credentials, _lookup_vault_credential

        anima_name = Path(anima_dir).name
        per_key = f"SLACK_BOT_TOKEN__{anima_name}"
        token = _lookup_vault_credential(per_key) or _lookup_shared_credentials(per_key) or ""
        if token:
            return token

    # Fall back to credentials.json / vault / env
    from core.tools._base import get_credential

    try:
        return get_credential("slack", "call_human", env_var="SLACK_BOT_TOKEN")
    except Exception:
        return ""


def _resolve_cli_anima_name() -> str:
    anima_dir = os.environ.get("ANIMAWORKS_ANIMA_DIR")
    if anima_dir:
        return Path(anima_dir).name
    return ""


def _resolve_cli_anima_identity(channel_cfg: dict) -> tuple[str, str]:
    """Resolve Anima name and icon URL for CLI call_human invocations.

    Returns (username, icon_url) — either may be empty string.
    """
    from core.tools._anima_icon_url import resolve_anima_icon_identity

    anima_dir = os.environ.get("ANIMAWORKS_ANIMA_DIR")
    if not anima_dir:
        return ("", "")

    return resolve_anima_icon_identity(Path(anima_dir).name, channel_cfg)


def _button_emoji(opt: str) -> str:
    return {"approve": "✅", "reject": "❌", "comment": "💬"}.get(opt, "▶️")


def _build_slack_blocks(text: str, interaction: InteractionRequest) -> list[dict[str, Any]]:
    """Build Slack Block Kit blocks with action buttons."""
    style_map = {"approve": "primary", "reject": "danger"}
    elements: list[dict[str, Any]] = []
    for opt in interaction.options:
        btn: dict[str, Any] = {
            "type": "button",
            "text": {"type": "plain_text", "text": _button_emoji(opt) + " " + opt.capitalize()},
            "action_id": f"aw_interact_{opt}",
            "value": interaction.callback_id,
        }
        if opt in style_map:
            btn["style"] = style_map[opt]
        elements.append(btn)
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "actions", "block_id": f"aw_interact:{interaction.callback_id}", "elements": elements},
    ]


async def _send_slack(
    channel: str,
    token: str,
    text: str,
    *,
    username: str = "",
    icon_url: str = "",
    notification_text: str = "",
    interaction: InteractionRequest | None = None,
) -> tuple[str, str | None]:
    """Post to Slack; returns (status_message, message_ts_on_success)."""
    try:
        payload: dict[str, Any] = {"channel": channel, "text": text}
        if username:
            payload["username"] = username
        if icon_url:
            payload["icon_url"] = icon_url
        if interaction is not None:
            payload["blocks"] = _build_slack_blocks(text, interaction)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                return f"ERROR: {data.get('error', 'unknown')}", None

            ts_val: str | None = str(data["ts"]) if data.get("ts") else None

            # Save ts→anima mapping so Slack thread replies are routed back.
            # Direct file write fails inside execution sandboxes (read-only
            # {data_dir}/run/), so fall back to the server internal API.
            if username and ts_val and notification_text:
                try:
                    from core.notification.reply_routing import (
                        save_notification_mapping_resilient,
                    )

                    saved = save_notification_mapping_resilient(
                        ts=ts_val,
                        channel=data.get("channel", channel),
                        anima_name=username,
                        notification_text=notification_text[:2000],
                        callback_id=interaction.callback_id if interaction is not None else "",
                    )
                    if not saved:
                        print(
                            "WARNING: notification mapping not saved; Slack thread replies will not be routed back",
                            file=sys.stderr,
                        )
                except Exception:
                    logger.debug("Failed to save notification mapping", exc_info=True)

            return "OK", ts_val
    except Exception as e:
        return f"ERROR: {e}", None


def _create_interaction_via_server(
    anima_name: str,
    category: str,
    options: list[str],
    allowed_users: dict[str, list[str]] | None,
    callback_id: str,
) -> InteractionRequest:
    """Create an interaction record, preferring the server internal API.

    The CLI usually runs inside an execution sandbox where the interaction
    map ({data_dir}/run/interaction_map.json) is read-only; a local write
    would fail silently and the server could never resolve the buttons.
    Falls back to the local router when the server is unreachable.
    """
    from core.notification.interactive import create_interaction_resilient

    return create_interaction_resilient(
        anima_name,
        category,
        options,
        allowed_users=allowed_users,
        callback_id=callback_id,
    )


def _persist_interaction_slack_ts(callback_id: str, ts_val: str) -> None:
    """Store Slack message ts on the interaction record (CLI path).

    Prefers the server internal API for the same sandbox reason as
    :func:`_create_interaction_via_server`.
    """
    from core.notification.interactive import update_interaction_message_ts_resilient

    update_interaction_message_ts_resilient(callback_id, "slack", ts_val)


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
        "--priority",
        choices=["low", "normal", "high", "urgent"],
        default="normal",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Send Slack Block Kit action buttons (bot token mode)",
    )
    parser.add_argument(
        "--callback-id",
        default="",
        help="Stable callback id for interactive mode (auto-generated if omitted)",
    )
    parser.add_argument(
        "--options",
        default="approve,reject,comment",
        help="Comma-separated button labels for interactive mode",
    )
    parser.add_argument(
        "--category",
        default="approval",
        help="Interaction category label",
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

    interaction: InteractionRequest | None = None
    if ns.interactive:
        anima_name = _resolve_cli_anima_name()
        opts_list = [p.strip() for p in ns.options.split(",") if p.strip()]

        try:
            interaction = _create_interaction_via_server(
                anima_name,
                ns.category,
                opts_list,
                {"slack": []},
                ns.callback_id,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    results: list[str] = []
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
            username, icon_url = _resolve_cli_anima_identity(ch_cfg)

            status, slack_ts = asyncio.run(
                _send_slack(
                    channel_id,
                    token,
                    text,
                    username=username,
                    icon_url=icon_url,
                    notification_text=f"{ns.subject}\n{ns.body}",
                    interaction=interaction if ns.interactive else None,
                )
            )
            results.append(f"slack: {status}")

            if ns.interactive and interaction is not None and status == "OK" and slack_ts:
                _persist_interaction_slack_ts(interaction.callback_id, str(slack_ts))
        else:
            results.append(f"{ch_type}: not supported in CLI mode")

    for r in results:
        print(r)

    if ns.interactive and interaction is not None:
        print(t("tools.call_human.callback_id", callback_id=interaction.callback_id))

    sent_ok = any("OK" in r for r in results)
    has_error = any("ERROR" in r for r in results)
    if has_error or not sent_ok:
        sys.exit(1)


__all__ = ["_build_slack_blocks", "cli_main", "get_cli_guide"]
