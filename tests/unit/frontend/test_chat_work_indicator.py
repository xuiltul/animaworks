"""Static checks for chat work-indicator (input-adjacent live tool strip)."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHAT_DIR = PROJECT_ROOT / "server" / "static" / "pages" / "chat"
CHAT_CSS = PROJECT_ROOT / "server" / "static" / "styles" / "chat.css"
I18N_DIR = PROJECT_ROOT / "server" / "static" / "i18n"

PANE_HOST = CHAT_DIR / "pane-host.js"
STREAMING = CHAT_DIR / "streaming-controller.js"
WORK_INDICATOR = CHAT_DIR / "work-indicator-controller.js"
ANIMA_CTRL = CHAT_DIR / "anima-controller.js"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── DOM / wiring ──────────────────────────────────────────


def test_pane_host_has_work_indicator_placeholder() -> None:
    js = _read(PANE_HOST)
    assert 'data-chat-id="chatWorkIndicator"' in js
    assert 'class="chat-work-indicator"' in js
    assert "createWorkIndicatorController" in js
    assert "ctx.controllers.workIndicator" in js
    # Placed before pending-queue-bar inside chat-input-form
    work_idx = js.index("chatWorkIndicator")
    pending_idx = js.index("chatPagePending")
    assert work_idx < pending_idx


def test_pane_host_destroys_work_indicator() -> None:
    js = _read(PANE_HOST)
    destroy_idx = js.index("function _destroyPane")
    section = js[destroy_idx : destroy_idx + 800]
    assert "workIndicator?.destroy()" in section or "workIndicator.destroy()" in section


def test_streaming_forwards_tool_and_thinking_events() -> None:
    js = _read(STREAMING)
    for hook in (
        "workIndicator?.onToolStart",
        "workIndicator?.onToolDetail",
        "workIndicator?.onToolEnd",
        "workIndicator?.onThinkingStart",
        "workIndicator?.onThinkingEnd",
        "workIndicator?.onStreamSettled",
    ):
        assert hook in js, f"missing forward: {hook}"


def test_anima_change_clears_indicator() -> None:
    js = _read(ANIMA_CTRL)
    select_idx = js.index("async function selectAnima")
    section = js[select_idx : select_idx + 1200]
    assert "workIndicator?.onAnimaChange()" in section


# ── Controller behaviour (source-level) ───────────────────


def test_work_indicator_controller_exists_and_exports() -> None:
    js = _read(WORK_INDICATOR)
    assert "export function createWorkIndicatorController" in js
    assert "anima-tool-activity" in js
    assert "running-tasks" in js
    assert "CHAT_POLL_INTERVAL_MS" in js
    assert "destroy()" in js or "function destroy" in js
    # Multi-pane safety: use ctx.$ not document.getElementById for container
    assert 'document.getElementById("chatWorkIndicator")' not in js
    assert '$("chatWorkIndicator")' in js or "chatWorkIndicator" in js
    # Filter other anima WS events
    assert "selectedAnima" in js
    # Stale auto-clear for broken streams
    assert "30000" in js or "STREAM_STALE" in js
    # Hold timings
    assert "1500" in js  # tool_end hold
    assert "3000" in js  # WS flash hold


def test_work_indicator_uses_get_icon() -> None:
    js = _read(WORK_INDICATOR)
    assert "getIcon(" in js
    assert "lucide.createIcons" in js


def test_work_indicator_api_failure_is_non_fatal() -> None:
    js = _read(WORK_INDICATOR)
    assert "console.error" in js
    assert "Failed to load running activity tasks" in js


# ── CSS ───────────────────────────────────────────────────


def test_chat_css_styles_work_indicator() -> None:
    css = _read(CHAT_CSS)
    assert ".chat-work-indicator" in css
    assert ".chat-work-indicator[hidden]" in css
    assert "display: none" in css[css.index(".chat-work-indicator[hidden]") :][:200]
    assert ".chat-work-chip" in css
    assert "chat-work-pulse" in css
    # Mobile hides bg task chips
    assert "chat-work-chip--bg" in css
    mobile_idx = css.index("@media (max-width: 768px)")
    # Find a media block that mentions work-chip--bg
    assert "chat-work-chip--bg" in css[mobile_idx:]


# ── i18n ──────────────────────────────────────────────────


def test_i18n_contains_work_indicator_keys() -> None:
    required = (
        "chat.work_tool_running",
        "chat.work_thinking",
        "chat.work_bg_tasks",
    )
    for locale in ("ja", "en", "ko"):
        data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
        for key in required:
            assert key in data, f"{locale}: missing {key}"
            assert "{tool}" in data["chat.work_tool_running"] or locale != "ja"
            assert data[key]
        assert "{count}" in data["chat.work_bg_tasks"]
        assert "{tool}" in data["chat.work_tool_running"]
