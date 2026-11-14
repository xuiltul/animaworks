/* ── Sidebar: External agents (hub) item ───────
   Shown only while the hub is enabled. The badge is the human's unread count
   on the hub board, refreshed on load, every minute, and shortly after a
   board post / read event for that board. */

import { api } from "./api.js";
import { onEvent } from "./websocket.js";

const REFRESH_MS = 60_000;
const EVENT_DEBOUNCE_MS = 1_200;
const BADGE_MAX = 99;

let _board = "";
let _started = false;
let _loading = false;
let _eventTimer = null;

function _render(info) {
  const item = document.getElementById("navHubItem");
  const badge = document.getElementById("navHubBadge");
  if (!item) return;
  const enabled = Boolean(info && info.enabled === true);
  item.hidden = !enabled;
  _board = enabled ? String(info.board || "") : "";
  if (!badge) return;
  const unread = enabled ? Math.max(0, Number(info.unread) || 0) : 0;
  badge.hidden = unread === 0;
  badge.textContent = unread > BADGE_MAX ? `${BADGE_MAX}+` : String(unread);
}

export async function refreshHubNav() {
  if (_loading) return;
  _loading = true;
  try {
    _render(await api("/api/hub/info"));
  } catch {
    // Older servers have no /api/hub/info; keep the item hidden.
  } finally {
    _loading = false;
  }
}

function _scheduleRefresh() {
  if (_eventTimer) clearTimeout(_eventTimer);
  _eventTimer = setTimeout(() => {
    _eventTimer = null;
    void refreshHubNav();
  }, EVENT_DEBOUNCE_MS);
}

function _onBoardEvent(data) {
  if (!_board || !data || data.channel !== _board) return;
  _scheduleRefresh();
}

export function initHubNav() {
  if (_started) return;
  _started = true;
  void refreshHubNav();
  onEvent("board.post", _onBoardEvent);
  onEvent("board.read", _onBoardEvent);
  setInterval(() => {
    if (document.visibilityState === "visible") void refreshHubNav();
  }, REFRESH_MS);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") void refreshHubNav();
  });
}
