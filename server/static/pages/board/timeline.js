// ── Company timeline ("what happened at the company") ──
// A compact read-only column at the top of the board home. Items come from the
// company timeline API, arrive live over the `timeline.item` WebSocket event,
// and are also refreshed with `since` every minute / on focus because Anima
// processes can append items without a WebSocket broadcast.
import { api } from "../../modules/api.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { t } from "/shared/i18n.js";
import { onEvent } from "../../modules/websocket.js";
import { avatarMarkup } from "./avatar.js";
import { clockTime, numberText, relativeTime } from "./time.js";

const POLL_MS = 60_000;
const DAY_MS = 24 * 60 * 60 * 1000;
// The server supports ``limit`` and returns newest first. We fetch a generous
// window so older unread items are visible even when many recent items exist.
const FETCH_LIMIT = 200;
const DISPLAY_LIMIT = 50;
const READ_DEBOUNCE_MS = 800;

const KIND_KEYS = {
  notified: "board.timeline_kind_notified",
  done: "board.timeline_kind_done",
  decided: "board.timeline_kind_decided",
};

function _ts(value) {
  const parsed = Date.parse(String(value || ""));
  return Number.isNaN(parsed) ? null : parsed;
}

function _atOrAfter(candidate, reference) {
  if (!reference) return false;
  if (!candidate) return false;
  const a = _ts(candidate);
  const b = _ts(reference);
  if (a !== null && b !== null) return a >= b;
  return String(candidate) >= String(reference);
}

function _newer(a, b) {
  const left = _ts(a?.ts) || 0;
  const right = _ts(b?.ts) || 0;
  return left - right;
}

function _kindLabel(kind) {
  return t(KIND_KEYS[kind] || "board.timeline_kind_notified");
}

function _isUrgent(item) {
  return Boolean(item.urgent) || String(item.priority || "") === "urgent";
}

function _isLong(text) {
  return String(text || "").length > 90 || String(text || "").includes("\n");
}

function _safeRef(ref) {
  const value = String(ref || "").trim();
  if (!value) return "";
  if (/^(javascript|data|vbscript):/i.test(value)) return "";
  return value;
}

function _sourceKind(store, item) {
  if (item.from === "system" || item.source === "system") return "system";
  if (store && store.hasAnima(item.from)) return "anima";
  if (item.from === "human") return "human";
  return "anima";
}

function _fromMarkup(item, store) {
  const name = String(item.from || "");
  if (_sourceKind(store, item) === "system") {
    return `<span class="board-timeline-system" title="${escapeAttr(t("board.kind_operator"))}">${escapeHtml(t("board.kind_operator"))}</span>`;
  }
  const kind = store && store.hasAnima(name) ? "anima" : "human";
  const url = store ? store.avatarUrl(name) : null;
  return `<span class="board-timeline-from">${avatarMarkup({ name, kind, url, size: "xs" })}<span class="board-timeline-from-name">${escapeHtml(name || "?")}</span></span>`;
}

function _kindMarkup(kind) {
  const safe = KIND_KEYS[kind] ? kind : "notified";
  return `<span class="board-timeline-kind board-timeline-kind-${safe}" data-timeline-kind="${escapeAttr(safe)}">${escapeHtml(_kindLabel(safe))}</span>`;
}

function _marks(item) {
  const marks = [];
  if (_isUrgent(item)) {
    marks.push(`<span class="board-timeline-mark board-timeline-mark-urgent">${escapeHtml(t("board.timeline_urgent"))}</span>`);
  }
  if (item.delivered === false) {
    marks.push(`<span class="board-timeline-mark board-timeline-mark-undelivered">${escapeHtml(t("board.timeline_undelivered"))}</span>`);
  }
  if (item.changed) {
    marks.push(`<span class="board-timeline-mark board-timeline-mark-changed">${escapeHtml(t("board.timeline_changed"))}</span>`);
  }
  if (!item.read) {
    marks.push(`<span class="board-timeline-mark board-timeline-mark-unread" data-timeline-unread-mark>${escapeHtml(t("board.timeline_unread"))}</span>`);
  }
  return marks.length ? `<span class="board-timeline-marks">${marks.join("")}</span>` : "";
}

function _textMarkup(text) {
  const value = String(text || "");
  const more = _isLong(value)
    ? `<button type="button" class="board-timeline-more" data-timeline-expand data-timeline-more aria-expanded="false">${escapeHtml(t("board.timeline_show_full"))}</button>`
    : "";
  return `<span class="board-timeline-text-wrap">
    <button type="button" class="board-timeline-text" data-timeline-expand data-timeline-text aria-expanded="false">${escapeHtml(value)}</button>
    ${more}
  </span>`;
}

function _itemMarkup(item, updates, { nested = false, store = null } = {}) {
  const id = escapeAttr(item.id);
  const itemAttr = nested ? "data-timeline-update" : "data-timeline-item";
  const ref = _safeRef(item.ref);
  const refMarkup = ref
    ? `<a class="board-timeline-ref" href="${escapeAttr(ref)}" target="_blank" rel="noopener noreferrer">${escapeHtml(t("board.timeline_ref"))}</a>`
    : "";
  const classes = [
    "board-timeline-item",
    nested ? "board-timeline-update" : "",
    item.read ? "" : "is-unread",
    _isUrgent(item) ? "is-urgent" : "",
  ].filter(Boolean).join(" ");
  const updateRows = (updates || []).map((update) => _itemMarkup(update, [], { nested: true, store })).join("");
  const updateBlock = updateRows
    ? `<div class="board-timeline-updates" data-timeline-updates="${id}">
      <span class="board-timeline-updates-label">${escapeHtml(t("board.timeline_update"))}</span>
      ${updateRows}
    </div>`
    : "";
  const time = clockTime(item.ts);
  return `<article class="${classes}" ${itemAttr}="${id}" data-timeline-id="${id}" data-timeline-kind="${escapeAttr(item.kind || "")}">
    <div class="board-timeline-row" data-timeline-row>
      <span class="board-timeline-time" title="${escapeAttr(time)}">${escapeHtml(relativeTime(item.ts))}</span>
      ${_kindMarkup(item.kind)}
      ${_fromMarkup(item, store)}
      ${_textMarkup(item.text)}
      ${_marks(item)}
      ${refMarkup}
    </div>
    ${updateBlock}
  </article>`;
}

export function createTimelineView({ element, store }) {
  const state = {
    enabled: false,
    company: "",
    items: [],
    unread: 0,
    reader: { upto_id: "", upto_ts: "" },
    latestTs: "",
    destroyed: false,
    loading: false,
    readTimer: null,
    marking: false,
    lastMarkedId: "",
  };
  const abort = new AbortController();
  let interval = null;
  let unsubscribe = null;

  element.hidden = true;
  element.dataset.boardTimeline = "";

  function _itemById(id) {
    return state.items.find((item) => String(item.id) === String(id)) || null;
  }

  function _normalize(raw) {
    const item = { ...raw };
    item.id = String(item.id || "");
    item.ts = String(item.ts || "");
    item.kind = String(item.kind || "notified");
    item.from = String(item.from || "");
    item.text = String(item.text || "");
    item.updates = Array.isArray(item.updates) ? item.updates.map(String) : [];
    item.read = Boolean(item.read);
    item.changed = Boolean(item.changed);
    item.urgent = Boolean(item.urgent);
    if (item.delivered === undefined) item.delivered = null;
    return item;
  }

  function _sort() {
    state.items.sort((a, b) => _newer(b, a) || String(b.id).localeCompare(String(a.id)));
    state.latestTs = state.items.reduce((latest, item) => (_atOrAfter(item.ts, latest) ? item.ts : latest), "");
  }

  function _upsert(raw, { authoritative = false } = {}) {
    if (!raw || !raw.id) return false;
    const normalized = _normalize(raw);
    const index = state.items.findIndex((item) => item.id === normalized.id);
    if (index >= 0) {
      if (authoritative) state.items[index] = normalized;
      return false;
    }
    state.items.push(normalized);
    return true;
  }

  function _isRecent(item) {
    const value = _ts(item.ts);
    return value !== null && Date.now() - value <= DAY_MS;
  }

  /** Build the top-level rows plus updates nested under their original. */
  function _displayRows() {
    if (!state.enabled) return [];
    const byId = new Map(state.items.map((item) => [item.id, item]));
    const candidates = new Map();
    for (const item of state.items) {
      if (!item.read || _isRecent(item)) candidates.set(item.id, item);
    }
    // A newly arrived update keeps its original on screen even when the
    // original itself is older than 24h and already read.
    const updatesByBase = new Map();
    for (const item of state.items) {
      if (!item.about || !candidates.has(item.id)) continue;
      const original = byId.get(String(item.about));
      if (!original) continue;
      candidates.set(original.id, original);
    }
    for (const item of state.items) {
      if (!item.about) continue;
      const base = String(item.about);
      if (!candidates.has(base) || !byId.has(base)) continue;
      const list = updatesByBase.get(base) || [];
      if (!list.some((update) => update.id === item.id)) list.push(item);
      updatesByBase.set(base, list);
    }
    const rows = [...candidates.values()]
      .filter((item) => !(item.about && candidates.has(String(item.about)) && byId.has(String(item.about))))
      .sort((a, b) => _newer(b, a) || String(b.id).localeCompare(String(a.id)))
      .slice(0, DISPLAY_LIMIT);
    return rows.map((item) => ({
      item,
      updates: (updatesByBase.get(item.id) || []).sort((a, b) => _newer(b, a)),
    }));
  }

  function _visibleItems() {
    if (state.destroyed || element.hidden || !element.isConnected) return [];
    if (!element.getClientRects().length) return [];
    const viewportWidth = window.innerWidth || 0;
    const viewportHeight = window.innerHeight || 0;
    // The board home is its own scroll container; clipping by it matters just
    // as much as clipping by the browser viewport.
    const scroller = element.closest(".board-home") || element;
    const bounds = scroller.getBoundingClientRect();
    const top = Math.max(bounds.top, 0);
    const bottom = Math.min(bounds.bottom, viewportHeight);
    const left = Math.max(bounds.left, 0);
    const right = Math.min(bounds.right, viewportWidth);
    const visible = [];
    for (const row of element.querySelectorAll("[data-timeline-id]")) {
      const rect = row.getBoundingClientRect();
      if (rect.height > 0 && rect.bottom > top && rect.top < bottom && rect.right > left && rect.left < right) {
        const item = _itemById(row.dataset.timelineId);
        if (item) visible.push(item);
      }
    }
    return visible;
  }

  function _scheduleRead() {
    if (state.destroyed || !state.enabled) return;
    if (state.readTimer) clearTimeout(state.readTimer);
    state.readTimer = setTimeout(() => {
      state.readTimer = null;
      void _markVisibleRead();
    }, READ_DEBOUNCE_MS);
  }

  async function _markVisibleRead() {
    if (state.destroyed || state.marking) return;
    const visible = _visibleItems();
    if (visible.length === 0) return;
    visible.sort((a, b) => _newer(b, a));
    const candidate = visible.find(
      (item) => !item.read && item.id !== state.lastMarkedId && !_atOrAfter(state.reader.upto_ts, item.ts),
    );
    if (!candidate) return;
    state.marking = true;
    try {
      const result = await api("/api/timeline/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ upto_id: candidate.id }),
        signal: abort.signal,
      });
      if (state.destroyed) return;
      const uptoId = String(result?.upto_id || candidate.id);
      const uptoTs = String(result?.upto_ts || candidate.ts);
      state.reader = { upto_id: uptoId, upto_ts: uptoTs };
      state.lastMarkedId = candidate.id;
      let newlyRead = 0;
      for (const item of state.items) {
        if (!item.read && _atOrAfter(uptoTs, item.ts)) {
          item.read = true;
          item.changed = false;
          newlyRead += 1;
        }
      }
      state.unread = Math.max(0, state.unread - newlyRead);
      render();
      // The response only reports the position; refresh once so the heading
      // reflects the server's authoritative unread count (older items may be
      // outside the fetched window).
      void fetchTimeline();
    } catch {
      // Keep the unread marks and retry on the next render / scroll.
    } finally {
      state.marking = false;
    }
  }

  function _loadAvatars() {
    if (!store || typeof store.ensureAvatars !== "function") return;
    const names = state.items
      .filter((item) => _sourceKind(store, item) !== "system")
      .map((item) => item.from)
      .filter(Boolean);
    Promise.resolve(store.ensureAvatars(names))
      .then((changed) => {
        if (changed && !state.destroyed) render();
      })
      .catch(() => {});
  }

  function render() {
    if (state.destroyed) return;
    if (!state.enabled) {
      element.hidden = true;
      element.innerHTML = "";
      return;
    }
    element.hidden = false;
    const rows = _displayRows();
    const unread = Math.max(0, Number(state.unread) || 0);
    const unreadMarkup = unread > 0
      ? `<span class="board-timeline-unread-count" data-timeline-unread data-timeline-unread-count>${escapeHtml(t("board.timeline_unread_count", { count: numberText(unread) }))}</span>`
      : "";
    const body = rows.length
      ? rows.map(({ item, updates }) => _itemMarkup(item, updates, { store })).join("")
      : `<div class="board-timeline-empty" data-timeline-empty>${escapeHtml(t("board.timeline_empty"))}</div>`;
    element.innerHTML = `<header class="board-timeline-head">
      <h2 class="board-timeline-title" data-timeline-title>${escapeHtml(t("board.timeline_title"))}</h2>
      ${unreadMarkup}
    </header>
    <div class="board-timeline-list" data-timeline-list>${body}</div>`;
    _scheduleRead();
  }

  async function fetchTimeline({ since = "" } = {}) {
    if (state.destroyed || state.loading) return;
    state.loading = true;
    const query = new URLSearchParams({ limit: String(FETCH_LIMIT) });
    if (since) query.set("since", since);
    try {
      const data = await api(`/api/timeline?${query.toString()}`, { signal: abort.signal });
      if (state.destroyed) return;
      if (!data || data.enabled !== true) {
        state.enabled = false;
        state.items = [];
        state.unread = 0;
        state.latestTs = "";
        render();
        return;
      }
      state.enabled = true;
      state.company = String(data.company || "");
      const incoming = Array.isArray(data.items) ? data.items : [];
      for (const raw of incoming) _upsert(raw, { authoritative: true });
      _sort();
      if (typeof data.unread === "number") state.unread = Math.max(0, data.unread);
      state.reader = {
        upto_id: String(data.reader?.upto_id || ""),
        upto_ts: String(data.reader?.upto_ts || ""),
      };
      render();
      _loadAvatars();
    } catch {
      // A disabled timeline endpoint (404) or a transient failure leaves the
      // column hidden / unchanged; the next poll retries.
    } finally {
      state.loading = false;
    }
  }

  function handleEvent(data) {
    if (state.destroyed || !data || typeof data !== "object") return;
    const raw = Array.isArray(data) ? data[0] : data;
    if (!raw || !raw.id || !raw.kind || String(raw.id).startsWith("board:")) return;
    if (state.company && raw.company && raw.company !== state.company) return;
    const added = _upsert(raw, { authoritative: false });
    if (!added) return;
    _sort();
    if (!raw.read && !_atOrAfter(state.reader.upto_ts, raw.ts)) state.unread += 1;
    render();
    _loadAvatars();
  }

  element.addEventListener("click", (event) => {
    const expand = event.target.closest("[data-timeline-expand]");
    if (!expand) return;
    const wrap = expand.closest(".board-timeline-text-wrap");
    const text = wrap?.querySelector(".board-timeline-text");
    if (!text) return;
    const open = !text.classList.contains("is-open");
    text.classList.toggle("is-open", open);
    wrap.querySelectorAll("[data-timeline-expand]").forEach((button) => {
      button.setAttribute("aria-expanded", String(open));
    });
  });

  function onFocus() {
    void fetchTimeline({ since: state.latestTs });
  }

  function onVisibility() {
    if (document.visibilityState === "visible") onFocus();
  }

  function onScroll() {
    _scheduleRead();
  }

  unsubscribe = onEvent("timeline.item", handleEvent);
  interval = setInterval(() => {
    if (document.visibilityState !== "hidden") void fetchTimeline({ since: state.latestTs });
  }, POLL_MS);
  window.addEventListener("focus", onFocus);
  document.addEventListener("visibilitychange", onVisibility);
  document.addEventListener("scroll", onScroll, true);
  void fetchTimeline();

  function destroy() {
    state.destroyed = true;
    abort.abort();
    if (interval) clearInterval(interval);
    if (state.readTimer) clearTimeout(state.readTimer);
    if (unsubscribe) unsubscribe();
    window.removeEventListener("focus", onFocus);
    document.removeEventListener("visibilitychange", onVisibility);
    document.removeEventListener("scroll", onScroll, true);
    element.innerHTML = "";
    element.hidden = true;
  }

  return { render, destroy, handleEvent, fetchTimeline };
}
