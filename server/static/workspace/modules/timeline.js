/**
 * timeline.js — Activity timeline orchestrator.
 *
 * Owns the canonical _events array and module-level DOM references.
 * Delegates DOM construction to timeline-dom.js, 3D replay to
 * timeline-replay.js, and history fetching to timeline-history.js.
 *
 * Public API (unchanged):
 *   initTimeline, addTimelineEvent, loadHistory, dispose, localISOString
 */

import { buildTimelineDOM, renderList, createEventElement, updateLoadMoreButton } from "./timeline-dom.js";
import { replayEvent, ensureHighlightFns } from "./timeline-replay.js";
import { fetchHistory as _fetchHistory, fetchMore as _fetchMore, hasMore, totalCount, resetPagination } from "./timeline-history.js";

// ── Timestamp helper ──────────────────────────────

/**
 * Generate a naive local ISO string (no "Z" suffix) matching the server's
 * ``datetime.now().isoformat()`` format.
 * @returns {string} e.g. "2026-02-18T21:41:02.123"
 */
export function localISOString() {
  const d = new Date();
  const pad = (n, len = 2) => String(n).padStart(len, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` +
    `.${pad(d.getMilliseconds(), 3)}`
  );
}

function _tsDescending(a, b) {
  const ta = new Date(a.ts || 0).getTime();
  const tb = new Date(b.ts || 0).getTime();
  return tb - ta;
}

// ── Module state ───────────────────────────────────

/** @type {import("./timeline-dom.js").TimelineEvent[]} */
const _events = [];

const MAX_EVENTS = 200;

const filterDefs = [
  { label: "All", types: [] },
  { label: "💬", types: ["message_received", "response_sent", "dm_received", "dm_sent", "message", "chat"] },
  { label: "📋", types: ["channel_read", "channel_post", "board"] },
  { label: "💗", types: ["heartbeat_start", "heartbeat_end", "heartbeat", "heartbeat_reflection"] },
  { label: "⏰", types: ["cron_executed", "cron"] },
  { label: "🔧", types: ["tool_use", "memory_write"] },
  { label: "📣", types: ["human_notify", "notification"] },
  { label: "⚠️", types: ["error", "issue_resolved"] },
];

/** @type {string[]} current filter — empty array means "all" */
let _currentFilter = [];

/** @type {HTMLElement|null} */
let _container = null;
/** @type {HTMLElement|null} */
let _listEl = null;
/** @type {HTMLElement|null} */
let _countEl = null;
/** @type {HTMLElement|null} */
let _bodyEl = null;

/** @type {object|null} */
let _interactionManager = null;

// ── Internal helpers ───────────────────────────────

function _makeElement(evt) {
  return createEventElement(evt, {
    onReplay: (e, el) => replayEvent(e, el, { interactionManager: _interactionManager }),
  });
}

function _render() {
  renderList(_listEl, _events, _currentFilter, _makeElement);
}

function _dedup() {
  const seen = new Set();
  for (let i = _events.length - 1; i >= 0; i--) {
    if (seen.has(_events[i].id)) {
      _events.splice(i, 1);
    } else {
      seen.add(_events[i].id);
    }
  }
}

function _updateCount() {
  if (!_countEl) return;
  const total = totalCount();
  _countEl.textContent = total > 0 ? `${_events.length}/${total}` : _events.length.toString();
}

// ── Public API ─────────────────────────────────────

/**
 * Initialize the timeline UI.
 * @param {HTMLElement} officePanel — the office panel container
 * @param {object} interactionManager — { showMessageEffect, showConversation }
 */
export function initTimeline(officePanel, interactionManager) {
  _interactionManager = interactionManager;

  const dom = buildTimelineDOM(officePanel, {
    filterDefs,
    onFilterClick: (index) => {
      const fd = filterDefs[index];
      _currentFilter = (index === 0 || !fd) ? [] : (fd.types || []);
      _render();
    },
    onLoadMore: () => _loadMoreAndRender(),
  });

  _container = dom.container;
  _listEl = dom.listEl;
  _countEl = dom.countEl;
  _bodyEl = dom.bodyEl;

  ensureHighlightFns();
}

/**
 * Add a real-time event to the timeline.
 * @param {object} event
 */
export function addTimelineEvent(event) {
  _events.unshift(event);

  if (_events.length > MAX_EVENTS) {
    _events.length = MAX_EVENTS;
  }

  if (_countEl) {
    _countEl.textContent = _events.length.toString();
  }

  if (_listEl && (_currentFilter.length === 0 || _currentFilter.includes(event.type))) {
    const el = _makeElement(event);
    _listEl.insertBefore(el, _listEl.firstChild);
  }
}

/**
 * Load historical events from the backend.
 * @param {number} [hours=48]
 */
export async function loadHistory(hours = 48) {
  const result = await _fetchHistory(hours);

  for (const evt of result.events) {
    _events.push(evt);
  }

  _dedup();
  _events.sort(_tsDescending);

  if (_events.length > MAX_EVENTS) {
    _events.length = MAX_EVENTS;
  }

  _updateCount();
  updateLoadMoreButton(result.hasMore);
  _render();
}

async function _loadMoreAndRender() {
  const result = await _fetchMore();

  for (const evt of result.events) {
    _events.push(evt);
  }

  _dedup();
  _events.sort(_tsDescending);

  _updateCount();
  updateLoadMoreButton(result.hasMore);
  _render();
}

/**
 * Clean up the timeline DOM and state.
 */
export function dispose() {
  if (_container && _container.parentNode) {
    _container.parentNode.removeChild(_container);
  }
  _events.length = 0;
  resetPagination();
  _container = null;
  _listEl = null;
  _countEl = null;
  _bodyEl = null;
  _interactionManager = null;
}
