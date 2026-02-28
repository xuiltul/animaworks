/**
 * timeline-history.js — Historical event fetching + pagination.
 *
 * Owns all pagination state (_currentOffset, _hasMore, _totalCount,
 * _currentHours).  Returns fetched events to the caller; the caller
 * (timeline.js) is responsible for dedup, sort, and rendering.
 */

import { normalizeEvent } from "../../shared/activity-types.js";
import { t } from "/shared/i18n.js";

// ── Pagination state ───────────────────────────────

let _currentOffset = 0;
let _hasMore = false;
let _totalCount = 0;
let _currentHours = 48;

// ── Accessors ──────────────────────────────────────

export function hasMore() {
  return _hasMore;
}

export function totalCount() {
  return _totalCount;
}

/**
 * Reset pagination state (called on dispose).
 */
export function resetPagination() {
  _currentOffset = 0;
  _hasMore = false;
  _totalCount = 0;
  _currentHours = 48;
}

// ── Fetch helpers ──────────────────────────────────

/**
 * Normalize raw API events: assign IDs where missing.
 * @param {object[]} rawEvents
 * @returns {object[]}
 */
function _normalizeRawEvents(rawEvents) {
  const result = [];
  for (const raw of rawEvents) {
    const evt = normalizeEvent(raw);
    if (!evt.id) {
      evt.id = evt.ts || Date.now().toString();
    }
    result.push(evt);
  }
  return result;
}

/**
 * Fetch initial history from the backend.
 *
 * @param {number} [hours=48]
 * @returns {Promise<{ events: object[], hasMore: boolean, total: number }>}
 */
export async function fetchHistory(hours = 48) {
  _currentHours = hours;
  try {
    const res = await fetch(`/api/activity/recent?hours=${hours}&limit=200&offset=0`);
    if (!res.ok) return { events: [], hasMore: false, total: 0 };
    const data = await res.json();
    const rawEvents = data.events || [];

    _currentOffset = rawEvents.length;
    _hasMore = data.has_more || false;
    _totalCount = data.total || 0;

    return {
      events: _normalizeRawEvents(rawEvents),
      hasMore: _hasMore,
      total: _totalCount,
    };
  } catch (err) {
    console.warn("[timeline] Failed to load history:", err);
    return { events: [], hasMore: false, total: 0 };
  }
}

/**
 * Fetch the next page of events.
 *
 * @returns {Promise<{ events: object[], hasMore: boolean, total: number }>}
 */
export async function fetchMore() {
  const btn = document.getElementById("wsTimelineLoadMore");
  if (btn) {
    btn.textContent = t("common.loading");
    btn.disabled = true;
  }
  try {
    const res = await fetch(`/api/activity/recent?hours=${_currentHours}&limit=200&offset=${_currentOffset}`);
    if (!res.ok) return { events: [], hasMore: _hasMore, total: _totalCount };
    const data = await res.json();
    const rawEvents = data.events || [];

    _hasMore = data.has_more || false;
    _totalCount = data.total || 0;
    _currentOffset += rawEvents.length;

    return {
      events: _normalizeRawEvents(rawEvents),
      hasMore: _hasMore,
      total: _totalCount,
    };
  } catch (err) {
    console.warn("[timeline] Failed to load more:", err);
    return { events: [], hasMore: _hasMore, total: _totalCount };
  } finally {
    if (btn) {
      btn.textContent = t("ws.load_more");
      btn.disabled = false;
    }
  }
}
