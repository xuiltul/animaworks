// ── Shared Thread Logic ──────────────────────
// Pure functions for thread CRUD and tab HTML generation.
import { t } from "../i18n.js";

/**
 * Parse a timestamp string to a numeric value for sorting.
 * @param {string} ts - ISO timestamp or empty
 * @returns {number}
 */
export function threadTimeValue(ts) {
  if (!ts) return 0;
  const v = Date.parse(ts);
  return Number.isNaN(v) ? 0 : v;
}

/**
 * Generate a default label for a thread.
 * @param {string} threadId
 * @param {string} [lastTs]
 * @param {function} [timeStr] - Formatter for timestamp display
 */
export function defaultThreadLabel(threadId, lastTs, timeStr) {
  if (threadId === "default") return t("thread.default_label");
  if (!lastTs || !timeStr) return t("pane.thread_label");
  return t("thread.label_with_time", { time: timeStr(lastTs) });
}

/**
 * Create a new thread entry.
 * @param {Array} threadList - Current thread list
 * @param {string} _animaName - (unused, kept for interface consistency)
 * @returns {{ updatedList: Array, newThreadId: string, newEntry: object }}
 */
export function createThread(threadList, _animaName) {
  // crypto.randomUUID requires secure context (HTTPS or localhost).
  // Fall back to crypto.getRandomValues for HTTP + LAN IP access.
  let threadId;
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    threadId = crypto.randomUUID().slice(0, 8);
  } else {
    const arr = new Uint8Array(4);
    crypto.getRandomValues(arr);
    threadId = Array.from(arr, b => b.toString(16).padStart(2, '0')).join('');
  }
  const newEntry = { id: threadId, label: t("thread.new"), unread: false };
  const updatedList = [...threadList, newEntry];
  return { updatedList, newThreadId: threadId, newEntry };
}

/**
 * Archive a thread (mark as archived instead of removing).
 * @param {Array} threadList
 * @param {string} threadId
 * @returns {Array} Updated list with thread marked archived
 */
export function archiveThread(threadList, threadId) {
  if (threadId === "default") return threadList;
  return threadList.map(th => th.id === threadId ? { ...th, archived: true } : th);
}

/**
 * Restore an archived thread.
 * @param {Array} threadList
 * @param {string} threadId
 * @returns {Array} Updated list with thread un-archived
 */
export function restoreThread(threadList, threadId) {
  return threadList.map(th => th.id === threadId ? { ...th, archived: false } : th);
}

/**
 * Close (archive) a thread from the list.
 * @param {Array} threadList
 * @param {string} threadId
 * @returns {Array} Updated list
 */
export function closeThread(threadList, threadId) {
  return archiveThread(threadList, threadId);
}

/**
 * Generate thread tabs HTML string.
 * @param {Array} threadList
 * @param {string} activeThreadId
 * @param {object} opts
 * @param {function} opts.escapeHtml
 * @param {number}  [opts.maxVisible=5] - Max visible non-default threads
 * @param {string}  [opts.newBtnId="chatNewThreadBtn"] - ID for the new-thread button
 * @param {string}  [opts.moreSelectId="chatThreadMoreSelect"] - ID for the more-threads select
 * @param {string}  [opts.streamingThreadId] - Thread currently streaming
 * @returns {string} HTML string
 */
export function renderThreadTabsHtml(threadList, activeThreadId, opts) {
  const { escapeHtml } = opts;
  const streamingThreadId = opts.streamingThreadId || null;
  const maxVisible = opts.maxVisible ?? 5;
  const newBtnId = opts.newBtnId || "chatNewThreadBtn";
  const moreSelectId = opts.moreSelectId || "chatThreadMoreSelect";

  const list = threadList.length > 0 ? threadList : [{ id: "default", label: t("thread.default_label"), unread: false }];
  const defaultThread = list.find(th => th.id === "default") || { id: "default", label: t("thread.default_label"), unread: false };
  const archived = list.filter(th => th.id !== "default" && th.archived);
  const nonDefault = list.filter(th => th.id !== "default" && !th.archived).sort((a, b) => {
    const diff = threadTimeValue(b.lastTs || "") - threadTimeValue(a.lastTs || "");
    if (diff !== 0) return diff;
    return String(a.label || "").localeCompare(String(b.label || ""), "ja");
  });

  let visibleNonDefault = nonDefault.slice(0, maxVisible);
  const activeHidden = activeThreadId !== "default" && !visibleNonDefault.some(th => th.id === activeThreadId);
  if (activeHidden) {
    const activeThread = nonDefault.find(th => th.id === activeThreadId);
    if (activeThread) {
      visibleNonDefault = [activeThread, ...visibleNonDefault.slice(0, Math.max(0, maxVisible - 1))];
      const unique = new Map();
      for (const th of visibleNonDefault) unique.set(th.id, th);
      visibleNonDefault = Array.from(unique.values());
    }
  }

  const visibleIds = new Set(visibleNonDefault.map(th => th.id));
  const hiddenThreads = nonDefault.filter(th => !visibleIds.has(th.id));

  let html = "";
  const visible = [defaultThread, ...visibleNonDefault];
  for (const th of visible) {
    const activeClass = th.id === activeThreadId ? " active" : "";
    const streamClass = streamingThreadId === th.id ? " is-streaming" : "";
    const completeClass = th.unread && th.id !== activeThreadId ? " has-unread-complete" : "";
    const star = th.unread ? ' <span class="tab-star" aria-label="unread">★</span>' : "";
    const closeBtn = th.id !== "default"
      ? ` <button type="button" class="thread-tab-close" data-thread="${escapeHtml(th.id)}" title="${escapeHtml(t("thread.close"))}" aria-label="${escapeHtml(t("thread.close_short"))}">&times;</button>`
      : "";
    html += `<span class="thread-tab-wrap"><button type="button" class="thread-tab${activeClass}${streamClass}${completeClass}" data-thread="${escapeHtml(th.id)}">${escapeHtml(th.label)}${star}</button>${closeBtn}</span>`;
  }

  if (hiddenThreads.length > 0) {
    html += `<span class="thread-more-wrap">` +
      `<label class="thread-more-label">${escapeHtml(t("thread.more_count", { count: hiddenThreads.length }))}</label>` +
      `<select data-chat-id="${moreSelectId}" class="thread-more-select">` +
      `<option value="">${escapeHtml(t("thread.select"))}</option>` +
      hiddenThreads.map(th => `<option value="${escapeHtml(th.id)}">${escapeHtml(th.label)}${th.unread ? " ★" : ""}</option>`).join("") +
      `</select></span>`;
  }

  html += `<button type="button" class="thread-tab-new" data-chat-id="${newBtnId}" title="${escapeHtml(t("thread.new"))}">＋</button>`;

  if (archived.length > 0) {
    const archiveBtnId = opts.archiveBtnId || "chatArchiveBtn";
    const archiveMenuId = opts.archiveMenuId || "chatArchiveMenu";
    html += `<span class="thread-archive-wrap">` +
      `<button type="button" class="thread-archive-btn" data-chat-id="${archiveBtnId}" title="${escapeHtml(t("thread.archive_count", { count: archived.length }))}">` +
      `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">` +
      `<polyline points="21 8 21 21 3 21 3 8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/>` +
      `</svg>` +
      `<span class="thread-archive-badge">${archived.length}</span>` +
      `</button>` +
      `<div class="thread-archive-menu" data-chat-id="${archiveMenuId}">` +
      archived.map(th =>
        `<button type="button" class="thread-archive-item" data-thread="${escapeHtml(th.id)}">${escapeHtml(th.label)}</button>`
      ).join("") +
      `</div></span>`;
  }

  return html;
}

/**
 * Merge thread data from sessions API response into existing thread list.
 * @param {Array} existingThreads
 * @param {object} sessionsData - { threads: [{ thread_id, last_timestamp }] }
 * @param {object} [helpers]
 * @param {function} [helpers.timeStr] - Timestamp formatter
 * @returns {Array} Merged and sorted thread list
 */
export function mergeThreadsFromSessions(existingThreads, sessionsData, helpers) {
  if (!sessionsData) return existingThreads;
  const timeStrFn = helpers?.timeStr;

  const existing = existingThreads.length > 0
    ? existingThreads
    : [{ id: "default", label: t("thread.default_label"), unread: false }];
  const byId = new Map(existing.map(th => [th.id, { ...th }]));

  if (!byId.has("default")) {
    byId.set("default", { id: "default", label: t("thread.default_label"), unread: false, lastTs: 0 });
  }

  for (const th of sessionsData.threads || []) {
    const id = th?.thread_id;
    if (!id || id === "default") continue;
    const prev = byId.get(id) || { id, unread: false };
    const nextTs = threadTimeValue(th.last_timestamp || "");
    const prevTs = threadTimeValue(prev.lastTs || "");
    byId.set(id, {
      ...prev,
      id,
      label: prev.label || defaultThreadLabel(id, th.last_timestamp || "", timeStrFn),
      lastTs: Math.max(prevTs, nextTs),
    });
  }

  const def = byId.get("default") || { id: "default", label: t("thread.default_label"), unread: false, lastTs: 0 };
  byId.delete("default");
  const rest = Array.from(byId.values()).sort((a, b) => {
    const diff = threadTimeValue(b.lastTs || "") - threadTimeValue(a.lastTs || "");
    if (diff !== 0) return diff;
    return String(a.label || "").localeCompare(String(b.label || ""), "ja");
  });

  return [def, ...rest];
}
