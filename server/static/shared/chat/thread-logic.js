// ── Shared Thread Logic ──────────────────────
// Pure functions for thread CRUD and tab HTML generation.

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
  if (threadId === "default") return "メイン";
  if (!lastTs || !timeStr) return "スレッド";
  return `スレッド ${timeStr(lastTs)}`;
}

/**
 * Create a new thread entry.
 * @param {Array} threadList - Current thread list
 * @param {string} _animaName - (unused, kept for interface consistency)
 * @returns {{ updatedList: Array, newThreadId: string, newEntry: object }}
 */
export function createThread(threadList, _animaName) {
  const threadId = crypto.randomUUID().slice(0, 8);
  const newEntry = { id: threadId, label: "新しいスレッド", unread: false };
  const updatedList = [...threadList, newEntry];
  return { updatedList, newThreadId: threadId, newEntry };
}

/**
 * Close (remove) a thread from the list.
 * @param {Array} threadList
 * @param {string} threadId
 * @returns {Array} Updated list (unchanged if threadId is "default" or not found)
 */
export function closeThread(threadList, threadId) {
  if (threadId === "default") return threadList;
  return threadList.filter(th => th.id !== threadId);
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
 * @returns {string} HTML string
 */
export function renderThreadTabsHtml(threadList, activeThreadId, opts) {
  const { escapeHtml } = opts;
  const maxVisible = opts.maxVisible ?? 5;
  const newBtnId = opts.newBtnId || "chatNewThreadBtn";
  const moreSelectId = opts.moreSelectId || "chatThreadMoreSelect";

  const list = threadList.length > 0 ? threadList : [{ id: "default", label: "メイン", unread: false }];
  const defaultThread = list.find(th => th.id === "default") || { id: "default", label: "メイン", unread: false };
  const nonDefault = list.filter(th => th.id !== "default").sort((a, b) => {
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
    const star = th.unread ? ' <span class="tab-star" aria-label="unread">★</span>' : "";
    const closeBtn = th.id !== "default"
      ? ` <button type="button" class="thread-tab-close" data-thread="${escapeHtml(th.id)}" title="スレッドを閉じる" aria-label="閉じる">&times;</button>`
      : "";
    html += `<span class="thread-tab-wrap"><button type="button" class="thread-tab${activeClass}" data-thread="${escapeHtml(th.id)}">${escapeHtml(th.label)}${star}</button>${closeBtn}</span>`;
  }

  if (hiddenThreads.length > 0) {
    html += `<span class="thread-more-wrap">` +
      `<label class="thread-more-label" for="${moreSelectId}">他 ${hiddenThreads.length} 件</label>` +
      `<select id="${moreSelectId}" class="thread-more-select">` +
      `<option value="">スレッドを選択...</option>` +
      hiddenThreads.map(th => `<option value="${escapeHtml(th.id)}">${escapeHtml(th.label)}${th.unread ? " ★" : ""}</option>`).join("") +
      `</select></span>`;
  }

  html += `<button type="button" class="thread-tab-new" id="${newBtnId}" title="新しいスレッド">＋</button>`;
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
    : [{ id: "default", label: "メイン", unread: false }];
  const byId = new Map(existing.map(th => [th.id, { ...th }]));

  if (!byId.has("default")) {
    byId.set("default", { id: "default", label: "メイン", unread: false, lastTs: 0 });
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

  const def = byId.get("default") || { id: "default", label: "メイン", unread: false, lastTs: 0 };
  byId.delete("default");
  const rest = Array.from(byId.values()).sort((a, b) => {
    const diff = threadTimeValue(b.lastTs || "") - threadTimeValue(a.lastTs || "");
    if (diff !== 0) return diff;
    return String(a.label || "").localeCompare(String(b.label || ""), "ja");
  });

  return [def, ...rest];
}
