// ── Shared History Loader ──────────────────────
// DOM-independent state management for conversation history pagination.

/**
 * Create a fresh history state object for a single thread.
 *
 * `loaded` means a fetch succeeded, so an empty state is a confirmed empty
 * history. `loadFailed` means the last full load failed with nothing to show;
 * `olderFailed` means the last older page failed while the rest stays on screen.
 * @returns {{ sessions: Array, feedItems: Array, isFeed: boolean, hasMore: boolean, nextBefore: string|null, loading: boolean, loaded: boolean, loadFailed: boolean, olderFailed: boolean }}
 */
export function createHistoryState() {
  return {
    sessions: [], feedItems: [], isFeed: false, hasMore: false, nextBefore: null,
    loading: false, loaded: false, loadFailed: false, olderFailed: false,
  };
}

/**
 * Whether a history state holds anything to show (feed background groups count).
 * @param {object} hs
 * @returns {boolean}
 */
export function hasHistoryContent(hs) {
  return hs?.isFeed ? (hs.feedItems || []).length > 0 : (hs?.sessions || []).length > 0;
}

/**
 * Whether the loaded feed pages hold a conversation message, rather than only
 * background work or a thread's compressed summary.
 * @param {object} hs
 * @returns {boolean}
 */
export function hasFeedConversation(hs) {
  return (hs?.feedItems || []).some(item => item?.kind === "message" && item.message?.type !== "thread_summary");
}

function _feedItemTs(item, isEnd = false) {
  if (item?.kind === "bg_group") return isEnd ? item.to_ts || item.from_ts || "" : item.from_ts || item.to_ts || "";
  return item?.message?.ts || "";
}

function _feedSessions(items) {
  return (items || [])
    .filter(item => item?.kind === "message" && item.message)
    .map(item => ({
      session_start: item.message.ts || "",
      session_end: item.message.ts || "",
      trigger: "chat",
      messages: [item.message],
    }));
}

/**
 * Apply API response data to a history state object.
 * Handles both initial load and pagination (prepend older sessions).
 * @param {object} hs - History state to mutate
 * @param {object} data - API response { sessions, has_more, next_before }
 * @param {object} [options]
 * @param {boolean} [options.prepend=false] - If true, prepend sessions (for loading older)
 * @returns {object} The mutated history state
 */
export function applyHistoryData(hs, data, options) {
  const prepend = options?.prepend || false;

  if (Array.isArray(data?.items)) {
    if (data.items.length > 0 || !prepend) {
      hs.feedItems = prepend ? [...data.items, ...(hs.feedItems || [])] : data.items;
    }
    hs.isFeed = true;
    hs.sessions = _feedSessions(hs.feedItems);
    hs.hasMore = Boolean(data.has_more);
    hs.nextBefore = data.next_before || null;
    hs.loading = false;
    return hs;
  }

  hs.isFeed = false;
  hs.feedItems = [];

  if (!data || !data.sessions || data.sessions.length === 0) {
    if (!prepend) {
      hs.sessions = [];
    }
    hs.hasMore = false;
    hs.loading = false;
    return hs;
  }

  if (prepend) {
    hs.sessions = [...data.sessions, ...hs.sessions];
  } else {
    hs.sessions = data.sessions;
  }

  hs.hasMore = data.has_more || false;
  hs.nextBefore = data.next_before || null;
  hs.loading = false;
  return hs;
}

/**
 * Merge polled conversation data with existing history state.
 * Preserves older sessions that were loaded via pagination while
 * updating the most recent sessions from the polling response.
 * @param {object} prev - Previous history state
 * @param {object} pollData - Poll API response { sessions, has_more, next_before }
 * @returns {{ changed: boolean, merged: object }} Whether data changed and the merged state
 */
function _sessionsEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].session_start !== b[i].session_start) return false;
    if (a[i].session_end !== b[i].session_end) return false;
    const aMsgs = a[i].messages || [];
    const bMsgs = b[i].messages || [];
    if (aMsgs.length !== bMsgs.length) return false;
    if (aMsgs.length > 0 && bMsgs.length > 0) {
      const aLast = aMsgs[aMsgs.length - 1];
      const bLast = bMsgs[bMsgs.length - 1];
      if (aLast.ts !== bLast.ts) return false;
    }
  }
  return true;
}

/**
 * Paging position after a poll of the newest page. Older pages that held no
 * items leave no trace but their cursor, so a cursor already past the poll's stays.
 */
function _polledPaging(prev, pollData) {
  const pollCursor = pollData.next_before || null;
  if (prev?.isFeed && prev.nextBefore && pollCursor && prev.nextBefore < pollCursor) {
    return { hasMore: prev.hasMore, nextBefore: prev.nextBefore };
  }
  return { hasMore: Boolean(pollData.has_more), nextBefore: pollCursor };
}

export function mergePolledHistory(prev, pollData) {
  if (pollData && Array.isArray(pollData.items)) {
    if (prev?.loading) return { changed: false, merged: prev };
    const previousItems = prev?.isFeed ? (prev.feedItems || []) : [];
    if (previousItems.length === 0) {
      // A poll is a successful fetch: it settles a pending or failed load. An
      // empty newest page once the load has settled adds nothing.
      if (pollData.items.length === 0 && prev?.isFeed && prev.loaded && !prev.loadFailed) {
        return { changed: false, merged: prev };
      }
      return {
        changed: true,
        merged: {
          ...createHistoryState(),
          feedItems: pollData.items,
          sessions: _feedSessions(pollData.items),
          isFeed: true,
          ..._polledPaging(prev, pollData),
          loaded: true,
        },
      };
    }

    if (pollData.items.length === 0 && previousItems.length > 0) {
      return { changed: false, merged: prev };
    }

    const pollOldestTs = _feedItemTs(pollData.items[0]);
    const olderItems = pollOldestTs
      ? previousItems.filter(item => _feedItemTs(item, true) && _feedItemTs(item, true) < pollOldestTs)
      : [];
    const currentItems = pollOldestTs
      ? previousItems.filter(item => !_feedItemTs(item, true) || _feedItemTs(item, true) >= pollOldestTs)
      : previousItems;
    const changed = JSON.stringify(currentItems) !== JSON.stringify(pollData.items);
    if (!changed) return { changed: false, merged: prev };

    const feedItems = [...olderItems, ...pollData.items];
    const merged = {
      ...prev,
      feedItems,
      sessions: _feedSessions(feedItems),
      isFeed: true,
      loaded: true,
      loadFailed: false,
    };
    if (olderItems.length === 0) Object.assign(merged, _polledPaging(prev, pollData));
    return { changed: true, merged };
  }

  if (!pollData || !Array.isArray(pollData.sessions)) {
    return { changed: false, merged: prev };
  }

  if (!prev || prev.sessions.length === 0) {
    return {
      changed: true,
      merged: {
        sessions: pollData.sessions,
        hasMore: pollData.has_more || false,
        nextBefore: pollData.next_before || null,
        loading: false,
      },
    };
  }

  if (prev.loading) return { changed: false, merged: prev };

  const pollOldestStart = pollData.sessions[0]?.session_start || "";
  const olderSessions = pollOldestStart
    ? prev.sessions.filter(s => s.session_start && s.session_start < pollOldestStart)
    : [];
  const currentPolledPart = pollOldestStart
    ? prev.sessions.filter(s => !s.session_start || s.session_start >= pollOldestStart)
    : prev.sessions;

  const changed = !_sessionsEqual(currentPolledPart, pollData.sessions);
  if (!changed) return { changed: false, merged: prev };

  const merged = {
    ...prev,
    sessions: [...olderSessions, ...pollData.sessions],
  };
  if (olderSessions.length === 0) {
    merged.hasMore = pollData.has_more || false;
    merged.nextBefore = pollData.next_before || null;
  }

  return { changed: true, merged };
}
