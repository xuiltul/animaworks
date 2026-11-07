// ── Board store ──────────────────────────────
// Single source of truth for the board page. All three message sources
// (initial fetch / incremental refetch / WebSocket / optimistic send) go
// through mergeMessages(), keyed by server id (falling back to ts+from), so
// the same post can never be rendered twice.
import { api } from "../../modules/api.js";
import { bustupCandidates, resolveCachedAvatar } from "../../modules/avatar-resolver.js";
import { kindGroup } from "./avatar.js";

const REFRESH_MS = 30_000;
const COMPACT_WINDOW_MS = 10 * 60 * 1000;
const USER_KEY = "animaworks_user";

export function currentUser() {
  try {
    return localStorage.getItem(USER_KEY) || "human";
  } catch {
    return "human";
  }
}

function _parseTs(value) {
  const time = Date.parse(String(value || ""));
  return Number.isNaN(time) ? null : time;
}

/** Whether *candidate* is at or after *reference* (ISO timestamps). */
export function timestampAtOrAfter(candidate, reference) {
  if (!reference) return false;
  if (!candidate) return false;
  const a = _parseTs(candidate);
  const b = _parseTs(reference);
  if (a !== null && b !== null) return a >= b;
  return String(candidate) >= String(reference);
}

const SEARCH_DEBOUNCE_MS = 300;
const KIND_FILTERS = ["anima", "human", "external", "operator"];

export function messageKey(message) {
  if (!message) return "";
  if (message.id !== undefined && message.id !== null && message.id !== "") {
    return `id:${message.id}`;
  }
  return `ts:${message.ts || ""}|from:${message.from || ""}`;
}

/** Map an /api/animas row (or anima.status event) to a board status. */
export function boardStatusOf(anima, override) {
  const raw = String(override ?? anima?.status ?? "").toLowerCase();
  if (anima?.enabled === false || raw === "disabled" || raw === "inactive") return "stopped";
  if (raw === "idle" || raw === "") return "idle";
  if (raw === "stopped" || raw === "not_found" || raw === "offline" || raw === "unknown" || raw === "sleeping" || raw === "sleep") {
    return "sleeping";
  }
  if (["running", "thinking", "processing", "busy", "bootstrapping", "starting", "restarting"].includes(raw)) {
    return "busy";
  }
  if (/run|work|think|process|busy/.test(raw)) return "busy";
  if (/sleep|stop|inactive|offline/.test(raw)) return "sleeping";
  return "idle";
}

export function createBoardStore() {
  const state = {
    channels: [],
    dmPairs: [],
    animas: [],
    animaByName: new Map(),
    statusOverrides: new Map(),
    avatars: new Map(),
    mode: "board", // "board" | "dm"
    view: "home", // "home" | "flows" (null while a channel/DM is open)
    selectedType: null, // "channel" | "dm"
    selectedName: null,
    messages: [],
    messageKeys: new Set(),
    total: 0,
    hasMore: false,
    loading: false,
    loadingOlder: false,
    error: "",
    lastTs: "",
    expanded: new Set(),
    personFilter: "",
    dmFilter: "",
    archiveOpen: false,
    pollTimer: null,
    loadToken: 0,
    // Read positions for the selected channel: readers[reader] = {upto_id,
    // upto_ts, updated_at}; picked[reader] = [post_id, ...].
    reads: { readers: {}, picked: {} },
    readsChannel: null,
    // "ここから新着" is fixed when the channel is opened, so marking read while
    // the page is open never moves the line until it is reopened.
    readLine: null,
    focusId: "",
    markTimer: null,
    lastMarkedId: "",
    // Keys added by a live event (WebSocket / poll) since the last render.
    liveKeys: new Set(),
    // 5-1: server-side search / kind filter and the per-kind counts.
    kindFilter: "all",
    query: "",
    filterCounts: { all: 0, anima: 0, human: 0, external: 0, operator: 0 },
    filterTotal: 0,
    boardTotal: 0,
    searchTimer: null,
    // Bumped when board.read / board.post arrives so the info drawer reloads.
    detailRevision: 0,
  };

  const listeners = new Set();

  function emit(reason) {
    for (const listener of [...listeners]) {
      try {
        listener(reason);
      } catch (error) {
        console.error("board store listener failed", error);
      }
    }
  }

  function subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  // ── Avatar cache ───────────────────────────
  async function ensureAvatars(names) {
    const unique = [...new Set((names || []).filter(Boolean))];
    const pending = [];
    for (const name of unique) {
      if (state.avatars.has(name)) continue;
      // Reserve the slot so concurrent loads do not probe the same name twice.
      state.avatars.set(name, null);
      pending.push(
        resolveCachedAvatar(name, bustupCandidates(), "S")
          .then((url) => {
            state.avatars.set(name, url || null);
          })
          .catch(() => {
            state.avatars.set(name, null);
          }),
      );
    }
    if (pending.length === 0) return false;
    await Promise.all(pending);
    emit("avatars");
    return true;
  }

  function avatarUrl(name) {
    return state.avatars.get(name) || null;
  }

  // ── Read positions ─────────────────────────
  function humanReader() {
    return `human:${currentUser()}`;
  }

  function readerName(reader) {
    const value = String(reader || "");
    const colon = value.indexOf(":");
    return colon >= 0 ? value.slice(colon + 1) : value;
  }

  function isAnimaReader(reader) {
    return String(reader || "").startsWith("anima:");
  }

  function _pickedFor(reader) {
    const values = state.reads.picked?.[reader];
    return Array.isArray(values) ? values : [];
  }

  function hasReadReader(reader, message) {
    if (!reader || !message) return false;
    if (_pickedFor(reader).includes(String(message.id))) return true;
    const position = state.reads.readers?.[reader];
    if (!position) return false;
    if (position.upto_id && String(position.upto_id) === String(message.id)) return true;
    if (!position.upto_ts) return false;
    return timestampAtOrAfter(position.upto_ts, message.ts);
  }

  function hasReadByName(name, message) {
    return hasReadReader(`anima:${name}`, message);
  }

  /** Anima names that have read *message* (author and the viewer excluded). */
  function readersForMessage(message) {
    if (!message) return [];
    return readerPoolForMessage(message).filter((name) => hasReadByName(name, message));
  }

  /**
   * Who may appear as a reader face for *message*: the members of a
   * member-only board, or every Anima with a recorded read position on an open
   * board. The author and the human viewing the board are never counted.
   */
  function readerPoolForMessage(message) {
    if (!message) return [];
    const author = String(message.from || "");
    const self = currentUser();
    const names = [];
    const selected = state.selectedType === "channel" ? channel(state.selectedName) : null;
    if (selected && selected.access === "members") {
      for (const name of selected.members || []) names.push(String(name));
    } else {
      for (const reader of Object.keys(state.reads.readers || {})) {
        if (isAnimaReader(reader)) names.push(readerName(reader));
      }
      for (const reader of Object.keys(state.reads.picked || {})) {
        if (isAnimaReader(reader)) names.push(readerName(reader));
      }
    }
    return [...new Set(names)].filter((name) => name && name !== author && name !== self);
  }

  /** The first reply in the loaded board from *name* to *message*. */
  function replyFor(message, name) {
    if (!message || message.id === undefined || message.id === null) return null;
    return state.messages.find(
      (row) => String(row.reply_to) === String(message.id) && row.from === name,
    ) || null;
  }

  function replyTarget(message) {
    if (!message || !message.reply_to) return null;
    return messageById(message.reply_to);
  }

  function _createdAtFromMessage(message, fallback) {
    const parsed = _parseTs(message?.ts) ?? _parseTs(fallback);
    if (parsed === null) return "";
    return new Date(parsed).toISOString();
  }

  function _computeReadLine() {
    const reader = humanReader();
    const position = state.reads.readers?.[reader];
    if (!position || !position.upto_id) return null;
    const message = messageById(position.upto_id);
    return {
      afterId: String(position.upto_id),
      name: message ? String(message.from || "") : "",
      ts: message ? String(message.ts || "") : String(position.upto_ts || ""),
      reader,
    };
  }

  function _applyRead(reader, payload, pickedId = "") {
    const current = state.reads.readers?.[reader] || {};
    const next = {
      upto_id: String(payload?.upto_id || current.upto_id || ""),
      upto_ts: String(payload?.upto_ts || current.upto_ts || ""),
      updated_at: _createdAtFromMessage(null, payload?.updated_at) || current.updated_at || "",
    };
    state.reads.readers[reader] = next;
    if (pickedId) {
      const picked = _pickedFor(reader);
      if (!picked.includes(String(pickedId))) picked.push(String(pickedId));
      state.reads.picked[reader] = picked;
    }
  }

  function _resetReads() {
    state.reads = { readers: {}, picked: {} };
    state.readsChannel = null;
    state.readLine = null;
    state.lastMarkedId = "";
    if (state.markTimer) {
      clearTimeout(state.markTimer);
      state.markTimer = null;
    }
  }

  async function _loadReads(name, token) {
    try {
      const data = await api(`/api/channels/${encodeURIComponent(name)}/reads`);
      if (token !== state.loadToken) return;
      state.reads = {
        readers: data && typeof data.readers === "object" && data.readers ? data.readers : {},
        picked: data && typeof data.picked === "object" && data.picked ? data.picked : {},
      };
      state.readsChannel = name;
      state.readLine = _computeReadLine();
    } catch {
      if (token !== state.loadToken) return;
      state.reads = { readers: {}, picked: {} };
      state.readsChannel = name;
      state.readLine = null;
    }
  }

  function handleBoardRead(data) {
    if (!data || !data.reader) return;
    if (state.selectedType !== "channel") return;
    if (data.channel && String(data.channel) !== String(state.selectedName)) return;
    const reader = String(data.reader);
    if (!reader.includes(":")) {
      // Accept both "anima:name" and a bare anima name.
      _applyRead(`anima:${reader}`, data, data.picked_id);
    } else {
      _applyRead(reader, data, data.picked_id);
    }
    if (state.selectedType === "channel") {
      emit("reads");
      // The info drawer reloads its read positions on this revision.
      state.detailRevision += 1;
      emit("read");
    }
  }

  function scheduleMarkRead() {
    if (state.selectedType !== "channel" || !state.selectedName) return;
    if (state.markTimer) clearTimeout(state.markTimer);
    const token = state.loadToken;
    state.markTimer = setTimeout(() => {
      state.markTimer = null;
      if (token !== state.loadToken) return;
      const last = state.messages[state.messages.length - 1];
      if (last && last.id !== undefined && last.id !== null && last.id !== "") {
        void markRead(last.id);
      }
    }, 500);
  }

  async function markRead(postId) {
    if (!postId || state.selectedType !== "channel" || !state.selectedName) return;
    const reader = humanReader();
    const current = state.reads.readers?.[reader];
    if (current && String(current.upto_id) === String(postId)) return;
    if (state.lastMarkedId === String(postId)) return;
    state.lastMarkedId = String(postId);
    const name = state.selectedName;
    const token = state.loadToken;
    try {
      const result = await api(`/api/channels/${encodeURIComponent(name)}/read`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ upto_id: String(postId) }),
      });
      if (token !== state.loadToken) return;
      if (result && typeof result === "object") {
        _applyRead(reader, {
          upto_id: result.upto_id || postId,
          upto_ts: result.upto_ts || "",
        });
      } else {
        _applyRead(reader, { upto_id: postId, upto_ts: "" });
      }
      const info = channel(name);
      if (info) info.unread = 0;
      emit("reads");
      emit("channels");
    } catch {
      if (token === state.loadToken) state.lastMarkedId = "";
    }
  }

  /** Open a post by id, fetching it with ``around`` when it is not loaded. */
  async function jumpTo(postId) {
    if (!postId) return;
    if (state.selectedType !== "channel" && state.selectedType !== "dm") return;
    if (state.selectedType === "channel" && !messageById(postId)) await _loadAround(postId);
    state.focusId = String(postId);
    emit("messages");
  }

  async function _loadAround(postId) {
    const name = state.selectedName;
    if (!name) return;
    const token = state.loadToken;
    try {
      const data = await api(
        `/api/channels/${encodeURIComponent(name)}?around=${encodeURIComponent(postId)}&limit=50`,
      );
      if (token !== state.loadToken) return;
      mergeMessages(data.messages || [], { type: "channel" });
      if (data.total !== undefined) state.total = Number(data.total) || state.total;
      if (data.has_more !== undefined) state.hasMore = Boolean(data.has_more);
      await ensureAvatars((data.messages || []).map((message) => message.from));
    } catch {
      // The link still points at the id; nothing else to do if it is gone.
    }
  }

  function clearFocus() {
    state.focusId = "";
  }

  /**
   * Rendering signature for the read/reply/status dependent parts of a post.
   * A change rebuilds the node so chips and faces update in place.
   */
  function stateSignature(message) {
    const delivery = message?.delivery || {};
    const notified = Array.isArray(delivery.notified) ? delivery.notified : [];
    const calls = notified.map((name) => {
      const reply = replyFor(message, name);
      return [
        name,
        statusOf(name),
        hasReadByName(name, message) ? "1" : "0",
        reply ? `${reply.id || ""}:${reply.ts || ""}` : "",
      ].join(":");
    });
    const readers = readersForMessage(message);
    const pool = readerPoolForMessage(message);
    const target = replyTarget(message);
    return JSON.stringify({
      delivery,
      hooks: message?.hooks || {},
      calls,
      readers,
      pool,
      target: target ? [target.id, target.from, target.ts] : null,
    });
  }

  // ── Anima directory / status ───────────────
  async function loadAnimas() {
    try {
      const list = await api("/api/animas");
      state.animas = Array.isArray(list) ? list : [];
    } catch {
      state.animas = [];
    }
    state.animaByName = new Map(state.animas.map((anima) => [anima.name, anima]));
    await ensureAvatars(state.animas.map((anima) => anima.name));
    emit("animas");
  }

  function anima(name) {
    return state.animaByName.get(name) || null;
  }

  function hasAnima(name) {
    return state.animaByName.has(name);
  }

  function statusOf(name) {
    if (state.statusOverrides.has(name)) {
      return boardStatusOf(anima(name), state.statusOverrides.get(name));
    }
    return boardStatusOf(anima(name));
  }

  function handleAnimaStatus(data) {
    if (!data) return;
    const name = data.name || data.anima;
    if (!name) return;
    state.statusOverrides.set(name, data.status ?? "");
    if (state.animaByName.has(name) && data.enabled !== undefined) {
      state.animaByName.get(name).enabled = data.enabled;
    }
    emit("status");
  }

  // ── Rail data ──────────────────────────────
  async function loadChannels() {
    try {
      const list = await api("/api/channels");
      state.channels = Array.isArray(list) ? list : [];
    } catch {
      state.channels = [];
    }
    emit("channels");
  }

  async function loadDmPairs() {
    try {
      const list = await api("/api/dm");
      state.dmPairs = Array.isArray(list) ? list : [];
    } catch {
      state.dmPairs = [];
    }
    const names = state.dmPairs.flatMap((pair) => pair.participants || []);
    await ensureAvatars(names);
    emit("dm");
  }

  function channel(name) {
    return state.channels.find((item) => item.name === name) || null;
  }

  function hasChannel(name) {
    return state.channels.some((item) => item.name === name);
  }

  function pair(name) {
    return state.dmPairs.find((item) => item.pair === name) || null;
  }

  // ── Message merging ────────────────────────
  function _normalize(row, type) {
    const message = { ...row };
    if (!message.ts) message.ts = new Date().toISOString();
    if (!message.from) message.from = "unknown";
    if (type === "dm") {
      if (
        (message.id === undefined || message.id === null || message.id === "") &&
        message.message_id !== undefined &&
        message.message_id !== null &&
        message.message_id !== ""
      ) {
        message.id = message.message_id;
      }
      if (!message.kind) {
        message.kind = String(message.source || "").toLowerCase() === "human" ? "human" : "anima";
      }
    }
    return message;
  }

  function _updateLastTs() {
    let last = "";
    for (const message of state.messages) {
      if (message.ts && message.ts > last) last = message.ts;
    }
    state.lastTs = last;
  }

  /**
   * Merge rows into the current list. Returns the number of newly added rows.
   * Every source must call this exactly once per delivery.
   */
  function mergeMessages(rows, { type = state.selectedType, live = false } = {}) {
    let added = 0;
    for (const row of rows || []) {
      if (!row) continue;
      const key = messageKey(row);
      if (!key || state.messageKeys.has(key)) continue;
      state.messageKeys.add(key);
      state.messages.push(_normalize(row, type));
      if (live) state.liveKeys.add(key);
      added += 1;
    }
    if (added > 0) {
      state.messages.sort((a, b) => String(a.ts || "").localeCompare(String(b.ts || "")));
      _updateLastTs();
    }
    return added;
  }

  function messageById(id) {
    if (id === undefined || id === null || id === "") return null;
    return state.messages.find((message) => String(message.id) === String(id)) || null;
  }

  function visibleMessages() {
    const person = state.personFilter;
    if (!person) return state.messages;
    return state.messages.filter((message) => {
      if (message.from === person) return true;
      return String(message.text || "").includes(`@${person}`);
    });
  }

  function expansionKey(message) {
    return `${state.selectedType}:${state.selectedName}:${messageKey(message)}`;
  }

  function isExpanded(key) {
    return state.expanded.has(key);
  }

  function toggleExpanded(key) {
    if (state.expanded.has(key)) {
      state.expanded.delete(key);
      return false;
    }
    state.expanded.add(key);
    return true;
  }

  function setPersonFilter(name) {
    state.personFilter = name || "";
    emit("filter");
  }

  // ── 5-1: search / kind filters ─────────────
  function isFiltered() {
    return state.kindFilter !== "all" || Boolean(state.query) || Boolean(state.personFilter);
  }

  function _channelQuery({ limit = 50, offset = 0, since = "" } = {}) {
    const params = new URLSearchParams();
    params.set("limit", String(limit));
    params.set("offset", String(offset));
    if (since) params.set("since", since);
    if (state.query) params.set("q", state.query);
    if (state.kindFilter !== "all") params.set("kind", state.kindFilter);
    return params.toString();
  }

  function _matchesActiveFilters(row) {
    if (!row) return false;
    if (state.kindFilter !== "all" && kindGroup(row.kind) !== state.kindFilter) return false;
    const needle = state.query.trim().toLowerCase();
    if (!needle) return true;
    return String(row.text || "").toLowerCase().includes(needle)
      || String(row.from || "").toLowerCase().includes(needle);
  }

  function _resetMessages() {
    state.messages = [];
    state.messageKeys = new Set();
    state.total = 0;
    state.filterTotal = 0;
    state.hasMore = false;
    state.loading = false;
    state.loadingOlder = false;
    state.error = "";
    state.lastTs = "";
  }

  function reloadFiltered() {
    if (state.selectedType !== "channel" || !state.selectedName) return;
    state.loadToken += 1;
    state.messages = [];
    state.messageKeys = new Set();
    state.total = 0;
    state.filterTotal = 0;
    state.hasMore = false;
    state.loading = true;
    state.loadingOlder = false;
    state.error = "";
    state.lastTs = "";
    emit("filters");
    void loadMessages();
    void loadFilterCounts();
  }

  function setKindFilter(kind) {
    const next = kind === "all" || KIND_FILTERS.includes(kind) ? kind : "all";
    if (next === state.kindFilter) return;
    state.kindFilter = next;
    reloadFiltered();
  }

  function setQuery(value) {
    state.query = String(value || "");
    if (state.searchTimer) clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(() => {
      state.searchTimer = null;
      reloadFiltered();
    }, SEARCH_DEBOUNCE_MS);
  }

  async function loadFilterCounts() {
    if (state.selectedType !== "channel" || !state.selectedName) return;
    const token = state.loadToken;
    const name = state.selectedName;
    const q = state.query;
    const counts = await Promise.all(
      KIND_FILTERS.map(async (kind) => {
        const params = new URLSearchParams({ limit: "1", offset: "0", kind });
        if (q) params.set("q", q);
        try {
          const data = await api(`/api/channels/${encodeURIComponent(name)}?${params.toString()}`);
          return Number(data?.total) || 0;
        } catch {
          return 0;
        }
      }),
    );
    if (token !== state.loadToken || name !== state.selectedName) return;
    const next = {
      anima: counts[0],
      human: counts[1],
      external: counts[2],
      operator: counts[3],
    };
    next.all = next.anima + next.human + next.external + next.operator;
    state.filterCounts = next;
    emit("filters");
  }

  function setDmFilter(name) {
    state.dmFilter = name || "";
    emit("dm-filter");
  }

  function setArchiveOpen(open) {
    state.archiveOpen = Boolean(open);
  }

  function setMode(mode) {
    state.mode = mode === "dm" ? "dm" : "board";
    emit("mode");
  }

  function setView(view) {
    if (view !== "home" && view !== "flows") return;
    state.loadToken += 1;
    if (state.searchTimer) clearTimeout(state.searchTimer);
    state.searchTimer = null;
    state.view = view;
    state.selectedType = null;
    state.selectedName = null;
    state.personFilter = "";
    state.kindFilter = "all";
    state.query = "";
    state.filterTotal = 0;
    state.boardTotal = 0;
    state.filterCounts = { all: 0, anima: 0, human: 0, external: 0, operator: 0 };
    emit("view");
  }

  // ── Selection / loading ────────────────────
  function select(type, name) {
    if (type !== "channel" && type !== "dm") return;
    state.loadToken += 1;
    if (state.searchTimer) clearTimeout(state.searchTimer);
    state.searchTimer = null;
    state.view = null;
    state.selectedType = type;
    state.selectedName = name || null;
    state.mode = type === "dm" ? "dm" : "board";
    _resetMessages();
    // 5-1: switching boards always drops the previous search / kind filter.
    state.personFilter = "";
    state.focusId = "";
    state.liveKeys = new Set();
    _resetReads();
    state.kindFilter = "all";
    state.query = "";
    state.boardTotal = Number(channel(name)?.message_count) || 0;
    state.filterCounts = { all: 0, anima: 0, human: 0, external: 0, operator: 0 };
    emit("selection");
    emit("reads");
    if (name) {
      void loadMessages();
      if (type === "channel") void loadFilterCounts();
    }
  }

  async function loadMessages() {
    const token = ++state.loadToken;
    const type = state.selectedType;
    const name = state.selectedName;
    if (!type || !name) return;

    state.loading = true;
    state.error = "";
    emit("loading");

    try {
      const data = type === "channel"
        ? await api(`/api/channels/${encodeURIComponent(name)}?${_channelQuery({ limit: 50, offset: 0 })}`)
        : await api(`/api/dm/${encodeURIComponent(name)}?limit=50`);
      if (token !== state.loadToken) return;

      // Merge rather than replace: a WebSocket post may have arrived while
      // this fetch was in flight, and mergeMessages() keys it exactly once.
      mergeMessages(data.messages || [], { type });
      state.total = Number(data.total ?? state.messages.length) || 0;
      state.hasMore = Boolean(data.has_more);
      if (type === "channel") {
        state.filterTotal = state.total;
        if (state.kindFilter === "all" && !state.query) {
          state.boardTotal = state.total;
        }
      }

      if (type === "channel") {
        await _loadReads(name, token);
        // The previous read position may be older than the initial window.
        // Pull it (and its neighbours) so the "new from here" line can be
        // drawn, then describe the line from the fetched row.
        const line = state.readLine;
        // While a search or kind filter is active the list only holds matching
        // posts, so do not pull unfiltered neighbours in for the read line.
        const filtering = state.kindFilter !== "all" || Boolean(state.query);
        if (line && !filtering && !messageById(line.afterId)) {
          await _loadAround(line.afterId);
          if (token !== state.loadToken) return;
          state.readLine = _computeReadLine();
        }
      }
      await ensureAvatars(state.messages.map((message) => message.from));
      if (token !== state.loadToken) return;

      state.loading = false;
      emit("reads");
      emit("messages");
    } catch (error) {
      if (token !== state.loadToken) return;
      state.loading = false;
      state.error = error?.message || String(error);
      emit("error");
    }
  }

  async function loadOlder() {
    if (state.loadingOlder || state.selectedType !== "channel" || !state.hasMore) return;
    const token = state.loadToken;
    const name = state.selectedName;
    state.loadingOlder = true;
    try {
      const data = await api(
        `/api/channels/${encodeURIComponent(name)}?${_channelQuery({ limit: 50, offset: state.messages.length })}`,
      );
      if (token !== state.loadToken) return;
      const rows = data.messages || [];
      const added = mergeMessages(rows, { type: "channel" });
      state.hasMore = Boolean(data.has_more);
      if (added > 0) {
        await ensureAvatars(rows.map((message) => message.from));
        if (token !== state.loadToken) return;
      }
      state.loadingOlder = false;
      emit("older");
    } catch {
      if (token === state.loadToken) {
        state.loadingOlder = false;
        emit("older");
      }
    }
  }

  /** Fetch only posts newer than the last rendered one (dedup-safe). */
  async function refresh() {
    if (!state.selectedType || !state.selectedName) return;
    // While a server-side filter is active, keep the filtered page intact;
    // board.post events still merge matching rows through handleBoardPost().
    if (state.selectedType === "channel" && (state.kindFilter !== "all" || state.query)) return;
    if (state.selectedType === "dm" || !state.lastTs) {
      await loadMessages();
      return;
    }
    const token = state.loadToken;
    const name = state.selectedName;
    try {
      const data = await api(
        `/api/channels/${encodeURIComponent(name)}?since=${encodeURIComponent(state.lastTs)}`,
      );
      if (token !== state.loadToken) return;
      const rows = data.messages || [];
      if (data.total !== undefined) {
        state.total = Number(data.total) || state.total;
        state.boardTotal = state.total;
      }
      const added = mergeMessages(rows, { type: "channel", live: true });
      if (added === 0) return;
      await ensureAvatars(rows.map((message) => message.from));
      if (token !== state.loadToken) return;
      emit("messages");
    } catch {
      // Polling failures are transient; the next tick retries.
    }
  }

  // ── WebSocket / submit ─────────────────────
  async function handleBoardPost(data) {
    if (!data) return;
    if (state.selectedType !== "channel" || state.selectedName !== data.channel) return;
    const row = {
      id: data.id,
      ts: data.ts,
      from: data.from,
      text: data.text,
      source: data.source,
      kind: data.kind,
      reply_to: data.reply_to,
    };
    if (data.delivery && typeof data.delivery === "object") {
      row.delivery = data.delivery;
    } else if (data.notified || data.not_notified) {
      row.delivery = {
        notified: Array.isArray(data.notified) ? data.notified : [],
        not_notified: data.not_notified && typeof data.not_notified === "object" ? data.not_notified : {},
      };
    }
    if (data.hooks && typeof data.hooks === "object") row.hooks = data.hooks;

    const channelInfo = channel(data.channel);
    const filtered = state.kindFilter !== "all" || Boolean(state.query);
    const matches = !filtered || _matchesActiveFilters(row);
    const added = matches ? mergeMessages([row], { type: "channel", live: true }) : 0;

    if (channelInfo) {
      channelInfo.message_count = (channelInfo.message_count || 0) + 1;
      if (!channelInfo.last_post_ts || String(data.ts || "") >= channelInfo.last_post_ts) {
        channelInfo.last_post_ts = data.ts;
        channelInfo.last_from = data.from;
      }
      state.boardTotal = Number(channelInfo.message_count) || state.boardTotal + 1;
    } else {
      state.boardTotal += 1;
    }
    state.detailRevision += 1;

    if (added === 0) {
      emit("channels");
      return;
    }
    state.total += added;
    state.filterTotal += added;
    await ensureAvatars([data.from]);
    emit("messages");
    emit("channels");
  }

  async function submit(text, replyTo) {
    if (state.selectedType !== "channel" || !state.selectedName) {
      return { ok: false, error: "no channel" };
    }
    const name = state.selectedName;
    const token = state.loadToken;
    const user = currentUser();
    const body = { text, from_name: user };
    if (replyTo !== undefined && replyTo !== null && replyTo !== "") body.reply_to = replyTo;
    try {
      const result = await api(`/api/channels/${encodeURIComponent(name)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const post = result && result.post ? result.post : {};
      if (post.error) return { ok: false, error: String(post.error) };
      const hooks = post.hooks && typeof post.hooks === "object" ? post.hooks : {};
      const delivery = {
        notified: Array.isArray(post.notified) ? post.notified : [],
        not_notified: post.not_notified && typeof post.not_notified === "object" ? post.not_notified : {},
      };
      if (token !== state.loadToken) return { ok: true, delivery, hooks };

      if (post.id !== undefined && post.id !== null && post.id !== "") {
        const row = {
          id: post.id,
          ts: new Date().toISOString(),
          from: user,
          text,
          source: "human",
          kind: "human",
          reply_to: replyTo || undefined,
          delivery,
          hooks,
        };
        const filtered = state.kindFilter !== "all" || Boolean(state.query);
        const added = !filtered || _matchesActiveFilters(row)
          ? mergeMessages([row], { type: "channel" })
          : 0;
        const stored = messageById(post.id);
        if (stored) {
          stored.delivery = delivery;
          stored.hooks = hooks;
        }
        const channelInfo = channel(name);
        if (channelInfo) {
          channelInfo.message_count = (channelInfo.message_count || 0) + 1;
          state.boardTotal = channelInfo.message_count;
        } else {
          state.boardTotal += 1;
        }
        state.detailRevision += 1;
        if (added) {
          state.total += added;
          state.filterTotal += added;
          await ensureAvatars([user]);
          emit("messages");
          emit("channels");
        } else if (stored) {
          emit("messages");
        } else {
          emit("channels");
        }
      }
      return { ok: true, delivery, hooks };
    } catch (error) {
      return { ok: false, error: error?.message || String(error) };
    }
  }

  async function updateSettings(name, payload) {
    const updated = await api(`/api/channels/${encodeURIComponent(name)}/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (updated && typeof updated === "object") {
      const index = state.channels.findIndex((item) => item.name === name);
      if (index >= 0) state.channels[index] = updated;
      else state.channels.push(updated);
      emit("channels");
    }
    return updated;
  }

  // ── Lifecycle ──────────────────────────────
  function start() {
    void loadChannels();
    void loadDmPairs();
    void loadAnimas();
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, REFRESH_MS);
  }

  function stop() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    if (state.markTimer) {
      clearTimeout(state.markTimer);
      state.markTimer = null;
    }
    state.loadToken += 1;
  }

  return {
    state,
    subscribe,
    start,
    stop,
    emit,
    // avatars
    ensureAvatars,
    avatarUrl,
    // directory
    anima,
    hasAnima,
    statusOf,
    handleAnimaStatus,
    // rail
    channel,
    hasChannel,
    pair,
    loadChannels,
    loadDmPairs,
    setDmFilter,
    setArchiveOpen,
    setMode,
    setView,
    // messages
    mergeMessages,
    messageKey,
    messageById,
    visibleMessages,
    expansionKey,
    isExpanded,
    toggleExpanded,
    setPersonFilter,
    // filters
    isFiltered,
    setKindFilter,
    setQuery,
    loadFilterCounts,
    // selection / reading
    select,
    loadMessages,
    loadOlder,
    refresh,
    handleBoardPost,
    handleBoardRead,
    submit,
    updateSettings,
    // reads / replies / jump
    humanReader,
    readerName,
    hasReadByName,
    readersForMessage,
    readerPoolForMessage,
    replyFor,
    replyTarget,
    stateSignature,
    markRead,
    scheduleMarkRead,
    jumpTo,
    clearFocus,
  };
}

export function isCompactWithPrevious(previous, message, dayKeyChanged) {
  if (!previous || !message || dayKeyChanged) return false;
  if (previous.from !== message.from) return false;
  if ((previous.kind || "anima") !== (message.kind || "anima")) return false;
  const prevTs = new Date(previous.ts).getTime();
  const currentTs = new Date(message.ts).getTime();
  if (Number.isNaN(prevTs) || Number.isNaN(currentTs)) return false;
  return Math.abs(currentTs - prevTs) <= COMPACT_WINDOW_MS;
}
