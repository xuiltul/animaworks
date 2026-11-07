// ── Board store ──────────────────────────────
// Single source of truth for the board page. All three message sources
// (initial fetch / incremental refetch / WebSocket / optimistic send) go
// through mergeMessages(), keyed by server id (falling back to ts+from), so
// the same post can never be rendered twice.
import { api } from "../../modules/api.js";
import { bustupCandidates, resolveCachedAvatar } from "../../modules/avatar-resolver.js";

const REFRESH_MS = 30_000;
const COMPACT_WINDOW_MS = 10 * 60 * 1000;

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
    if (type === "dm" && !message.kind) {
      message.kind = String(message.source || "").toLowerCase() === "human" ? "human" : "anima";
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
  function mergeMessages(rows, { type = state.selectedType } = {}) {
    let added = 0;
    for (const row of rows || []) {
      if (!row) continue;
      const key = messageKey(row);
      if (!key || state.messageKeys.has(key)) continue;
      state.messageKeys.add(key);
      state.messages.push(_normalize(row, type));
      added += 1;
    }
    if (added > 0) {
      state.messages.sort((a, b) => String(a.ts || "").localeCompare(String(b.ts || "")));
      _updateLastTs();
    }
    return added;
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
    state.view = view;
    state.selectedType = null;
    state.selectedName = null;
    state.personFilter = "";
    emit("view");
  }

  // ── Selection / loading ────────────────────
  function select(type, name) {
    if (type !== "channel" && type !== "dm") return;
    state.loadToken += 1;
    state.view = null;
    state.selectedType = type;
    state.selectedName = name || null;
    state.mode = type === "dm" ? "dm" : "board";
    state.messages = [];
    state.messageKeys = new Set();
    state.total = 0;
    state.hasMore = false;
    state.loading = false;
    state.loadingOlder = false;
    state.error = "";
    state.lastTs = "";
    state.personFilter = "";
    emit("selection");
    if (name) void loadMessages();
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
        ? await api(`/api/channels/${encodeURIComponent(name)}?limit=50&offset=0`)
        : await api(`/api/dm/${encodeURIComponent(name)}?limit=50`);
      if (token !== state.loadToken) return;

      // Merge rather than replace: a WebSocket post may have arrived while
      // this fetch was in flight, and mergeMessages() keys it exactly once.
      mergeMessages(data.messages || [], { type });
      state.total = Number(data.total ?? state.messages.length) || 0;
      state.hasMore = Boolean(data.has_more);

      await ensureAvatars(state.messages.map((message) => message.from));
      if (token !== state.loadToken) return;

      state.loading = false;
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
        `/api/channels/${encodeURIComponent(name)}?limit=50&offset=${state.messages.length}`,
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
      if (data.total !== undefined) state.total = Number(data.total) || state.total;
      const added = mergeMessages(rows, { type: "channel" });
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
    };
    const added = mergeMessages([row], { type: "channel" });
    if (added === 0) return;
    state.total += added;

    const channelInfo = channel(data.channel);
    if (channelInfo) {
      channelInfo.message_count = (channelInfo.message_count || 0) + added;
      if (!channelInfo.last_post_ts || String(data.ts || "") >= channelInfo.last_post_ts) {
        channelInfo.last_post_ts = data.ts;
        channelInfo.last_from = data.from;
      }
    }
    await ensureAvatars([data.from]);
    emit("messages");
    emit("channels");
  }

  async function submit(text) {
    if (state.selectedType !== "channel" || !state.selectedName) {
      return { ok: false, error: "no channel" };
    }
    const name = state.selectedName;
    const token = state.loadToken;
    const currentUser = localStorage.getItem("animaworks_user") || "human";
    try {
      const result = await api(`/api/channels/${encodeURIComponent(name)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, from_name: currentUser }),
      });
      const post = result && result.post ? result.post : {};
      if (post.error) return { ok: false, error: String(post.error) };
      if (token !== state.loadToken) return { ok: true };

      if (post.id !== undefined && post.id !== null && post.id !== "") {
        const added = mergeMessages([{
          id: post.id,
          ts: new Date().toISOString(),
          from: currentUser,
          text,
          source: "human",
          kind: "human",
        }], { type: "channel" });
        if (added) {
          state.total += added;
          await ensureAvatars([currentUser]);
          emit("messages");
          emit("channels");
        }
      }
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error?.message || String(error) };
    }
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
    visibleMessages,
    expansionKey,
    isExpanded,
    toggleExpanded,
    setPersonFilter,
    select,
    loadMessages,
    loadOlder,
    refresh,
    handleBoardPost,
    submit,
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
