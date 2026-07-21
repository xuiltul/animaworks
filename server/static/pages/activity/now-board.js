const THINKING_STALE_MS = 120_000;
const MAX_TICKER_ITEMS = 5;

const STATUS_META = {
  chat: { key: "activity.now_chat", emoji: "💬", lucide: "message-circle" },
  task: { key: "activity.now_task", emoji: "🔨", lucide: "hammer" },
  cron: { key: "activity.now_cron", emoji: "⏰", lucide: "clock" },
  error: { key: "activity.now_error", emoji: "⚠️", lucide: "triangle-alert" },
  idle: { key: "activity.now_idle", emoji: "💤", lucide: "circle-pause" },
};

function _timestamp(value) {
  const result = new Date(value || 0).getTime();
  return Number.isFinite(result) ? result : 0;
}

function _animaNames(group) {
  if (typeof group?.anima === "string") return [group.anima];
  if (Array.isArray(group?.animas)) return group.animas.filter((name) => typeof name === "string");
  return [];
}

function _activityStatus(ctx, fallback = "task") {
  if (typeof ctx !== "string") return fallback;
  if (ctx === "chat") return "chat";
  if (ctx === "heartbeat" || ctx.startsWith("cron:")) return "cron";
  if (ctx.startsWith("task:") || ctx.startsWith("inbox")) return "task";
  return fallback;
}

export function deriveNowStatus(card, now = Date.now()) {
  const runtimeStatus = String(card?.runtimeStatus || "").toLowerCase();
  if (runtimeStatus === "error") return "error";
  if ((runtimeStatus === "thinking" || runtimeStatus === "streaming" || runtimeStatus === "processing")
      && now - card.statusUpdatedAt <= THINKING_STALE_MS) {
    return "chat";
  }
  if (card.runningTasks?.length) return "task";
  if (card.transientStatus && now - card.lastActivityAt <= THINKING_STALE_MS) {
    return card.transientStatus;
  }
  return "idle";
}

function _tickerFromEvent(event, fallbackAnima = "") {
  const kind = event?.kind || event?.type || event?.event || "";
  if (kind !== "tool_use" && kind !== "tool_result") return null;
  const summary = String(event.summary || event.content || "").slice(0, 200);
  return {
    id: String(event.id || `${event.ts || Date.now()}-${kind}-${event.tool || event.tool_name || "tool"}`),
    anima: event.anima || event.name || fallbackAnima,
    kind,
    tool: event.tool || event.tool_name || "tool",
    summary,
    isError: Boolean(event.is_error || event.meta?.is_error || event.meta?.result_status === "fail"),
    dropped: Math.max(0, Number(event.dropped) || 0),
    ts: event.ts || new Date().toISOString(),
  };
}

function _createCard(name, anima = {}) {
  const lastActivityAt = _timestamp(anima.last_busy_since);
  return {
    name,
    runtimeStatus: anima.status || "idle",
    statusUpdatedAt: lastActivityAt || 0,
    transientStatus: "",
    lastActivityAt,
    lastActivityTs: anima.last_busy_since || "",
    error: "",
    runningTasks: [],
    ticker: [],
  };
}

function _icon(documentRef, meta) {
  const wrap = documentRef.createElement("span");
  wrap.className = "now-status-icon";
  wrap.setAttribute("aria-hidden", "true");
  const emoji = documentRef.createElement("span");
  emoji.className = "now-status-emoji";
  emoji.textContent = meta.emoji;
  const lucide = documentRef.createElement("i");
  lucide.dataset.lucide = meta.lucide;
  wrap.appendChild(emoji);
  wrap.appendChild(lucide);
  return wrap;
}

function _statusLabel(card, status, translate) {
  if (status === "task" && card.runningTasks.length) {
    const first = card.runningTasks[0];
    return `${translate(STATUS_META.task.key)} · ${first.title || first.task_id || ""}`;
  }
  return translate(STATUS_META[status].key);
}

function _renderTicker(documentRef, card, translate, newTickerId) {
  const ticker = documentRef.createElement("div");
  ticker.className = "now-ticker";
  ticker.setAttribute("aria-live", "polite");

  if (!card.ticker.length) {
    const empty = documentRef.createElement("span");
    empty.className = "now-ticker-empty";
    empty.textContent = translate("activity.now_no_tools");
    ticker.appendChild(empty);
    return ticker;
  }

  for (const item of card.ticker.slice(0, MAX_TICKER_ITEMS)) {
    const row = documentRef.createElement("div");
    row.className = `now-ticker-row now-ticker-row--${item.kind}`;
    if (item.id === newTickerId) row.classList.add("now-ticker-row--new");

    const result = documentRef.createElement("span");
    result.className = "now-ticker-result";
    result.textContent = item.kind === "dropped" ? "" : (item.kind === "tool_use" ? "…" : (item.isError ? "✗" : "✓"));
    if (item.isError) result.classList.add("is-error");

    const tool = documentRef.createElement("code");
    tool.className = "now-ticker-tool";
    tool.textContent = item.tool;

    const summary = documentRef.createElement("span");
    summary.className = "now-ticker-summary";
    summary.textContent = item.summary;
    summary.title = item.summary;

    row.appendChild(result);
    row.appendChild(tool);
    row.appendChild(summary);
    ticker.appendChild(row);

    if (item.dropped > 0) {
      const droppedRow = documentRef.createElement("div");
      droppedRow.className = "now-ticker-row now-ticker-row--dropped";
      if (item.id === newTickerId) droppedRow.classList.add("now-ticker-row--new");

      const droppedResult = documentRef.createElement("span");
      droppedResult.className = "now-ticker-result";
      const droppedTool = documentRef.createElement("code");
      droppedTool.className = "now-ticker-tool";
      droppedTool.textContent = "+";
      const droppedSummary = documentRef.createElement("span");
      droppedSummary.className = "now-ticker-summary";
      droppedSummary.textContent = translate("activity.now_dropped", { count: item.dropped });

      droppedRow.appendChild(droppedResult);
      droppedRow.appendChild(droppedTool);
      droppedRow.appendChild(droppedSummary);
      ticker.appendChild(droppedRow);
    }
  }
  return ticker;
}

/** Render and manage the activity page's independent live fleet overview. */
export function renderNowBoard(container, ctx = {}) {
  const documentRef = ctx.documentRef || document;
  const translate = ctx.t || ((key) => key);
  const api = ctx.api;
  const now = ctx.now || (() => Date.now());
  const cards = new Map();
  let destroyed = false;
  let newTickerId = "";

  container.classList.add("now-board");

  const ensureCard = (name, anima = {}) => {
    if (!name) return null;
    if (!cards.has(name)) cards.set(name, _createCard(name, anima));
    const card = cards.get(name);
    if (anima.status) card.runtimeStatus = anima.status;
    return card;
  };

  const render = () => {
    if (destroyed) return;
    const heading = documentRef.createElement("div");
    heading.className = "now-board-heading";
    const title = documentRef.createElement("h3");
    title.textContent = translate("activity.now_title");
    const live = documentRef.createElement("span");
    live.className = "now-board-live";
    live.textContent = translate("activity.now_live");
    heading.appendChild(title);
    heading.appendChild(live);

    const grid = documentRef.createElement("div");
    grid.className = "now-board-grid";
    const ordered = [...cards.values()].sort((left, right) => {
      const leftStatus = deriveNowStatus(left, now());
      const rightStatus = deriveNowStatus(right, now());
      if ((leftStatus === "idle") !== (rightStatus === "idle")) return leftStatus === "idle" ? 1 : -1;
      return right.lastActivityAt - left.lastActivityAt || left.name.localeCompare(right.name);
    });

    for (const card of ordered) {
      const status = deriveNowStatus(card, now());
      const element = documentRef.createElement("article");
      element.className = `now-card now-card--${status}`;
      element.dataset.anima = card.name;
      element.dataset.status = status;
      if (status === "idle") element.classList.add("now-card--compact");

      const header = documentRef.createElement("div");
      header.className = "now-card-header";
      header.appendChild(_icon(documentRef, STATUS_META[status]));
      const name = documentRef.createElement("strong");
      name.className = "now-card-name";
      name.textContent = card.name;
      const statusText = documentRef.createElement("span");
      statusText.className = "now-card-status";
      statusText.textContent = _statusLabel(card, status, translate);
      header.appendChild(name);
      header.appendChild(statusText);
      element.appendChild(header);

      const last = documentRef.createElement("div");
      last.className = "now-card-last";
      last.textContent = card.lastActivityTs
        ? translate("activity.now_last_activity", { time: ctx.smartTimestamp?.(card.lastActivityTs) || card.lastActivityTs })
        : translate("activity.now_no_activity");
      element.appendChild(last);

      if (status === "error" && card.error) {
        const error = documentRef.createElement("div");
        error.className = "now-card-error";
        error.textContent = card.error;
        element.appendChild(error);
      }
      if (status !== "idle") element.appendChild(_renderTicker(documentRef, card, translate, newTickerId));
      grid.appendChild(element);
    }

    container.replaceChildren(heading, grid);
    if (globalThis.window?.lucide) globalThis.window.lucide.createIcons({ nodes: [container] });
    newTickerId = "";
  };

  const addTicker = (event, fallbackAnima = "") => {
    const item = _tickerFromEvent(event, fallbackAnima);
    if (!item?.anima) return;
    const card = ensureCard(item.anima);
    card.lastActivityAt = _timestamp(item.ts) || now();
    card.lastActivityTs = item.ts;
    card.transientStatus = _activityStatus(event.ctx, "task");
    card.ticker = [item, ...card.ticker.filter((current) => current.id !== item.id)].slice(0, MAX_TICKER_ITEMS);
    newTickerId = item.id;
    render();
  };

  const applySnapshot = (animas, runningPayload, recentPayload) => {
    for (const anima of animas || []) ensureCard(anima.name, anima);
    for (const card of cards.values()) {
      card.runningTasks = [];
      card.transientStatus = "";
      if (["thinking", "streaming", "processing"].includes(String(card.runtimeStatus).toLowerCase())
          && !card.statusUpdatedAt) {
        card.statusUpdatedAt = now();
      }
    }
    for (const anima of runningPayload?.animas || []) {
      const card = ensureCard(anima.name);
      card.runningTasks = Array.isArray(anima.tasks) ? anima.tasks : [];
      for (const task of card.runningTasks) {
        const timestamp = _timestamp(task.started_at);
        if (timestamp > card.lastActivityAt) {
          card.lastActivityAt = timestamp;
          card.lastActivityTs = task.started_at;
        }
      }
    }

    const groups = Array.isArray(recentPayload?.groups) ? recentPayload.groups : [];
    for (const group of [...groups].reverse()) {
      for (const name of _animaNames(group)) {
        const card = ensureCard(name);
        const groupTime = _timestamp(group.end_ts || group.start_ts);
        if (groupTime >= card.lastActivityAt) {
          card.lastActivityAt = groupTime;
          card.lastActivityTs = group.end_ts || group.start_ts || card.lastActivityTs;
          if (group.is_open && group.type === "cron") card.transientStatus = "cron";
          if (group.is_open && ["task", "task_exec", "inbox"].includes(group.type)) card.transientStatus = "task";
          if (group.is_open && group.type === "chat") card.transientStatus = "chat";
        }
        for (const event of group.events || []) {
          const item = _tickerFromEvent(event, name);
          if (item && !card.ticker.some((current) => current.id === item.id)) card.ticker.push(item);
        }
        card.ticker.sort((left, right) => _timestamp(right.ts) - _timestamp(left.ts));
        card.ticker = card.ticker.slice(0, MAX_TICKER_ITEMS);
      }
    }
    render();
  };

  const refresh = async () => {
    if (!api || destroyed) return;
    try {
      const [animas, running, recent] = await Promise.all([
        api("/api/animas"),
        api("/api/activity/running-tasks"),
        api("/api/activity/recent?hours=48&grouped=true&group_limit=50"),
      ]);
      applySnapshot(animas, running, recent);
    } catch (error) {
      console.error("Failed to refresh Now board:", error);
    }
  };

  const statusHandler = (event) => {
    const name = event?.name || event?.anima;
    const card = ensureCard(name);
    if (!card) return;
    card.runtimeStatus = event.status || "idle";
    card.statusUpdatedAt = now();
    card.error = event.error || "";
    if (["thinking", "streaming", "processing"].includes(card.runtimeStatus)) {
      card.lastActivityAt = now();
      card.lastActivityTs = new Date(now()).toISOString();
    }
    if (card.runtimeStatus === "idle") card.transientStatus = "";
    render();
  };
  const toolHandler = (event) => addTicker(event.detail || event);

  const unsubscribeStatus = ctx.onEvent?.("anima.status", statusHandler) || (() => {});
  documentRef.addEventListener?.("anima-tool-activity", toolHandler);
  const staleInterval = setInterval(render, 30_000);
  render();
  refresh();

  return {
    refresh,
    applySnapshot,
    handleStatus: statusHandler,
    handleToolActivity: addTicker,
    destroy() {
      destroyed = true;
      clearInterval(staleInterval);
      unsubscribeStatus();
      documentRef.removeEventListener?.("anima-tool-activity", toolHandler);
      cards.clear();
    },
  };
}
