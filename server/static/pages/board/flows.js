// ── Call-flow view ──────────────────────────
// One API result drives every figure: the summary tiles, the stage ribbon,
// the map, the day lanes, the list and the trigger chain.
import { api } from "../../modules/api.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { t } from "/shared/i18n.js";
import { clockTime, dayLabel, localDayKey, numberText } from "./time.js";
import { lanesMarkup, mapMarkup, sankeyMarkup } from "./flows-charts.js";

const PAGE_SIZE = 25;
const WAITING_MS = 2 * 24 * 60 * 60 * 1000;

function _waiting(call) {
  if (!call.delivered || call.reply) return false;
  const start = new Date(call.t).getTime();
  const age = Date.now() - start;
  return !Number.isNaN(start) && age >= 0 && age <= WAITING_MS;
}

function _preview(text, maxLines = 3) {
  return String(text || "")
    .replace(/\r/g, "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, maxLines)
    .join("\n");
}

function _safeCalls(data) {
  return Array.isArray(data?.calls) ? data.calls.filter((call) => call && call.id) : [];
}

function _fallbackSummary(calls) {
  const delivered = calls.filter((call) => call.delivered).length;
  const read = calls.filter((call) => call.read).length;
  const replied = calls.filter((call) => call.reply).length;
  const replySeconds = [];
  let waiting = 0;
  for (const call of calls) {
    const start = new Date(call.t).getTime();
    const end = call.reply ? new Date(call.reply).getTime() : 0;
    if (call.reply && !Number.isNaN(start) && !Number.isNaN(end)) {
      replySeconds.push(Math.max(0, Math.round((end - start) / 1000)));
    }
    if (_waiting(call)) waiting += 1;
  }
  replySeconds.sort((a, b) => a - b);
  const median = replySeconds.length
    ? replySeconds[Math.floor(replySeconds.length / 2)]
    : null;
  return {
    calls: calls.length,
    delivered,
    read,
    replied,
    median_reply_seconds: median,
    waiting,
  };
}

function _summary(data, calls) {
  const fallback = _fallbackSummary(calls);
  const raw = data?.summary;
  if (!raw || typeof raw !== "object") return fallback;
  return {
    calls: Number(raw.calls ?? fallback.calls) || 0,
    delivered: Number(raw.delivered ?? fallback.delivered) || 0,
    read: Number(raw.read ?? fallback.read) || 0,
    replied: Number(raw.replied ?? fallback.replied) || 0,
    median_reply_seconds: raw.median_reply_seconds === null || raw.median_reply_seconds === undefined
      ? null
      : Number(raw.median_reply_seconds),
    waiting: Number(raw.waiting ?? fallback.waiting) || 0,
  };
}

function _percent(part, whole) {
  if (!whole) return "—";
  return `${Math.round((part / whole) * 100)}%`;
}

function _duration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
    return t("board.flows_duration_unknown");
  }
  const value = Math.max(0, Math.round(Number(seconds)));
  if (value < 60) return t("board.flows_duration_seconds", { count: value });
  if (value < 3_600) return t("board.flows_duration_minutes", { count: Math.round(value / 60) });
  if (value < 86_400) return t("board.flows_duration_hours", { count: Math.round(value / 3_600) });
  return t("board.flows_duration_days", { count: Math.round(value / 86_400) });
}

function _medianReply(calls) {
  const values = calls
    .filter((call) => call.reply)
    .map((call) => {
      const start = new Date(call.t).getTime();
      const end = new Date(call.reply).getTime();
      if (Number.isNaN(start) || Number.isNaN(end)) return null;
      return Math.max(0, Math.round((end - start) / 1000));
    })
    .filter((value) => value !== null)
    .sort((a, b) => a - b);
  return values.length ? values[Math.floor(values.length / 2)] : null;
}

function _stageMatch(call, stage) {
  if (!stage) return true;
  if (stage === "delivered") return Boolean(call.delivered);
  if (stage === "undelivered") return !call.delivered;
  if (stage === "read") return Boolean(call.read);
  if (stage === "unread") return Boolean(call.delivered) && !call.read;
  if (stage === "replied") return Boolean(call.reply);
  if (stage === "noreply") return Boolean(call.read) && !call.reply;
  return true;
}

function _involves(call, person) {
  return !person || call.from === person || call.to === person;
}

function _badge(kind, label) {
  return `<span class="board-flow-badge board-flow-badge-${kind}">${escapeHtml(label)}</span>`;
}

function _stageBadges(call) {
  const badges = [];
  if (!call.delivered) {
    const label = t("board.flows_stage_undelivered");
    const reason = call.reason ? ` · ${call.reason}` : "";
    badges.push(`<span class="board-flow-badge board-flow-badge-undelivered" title="${escapeAttr(reason.replace(/^ · /, ""))}">${escapeHtml(label)}</span>`);
  } else {
    badges.push(_badge("delivered", t("board.flows_stage_delivered")));
    badges.push(call.read
      ? _badge("read", t("board.flows_stage_read"))
      : _badge("unread", t("board.flows_stage_unread")));
    if (call.reply) badges.push(_badge("replied", t("board.flows_stage_replied")));
    else if (_waiting(call)) badges.push(_badge("waiting", t("board.flows_stage_waiting")));
    else if (call.read) badges.push(_badge("waiting", t("board.flows_stage_noreply")));
  }
  if (call.inferred) badges.push(_badge("inferred", t("board.flows_inferred")));
  return badges.join("");
}

function _viaMarkup(call) {
  if (call.via === "dm") {
    return `<span class="board-flow-via board-flow-via-dm">${escapeHtml(t("board.mode_dm"))}</span>`;
  }
  return `<span class="board-flow-via">#${escapeHtml(call.place || t("board.flows_legend_board"))}</span>`;
}

function _stageLabel(stage) {
  if (!stage) return "";
  const key = {
    delivered: "board.flows_stage_delivered",
    undelivered: "board.flows_stage_undelivered",
    read: "board.flows_stage_read",
    unread: "board.flows_stage_unread",
    replied: "board.flows_stage_replied",
    noreply: "board.flows_stage_noreply",
  }[stage];
  return key ? t(key) : "";
}

function _lastDays(count) {
  const today = new Date();
  const days = [];
  for (let offset = count - 1; offset >= 0; offset -= 1) {
    const date = new Date(today.getFullYear(), today.getMonth(), today.getDate() - offset);
    days.push(localDayKey(date));
  }
  return days;
}

function _tiles(summary, calls) {
  const boardCount = calls.filter((call) => call.via === "board").length;
  const dmCount = calls.filter((call) => call.via === "dm").length;
  return `<div class="board-flow-tiles">
    <div class="board-flow-tile" data-flow-tile="calls">
      <span class="board-flow-tile-label">${escapeHtml(t("board.flows_tile_calls"))}</span>
      <b class="board-flow-tile-value">${numberText(summary.calls)}</b>
      <small>${escapeHtml(t("board.flows_tile_calls_breakdown", { board: numberText(boardCount), dm: numberText(dmCount) }))}</small>
    </div>
    <div class="board-flow-tile" data-flow-tile="delivered">
      <span class="board-flow-tile-label">${escapeHtml(t("board.flows_tile_delivered"))}</span>
      <b class="board-flow-tile-value">${_percent(summary.delivered, summary.calls)}</b>
      <small>${escapeHtml(t("board.flows_tile_count", { count: numberText(summary.delivered) }))}</small>
    </div>
    <div class="board-flow-tile" data-flow-tile="read">
      <span class="board-flow-tile-label">${escapeHtml(t("board.flows_tile_read"))}</span>
      <b class="board-flow-tile-value">${_percent(summary.read, summary.delivered)}</b>
      <small>${escapeHtml(t("board.flows_tile_read_note", { count: numberText(summary.read) }))}</small>
    </div>
    <div class="board-flow-tile" data-flow-tile="replied">
      <span class="board-flow-tile-label">${escapeHtml(t("board.flows_tile_replied"))}</span>
      <b class="board-flow-tile-value">${_percent(summary.replied, summary.read)}</b>
      <small>${escapeHtml(t("board.flows_tile_replied_note", { count: numberText(summary.replied) }))}</small>
    </div>
    <div class="board-flow-tile" data-flow-tile="median">
      <span class="board-flow-tile-label">${escapeHtml(t("board.flows_tile_median"))}</span>
      <b class="board-flow-tile-value">${escapeHtml(_duration(summary.median_reply_seconds))}</b>
      <small>${escapeHtml(t("board.flows_tile_median_note"))}</small>
    </div>
    <div class="board-flow-tile is-warn" data-flow-tile="waiting">
      <span class="board-flow-tile-label">${escapeHtml(t("board.flows_tile_waiting"))}</span>
      <b class="board-flow-tile-value">${numberText(summary.waiting)}</b>
      <small>${escapeHtml(t("board.flows_tile_waiting_note"))}</small>
    </div>
  </div>`;
}

function _pairsMarkup(calls) {
  const pairs = new Map();
  for (const call of calls) {
    const key = `${call.from} → ${call.to}`;
    if (!pairs.has(key)) pairs.set(key, { n: 0, replied: 0 });
    const row = pairs.get(key);
    row.n += 1;
    if (call.reply) row.replied += 1;
  }
  const rows = [...pairs.entries()]
    .sort((a, b) => b[1].n - a[1].n)
    .slice(0, 8);
  if (rows.length === 0) {
    return `<div class="board-flow-pairs-empty">${escapeHtml(t("board.flows_empty"))}</div>`;
  }
  return `<div class="board-flow-pairs">${rows.map(([key, row]) => `<div class="board-flow-pair">
    <span>${escapeHtml(key)}</span>
    <b>${numberText(row.n)}</b>
    <small>${escapeHtml(_percent(row.replied, row.n))}</small>
  </div>`).join("")}</div>`;
}

function _personPanel(calls, person) {
  if (!person) {
    return `<div class="board-flow-panel">
      <h3>${escapeHtml(t("board.flows_pairs_title"))}</h3>
      ${_pairsMarkup(calls)}
      <span class="board-flow-panel-note">${escapeHtml(t("board.flows_pairs_note"))}</span>
    </div>`;
  }
  const out = calls.filter((call) => call.from === person);
  const incoming = calls.filter((call) => call.to === person);
  const repliedOut = out.filter((call) => call.reply).length;
  const repliedIn = incoming.filter((call) => call.reply).length;
  const waiting = incoming.filter(_waiting).length;
  return `<div class="board-flow-panel is-person">
    <h3>${escapeHtml(person)}</h3>
    <div class="board-flow-panel-grid">
      <span>${escapeHtml(t("board.flows_person_called"))}</span><b>${numberText(out.length)}</b>
      <span>${escapeHtml(t("board.flows_person_called_replied"))}</span><b>${_percent(repliedOut, out.length)}</b>
      <span>${escapeHtml(t("board.flows_person_called_by"))}</span><b>${numberText(incoming.length)}</b>
      <span>${escapeHtml(t("board.flows_person_replied"))}</span><b>${_percent(repliedIn, incoming.length)}</b>
      <span>${escapeHtml(t("board.flows_person_median"))}</span><b>${escapeHtml(_duration(_medianReply(incoming)))}</b>
      <span>${escapeHtml(t("board.flows_tile_waiting"))}</span><b>${numberText(waiting)}</b>
    </div>
    <button type="button" class="board-flow-clear" data-flow-clear>${escapeHtml(t("board.flows_person_clear"))}</button>
  </div>`;
}

function _listRows(calls, state) {
  return calls
    .filter((call) => _stageMatch(call, state.stage) && _involves(call, state.person))
    .sort((a, b) => String(b.t || "").localeCompare(String(a.t || "")));
}

function _listMarkup(rows, state) {
  const shown = rows.slice(0, state.limit);
  const filterParts = [];
  const stageLabel = _stageLabel(state.stage);
  if (stageLabel) filterParts.push(stageLabel);
  if (state.person) filterParts.push(t("board.flows_person_filter", { name: state.person }));
  const filterLabel = filterParts.length
    ? `${filterParts.join(" · ")} · `
    : "";
  const clear = state.stage || state.person
    ? `<button type="button" class="board-flow-clear" data-flow-clear>${escapeHtml(t("board.flows_clear"))}</button>`
    : "";
  const body = shown.length === 0
    ? `<tr><td class="board-flow-table-empty" colspan="4">${escapeHtml(t("board.flows_list_empty"))}</td></tr>`
    : shown.map((call) => {
      const selected = call.id === state.selected;
      const parentMark = call.parent ? `<span class="board-flow-parent-mark" title="${escapeAttr(t("board.flows_triggered"))}">↳</span>` : "";
      return `<tr class="board-flow-row${selected ? " is-selected" : ""}" data-flow-call="${escapeAttr(call.id)}" tabindex="0">
        <td class="board-flow-time">${escapeHtml(clockTime(call.t))}</td>
        <td>${_viaMarkup(call)}</td>
        <td class="board-flow-who">${escapeHtml(call.from)} → ${escapeHtml(call.to)}${parentMark}</td>
        <td class="board-flow-status">${_stageBadges(call)}</td>
      </tr>`;
    }).join("");
  const more = rows.length > shown.length
    ? `<button type="button" class="board-flow-more" data-flow-more>${escapeHtml(t("board.flows_more", { count: Math.min(PAGE_SIZE, rows.length - shown.length) }))}</button>`
    : "";
  return `<div class="board-flow-section-head">
      <h2>${escapeHtml(t("board.flows_list_title"))}</h2>
      <span>${escapeHtml(`${filterLabel}${rows.length}`)} ${escapeHtml(t("board.flows_list_count"))}</span>
      ${clear}
    </div>
    <div class="board-flow-tablewrap"><table class="board-flow-table">
      <thead><tr>
        <th>${escapeHtml(t("board.flows_col_time"))}</th>
        <th>${escapeHtml(t("board.flows_col_via"))}</th>
        <th>${escapeHtml(t("board.flows_col_who"))}</th>
        <th>${escapeHtml(t("board.flows_col_status"))}</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table></div>${more}`;
}

function _traceMarkup(calls, selectedId) {
  if (!selectedId) {
    return `<div class="board-flow-section-head"><h2>${escapeHtml(t("board.flows_trace_title"))}</h2></div>
      <div class="board-flow-empty">${escapeHtml(t("board.flows_trace_empty"))}</div>`;
  }
  const byId = new Map(calls.map((call) => [call.id, call]));
  const selected = byId.get(selectedId);
  if (!selected) {
    return `<div class="board-flow-section-head"><h2>${escapeHtml(t("board.flows_trace_title"))}</h2></div>
      <div class="board-flow-empty">${escapeHtml(t("board.flows_trace_empty"))}</div>`;
  }
  const childrenOf = new Map();
  for (const call of calls) {
    if (!call.parent) continue;
    if (!childrenOf.has(call.parent)) childrenOf.set(call.parent, []);
    childrenOf.get(call.parent).push(call);
  }
  const seen = new Set();
  let root = selected;
  while (root.parent && byId.has(root.parent) && !seen.has(root.parent)) {
    seen.add(root.id);
    root = byId.get(root.parent);
  }
  seen.clear();

  function nodeMarkup(call) {
    if (!call || seen.has(call.id)) return "";
    seen.add(call.id);
    const kids = (childrenOf.get(call.id) || [])
      .sort((a, b) => String(a.t || "").localeCompare(String(b.t || "")));
    const reply = call.reply_text
      ? `<div class="board-flow-trace-reply">↩ ${escapeHtml(call.to)}: ${escapeHtml(_preview(call.reply_text, 2))}</div>`
      : "";
    const text = call.text ? escapeHtml(_preview(call.text)) : "";
    return `<li class="board-flow-trace-node">
      <div class="board-flow-trace-box${call.id === selectedId ? " is-selected" : ""}" data-flow-call="${escapeAttr(call.id)}" tabindex="0">
        <div class="board-flow-trace-head">
          <time>${escapeHtml(clockTime(call.t))}</time>
          <b>${escapeHtml(call.from)}</b><span>→</span><b>${escapeHtml(call.to)}</b>
          ${_viaMarkup(call)}
        </div>
        <p class="board-flow-trace-text">${text}</p>
        <div class="board-flow-trace-steps">${_stageBadges(call)}</div>
        ${reply}
      </div>
      ${kids.length ? `<ol class="board-flow-trace-children">${kids.map(nodeMarkup).join("")}</ol>` : ""}
    </li>`;
  }

  const total = (function count(node) {
    const kids = childrenOf.get(node.id) || [];
    return 1 + kids.reduce((sum, child) => sum + count(child), 0);
  })(root);

  return `<div class="board-flow-section-head">
      <h2>${escapeHtml(t("board.flows_trace_title"))}</h2>
      <span>${escapeHtml(t("board.flows_trace_count", { count: numberText(total) }))}</span>
    </div>
    <ol class="board-flow-trace">${nodeMarkup(root)}</ol>`;
}

function _legend() {
  return `<div class="board-flow-legend">
    <span><svg width="26" height="8" aria-hidden="true"><path d="M0 4h26" class="board-flow-legend-board"></path></svg>${escapeHtml(t("board.flows_legend_board"))}</span>
    <span><svg width="26" height="8" aria-hidden="true"><path d="M0 4h26" class="board-flow-legend-dm"></path></svg>${escapeHtml(t("board.flows_legend_dm"))}</span>
    <span><svg width="26" height="8" aria-hidden="true"><path d="M0 4h26" class="board-flow-legend-reply"></path></svg>${escapeHtml(t("board.flows_legend_reply"))}</span>
  </div>`;
}

export function createFlowsView({ element, store }) {
  const state = {
    days: 30,
    via: "all",
    loaded: false,
    loading: false,
    error: "",
    data: null,
    stage: "",
    person: "",
    selected: "",
    day: localDayKey(new Date()),
    limit: PAGE_SIZE,
    token: 0,
  };
  let destroyed = false;

  function isHuman(name) {
    return name === "human" || !store.hasAnima(name);
  }

  function calls() {
    return _safeCalls(state.data);
  }

  function _controls() {
    const dayButtons = [7, 30].map((days) => `<button type="button" class="board-flow-seg-btn${state.days === days ? " is-active" : ""}" data-flow-days="${days}" aria-pressed="${state.days === days}">${escapeHtml(t("board.flows_days", { count: days }))}</button>`).join("");
    const viaButtons = [
      ["all", "board.flows_via_all"],
      ["board", "board.flows_via_board"],
      ["dm", "board.flows_via_dm"],
    ].map(([via, key]) => `<button type="button" class="board-flow-seg-btn${state.via === via ? " is-active" : ""}" data-flow-via="${via}" aria-pressed="${state.via === via}">${escapeHtml(t(key))}</button>`).join("");
    return `<div class="board-flow-controls">
      <div class="board-flow-seg" role="group" aria-label="${escapeAttr(t("board.flows_period"))}">${dayButtons}</div>
      <div class="board-flow-seg" role="group" aria-label="${escapeAttr(t("board.flows_via"))}">${viaButtons}</div>
    </div>`;
  }

  function _header() {
    return `<div class="board-flow-head">
      <h1>${escapeHtml(t("board.flows_title"))}</h1>
      <p>${escapeHtml(t("board.flows_subtitle"))}</p>
      ${_controls()}
    </div>`;
  }

  function _body() {
    if (state.loading && !state.data) {
      return `<div class="board-flow-loading">${escapeHtml(t("board.loading"))}</div>`;
    }
    if (state.error && !state.data) {
      return `<div class="board-flow-error">${escapeHtml(t("board.load_failed"))}: ${escapeHtml(state.error)} <button type="button" class="board-flow-clear" data-flow-retry>${escapeHtml(t("board.flows_retry"))}</button></div>`;
    }
    const allCalls = calls();
    if (allCalls.length === 0) {
      return `<div class="board-flow-empty">${escapeHtml(t("board.flows_empty"))}</div>`;
    }
    const summary = _summary(state.data, allCalls);
    const rows = _listRows(allCalls, state);
    const dayChips = _lastDays(7).map((day) => `<button type="button" class="board-flow-chip${state.day === day ? " is-active" : ""}" data-flow-day="${day}" aria-pressed="${state.day === day}">${escapeHtml(dayLabel(day))}</button>`).join("");
    return `${_tiles(summary, allCalls)}
      <div class="board-flow-grid">
        <section class="board-flow-card">
          <div class="board-flow-section-head"><h2>${escapeHtml(t("board.flows_sankey_title"))}</h2><span>${escapeHtml(t("board.flows_sankey_note"))}</span></div>
          <div class="board-flow-svgwrap">${sankeyMarkup(allCalls, state.stage)}</div>
        </section>
        <section class="board-flow-card">
          <div class="board-flow-section-head"><h2>${escapeHtml(t("board.flows_map_title"))}</h2><span>${escapeHtml(t("board.flows_map_note"))}</span></div>
          <div class="board-flow-mapgrid">
            <div class="board-flow-svgwrap">${mapMarkup(allCalls, { person: state.person, isHuman })}</div>
            ${_personPanel(allCalls, state.person)}
          </div>
        </section>
      </div>
      <section class="board-flow-card">
        <div class="board-flow-section-head"><h2>${escapeHtml(t("board.flows_lanes_title"))}</h2><span>${escapeHtml(t("board.flows_lanes_note"))}</span></div>
        <div class="board-flow-days">${dayChips}</div>
        ${lanesMarkup(allCalls, { dayKey: state.day, selectedId: state.selected })}
        ${_legend()}
      </section>
      <div class="board-flow-grid is-list">
        <section class="board-flow-card">${_listMarkup(rows, state)}</section>
        <section class="board-flow-card">${_traceMarkup(allCalls, state.selected)}</section>
      </div>`;
  }

  function render() {
    if (destroyed) return;
    const top = element.scrollTop;
    element.innerHTML = `${_header()}<div class="board-flow-body">${_body()}</div>`;
    element.scrollTop = top;
  }

  async function load() {
    const token = ++state.token;
    state.loading = true;
    state.error = "";
    state.loaded = true;
    state.data = null;
    render();
    try {
      const data = await api(`/api/flows?days=${state.days}&via=${state.via}`);
      if (destroyed || token !== state.token) return;
      state.data = data && typeof data === "object" ? data : { calls: [], summary: {} };
      state.loading = false;
      render();
    } catch (error) {
      if (destroyed || token !== state.token) return;
      state.loading = false;
      state.error = error?.message || String(error);
      render();
    }
  }

  function show() {
    if (!state.loaded && !state.loading) {
      void load();
    } else {
      render();
    }
  }

  function onClick(event) {
    const retry = event.target.closest("[data-flow-retry]");
    if (retry) {
      void load();
      return;
    }
    const daysButton = event.target.closest("[data-flow-days]");
    if (daysButton) {
      const days = Number(daysButton.dataset.flowDays);
      if (days && days !== state.days) {
        state.days = days;
        state.limit = PAGE_SIZE;
        void load();
      }
      return;
    }
    const viaButton = event.target.closest("[data-flow-via]");
    if (viaButton) {
      const via = viaButton.dataset.flowVia;
      if (via && via !== state.via) {
        state.via = via;
        state.limit = PAGE_SIZE;
        void load();
      }
      return;
    }
    const stage = event.target.closest("[data-flow-stage]");
    if (stage) {
      const value = stage.dataset.flowStage || "";
      state.stage = state.stage === value ? "" : value;
      state.limit = PAGE_SIZE;
      render();
      return;
    }
    const person = event.target.closest("[data-flow-person]");
    if (person) {
      const value = person.dataset.flowPerson || "";
      state.person = state.person === value ? "" : value;
      state.limit = PAGE_SIZE;
      render();
      return;
    }
    const day = event.target.closest("[data-flow-day]");
    if (day) {
      state.day = day.dataset.flowDay || state.day;
      render();
      return;
    }
    const call = event.target.closest("[data-flow-call]");
    if (call) {
      state.selected = call.dataset.flowCall || "";
      render();
      return;
    }
    const more = event.target.closest("[data-flow-more]");
    if (more) {
      state.limit += PAGE_SIZE;
      render();
      return;
    }
    const clear = event.target.closest("[data-flow-clear]");
    if (clear) {
      state.stage = "";
      state.person = "";
      state.limit = PAGE_SIZE;
      render();
    }
  }

  element.addEventListener("click", onClick);

  function destroy() {
    destroyed = true;
    state.token += 1;
    element.removeEventListener("click", onClick);
    element.innerHTML = "";
  }

  return { show, render, destroy };
}
