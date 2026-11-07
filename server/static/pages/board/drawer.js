// ── Board info drawer ────────────────────────
// Right-side panel for "about this board", read positions, the anima-facing
// snippet and the post statistics, plus the settings editor. The panel is
// section based so the 5-3 blocks can be appended without rewriting it.
import { getLocale, t } from "/shared/i18n.js";
import { api } from "../../modules/api.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { avatarMarkup, statusLabel } from "./avatar.js";
import { clockTime, numberText, relativeTime } from "./time.js";
import { memberCandidates } from "./mentions.js";

const ACTIVITY_DAYS = 14;
const KIND_ORDER = ["anima", "human", "external", "operator"];

function _accessLabel(channel) {
  if (channel.access === "closed") return t("board.access_closed");
  if (channel.access === "members") {
    const members = Array.isArray(channel.members) ? channel.members : [];
    return t("board.access_members", { count: members.length });
  }
  return t("board.access_open");
}

function _slackValue(channel) {
  const slack = channel.slack || {};
  if (!slack.inbound && !slack.outbound) return t("board.drawer_slack_none");
  return t("board.drawer_slack_value", {
    inbound: slack.inbound ? t("board.yes") : t("board.no"),
    outbound: slack.outbound ? t("board.yes") : t("board.no"),
  });
}

function _monthDay(date) {
  return new Intl.DateTimeFormat(getLocale() || "ja", { month: "numeric", day: "numeric" }).format(date);
}

function _niceMax(value) {
  if (value <= 4) return 4;
  if (value <= 10) return 10;
  const step = 10 ** Math.floor(Math.log10(value));
  const normalized = value / step;
  const nice = normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return nice * step;
}

function _headline(text) {
  const line = String(text || "").split(/\r?\n/, 1)[0].trim();
  return line.length > 64 ? `${line.slice(0, 64)}…` : line;
}

function _readerName(reader) {
  const raw = String(reader || "");
  if (raw.startsWith("anima:")) return raw.slice(6);
  if (raw.startsWith("human:")) return "";
  return raw;
}

export function createDrawer({ store, onToggle, onSaved }) {
  const element = document.createElement("aside");
  element.className = "board-drawer";
  element.id = "boardDrawer";
  element.hidden = true;
  element.setAttribute("aria-label", t("board.drawer_title"));

  let open = false;
  let channelName = "";
  let current = null;
  let editing = false;
  let draft = null;
  let pendingClose = false;
  let error = "";
  let flash = "";
  let saving = false;
  // 5-3: details are fetched when the drawer opens and refreshed after
  // board.read / board.post bumps store.state.detailRevision.
  let info = null;
  let reads = null;
  let detailLoading = false;
  let detailError = "";
  let sample = null;
  let detailToken = 0;
  let detailRevision = -1;

  function setToggle(openState) {
    open = openState;
    if (typeof onToggle === "function") onToggle(open);
  }

  function _closeButton(attributes = "") {
    return `<button type="button" class="board-drawer-close" ${attributes} aria-label="${escapeHtml(t("board.info_close"))}">×</button>`;
  }

  function _section(title, body, extraClass = "") {
    return `<section class="board-drawer-section${extraClass ? ` ${extraClass}` : ""}">
      <div class="board-drawer-label">${escapeHtml(title)}</div>
      ${body}
    </section>`;
  }

  function _infoSection(channel) {
    const members = Array.isArray(channel.members) ? channel.members : [];
    const description = channel.description
      ? `<p class="board-drawer-description">${escapeHtml(channel.description)}</p>`
      : `<p class="board-drawer-description board-drawer-muted">—</p>`;
    const rows = `
      <dl class="board-drawer-kv">
        <dt>${escapeHtml(t("board.drawer_access"))}</dt><dd>${escapeHtml(_accessLabel(channel))}</dd>
        <dt>${escapeHtml(t("board.drawer_company"))}</dt><dd>${escapeHtml(channel.company || "—")}</dd>
        <dt>${escapeHtml(t("board.drawer_members"))}</dt><dd>${escapeHtml(String(members.length))}</dd>
        <dt>${escapeHtml(t("board.drawer_slack"))}</dt><dd>${escapeHtml(_slackValue(channel))}</dd>
      </dl>`;
    const flashMessage = flash
      ? `<div class="board-drawer-flash" role="status">${escapeHtml(flash)}</div>`
      : "";
    return _section(
      t("board.drawer_about"),
      `${description}${rows}${flashMessage}
       <button type="button" class="board-drawer-edit" data-board-drawer-edit>${escapeHtml(t("board.drawer_edit"))}</button>
       <span class="board-drawer-hint">${escapeHtml(t("board.drawer_edit_hint"))}</span>`,
      "board-drawer-about",
    );
  }

  // ── 5-3 detail sections ────────────────────
  function _readerRows() {
    const readers = Array.isArray(info?.readers) ? info.readers : [];
    return readers
      .map((entry) => {
        const name = _readerName(entry?.reader);
        if (!name) return null;
        const record = reads?.readers?.[entry.reader] || {};
        return {
          reader: String(entry.reader || name),
          name,
          unread: Number(entry.unread) || 0,
          updated_at: entry.updated_at || record.updated_at || "",
        };
      })
      .filter(Boolean);
  }

  function _readerSection() {
    if (detailLoading && !info) {
      return _section(t("board.drawer_reads"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_loading"))}</p>`);
    }
    if (detailError && !info) {
      return _section(t("board.drawer_reads"), `<p class="board-drawer-error" role="alert">${escapeHtml(t("board.drawer_load_failed"))}</p>`);
    }
    const rows = _readerRows();
    if (!rows.length) {
      return _section(t("board.drawer_reads"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_reads_empty"))}</p>`);
    }
    const body = rows.map((row) => {
      const status = store.statusOf(row.name);
      const updated = row.updated_at
        ? t("board.drawer_read_last", { time: relativeTime(row.updated_at) })
        : t("board.drawer_read_never");
      const readState = row.unread > 0
        ? `<span class="board-drawer-unread">${escapeHtml(t("board.drawer_read_unread", { count: numberText(row.unread) }))}</span>`
        : `<span class="board-drawer-caught">${escapeHtml(t("board.drawer_read_latest"))}</span>`;
      return `<div class="board-drawer-reader" data-board-reader="${escapeAttr(row.reader)}">
        ${avatarMarkup({ name: row.name, kind: "anima", url: store.avatarUrl(row.name), size: "xs" })}
        <div class="board-drawer-reader-main">
          <span class="board-drawer-reader-name">${escapeHtml(row.name)}</span>
          <span class="board-drawer-reader-meta">
            <span class="board-status-dot board-status-${escapeAttr(status)}"></span>
            ${escapeHtml(statusLabel(status))} · ${escapeHtml(updated)}
          </span>
        </div>
        ${readState}
      </div>`;
    }).join("");
    return _section(t("board.drawer_reads"), `<div class="board-drawer-readers">${body}</div>`);
  }

  function _buildSample() {
    const rows = _readerRows().filter((row) => row.unread > 0);
    if (!rows.length) return null;
    const pick = rows[0];
    const record = reads?.readers?.[pick.reader] || {};
    const upto = String(record.upto_ts || "");
    const loaded = store.state.messages
      .filter((message) => !upto || String(message.ts || "") > upto)
      .slice()
      .sort((a, b) => String(a.ts || "").localeCompare(String(b.ts || "")));
    return {
      reader: pick.reader,
      name: pick.name,
      unread: pick.unread,
      upto_ts: upto,
      messages: loaded.slice(0, 4),
      mentions: 0,
    };
  }

  function _countMentions(messages, name) {
    const needle = `@${name}`;
    return messages.filter((message) => String(message.text || "").includes(needle)).length;
  }

  async function loadSampleMessages(channel, token) {
    const target = sample;
    if (!target) return;
    let rows = [];
    try {
      const data = target.upto_ts
        ? await api(`/api/channels/${encodeURIComponent(channel.name)}?since=${encodeURIComponent(target.upto_ts)}`)
        : await api(`/api/channels/${encodeURIComponent(channel.name)}?limit=50&offset=0`);
      rows = Array.isArray(data?.messages) ? data.messages : [];
    } catch {
      return;
    }
    if (token !== detailToken || !open || sample !== target) return;
    const messages = rows
      .filter((message) => !target.upto_ts || String(message.ts || "") > target.upto_ts)
      .slice()
      .sort((a, b) => String(a.ts || "").localeCompare(String(b.ts || "")));
    target.messages = messages.slice(0, 4);
    target.mentions = _countMentions(messages, target.name);
    if (!editing) renderInfo(current || channel);
  }

  function _sampleSection(channel) {
    if (detailLoading && !info) {
      return _section(t("board.drawer_from_anima"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_loading"))}</p>`);
    }
    if (!sample) {
      return _section(t("board.drawer_from_anima"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_read_none"))}</p>`);
    }
    const count = numberText(sample.unread);
    const prompt = t("board.drawer_prompt_line", {
      name: channel.name,
      count,
      to: sample.name,
      mentions: numberText(sample.mentions || 0),
    });
    const intro = sample.unread > 0
      ? t("board.drawer_read_command", { count })
      : t("board.drawer_read_none");
    const lines = sample.messages.map((message) => (
      `${clockTime(message.ts)} ${message.from}: ${_headline(message.text)}`
    ));
    const snippet = [
      prompt,
      "",
      `$ animaworks-tool board read ${channel.name}`,
      intro,
      ...lines,
    ].join("\n");
    return _section(
      t("board.drawer_from_anima"),
      `<pre class="board-drawer-code" data-board-read-snippet>${escapeHtml(snippet)}</pre>`,
    );
  }

  function _activitySection(channel) {
    const raw = Array.isArray(channel.activity_14d) ? channel.activity_14d : [];
    const values = [
      ...Array(Math.max(0, ACTIVITY_DAYS - raw.length)).fill(0),
      ...raw.slice(-ACTIVITY_DAYS),
    ].map((value) => Number(value) || 0);
    const total = values.reduce((sum, value) => sum + value, 0);
    const max = _niceMax(Math.max(0, ...values));
    const days = [];
    const today = new Date();
    const midnight = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    for (let index = 0; index < ACTIVITY_DAYS; index += 1) {
      const date = new Date(midnight);
      date.setDate(midnight.getDate() - (ACTIVITY_DAYS - 1 - index));
      days.push(date);
    }

    const left = 34;
    const right = 272;
    const top = 10;
    const bottom = 90;
    const plotHeight = bottom - top;
    const slot = (right - left) / ACTIVITY_DAYS;
    const barWidth = Math.max(3, slot - 3);
    const axis = (value) => bottom - (value / max) * plotHeight;
    const mid = max / 2;

    const bars = values.map((value, index) => {
      const date = days[index];
      const weekend = date.getDay() === 0 || date.getDay() === 6;
      const height = value > 0 ? Math.max(1, (value / max) * plotHeight) : 0;
      const x = left + index * slot + (slot - barWidth) / 2;
      const y = bottom - height;
      const title = `${_monthDay(date)} · ${numberText(value)}`;
      return `<rect class="board-drawer-bar${weekend ? " board-drawer-bar-weekend" : ""}" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${height.toFixed(2)}" rx="1.5"><title>${escapeHtml(title)}</title></rect>`;
    }).join("");

    const grid = [0, mid, max].map((value) => {
      const y = axis(value);
      return `<line class="board-drawer-chart-grid" x1="${left}" y1="${y.toFixed(2)}" x2="${right}" y2="${y.toFixed(2)}"></line>
        <text class="board-drawer-chart-axis" x="${left - 5}" y="${(y + 3).toFixed(2)}" text-anchor="end">${escapeHtml(numberText(value))}</text>`;
    }).join("");

    const firstDate = _monthDay(days[0]);
    const lastDate = _monthDay(days[days.length - 1]);
    return _section(
      t("board.drawer_activity"),
      `<svg class="board-drawer-chart" viewBox="0 0 280 112" role="img" aria-label="${escapeAttr(t("board.drawer_activity_label"))}">
        ${grid}
        ${bars}
        <text class="board-drawer-chart-axis" x="${left}" y="108" text-anchor="start">${escapeHtml(firstDate)}</text>
        <text class="board-drawer-chart-axis" x="${right}" y="108" text-anchor="end">${escapeHtml(lastDate)}</text>
      </svg>
      <div class="board-drawer-chart-note">${escapeHtml(t("board.drawer_activity_note", { count: numberText(total) }))}</div>`,
    );
  }

  function _mentionEntries() {
    const list = Array.isArray(info?.mentions_30d) ? info.mentions_30d : [];
    return list
      .map((item) => {
        if (Array.isArray(item)) return { name: String(item[0] || ""), count: Number(item[1]) || 0 };
        return { name: String(item?.name || ""), count: Number(item?.count) || 0 };
      })
      .filter((item) => item.name && item.count > 0);
  }

  function _mentionsSection() {
    if (detailLoading && !info) {
      return _section(t("board.drawer_mentions"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_loading"))}</p>`);
    }
    const entries = _mentionEntries();
    if (!entries.length) {
      return _section(t("board.drawer_mentions"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_mentions_empty"))}</p>`);
    }
    const max = Math.max(1, ...entries.map((entry) => entry.count));
    const body = entries.map((entry) => {
      const percent = Math.max(4, Math.round((entry.count / max) * 100));
      return `<div class="board-drawer-hbar" role="listitem">
        <span class="board-drawer-hbar-label">@${escapeHtml(entry.name)}</span>
        <span class="board-drawer-hbar-track"><span class="board-drawer-hbar-fill" style="width:${percent}%"></span></span>
        <span class="board-drawer-hbar-value">${escapeHtml(numberText(entry.count))}</span>
      </div>`;
    }).join("");
    return _section(
      t("board.drawer_mentions"),
      `<div class="board-drawer-hbars" role="list" aria-label="${escapeAttr(t("board.drawer_mentions_label"))}">${body}</div>`,
    );
  }

  function _kindEntries() {
    const kinds = info?.kinds && typeof info.kinds === "object" ? info.kinds : {};
    return KIND_ORDER
      .map((kind) => ({ kind, count: Number(kinds[kind]) || 0 }))
      .filter((entry) => entry.count > 0);
  }

  function _kindsSection() {
    if (detailLoading && !info) {
      return _section(t("board.drawer_kinds"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_loading"))}</p>`);
    }
    const entries = _kindEntries();
    if (!entries.length) {
      return _section(t("board.drawer_kinds"), `<p class="board-drawer-muted">${escapeHtml(t("board.drawer_reads_empty"))}</p>`);
    }
    const total = entries.reduce((sum, entry) => sum + entry.count, 0);
    const segments = entries.map((entry) => {
      const percent = (entry.count / total) * 100;
      const label = t("board.drawer_kinds_legend", {
        name: t(`board.filter_kind_${entry.kind}`),
        count: numberText(entry.count),
      });
      return `<span class="board-drawer-kind-seg board-drawer-kind-${escapeAttr(entry.kind)}" style="width:${percent.toFixed(2)}%" title="${escapeAttr(label)}"></span>`;
    }).join("");
    const legend = entries.map((entry) => {
      const label = t("board.drawer_kinds_legend", {
        name: t(`board.filter_kind_${entry.kind}`),
        count: numberText(entry.count),
      });
      return `<span class="board-drawer-kind-legend">
        <span class="board-drawer-kind-swatch board-drawer-kind-${escapeAttr(entry.kind)}"></span>
        ${escapeHtml(label)}
      </span>`;
    }).join("");
    return _section(
      t("board.drawer_kinds"),
      `<div class="board-drawer-kinds" role="img" aria-label="${escapeAttr(t("board.drawer_kinds_label"))}">${segments}</div>
       <div class="board-drawer-kind-legend-list">${legend}</div>`,
    );
  }

  function _detailSections(channel) {
    return [
      _readerSection(),
      _sampleSection(channel),
      _activitySection(channel),
      _mentionsSection(),
      _kindsSection(),
    ].join("");
  }

  function _accessRadio(value, label, hint) {
    const checked = draft.access === value;
    return `<label class="board-drawer-radio">
      <input type="radio" name="boardSetAccess" value="${escapeAttr(value)}"${checked ? " checked" : ""}>
      <span><b>${escapeHtml(label)}</b><small>${escapeHtml(hint)}</small></span>
    </label>`;
  }

  function _membersField(channel) {
    if (draft.access !== "members") return "";
    const chips = draft.members.length
      ? draft.members.map((name) => `<span class="board-drawer-chip">
          ${avatarMarkup({ name, kind: "anima", url: store.avatarUrl(name), size: "xs" })}
          <span>${escapeHtml(name)}</span>
          <button type="button" class="board-drawer-chip-remove" data-board-member-remove="${escapeAttr(name)}" aria-label="${escapeHtml(t("board.settings_remove_member", { name }))}">×</button>
        </span>`).join("")
      : `<span class="board-drawer-muted">${escapeHtml(t("board.members_none"))}</span>`;
    const candidates = memberCandidates(store, { ...channel, members: draft.members });
    const options = candidates.map((anima) => `<option value="${escapeAttr(anima.name)}">${escapeHtml(anima.name)}${anima.role ? `（${escapeHtml(anima.role)}）` : ""}</option>`).join("");
    const select = candidates.length
      ? `<select class="board-drawer-add-member" id="boardSetAdd" aria-label="${escapeHtml(t("board.members_add_select"))}">
          <option value="">${escapeHtml(t("board.settings_add_member"))}</option>${options}
        </select>`
      : "";
    return `<div class="board-drawer-field">
      <span class="board-drawer-field-label">${escapeHtml(t("board.settings_members", { count: draft.members.length }))}</span>
      <div class="board-drawer-chips">${chips}</div>
      ${select}
    </div>`;
  }

  function _settingsSection(channel) {
    const companyHint = channel.company
      ? t("board.settings_access_open_hint", { company: channel.company })
      : t("board.settings_access_open_hint_all");
    const warning = pendingClose
      ? `<div class="board-drawer-warning" role="alert">${escapeHtml(t("board.settings_confirm_close"))}</div>`
      : "";
    const errorMessage = error
      ? `<div class="board-drawer-error" role="alert">${escapeHtml(error)}</div>`
      : "";
    const saveLabel = pendingClose ? t("board.settings_save_close") : t("board.settings_save");
    return _section(
      t("board.drawer_edit"),
      `<form class="board-drawer-form" id="boardSettingsForm">
        <label class="board-drawer-field" for="boardSetDescription">
          <span class="board-drawer-field-label">${escapeHtml(t("board.settings_description"))}</span>
          <textarea id="boardSetDescription" class="board-drawer-textarea" rows="3">${escapeHtml(draft.description)}</textarea>
        </label>
        <div class="board-drawer-field">
          <span class="board-drawer-field-label">${escapeHtml(t("board.settings_access"))}</span>
          <div class="board-drawer-radios" role="radiogroup" aria-label="${escapeHtml(t("board.settings_access"))}">
            ${_accessRadio("open", t("board.settings_access_open"), companyHint)}
            ${_accessRadio("members", t("board.settings_access_members"), t("board.settings_access_members_hint"))}
            ${_accessRadio("closed", t("board.settings_access_closed"), t("board.settings_access_closed_hint"))}
          </div>
        </div>
        ${_membersField(channel)}
        ${warning}${errorMessage}
        <div class="board-drawer-actions">
          <button type="button" class="board-drawer-cancel" data-board-drawer-cancel>${escapeHtml(t("board.settings_cancel"))}</button>
          <button type="submit" class="board-drawer-save${pendingClose ? " board-drawer-danger" : ""}"${saving ? " disabled" : ""}>${escapeHtml(saveLabel)}</button>
        </div>
      </form>`,
      "board-drawer-settings",
    );
  }

  function renderInfo(channel) {
    element.innerHTML = `
      <div class="board-drawer-head">
        <h2 class="board-drawer-title">${escapeHtml(t("board.drawer_title"))}</h2>
        ${_closeButton("data-board-drawer-close")}
      </div>
      ${_infoSection(channel)}
      ${_detailSections(channel)}`;
  }

  function renderSettings(channel) {
    element.innerHTML = `
      <div class="board-drawer-head">
        <h2 class="board-drawer-title">${escapeHtml(t("board.settings_title", { name: channel.name }))}</h2>
        ${_closeButton("data-board-drawer-cancel")}
      </div>
      ${_settingsSection(channel)}`;
  }

  async function loadDetail(channel) {
    const token = ++detailToken;
    detailRevision = Number(store.state.detailRevision) || 0;
    detailLoading = true;
    detailError = "";
    info = null;
    reads = null;
    sample = null;
    if (!editing) renderInfo(channel);

    const [infoResult, readsResult] = await Promise.allSettled([
      api(`/api/channels/${encodeURIComponent(channel.name)}/info`),
      api(`/api/channels/${encodeURIComponent(channel.name)}/reads`),
    ]);
    if (token !== detailToken || !open) return;

    if (infoResult.status === "fulfilled") info = infoResult.value || {};
    else detailError = infoResult.reason?.message || String(infoResult.reason || "");
    if (readsResult.status === "fulfilled") reads = readsResult.value || {};

    detailLoading = false;
    sample = _buildSample();
    if (!editing) renderInfo(current || channel);
    if (sample) void loadSampleMessages(channel, token);
  }

  function render(channel) {
    if (!open) return;
    if (!channel) return;
    const changed = channel.name !== channelName;
    if (changed) {
      channelName = channel.name;
      editing = false;
      draft = null;
      pendingClose = false;
      error = "";
      flash = "";
      saving = false;
      info = null;
      reads = null;
      sample = null;
      detailLoading = false;
      detailError = "";
      detailRevision = -1;
    }
    current = channel;
    const revision = Number(store.state.detailRevision) || 0;
    if (!editing && (changed || revision !== detailRevision)) {
      void loadDetail(channel);
      return;
    }
    if (editing && !changed && draft) return;
    if (editing) renderSettings(channel);
    else renderInfo(channel);
  }

  function startEditing(channel) {
    editing = true;
    pendingClose = false;
    error = "";
    flash = "";
    draft = {
      description: channel.description || "",
      access: channel.access || "open",
      members: Array.isArray(channel.members) ? [...channel.members] : [],
    };
    renderSettings(channel);
  }

  async function saveSettings() {
    if (!draft || !current || saving) return;
    if (draft.access === "closed" && current.access !== "closed" && !pendingClose) {
      pendingClose = true;
      renderSettings(current);
      return;
    }
    saving = true;
    error = "";
    const channel = current;
    const payload = { description: draft.description, access: draft.access };
    if (draft.access === "members") payload.members = [...draft.members];
    try {
      await store.updateSettings(channel.name, payload);
      await store.loadChannels();
      const updated = store.channel(channel.name) || channel;
      current = updated;
      editing = false;
      draft = null;
      pendingClose = false;
      saving = false;
      flash = t("board.settings_saved");
      renderInfo(updated);
      if (typeof onSaved === "function") onSaved(updated);
    } catch (err) {
      saving = false;
      error = err?.message || t("board.settings_failed");
      renderSettings(channel);
    }
  }

  function close() {
    detailToken += 1;
    open = false;
    element.hidden = true;
    element.innerHTML = "";
    editing = false;
    draft = null;
    info = null;
    reads = null;
    sample = null;
    detailLoading = false;
    detailError = "";
    if (typeof onToggle === "function") onToggle(false);
  }

  function openFor(channel) {
    detailToken += 1;
    element.hidden = false;
    channelName = "";
    editing = false;
    draft = null;
    pendingClose = false;
    error = "";
    flash = "";
    saving = false;
    info = null;
    reads = null;
    sample = null;
    detailLoading = false;
    detailError = "";
    detailRevision = Number(store.state.detailRevision) || 0;
    setToggle(true);
    render(channel);
  }

  element.addEventListener("click", (event) => {
    if (event.target.closest("[data-board-drawer-close]")) {
      close();
      return;
    }
    if (event.target.closest("[data-board-drawer-edit]")) {
      if (current) startEditing(current);
      return;
    }
    if (event.target.closest("[data-board-drawer-cancel]")) {
      editing = false;
      draft = null;
      pendingClose = false;
      error = "";
      if (current) {
        if ((Number(store.state.detailRevision) || 0) !== detailRevision) void loadDetail(current);
        else renderInfo(current);
      }
      return;
    }
    const remove = event.target.closest("[data-board-member-remove]");
    if (remove && draft) {
      const name = remove.dataset.boardMemberRemove;
      draft.members = draft.members.filter((member) => member !== name);
      error = "";
      pendingClose = false;
      if (current) renderSettings(current);
    }
  });

  element.addEventListener("input", (event) => {
    if (!draft || event.target.id !== "boardSetDescription") return;
    draft.description = event.target.value;
  });

  element.addEventListener("change", (event) => {
    if (!draft || !current) return;
    if (event.target.name === "boardSetAccess") {
      draft.access = event.target.value;
      pendingClose = false;
      error = "";
      renderSettings(current);
      return;
    }
    if (event.target.id === "boardSetAdd" && event.target.value) {
      if (!draft.members.includes(event.target.value)) draft.members.push(event.target.value);
      error = "";
      pendingClose = false;
      renderSettings(current);
    }
  });

  element.addEventListener("submit", (event) => {
    if (event.target.id !== "boardSettingsForm") return;
    event.preventDefault();
    void saveSettings();
  });

  return {
    element,
    open: openFor,
    close,
    render,
    isOpen: () => open,
    destroy: () => element.remove(),
  };
}
