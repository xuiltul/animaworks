// ── Board info drawer ────────────────────────
// Right-side panel for "about this board" plus the settings editor. The panel
// is section based so later PRs can append read position / post-count charts
// without rewriting it.
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { avatarMarkup } from "./avatar.js";
import { memberCandidates } from "./mentions.js";

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
      ${_infoSection(channel)}`;
  }

  function renderSettings(channel) {
    element.innerHTML = `
      <div class="board-drawer-head">
        <h2 class="board-drawer-title">${escapeHtml(t("board.settings_title", { name: channel.name }))}</h2>
        ${_closeButton("data-board-drawer-cancel")}
      </div>
      ${_settingsSection(channel)}`;
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
    }
    current = channel;
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
    open = false;
    element.hidden = true;
    element.innerHTML = "";
    editing = false;
    draft = null;
    if (typeof onToggle === "function") onToggle(false);
  }

  function openFor(channel) {
    element.hidden = false;
    channelName = "";
    editing = false;
    draft = null;
    pendingClose = false;
    error = "";
    flash = "";
    saving = false;
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
      if (current) renderInfo(current);
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
