// ── Settings: Secrets tab ───────────────────
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";
import { t } from "/shared/i18n.js";

const _KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;

let _container = null;
let _overview = null;
let _syncPlan = null;
let _syncResult = null;
let _abort = null;
let _statusTimer = null;

export function render(container) {
  _container = container;
  _abort?.abort();
  _abort = new AbortController();
  _overview = null;
  _syncPlan = null;
  _syncResult = null;

  container.addEventListener("click", _onClick, { signal: _abort.signal });
  container.addEventListener("submit", _onSubmit, { signal: _abort.signal });

  container.innerHTML = `
    <div id="secretsRoot">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;
  _loadOverview();
}

export function destroy() {
  _abort?.abort();
  _abort = null;
  if (_statusTimer) clearTimeout(_statusTimer);
  _statusTimer = null;
  _container = null;
  _overview = null;
  _syncPlan = null;
  _syncResult = null;
}

// ── Loading ────────────────────────────────

function _root() {
  return _container ? _container.querySelector("#secretsRoot") : null;
}

async function _loadOverview() {
  const root = _root();
  if (!root) return;
  try {
    const data = await api("/api/vault/overview");
    if (!_container) return;
    _overview = data || {};
    _renderAll();
  } catch (err) {
    if (!_container) return;
    const message = _errorText(err, "secrets.error.load");
    root.innerHTML = `
      <div class="secrets-load-error">${escapeHtml(message)}</div>
      <button type="button" class="preset-btn" data-action="reload">${t("secrets.retry")}</button>
    `;
  }
}

// ── Rendering ──────────────────────────────

function _renderAll() {
  const root = _root();
  if (!root) return;

  const data = _overview || {};
  const write = data.write || {};
  const blocked = write.allowed === false;
  const requiresPassword = write.requires_password === true;

  const writeWarningHtml = blocked
    ? `<div class="secrets-write-warning">${escapeHtml(
      typeof write.reason === "string" && write.reason ? write.reason : t("secrets.write.not_allowed"),
    )}</div>`
    : "";
  const onepasswordHtml = _renderOnepassword(data.onepassword || {}, blocked);
  const entriesHtml = _renderEntries(data.entries || [], blocked);
  const addHtml = _renderAddForm(blocked, requiresPassword);
  const animaHtml = _renderAnimaSections(data.anima_sections || []);
  const hubHtml = _renderHubMembers(data.hub_members || [], blocked);
  const plaintextHtml = _renderPlaintext(data.plaintext_config || [], blocked, requiresPassword);

  root.innerHTML = `
    <div id="secretsStatus" class="settings-status secrets-status hidden"></div>
    ${writeWarningHtml}

    <section class="settings-section">
      <h3 class="settings-section-title">${t("secrets.onepassword.title")}</h3>
      ${onepasswordHtml}
    </section>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("secrets.table.title")}</h3>
      <p class="settings-section-desc">${t("secrets.table.desc")}</p>
      ${entriesHtml}
    </section>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("secrets.add.title")}</h3>
      <p class="settings-section-desc">${t("secrets.add.desc")}</p>
      ${addHtml}
    </section>

    <section class="settings-section">
      <h3 class="settings-section-title">${t("secrets.anima.title")}</h3>
      <p class="settings-section-desc">${t("secrets.anima.desc")}</p>
      ${animaHtml}
    </section>

    ${hubHtml}
    ${plaintextHtml}
  `;
}

function _renderOnepassword(op, blocked) {
  if (op.available !== true) {
    return `<p class="secrets-empty">${t("secrets.onepassword.unavailable")}</p>`;
  }

  const count = Number(op.sources_count || 0);
  const lastCheck = _formatDate(op.last_check_at);
  const syncPanelHtml = _renderSyncPanel(blocked);

  return `
    <div class="secrets-op-block">
      <div class="secrets-op-row">
        <span class="secrets-op-dot" aria-hidden="true"></span>
        <span>${t("secrets.onepassword.connected")}</span>
      </div>
      <div class="secrets-op-row">
        <span class="secrets-op-label">${t("secrets.onepassword.sources")}</span>
        <span>${t("secrets.onepassword.sources_count", { count })}</span>
      </div>
      <div class="secrets-op-row">
        <span class="secrets-op-label">${t("secrets.onepassword.last_check")}</span>
        <span>${escapeHtml(lastCheck)}</span>
      </div>
      <div class="secrets-form-actions">
        <button type="button" class="preset-btn" data-action="check">${t("secrets.check.button")}</button>
        <button type="button" class="preset-btn" data-action="sync-plan">${t("secrets.sync.button")}</button>
      </div>
    </div>
    ${syncPanelHtml}
  `;
}

function _renderSyncPanel(blocked) {
  if (_syncResult) {
    const applied = Array.isArray(_syncResult.applied) ? _syncResult.applied.length : 0;
    const skipped = Array.isArray(_syncResult.skipped) ? _syncResult.skipped.length : 0;
    return `
      <div class="secrets-sync-panel">
        <h4 class="settings-subsection-title">${t("secrets.sync.result_title")}</h4>
        <p class="secrets-sync-line">${t("secrets.sync.applied", { count: applied })}</p>
        <p class="secrets-sync-line">${t("secrets.sync.skipped", { count: skipped })}</p>
      </div>
    `;
  }

  if (!_syncPlan) return "";

  const items = Array.isArray(_syncPlan.items) ? _syncPlan.items : [];
  if (items.length === 0) {
    return `<div class="secrets-sync-panel"><p class="secrets-empty">${t("secrets.sync.none")}</p></div>`;
  }

  const planRowsHtml = items.map((item) => {
    const key = String(item.key || "");
    const status = String(item.status || "");
    const badgeClass = status === "new" || status === "changed"
      ? "secrets-plan-badge secrets-plan-change"
      : "secrets-plan-badge";
    const errorText = item.error ? escapeHtml(item.error) : "";
    return `
      <tr>
        <td class="secrets-key-cell"><code>${escapeHtml(key)}</code></td>
        <td><span class="${escapeHtml(badgeClass)}">${escapeHtml(_syncStatusLabel(status))}</span></td>
        <td class="secrets-muted">${errorText}</td>
      </tr>
    `;
  }).join("");

  const changeKeys = items.filter((item) => item.status === "new" || item.status === "changed");
  if (changeKeys.length === 0) {
    return `<div class="secrets-sync-panel"><p class="secrets-empty">${t("secrets.sync.none")}</p></div>`;
  }

  const disabledAttr = blocked ? " disabled" : "";
  const passwordFieldHtml = _requiresPassword() ? _passwordFieldHtml() : "";

  return `
    <div class="secrets-sync-panel">
      <h4 class="settings-subsection-title">${t("secrets.sync.plan_title")}</h4>
      <div class="secrets-table-wrap">
        <table class="data-table secrets-table">
          <thead>
            <tr>
              <th>${t("secrets.table.key")}</th>
              <th>${t("secrets.table.status")}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>${planRowsHtml}</tbody>
        </table>
      </div>
      <form class="secrets-form" data-form="sync-apply">
        ${passwordFieldHtml}
        <div class="secrets-form-actions">
          <button type="submit" class="btn-primary"${disabledAttr}>${t("secrets.sync.apply")}</button>
          <button type="button" class="preset-btn" data-action="sync-cancel">${t("secrets.cancel")}</button>
        </div>
      </form>
    </div>
  `;
}

function _renderEntries(entries, blocked) {
  if (entries.length === 0) {
    return `<p class="secrets-empty">${t("secrets.table.empty")}</p>`;
  }
  const rowsHtml = entries.map((entry) => _renderEntryRow(entry, blocked)).join("");
  return `
    <div class="secrets-table-wrap">
      <table class="data-table secrets-table">
        <thead>
          <tr>
            <th>${t("secrets.table.key")}</th>
            <th>${t("secrets.table.status")}</th>
            <th>${t("secrets.table.owner")}</th>
            <th>${t("secrets.table.updated")}</th>
            <th>${t("secrets.table.check")}</th>
            <th>${t("secrets.table.actions")}</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
  `;
}

function _renderEntryRow(entry, blocked) {
  const key = String(entry.key || "");
  const isSet = entry.is_set === true;
  const statusLabel = isSet ? t("secrets.status.set") : t("secrets.status.unset");
  const statusClass = isSet ? "secrets-status-set" : "secrets-status-unset";
  const ownerHtml = entry.owner === "onepassword"
    ? `<span class="secrets-badge secrets-badge-op"><span aria-hidden="true">\u{1F511}</span> ${t("secrets.owner.onepassword")}</span>`
    : `<span class="secrets-badge secrets-badge-local">${t("secrets.owner.local")}</span>`;
  const updated = _formatUpdated(entry);
  const checkLabel = _checkLabel(entry.check);
  const actionsHtml = _renderEntryActions(key, entry.owner, blocked);

  return `
    <tr id="secretsRow-${escapeHtml(key)}" data-row-key="${escapeHtml(key)}">
      <td class="secrets-key-cell"><code>${escapeHtml(key)}</code></td>
      <td><span class="secrets-status-badge ${escapeHtml(statusClass)}">${escapeHtml(statusLabel)}</span></td>
      <td>${ownerHtml}</td>
      <td>${escapeHtml(updated)}</td>
      <td>${escapeHtml(checkLabel)}</td>
      <td>${actionsHtml}</td>
    </tr>
  `;
}

function _renderEntryActions(key, owner, blocked) {
  if (owner === "onepassword") {
    return `<span class="secrets-note">${t("secrets.action.onepassword_hint")}</span>`;
  }
  const disabledAttr = blocked ? " disabled" : "";
  return `
    <div class="secrets-row-actions">
      <button type="button" class="secrets-link-btn" data-action="edit" data-key="${escapeHtml(key)}"${disabledAttr}>${t("secrets.action.edit")}</button>
      <button type="button" class="secrets-link-btn" data-action="delete" data-key="${escapeHtml(key)}"${disabledAttr}>${t("secrets.action.delete")}</button>
      <button type="button" class="secrets-link-btn" data-action="rollback" data-key="${escapeHtml(key)}"${disabledAttr}>${t("secrets.action.rollback")}</button>
    </div>
  `;
}

function _renderAddForm(blocked, requiresPassword) {
  const disabledAttr = blocked ? " disabled" : "";
  const passwordFieldHtml = requiresPassword ? _passwordFieldHtml() : "";
  return `
    <form class="secrets-form" data-form="add">
      <label class="secrets-field">
        <span>${t("secrets.add.key_label")}</span>
        <input type="text" data-field="key" autocomplete="off" placeholder="${escapeHtml(t("secrets.add.key_placeholder"))}">
      </label>
      <label class="secrets-field">
        <span>${t("secrets.add.value_label")}</span>
        <div class="secrets-value-row">
          <input type="password" data-field="value" autocomplete="new-password">
          <button type="button" class="preset-btn" data-action="toggle-value">${t("secrets.edit.show")}</button>
        </div>
      </label>
      ${passwordFieldHtml}
      <div class="secrets-form-actions">
        <button type="submit" class="btn-primary"${disabledAttr}>${t("secrets.add.button")}</button>
      </div>
    </form>
  `;
}

function _renderAnimaSections(sections) {
  if (sections.length === 0) {
    return `<p class="secrets-empty">${t("secrets.anima.empty")}</p>`;
  }
  const rowsHtml = sections.map((section) => `
    <tr>
      <td>${escapeHtml(section.section)}</td>
      <td>${escapeHtml(t("secrets.anima.count", { count: section.count }))}</td>
    </tr>
  `).join("");
  return `
    <div class="secrets-table-wrap">
      <table class="data-table secrets-table">
        <thead>
          <tr>
            <th>${t("secrets.anima.name")}</th>
            <th>${t("secrets.anima.count_label")}</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
  `;
}

function _renderHubMembers(members, blocked) {
  if (members.length === 0) return "";

  const disabledAttr = blocked ? " disabled" : "";
  const rowsHtml = members.map((member) => {
    const id = String(member.id || "");
    const name = String(member.display_name || id);
    const tokenLabel = member.has_token ? t("secrets.hub.token_yes") : t("secrets.hub.token_no");
    const enabledHtml = member.enabled === false
      ? `<span class="secrets-muted">(${t("secrets.hub.disabled")})</span>`
      : "";
    const lastSeen = _formatDate(member.last_seen);
    const urlKey = member.webhook_url_key ? String(member.webhook_url_key) : "";
    const keyKey = member.webhook_key_key ? String(member.webhook_key_key) : "";
    const urlHtml = urlKey
      ? `<a class="secrets-link" role="link" tabindex="0" data-action="jump-key" data-key="${escapeHtml(urlKey)}">${escapeHtml(urlKey)}</a>`
      : `<span class="secrets-muted">${t("secrets.hub.webhook_none")}</span>`;
    const keyHtml = keyKey
      ? `<a class="secrets-link" role="link" tabindex="0" data-action="jump-key" data-key="${escapeHtml(keyKey)}">${escapeHtml(keyKey)}</a>`
      : `<span class="secrets-muted">${t("secrets.hub.webhook_none")}</span>`;
    const webhookHtml = `
      <div class="secrets-webhook">
        <div>${t("secrets.hub.webhook_url")} ${urlHtml}</div>
        <div>${t("secrets.hub.webhook_key")} ${keyHtml}</div>
      </div>
    `;
    return `
      <tr data-member-row="${escapeHtml(id)}">
        <td>${escapeHtml(name)} ${enabledHtml}</td>
        <td class="secrets-key-cell"><code>${escapeHtml(id)}</code></td>
        <td>${escapeHtml(tokenLabel)}</td>
        <td>${escapeHtml(lastSeen)}</td>
        <td>${webhookHtml}</td>
        <td>
          <button type="button" class="preset-btn" data-action="reissue" data-id="${escapeHtml(id)}"${disabledAttr}>${t("secrets.hub.reissue")}</button>
        </td>
      </tr>
    `;
  }).join("");

  return `
    <section class="settings-section">
      <h3 class="settings-section-title">${t("secrets.hub.title")}</h3>
      <p class="settings-section-desc">${t("secrets.hub.desc")}</p>
      <div class="secrets-table-wrap">
        <table class="data-table secrets-table">
          <thead>
            <tr>
              <th>${t("secrets.hub.name")}</th>
              <th>${t("secrets.hub.id")}</th>
              <th>${t("secrets.hub.token")}</th>
              <th>${t("secrets.hub.last_seen")}</th>
              <th>${t("secrets.hub.webhook")}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>
      <div id="secretsTokenNotice" class="secrets-token-notice hidden"></div>
    </section>
  `;
}

function _renderPlaintext(items, blocked, requiresPassword) {
  if (items.length === 0) return "";

  const disabledAttr = blocked ? " disabled" : "";
  const passwordFieldHtml = requiresPassword ? _passwordFieldHtml() : "";
  const rowsHtml = items.map((item) => `
    <tr>
      <td><code>${escapeHtml(item.path)}</code></td>
      <td><code>${escapeHtml(item.vault_key)}</code></td>
    </tr>
  `).join("");

  return `
    <section class="settings-section">
      <h3 class="settings-section-title">${t("secrets.plaintext.title")}</h3>
      <p class="settings-section-desc">${t("secrets.plaintext.desc")}</p>
      <div class="secrets-table-wrap">
        <table class="data-table secrets-table">
          <thead>
            <tr>
              <th>${t("secrets.plaintext.path")}</th>
              <th>${t("secrets.plaintext.vault_key")}</th>
            </tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>
      <form class="secrets-form" data-form="migrate">
        ${passwordFieldHtml}
        <div class="secrets-form-actions">
          <button type="submit" class="btn-primary"${disabledAttr}>${t("secrets.plaintext.button")}</button>
        </div>
      </form>
    </section>
  `;
}

function _passwordFieldHtml() {
  return `
    <label class="secrets-field">
      <span>${t("secrets.password")}</span>
      <input type="password" data-field="password" autocomplete="current-password">
    </label>
  `;
}

// ── Form rows ──────────────────────────────

function _editFormHtml(key) {
  const passwordFieldHtml = _requiresPassword() ? _passwordFieldHtml() : "";
  return `
    <tr class="secrets-form-row">
      <td colspan="6">
        <form class="secrets-form" data-form="edit" data-key="${escapeHtml(key)}">
          <label class="secrets-field">
            <span>${t("secrets.edit.new_value")}</span>
            <div class="secrets-value-row">
              <input type="password" data-field="value" autocomplete="new-password">
              <button type="button" class="preset-btn" data-action="toggle-value">${t("secrets.edit.show")}</button>
            </div>
          </label>
          ${passwordFieldHtml}
          <div class="secrets-form-actions">
            <button type="submit" class="btn-primary">${t("secrets.edit.save")}</button>
            <button type="button" class="preset-btn" data-action="cancel-row">${t("secrets.cancel")}</button>
          </div>
        </form>
      </td>
    </tr>
  `;
}

function _deleteFormHtml(key) {
  const passwordFieldHtml = _requiresPassword() ? _passwordFieldHtml() : "";
  return `
    <tr class="secrets-form-row">
      <td colspan="6">
        <form class="secrets-form" data-form="delete" data-key="${escapeHtml(key)}">
          <p class="secrets-confirm">${escapeHtml(t("secrets.delete.confirm", { key }))}</p>
          ${passwordFieldHtml}
          <div class="secrets-form-actions">
            <button type="submit" class="btn-primary">${t("secrets.delete.button")}</button>
            <button type="button" class="preset-btn" data-action="cancel-row">${t("secrets.cancel")}</button>
          </div>
        </form>
      </td>
    </tr>
  `;
}

function _rollbackFormHtml(key) {
  const passwordFieldHtml = _requiresPassword() ? _passwordFieldHtml() : "";
  return `
    <tr class="secrets-form-row">
      <td colspan="6">
        <form class="secrets-form" data-form="rollback" data-key="${escapeHtml(key)}">
          <p class="secrets-confirm">${escapeHtml(t("secrets.rollback.confirm", { key }))}</p>
          ${passwordFieldHtml}
          <div class="secrets-form-actions">
            <button type="submit" class="btn-primary">${t("secrets.rollback.button")}</button>
            <button type="button" class="preset-btn" data-action="cancel-row">${t("secrets.cancel")}</button>
          </div>
        </form>
      </td>
    </tr>
  `;
}

function _reissueFormHtml(member) {
  const id = String(member.id || "");
  const name = String(member.display_name || id);
  const passwordFieldHtml = _requiresPassword() ? _passwordFieldHtml() : "";
  return `
    <tr class="secrets-form-row">
      <td colspan="6">
        <form class="secrets-form" data-form="reissue" data-id="${escapeHtml(id)}" data-name="${escapeHtml(name)}">
          <p class="secrets-confirm">${escapeHtml(t("secrets.hub.reissue_confirm", { name }))}</p>
          ${passwordFieldHtml}
          <div class="secrets-form-actions">
            <button type="submit" class="btn-primary">${t("secrets.hub.reissue_button")}</button>
            <button type="button" class="preset-btn" data-action="cancel-row">${t("secrets.cancel")}</button>
          </div>
        </form>
      </td>
    </tr>
  `;
}

function _tokenNoticeHtml(token) {
  return `
    <h4 class="settings-subsection-title">${t("secrets.hub.token_label")}</h4>
    <p class="secrets-token-warning">${t("secrets.hub.token_once")}</p>
    <code class="secrets-token-value">${escapeHtml(token)}</code>
    <div class="secrets-form-actions">
      <button type="button" class="preset-btn" data-action="copy-token">${t("secrets.hub.copy")}</button>
      <button type="button" class="preset-btn" data-action="close-token">${t("secrets.hub.close")}</button>
    </div>
  `;
}

// ── Click handling ─────────────────────────

function _onClick(event) {
  const target = event.target.closest("[data-action]");
  if (!target || !_container || !_container.contains(target)) return;
  const action = target.dataset.action;
  const key = target.dataset.key || "";
  const id = target.dataset.id || "";

  if (action === "reload") {
    event.preventDefault();
    _loadOverview();
  } else if (action === "check") {
    event.preventDefault();
    _runCheck(target);
  } else if (action === "sync-plan") {
    event.preventDefault();
    _runSyncPlan(target);
  } else if (action === "sync-cancel") {
    event.preventDefault();
    _syncPlan = null;
    _syncResult = null;
    _renderAll();
  } else if (action === "edit") {
    event.preventDefault();
    _openRowForm(key, "edit");
  } else if (action === "delete") {
    event.preventDefault();
    _openRowForm(key, "delete");
  } else if (action === "rollback") {
    event.preventDefault();
    _openRowForm(key, "rollback");
  } else if (action === "cancel-row") {
    event.preventDefault();
    _removeForms();
  } else if (action === "toggle-value") {
    event.preventDefault();
    _toggleValueInput(target);
  } else if (action === "reissue") {
    event.preventDefault();
    _openReissueForm(id);
  } else if (action === "copy-token") {
    event.preventDefault();
    _copyToken(target);
  } else if (action === "close-token") {
    event.preventDefault();
    _closeTokenNotice();
  } else if (action === "jump-key") {
    event.preventDefault();
    _jumpToKey(key);
  }
}

function _openRowForm(key, kind) {
  if (!_container) return;
  _removeForms();
  const row = _findRow(key);
  if (!row) return;
  let html = "";
  if (kind === "edit") html = _editFormHtml(key);
  else if (kind === "delete") html = _deleteFormHtml(key);
  else html = _rollbackFormHtml(key);
  row.insertAdjacentHTML("afterend", html);
}

function _openReissueForm(id) {
  if (!_container) return;
  _removeForms();
  const member = _findMember(id);
  if (!member) return;
  const row = _findMemberRow(id);
  if (!row) return;
  row.insertAdjacentHTML("afterend", _reissueFormHtml(member));
}

function _findRow(key) {
  if (!_container) return null;
  const rows = _container.querySelectorAll("tr[data-row-key]");
  for (const row of rows) {
    if (row.dataset.rowKey === key) return row;
  }
  return null;
}

function _findMemberRow(id) {
  if (!_container) return null;
  const rows = _container.querySelectorAll("tr[data-member-row]");
  for (const row of rows) {
    if (row.dataset.memberRow === id) return row;
  }
  return null;
}

function _findMember(id) {
  const members = (_overview && _overview.hub_members) || [];
  return members.find((member) => String(member.id || "") === id) || null;
}

function _removeForms() {
  if (!_container) return;
  _container.querySelectorAll("tr.secrets-form-row").forEach((row) => row.remove());
}

function _toggleValueInput(button) {
  const row = button.closest("tr");
  const input = row ? row.querySelector('input[data-field="value"]') : null;
  if (!input) return;
  const show = input.type === "password";
  input.type = show ? "text" : "password";
  button.textContent = show ? t("secrets.edit.hide") : t("secrets.edit.show");
}

function _jumpToKey(key) {
  const row = _findRow(key);
  if (!row) return;
  try {
    row.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch {
    /* ignore */
  }
  row.classList.add("secrets-row-target");
  setTimeout(() => row.classList.remove("secrets-row-target"), 2000);
}

// ── Submit handling ────────────────────────

function _onSubmit(event) {
  const form = event.target.closest("form[data-form]");
  if (!form || !_container) return;
  event.preventDefault();
  const kind = form.dataset.form;
  if (kind === "edit") _submitEdit(form);
  else if (kind === "delete") _submitDelete(form);
  else if (kind === "rollback") _submitRollback(form);
  else if (kind === "add") _submitAdd(form);
  else if (kind === "sync-apply") _submitSyncApply(form);
  else if (kind === "reissue") _submitReissue(form);
  else if (kind === "migrate") _submitMigrate(form);
}

function _formPassword(form) {
  const input = form.querySelector('input[data-field="password"]');
  return input ? input.value : "";
}

function _checkPassword(form) {
  if (!_requiresPassword()) return true;
  const passwordInput = form.querySelector('input[data-field="password"]');
  if (passwordInput && !passwordInput.value) {
    _showStatus(t("secrets.password_required"), true);
    passwordInput.focus();
    return false;
  }
  return true;
}

async function _submitEdit(form) {
  const key = form.dataset.key || "";
  const valueInput = form.querySelector('input[data-field="value"]');
  const value = valueInput ? valueInput.value : "";
  if (!value) {
    _showStatus(t("secrets.edit.empty_value"), true);
    if (valueInput) valueInput.focus();
    return;
  }
  if (!_checkPassword(form)) return;
  try {
    const path = "/api/vault/shared/" + encodeURIComponent(key);
    await api(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value, password: _formPassword(form) }),
    });
    if (valueInput) valueInput.value = "";
    form.reset();
    _removeForms();
    await _loadOverview();
    _showStatus(t("secrets.edit.saved", { key }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.save"), true);
  }
}

async function _submitDelete(form) {
  const key = form.dataset.key || "";
  if (!_checkPassword(form)) return;
  try {
    const path = "/api/vault/shared/" + encodeURIComponent(key) + "/delete";
    await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: _formPassword(form) }),
    });
    _removeForms();
    await _loadOverview();
    _showStatus(t("secrets.delete.done", { key }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.delete"), true);
  }
}

async function _submitRollback(form) {
  const key = form.dataset.key || "";
  if (!_checkPassword(form)) return;
  try {
    const path = "/api/vault/shared/" + encodeURIComponent(key) + "/rollback";
    await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: _formPassword(form) }),
    });
    _removeForms();
    await _loadOverview();
    _showStatus(t("secrets.rollback.done", { key }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.rollback"), true);
  }
}

async function _submitAdd(form) {
  const keyInput = form.querySelector('input[data-field="key"]');
  const valueInput = form.querySelector('input[data-field="value"]');
  const key = keyInput ? keyInput.value.trim() : "";
  const value = valueInput ? valueInput.value : "";
  if (!_KEY_RE.test(key)) {
    _showStatus(t("secrets.add.invalid_key"), true);
    if (keyInput) keyInput.focus();
    return;
  }
  if (!value) {
    _showStatus(t("secrets.add.empty_value"), true);
    if (valueInput) valueInput.focus();
    return;
  }
  if (!_checkPassword(form)) return;
  try {
    const path = "/api/vault/shared/" + encodeURIComponent(key);
    await api(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value, password: _formPassword(form) }),
    });
    form.reset();
    await _loadOverview();
    _showStatus(t("secrets.add.done", { key }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.add"), true);
  }
}

async function _submitSyncApply(form) {
  const items = (_syncPlan && _syncPlan.items) || [];
  const keys = items
    .filter((item) => item.status === "new" || item.status === "changed")
    .map((item) => item.key);
  if (keys.length === 0) {
    _showStatus(t("secrets.sync.none"), true);
    return;
  }
  if (!_checkPassword(form)) return;
  try {
    const data = await api("/api/vault/sync/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keys, password: _formPassword(form) }),
    });
    _syncResult = data || {};
    _syncPlan = null;
    await _loadOverview();
    const applied = Array.isArray(_syncResult.applied) ? _syncResult.applied.length : 0;
    _showStatus(t("secrets.sync.applied", { count: applied }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.sync"), true);
  }
}

async function _submitReissue(form) {
  const id = form.dataset.id || "";
  const name = form.dataset.name || id;
  if (!_checkPassword(form)) return;
  try {
    const path = "/api/vault/hub-members/" + encodeURIComponent(id) + "/reissue-token";
    const data = await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: _formPassword(form) }),
    });
    _removeForms();
    const notice = _container ? _container.querySelector("#secretsTokenNotice") : null;
    if (notice) {
      notice.innerHTML = _tokenNoticeHtml(String((data && data.token) || ""));
      notice.classList.remove("hidden");
    }
    _showStatus(t("secrets.hub.reissue_done", { name }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.reissue"), true);
  }
}

async function _submitMigrate(form) {
  if (!_checkPassword(form)) return;
  try {
    const data = await api("/api/vault/migrate-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: _formPassword(form) }),
    });
    const moved = Array.isArray(data && data.moved) ? data.moved.length : 0;
    await _loadOverview();
    _showStatus(t("secrets.plaintext.done", { count: moved }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.migrate"), true);
  }
}

// ── 1Password actions ──────────────────────

async function _runCheck(button) {
  button.disabled = true;
  try {
    const data = await api("/api/vault/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!_container || !_overview) return;
    const results = (data && data.results) || {};
    const entries = _overview.entries || [];
    for (const entry of entries) {
      if (Object.prototype.hasOwnProperty.call(results, entry.key)) {
        entry.check = results[entry.key];
      }
    }
    if (_overview.onepassword) {
      _overview.onepassword.last_check_at = new Date().toISOString();
    }
    _renderAll();
    _showStatus(t("secrets.check.done", { count: Object.keys(results).length }));
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.check"), true);
  }
}

async function _runSyncPlan(button) {
  button.disabled = true;
  try {
    const data = await api("/api/vault/sync/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!_container) return;
    _syncPlan = data || {};
    _syncResult = null;
    _renderAll();
  } catch (err) {
    _showStatus(_errorText(err, "secrets.error.sync"), true);
  }
}

// ── Token notice ───────────────────────────

async function _copyToken(button) {
  const notice = _container ? _container.querySelector("#secretsTokenNotice") : null;
  const code = notice ? notice.querySelector(".secrets-token-value") : null;
  if (!code) return;
  const text = code.textContent || "";
  try {
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== "function") {
      throw new Error("clipboard unavailable");
    }
    await navigator.clipboard.writeText(text);
    _showStatus(t("secrets.hub.copied"));
  } catch {
    _selectText(code);
    _showStatus(t("secrets.hub.copy_failed"), true);
  }
}

function _selectText(el) {
  try {
    const selection = window.getSelection();
    if (!selection) return;
    const range = document.createRange();
    range.selectNodeContents(el);
    selection.removeAllRanges();
    selection.addRange(range);
  } catch {
    /* ignore */
  }
}

function _closeTokenNotice() {
  const notice = _container ? _container.querySelector("#secretsTokenNotice") : null;
  if (!notice) return;
  notice.innerHTML = "";
  notice.classList.add("hidden");
}

// ── Helpers ────────────────────────────────

function _requiresPassword() {
  return !!(_overview && _overview.write && _overview.write.requires_password === true);
}

function _errorText(err, fallbackKey) {
  if (err && typeof err.message === "string" && err.message) return err.message;
  return t(fallbackKey);
}

function _formatDate(value) {
  if (!value) return t("secrets.unknown");
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString();
  } catch {
    return String(value);
  }
}

function _formatUpdated(entry) {
  const time = _formatDate(entry.updated_at);
  const by = entry.updated_by ? String(entry.updated_by) : t("secrets.unknown");
  return t("secrets.updated_format", { time, by });
}

function _checkLabel(check) {
  if (check === "same") return t("secrets.check.same");
  if (check === "differs") return t("secrets.check.differs");
  if (check === "missing_remote") return t("secrets.check.missing_remote");
  if (check === "missing_local") return t("secrets.check.missing_local");
  if (check === "error") return t("secrets.check.error");
  return t("secrets.check.none");
}

function _syncStatusLabel(status) {
  if (status === "new") return t("secrets.sync.status.new");
  if (status === "changed") return t("secrets.sync.status.changed");
  if (status === "unchanged") return t("secrets.sync.status.unchanged");
  if (status === "empty_remote") return t("secrets.sync.status.empty_remote");
  if (status === "error") return t("secrets.sync.status.error");
  return t("secrets.sync.status.unknown");
}

function _showStatus(message, isError) {
  const statusEl = _container ? _container.querySelector("#secretsStatus") : null;
  if (!statusEl) return;
  statusEl.textContent = message;
  statusEl.className = isError
    ? "settings-status secrets-status settings-status-error"
    : "settings-status secrets-status";
  statusEl.classList.remove("hidden");
  if (_statusTimer) clearTimeout(_statusTimer);
  _statusTimer = setTimeout(() => {
    if (statusEl.isConnected) statusEl.classList.add("hidden");
  }, 5000);
}
