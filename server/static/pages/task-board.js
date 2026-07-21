// ── TaskBoard Page (Kanban) ─────────────────
import { api } from "../modules/api.js";
import { escapeAttr, escapeHtml } from "../modules/state.js";
import { t } from "/shared/i18n.js";
import {
  COLUMNS,
  SUPPRESSED_VISIBILITIES,
  ageText,
  deadlineText,
  defaultLocalDateTime,
  isOverdue,
  shortId,
  splitTaskDescription,
  statusClassSuffix,
  taskKey,
  visibilityLabel,
  visibilityPayload,
} from "./task-board-utils.js";

const REFRESH_MS = 30000;

let _container = null;
let _pollTimer = null;
let _refreshing = false;
let _tasks = [];
let _allTasks = [];
let _animaNames = [];
let _activeColumn = "todo";
let _drag = null;
let _searchTimer = null; let _renderToken = 0;
/** @type {Set<string>} task keys whose card body is expanded */
let _expandedBodies = new Set();

export async function render(container) {
  const token = ++_renderToken;
  _container = container;
  _tasks = [];
  _allTasks = [];
  _animaNames = [];
  _activeColumn = "todo"; _drag = null; _refreshing = false;

  container.innerHTML = `
    <div class="taskboard-page" data-testid="taskboard-page">
      <div class="taskboard-header">
        <div>
          <h2>${t("taskboard.title")}</h2>
          <p>${t("taskboard.subtitle")}</p>
        </div>
        <button class="taskboard-icon-btn" id="taskboardRefresh" title="${t("taskboard.refresh")}">
          <i data-lucide="refresh-cw" aria-hidden="true"></i>
          <span>${t("taskboard.refresh")}</span>
        </button>
      </div>

      <div class="taskboard-toolbar" id="taskboardToolbar">
        <label class="taskboard-field">
          <span>${t("taskboard.assignee")}</span>
          <select id="taskboardAssignee">
            <option value="">${t("taskboard.assignee_all")}</option>
          </select>
        </label>

        <fieldset class="taskboard-segmented" aria-label="${t("taskboard.visibility")}">
          <button type="button" class="taskboard-segment is-active" data-visibility-filter="active">
            ${t("taskboard.filter_active")}
          </button>
          <button type="button" class="taskboard-segment" data-visibility-filter="snoozed">
            ${t("taskboard.filter_snoozed")}
          </button>
          <button type="button" class="taskboard-segment" data-visibility-filter="suppressed">
            ${t("taskboard.filter_suppressed")}
          </button>
          <button type="button" class="taskboard-segment" data-visibility-filter="all">
            ${t("taskboard.filter_all")}
          </button>
        </fieldset>

        <label class="taskboard-check">
          <input id="taskboardArchived" type="checkbox" />
          <span>${t("taskboard.archived")}</span>
        </label>

        <label class="taskboard-search">
          <i data-lucide="search" aria-hidden="true"></i>
          <input id="taskboardSearch" type="search" placeholder="${t("taskboard.search_placeholder")}" />
        </label>
      </div>

      <div class="taskboard-feedback" id="taskboardFeedback" role="status"></div>
      <div class="taskboard-mobile-tabs" id="taskboardMobileTabs"></div>
      <div class="taskboard-columns" id="taskboardColumns">
        <div class="taskboard-loading">${t("taskboard.loading")}</div>
      </div>
    </div>
  `;

  _bindEvents();
  await _loadAnimas();
  if (_container !== container || _renderToken !== token) return;
  await _loadBoard({ token });
  if (_container !== container || _renderToken !== token) return;
  _pollTimer = setInterval(() => {
    if (document.visibilityState === "visible") _loadBoard({ quiet: true });
  }, REFRESH_MS);
  _refreshIcons();
}
export function destroy() {
  _renderToken += 1;
  if (_pollTimer) clearInterval(_pollTimer);
  if (_searchTimer) clearTimeout(_searchTimer);
  _pollTimer = null; _searchTimer = null; _container = null; _refreshing = false;
  _tasks = [];
  _allTasks = [];
  _animaNames = [];
  _drag = null;
  _expandedBodies = new Set();
}
function _bindEvents() {
  _container.querySelector("#taskboardRefresh")?.addEventListener("click", () => _loadBoard());
  _container.querySelector("#taskboardAssignee")?.addEventListener("change", () => _loadBoard());
  _container.querySelector("#taskboardArchived")?.addEventListener("change", () => _loadBoard());
  _container.querySelector("#taskboardSearch")?.addEventListener("input", () => {
    if (_searchTimer) clearTimeout(_searchTimer);
    _searchTimer = setTimeout(() => _loadBoard(), 250);
  });

  _container.querySelectorAll("[data-visibility-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      _container.querySelectorAll("[data-visibility-filter]").forEach((el) => {
        el.classList.toggle("is-active", el === button);
      });
      _loadBoard();
    });
  });

  _container.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-mobile-column]");
    if (tab) {
      _setActiveColumn(tab.dataset.mobileColumn);
      return;
    }
    // Action buttons first so they never toggle body expand.
    const action = event.target.closest("[data-task-action]");
    if (action) {
      _handleAction(action);
      return;
    }
    const expandBtn = event.target.closest("[data-task-expand]");
    if (expandBtn) {
      event.preventDefault();
      _toggleCardBody(expandBtn.dataset.taskExpand);
      return;
    }
    const bodyWrap = event.target.closest("[data-task-body-wrap]");
    if (bodyWrap) {
      const card = bodyWrap.closest(".taskboard-card");
      const btn = card?.querySelector("[data-task-expand]");
      const body = card?.querySelector(".taskboard-card-body");
      // Only toggle when clamped (button visible) or already expanded.
      if (body?.classList.contains("is-expanded") || (btn && !btn.hidden)) {
        event.preventDefault();
        _toggleCardBody(bodyWrap.dataset.taskBodyWrap);
      }
    }
  });

  _container.addEventListener("dragstart", _onDragStart);
  _container.addEventListener("dragover", _onDragOver);
  _container.addEventListener("drop", _onDrop);
  _container.addEventListener("dragend", _onDragEnd);
}
async function _loadAnimas() {
  try {
    const data = await api("/api/animas");
    _animaNames = (data || []).map((item) => item.name).filter(Boolean).sort();
    _renderAssigneeOptions();
  } catch {
    _animaNames = [];
  }
}

async function _loadBoard({ quiet = false, token = _renderToken } = {}) {
  if (_refreshing) return;
  _refreshing = true;
  if (!quiet) _setFeedback(t("taskboard.loading"), "loading");
  try {
    const data = await api(`/api/task-board?${_queryParams().toString()}`);
    if (!_container || token !== _renderToken) return;
    _allTasks = data.tasks || [];
    _tasks = _filterClientVisibility(_allTasks);
    _syncAssigneeOptionsFromTasks();
    _ensureActiveColumnHasView();
    _renderBoard();
    _setFeedback(_summaryText(data), "ok");
  } catch (err) {
    if (!_container || token !== _renderToken) return;
    _setFeedback(`${t("taskboard.load_failed")}: ${err.message || err}`, "error");
    if (!_tasks.length) _renderBoard();
  } finally {
    _refreshing = false;
  }
}

function _queryParams() {
  const params = new URLSearchParams();
  const assignee = _container.querySelector("#taskboardAssignee")?.value || "";
  const search = _container.querySelector("#taskboardSearch")?.value.trim() || "";
  const visibility = _visibilityFilter();
  const includeArchived = _includeArchived();

  if (assignee) params.set("assignee", assignee);
  if (search) params.set("q", search);
  if (visibility === "active" || visibility === "snoozed") params.set("visibility", visibility);
  if (visibility === "all" || visibility === "suppressed" || includeArchived) {
    params.set("include_archived", "true");
  }
  params.set("include_missing", "true");
  return params;
}

function _visibilityFilter() {
  return _container.querySelector("[data-visibility-filter].is-active")?.dataset.visibilityFilter || "active";
}

function _includeArchived() {
  return _container.querySelector("#taskboardArchived")?.checked === true;
}

function _filterClientVisibility(tasks) {
  const visibility = _visibilityFilter();
  const includeArchived = _includeArchived();
  if (visibility === "active") return tasks.filter((task) => task.visibility === "active");
  if (visibility === "snoozed") return tasks.filter((task) => task.visibility === "snoozed");
  if (visibility === "suppressed") {
    return tasks.filter((task) => {
      if (!SUPPRESSED_VISIBILITIES.has(task.visibility)) return false;
      return task.visibility !== "archived" || includeArchived;
    });
  }
  if (!includeArchived) return tasks.filter((task) => task.visibility !== "archived");
  return tasks;
}

function _renderAssigneeOptions() {
  const select = _container?.querySelector("#taskboardAssignee");
  if (!select) return;
  const current = select.value;
  const options = [`<option value="">${t("taskboard.assignee_all")}</option>`];
  for (const name of _animaNames) {
    options.push(`<option value="${escapeAttr(name)}">${escapeHtml(name)}</option>`);
  }
  select.innerHTML = options.join("");
  if (_animaNames.includes(current)) select.value = current;
}

function _syncAssigneeOptionsFromTasks() {
  const names = new Set(_animaNames);
  for (const task of _allTasks) {
    if (task.anima_name) names.add(task.anima_name);
    if (task.assignee) names.add(task.assignee);
  }
  const next = [...names].sort();
  if (next.join("\n") !== _animaNames.join("\n")) {
    _animaNames = next;
    _renderAssigneeOptions();
  }
}

function _renderBoard() {
  _renderMobileTabs();
  const columnsEl = _container.querySelector("#taskboardColumns");
  if (!columnsEl) return;
  columnsEl.innerHTML = COLUMNS.map((column) => _columnHtml(column)).join("");
  _refreshIcons();
  _syncClampAffordances();
}

function _columnHtml(column) {
  const tasks = _tasks.filter((task) => (task.column || "todo") === column);
  const active = column === _activeColumn ? " is-active" : "";
  return `
    <section class="taskboard-column${active}" data-column="${column}">
      <header class="taskboard-column-header">
        <div>
          <h3>${t(`taskboard.column_${column}`)}</h3>
          <span>${t(`taskboard.column_${column}_hint`)}</span>
        </div>
        <strong>${tasks.length}</strong>
      </header>
      <div class="taskboard-card-list" data-card-list="${column}">
        ${tasks.length ? tasks.map(_cardHtml).join("") : _emptyColumnHtml(column)}
      </div>
    </section>
  `;
}

function _renderMobileTabs() {
  const tabs = _container.querySelector("#taskboardMobileTabs");
  if (!tabs) return;
  tabs.innerHTML = COLUMNS.map((column) => {
    const count = _tasks.filter((task) => (task.column || "todo") === column).length;
    const active = column === _activeColumn ? " is-active" : "";
    return `<button class="taskboard-mobile-tab${active}" data-mobile-column="${column}">
      <span>${t(`taskboard.column_${column}`)}</span><strong>${count}</strong>
    </button>`;
  }).join("");
}

function _cardHtml(task) {
  const key = taskKey(task);
  const column = task.column || "todo";
  const overdue = isOverdue(task.deadline);
  const description = task.summary || task.original_instruction || "";
  const { title, body } = splitTaskDescription(description);
  const displayTitle = title || t("taskboard.untitled");
  const expanded = _expandedBodies.has(key);
  const hasBody = Boolean(body);
  return `
    <article class="taskboard-card" draggable="true" data-task-key="${escapeAttr(key)}" data-column="${escapeAttr(column)}">
      <div class="taskboard-card-topline">
        <span class="taskboard-anima">${escapeHtml(task.anima_name || task.assignee || "-")}</span>
        <code>${escapeHtml(shortId(task.task_id))}</code>
      </div>
      <div class="taskboard-card-title-row">
        <h4 class="taskboard-card-title">${escapeHtml(displayTitle)}</h4>
        ${hasBody ? `<button type="button" class="taskboard-card-expand" data-task-expand="${escapeAttr(key)}" aria-expanded="${expanded ? "true" : "false"}" title="${escapeAttr(expanded ? t("taskboard.show_less") : t("taskboard.show_more"))}">${expanded ? t("taskboard.show_less") : t("taskboard.show_more")}</button>` : ""}
      </div>
      ${hasBody ? `
      <div class="taskboard-card-body-wrap" data-task-body-wrap="${escapeAttr(key)}">
        <p class="taskboard-card-body${expanded ? " is-expanded" : ""}">${escapeHtml(body)}</p>
      </div>
      ` : ""}
      <div class="taskboard-meta-row">
        <span class="taskboard-badge taskboard-badge--${statusClassSuffix(task.queue_status)}">
          ${escapeHtml(task.queue_status || t("taskboard.queue_missing"))}
        </span>
        <span class="taskboard-visibility">${escapeHtml(visibilityLabel(task.visibility))}</span>
      </div>
      <div class="taskboard-card-facts">
        <span class="${overdue ? "is-overdue" : ""}">
          <i data-lucide="calendar-clock" aria-hidden="true"></i>${deadlineText(task.deadline)}
        </span>
        <span><i data-lucide="clock-3" aria-hidden="true"></i>${ageText(task.updated_at)}</span>
      </div>
      <div class="taskboard-actions">
        ${_actionButtons(task, key)}
      </div>
    </article>
  `;
}

function _toggleCardBody(key) {
  if (!key || !_container) return;
  const next = !_expandedBodies.has(key);
  if (next) _expandedBodies.add(key);
  else _expandedBodies.delete(key);

  const card = [..._container.querySelectorAll(".taskboard-card")].find((el) => el.dataset.taskKey === key);
  if (!card) return;
  const body = card.querySelector(".taskboard-card-body");
  const btn = card.querySelector("[data-task-expand]");
  if (body) body.classList.toggle("is-expanded", next);
  if (btn) {
    btn.setAttribute("aria-expanded", next ? "true" : "false");
    const label = next ? t("taskboard.show_less") : t("taskboard.show_more");
    btn.textContent = label;
    btn.title = label;
  }
  _syncClampAffordances(card);
}

/** Hide expand controls when body does not overflow 3 lines (unless already expanded). */
function _syncClampAffordances(root = _container) {
  if (!root) return;
  const cards = root.classList?.contains("taskboard-card")
    ? [root]
    : [...root.querySelectorAll(".taskboard-card")];
  cards.forEach((card) => {
    const body = card.querySelector(".taskboard-card-body");
    const btn = card.querySelector("[data-task-expand]");
    if (!body || !btn) return;
    if (body.classList.contains("is-expanded")) {
      btn.hidden = false;
      return;
    }
    // scrollHeight > clientHeight means line-clamp is truncating.
    const clamped = body.scrollHeight > body.clientHeight + 1;
    btn.hidden = !clamped;
  });
}

function _actionButtons(task, key) {
  const attrs = `data-task-key="${escapeAttr(key)}"`;
  if (task.visibility === "active") {
    return `
      <button type="button" data-task-action="snooze" ${attrs}><i data-lucide="alarm-clock"></i>${t("taskboard.action_snooze")}</button>
      <button type="button" data-task-action="expire" ${attrs}><i data-lucide="timer-off"></i>${t("taskboard.action_expire")}</button>
      <button type="button" data-task-action="archive" ${attrs}><i data-lucide="archive"></i>${t("taskboard.action_archive")}</button>
      <button type="button" data-task-action="tombstone" ${attrs}><i data-lucide="ban"></i>${t("taskboard.action_tombstone")}</button>
    `;
  }
  return `<button type="button" data-task-action="reactivate" ${attrs}><i data-lucide="rotate-ccw"></i>${t("taskboard.action_reactivate")}</button>`;
}

function _emptyColumnHtml(column) {
  return `<div class="taskboard-empty-column">${t("taskboard.empty_column", { column: t(`taskboard.column_${column}`) })}</div>`;
}

async function _handleAction(button) {
  const task = _findTask(button.dataset.taskKey);
  if (!task) return;
  const action = button.dataset.taskAction;
  button.disabled = true;
  try {
    if (action === "reactivate") {
      await _patchTask(task, { visibility: "active", snoozed_until: null, actor: "dashboard" });
      _setFeedback(t("taskboard.reactivated"), "ok");
    } else if (action === "snooze") {
      const result = await _openActionModal({ mode: "snooze", datetime: true });
      if (!result) return;
      await _patchTask(task, {
        visibility: "snoozed",
        snoozed_until: result.datetime,
        reason: result.reason,
        actor: "dashboard",
      });
      _setFeedback(t("taskboard.snoozed"), "ok");
    } else {
      const result = await _openActionModal({
        mode: action,
        reasonRequired: action === "expire" || action === "tombstone",
        confirmRequired: action === "tombstone",
      });
      if (!result) return;
      await _patchTask(task, { visibility: visibilityPayload(action), reason: result.reason, actor: "dashboard" });
      _setFeedback(t(`taskboard.${action}d`), "ok");
    }
    await _loadBoard({ quiet: true });
  } catch (err) {
    _setFeedback(`${t("taskboard.action_failed")}: ${err.message || err}`, "error");
  } finally {
    button.disabled = false;
  }
}

function _openActionModal({ mode, datetime = false, reasonRequired = false, confirmRequired = false }) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "taskboard-modal";
    overlay.innerHTML = `
      <form class="taskboard-modal-panel">
        <header>
          <h3>${t(`taskboard.modal_${mode}`)}</h3>
          <button type="button" class="taskboard-modal-close" aria-label="${t("common.aria_close")}">&times;</button>
        </header>
        ${datetime ? `<label><span>${t("taskboard.snooze_until")}</span><input name="datetime" type="datetime-local" required value="${defaultLocalDateTime()}" /></label>` : ""}
        <label><span>${t("taskboard.reason")}${reasonRequired ? " *" : ""}</span><textarea name="reason" rows="3" ${reasonRequired ? "required" : ""}></textarea></label>
        ${confirmRequired ? `<label class="taskboard-modal-check"><input name="confirm" type="checkbox" required /><span>${t("taskboard.confirm_tombstone")}</span></label>` : ""}
        <footer>
          <button type="button" class="btn-secondary" data-cancel>${t("taskboard.cancel")}</button>
          <button type="submit" class="btn-primary">${t("taskboard.apply")}</button>
        </footer>
      </form>
    `;
    _container.appendChild(overlay);
    const form = overlay.querySelector("form");
    const close = () => {
      overlay.remove();
      resolve(null);
    };
    overlay.querySelector(".taskboard-modal-close")?.addEventListener("click", close);
    overlay.querySelector("[data-cancel]")?.addEventListener("click", close);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const fd = new FormData(form);
      const reason = String(fd.get("reason") || "").trim();
      if (reasonRequired && !reason) {
        form.elements.reason.setCustomValidity(t("taskboard.reason_required"));
        form.elements.reason.reportValidity();
        form.elements.reason.addEventListener("input", () => form.elements.reason.setCustomValidity(""), { once: true });
        return;
      }
      form.elements.reason.setCustomValidity("");
      overlay.remove();
      resolve({
        datetime: fd.get("datetime") || "",
        reason,
      });
    });
  });
}

async function _patchTask(task, body) {
  const anima = encodeURIComponent(task.anima_name);
  const taskId = encodeURIComponent(task.task_id);
  return api(`/api/task-board/${anima}/${taskId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function _onDragStart(event) {
  const card = event.target.closest(".taskboard-card");
  if (!card) return;
  _drag = { key: card.dataset.taskKey, column: card.dataset.column };
  card.classList.add("is-dragging");
  event.dataTransfer.effectAllowed = "move";
}

function _onDragOver(event) {
  const column = event.target.closest("[data-column]")?.dataset.column;
  if (_drag && column === _drag.column) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }
}

async function _onDrop(event) {
  const column = event.target.closest("[data-column]")?.dataset.column;
  if (!_drag || column !== _drag.column) return;
  event.preventDefault();
  const targetCard = event.target.closest(".taskboard-card");
  const targetKey = targetCard?.dataset.taskKey || "";
  if (targetKey === _drag.key) return;
  await _persistReorder(column, _drag.key, targetKey);
}

function _onDragEnd() {
  _container?.querySelectorAll(".is-dragging").forEach((card) => card.classList.remove("is-dragging"));
  _drag = null;
}

async function _persistReorder(column, movedKey, targetKey) {
  const columnTasks = _tasks.filter((task) => (task.column || "todo") === column);
  const movedIndex = columnTasks.findIndex((task) => taskKey(task) === movedKey);
  if (movedIndex < 0) return;
  const [moved] = columnTasks.splice(movedIndex, 1);
  const targetIndex = targetKey ? columnTasks.findIndex((task) => taskKey(task) === targetKey) : -1;
  columnTasks.splice(targetIndex >= 0 ? targetIndex : columnTasks.length, 0, moved);
  try {
    await Promise.all(columnTasks.map((task, index) => _patchTask(task, { position: (index + 1) * 1000, actor: "dashboard" })));
    _setFeedback(t("taskboard.reordered"), "ok");
    await _loadBoard({ quiet: true });
  } catch (err) {
    _setFeedback(`${t("taskboard.reorder_failed")}: ${err.message || err}`, "error");
  }
}

function _setActiveColumn(column) {
  _activeColumn = column;
  _container.querySelectorAll(".taskboard-column").forEach((el) => el.classList.toggle("is-active", el.dataset.column === column));
  _container.querySelectorAll(".taskboard-mobile-tab").forEach((el) => el.classList.toggle("is-active", el.dataset.mobileColumn === column));
}

function _ensureActiveColumnHasView() {
  if (!COLUMNS.includes(_activeColumn)) _activeColumn = "todo";
  const hasVisibleTask = (column) => _tasks.some((task) => (task.column || "todo") === column);
  if (hasVisibleTask(_activeColumn)) return;
  _activeColumn = COLUMNS.find(hasVisibleTask) || "todo";
}
function _findTask(key) { return _tasks.find((task) => taskKey(task) === key) || _allTasks.find((task) => taskKey(task) === key); }

function _summaryText(data) {
  const warnings = data?.meta?.warnings?.corrupt_task_queue_lines || 0;
  const base = t("taskboard.loaded_count", { count: _tasks.length });
  return warnings ? `${base} ${t("taskboard.corrupt_warning", { count: warnings })}` : base;
}

function _setFeedback(message, tone) {
  const el = _container?.querySelector("#taskboardFeedback");
  if (!el) return;
  el.textContent = message || "";
  el.dataset.tone = tone || "";
}

function _refreshIcons() { if (window.lucide && _container) window.lucide.createIcons({ nodes: [_container] }); }
