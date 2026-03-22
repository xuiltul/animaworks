/* ── Team Builder — 5-Screen Wizard ───────────
   Screens: 1=Template Select, 2=Member Confirm,
            3=Role Picker, 4=Complete, 5=Team Edit
   ─────────────────────────────────────────────── */

import { t, applyTranslations } from "/shared/i18n.js";
import { escapeHtml } from "../modules/state.js";
import { navigateTo } from "../modules/router.js";
import { api } from "../modules/api.js";
import {
  TEMPLATES,
  ROLES,
  getRoleById,
  getToolLabel,
  getAllTools,
  getTemplateById,
  createTeam,
  getTeam,
  addMember,
  removeMember,
  updateMemberRole,
  updateMemberName,
  updateMemberModel,
  updateMemberTitle,
  setTeamLead,
  updateTeamSettings,
  toggleMemberTool,
} from "../modules/team-data.js";

// ── Constants ────────────────────────────────

const ROLE_ICONS = {
  secretary: "\uD83D\uDCCB",
  customer_support: "\uD83C\uDFE7",
  back_office: "\uD83D\uDCC1",
  sales_assist: "\uD83D\uDCC8",
  pr_sns: "\uD83D\uDCE3",
  recruiter: "\uD83D\uDC65",
  accounting: "\uD83D\uDCB0",
  project_manager: "\uD83C\uDFAF",
  researcher: "\uD83D\uDD0D",
  content_writer: "\u270D\uFE0F",
  engineer: "\uD83D\uDCBB",
  data_analyst: "\uD83D\uDCCA",
  designer: "\uD83C\uDFA8",
  marketing: "\uD83D\uDCE1",
  hr: "\uD83E\uDDD1\u200D\uD83E\uDD1D\u200D\uD83E\uDDD1",
  legal: "\u2696\uFE0F",
  product_manager: "\uD83D\uDDE3\uFE0F",
  finance: "\uD83C\uDFDB\uFE0F",
};

const TEMPLATE_ICONS = {
  secretary: "\uD83D\uDCCB",
  customer_support: "\uD83C\uDFE7",
  sales_assist: "\uD83D\uDCC8",
  back_office: "\uD83D\uDCC1",
};

const AVATAR_COLORS = [
  "#6366f1", "#ec4899", "#f59e0b", "#10b981",
  "#3b82f6", "#8b5cf6", "#ef4444", "#14b8a6",
  "#f97316", "#06b6d4",
];

// ── State ────────────────────────────────────

const state = {
  currentStep: 1,
  selectedTemplateId: null,
  roleSelections: {},
  prefilled: false,
  createdTeam: null,
};

let _container = null;
let _availableModels = null;
let _orgInfo = null; // {departments:[], titles:[], animas:[]}

// ── Data Helpers ─────────────────────────────

async function _fetchModels() {
  if (_availableModels) return _availableModels;
  try {
    const data = await api("/api/system/available-models");
    _availableModels = data.models || [];
  } catch {
    _availableModels = [];
  }
  return _availableModels;
}

async function _fetchOrgInfo() {
  if (_orgInfo) return _orgInfo;
  try {
    _orgInfo = await api("/api/system/org-info");
  } catch {
    _orgInfo = { departments: [], titles: [], animas: [] };
  }
  return _orgInfo;
}

function _modelSelectHtml(memberId, currentModel) {
  if (!_availableModels || !_availableModels.length) return "";
  const options = _availableModels.map((m) => {
    const sel = m.id === currentModel ? "selected" : "";
    return `<option value="${escapeHtml(m.id)}" data-credential="${escapeHtml(m.credential)}" ${sel}>${escapeHtml(m.label)}</option>`;
  }).join("");
  const defaultLabel = escapeHtml(t("tb.model_default") || "System default");
  return `<select class="tb-model-select" data-member-id="${escapeHtml(memberId)}" style="font-size:0.85em;padding:2px 4px;border-radius:4px;border:1px solid var(--border-color,#ddd);max-width:200px;">
    <option value="" ${!currentModel ? "selected" : ""}>${defaultLabel}</option>
    ${options}
  </select>`;
}

function _datalistHtml(id, items) {
  const opts = items.map((v) => `<option value="${escapeHtml(v)}">`).join("");
  return `<datalist id="${escapeHtml(id)}">${opts}</datalist>`;
}

// ── Public API (Page Module) ─────────────────

export function render(container) {
  _container = container;
  state.currentStep = 1;
  state.selectedTemplateId = null;
  state.roleSelections = {};
  state.prefilled = false;
  state.createdTeam = null;
  _renderCurrentStep();
}

export function destroy() {
  _container = null;
}

// ── Step Router ──────────────────────────────

function _renderCurrentStep() {
  if (!_container) return;
  switch (state.currentStep) {
    case 1: _renderStep1(); break;
    case 2: _renderStep2(); break;
    case 3: _renderStep3(); break;
    case 4: _renderStep4(); break;
    case 5: _renderStep5(); break;
  }
  applyTranslations();
}

// ── Step Indicator ───────────────────────────

function _stepIndicator(active) {
  const totalSteps = 4;
  const clamp = Math.min(active, totalSteps);
  const parts = [];
  for (let i = 1; i <= totalSteps; i++) {
    let cls = "tb-step-dot";
    if (i === clamp) cls += " active";
    else if (i < clamp) cls += " done";
    parts.push(`<span class="${cls}"></span>`);
    if (i < totalSteps) {
      const connCls = i < clamp ? "tb-step-connector done" : "tb-step-connector";
      parts.push(`<span class="${connCls}"></span>`);
    }
  }
  return `<div class="tb-step-indicator">${parts.join("")}</div>`;
}

// ══════════════════════════════════════════════
// Screen 1: Template Selection
// ══════════════════════════════════════════════

function _renderStep1() {
  const templateCards = TEMPLATES.map((tpl) => {
    const members = tpl.members;
    const totalCount = members.reduce((s, m) => s + m.count, 0);
    const countLabel = totalCount === 1
      ? t("tb.count_one")
      : `${totalCount}${t("tb.count_suffix")}`;
    const recBadge = tpl.recommended
      ? `<span class="tb-template-badge">${escapeHtml(t("tb.recommended"))}</span>`
      : "";
    const cls = tpl.recommended ? "tb-template-card recommended" : "tb-template-card";
    const icon = TEMPLATE_ICONS[tpl.id] || "\uD83D\uDCCB";

    return `
      <div class="${cls}" data-tpl-id="${escapeHtml(tpl.id)}">
        <div class="tb-template-icon">${icon}</div>
        <div class="tb-template-info">
          <div class="tb-template-name">${escapeHtml(t(tpl.nameKey))} ${recBadge}</div>
          <div class="tb-template-desc">${escapeHtml(t(tpl.descKey))}</div>
        </div>
        <div class="tb-template-count">${escapeHtml(countLabel)}</div>
      </div>
    `;
  }).join("");

  const customCard = `
    <div class="tb-template-card" data-action="custom">
      <div class="tb-template-icon">\u2699\uFE0F</div>
      <div class="tb-template-info">
        <div class="tb-template-name">${escapeHtml(t("tb.tpl.custom"))}</div>
        <div class="tb-template-desc">${escapeHtml(t("tb.tpl.custom.desc"))}</div>
      </div>
    </div>
  `;

  _container.innerHTML = `
    <div class="tb-wizard">
      ${_stepIndicator(1)}
      <div class="tb-screen-title">${escapeHtml(t("tb.step1.title"))}</div>
      <div class="tb-screen-desc">${escapeHtml(t("tb.step1.desc"))}</div>
      <div class="tb-template-list">
        ${templateCards}
        ${customCard}
      </div>
      <div style="text-align:center;">
        <button class="tb-skip-link" data-action="skip">${escapeHtml(t("tb.skip"))}</button>
      </div>
    </div>
  `;

  _container.querySelectorAll("[data-tpl-id]").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedTemplateId = el.dataset.tplId;
      state.currentStep = 2;
      _renderCurrentStep();
    });
  });

  _container.querySelector("[data-action='custom']")?.addEventListener("click", () => {
    state.selectedTemplateId = null;
    state.roleSelections = {};
    state.prefilled = false;
    state.currentStep = 3;
    _renderCurrentStep();
  });

  _container.querySelector("[data-action='skip']")?.addEventListener("click", () => {
    navigateTo("#/");
  });
}

// ══════════════════════════════════════════════
// Screen 2: Member Confirmation
// ══════════════════════════════════════════════

function _renderStep2() {
  const tpl = getTemplateById(state.selectedTemplateId);
  if (!tpl) { state.currentStep = 1; _renderCurrentStep(); return; }

  const memberRows = tpl.members.map((m) => {
    const role = getRoleById(m.roleId);
    if (!role) return "";
    const tools = role.defaultTools.map((tid) => escapeHtml(getToolLabel(tid))).join(" / ");
    const icon = ROLE_ICONS[m.roleId] || "\uD83D\uDC64";
    return `
      <div class="tb-member-card">
        <div class="tb-member-avatar" style="background:${AVATAR_COLORS[0]}">${icon}</div>
        <div class="tb-member-info">
          <div class="tb-member-name">${escapeHtml(t(role.nameKey))} x ${m.count}</div>
          <div class="tb-member-role">${escapeHtml(t("tb.tools"))}: ${tools}</div>
        </div>
      </div>
    `;
  }).join("");

  _container.innerHTML = `
    <div class="tb-wizard">
      ${_stepIndicator(2)}
      <div class="tb-screen-title">${escapeHtml(t(tpl.nameKey))}${escapeHtml(t("tb.step2.title_suffix"))}</div>
      <div class="tb-screen-desc">${escapeHtml(t("tb.step2.desc"))}</div>
      <div class="tb-member-list">
        ${memberRows}
      </div>
      <div class="tb-wizard-nav">
        <button class="btn-secondary" data-action="back">${escapeHtml(t("tb.back"))}</button>
        <div class="tb-wizard-nav-right">
          <button class="btn-secondary" data-action="customize">${escapeHtml(t("tb.step2.customize"))}</button>
          <button class="btn-primary" data-action="create">${escapeHtml(t("tb.step2.create"))}</button>
        </div>
      </div>
    </div>
  `;

  _container.querySelector("[data-action='back']")?.addEventListener("click", () => {
    state.currentStep = 1;
    _renderCurrentStep();
  });

  _container.querySelector("[data-action='customize']")?.addEventListener("click", () => {
    state.roleSelections = {};
    for (const m of tpl.members) {
      state.roleSelections[m.roleId] = m.count;
    }
    state.prefilled = true;
    state.currentStep = 3;
    _renderCurrentStep();
  });

  _container.querySelector("[data-action='create']")?.addEventListener("click", () => {
    const selections = tpl.members.map((m) => ({ roleId: m.roleId, count: m.count }));
    state.createdTeam = createTeam(selections);
    state.currentStep = 4;
    _renderCurrentStep();
  });
}

// ══════════════════════════════════════════════
// Screen 3: Role Pickup
// ══════════════════════════════════════════════

function _renderStep3() {
  const roleItems = ROLES.map((role) => {
    const selected = state.roleSelections[role.id] != null;
    const count = state.roleSelections[role.id] || 1;
    const icon = ROLE_ICONS[role.id] || "\uD83D\uDC64";
    const cls = selected ? "tb-role-item selected" : "tb-role-item";
    const toolTags = role.defaultTools.map((tid) =>
      `<span class="tb-tool-tag">${escapeHtml(getToolLabel(tid))}</span>`
    ).join("");

    const options = [1, 2, 3].map((n) =>
      `<option value="${n}" ${n === count ? "selected" : ""}>${n}${escapeHtml(t("tb.count_suffix"))}</option>`
    ).join("");

    return `
      <div class="${cls}" data-role-id="${escapeHtml(role.id)}">
        <div class="tb-role-checkbox">\u2713</div>
        <div class="tb-role-info">
          <div class="tb-role-name">${icon} ${escapeHtml(t(role.nameKey))}</div>
          <div class="tb-role-desc">${escapeHtml(t(role.descKey))}</div>
          <div class="tb-role-tools">${toolTags}</div>
        </div>
        <select class="tb-role-count-select" data-count-for="${escapeHtml(role.id)}" ${selected ? "" : 'style="visibility:hidden"'}>
          ${options}
        </select>
      </div>
    `;
  }).join("");

  const hasSelection = Object.keys(state.roleSelections).length > 0;

  _container.innerHTML = `
    <div class="tb-wizard">
      ${_stepIndicator(3)}
      <div class="tb-screen-title">${escapeHtml(t("tb.step3.title"))}</div>
      <div class="tb-screen-desc">${escapeHtml(t("tb.step3.desc"))}</div>
      <div class="tb-role-grid">
        ${roleItems}
      </div>
      <div class="tb-wizard-nav">
        <button class="btn-secondary" data-action="back">${escapeHtml(t("tb.back"))}</button>
        <button class="btn-primary" data-action="create" ${hasSelection ? "" : "disabled"}>${escapeHtml(t("tb.step3.create"))}</button>
      </div>
    </div>
  `;

  _container.querySelectorAll(".tb-role-item").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "OPTION") return;
      const roleId = el.dataset.roleId;
      if (state.roleSelections[roleId] != null) {
        delete state.roleSelections[roleId];
      } else {
        state.roleSelections[roleId] = 1;
      }
      _renderCurrentStep();
    });
  });

  _container.querySelectorAll(".tb-role-count-select").forEach((sel) => {
    sel.addEventListener("change", (e) => {
      const roleId = sel.dataset.countFor;
      if (state.roleSelections[roleId] != null) {
        state.roleSelections[roleId] = parseInt(e.target.value, 10);
      }
    });
    sel.addEventListener("click", (e) => e.stopPropagation());
  });

  _container.querySelector("[data-action='back']")?.addEventListener("click", () => {
    if (state.selectedTemplateId) {
      state.currentStep = 2;
    } else {
      state.currentStep = 1;
    }
    _renderCurrentStep();
  });

  _container.querySelector("[data-action='create']")?.addEventListener("click", () => {
    const selections = Object.entries(state.roleSelections).map(([roleId, count]) => ({
      roleId,
      count,
    }));
    if (selections.length === 0) return;
    state.createdTeam = createTeam(selections);
    state.currentStep = 4;
    _renderCurrentStep();
  });
}

// ══════════════════════════════════════════════
// Screen 4: Creation Complete
// ══════════════════════════════════════════════

async function _renderStep4() {
  const team = state.createdTeam;
  if (!team || !team.members) {
    state.currentStep = 1;
    _renderCurrentStep();
    return;
  }

  // Fetch data in parallel
  let existingNames = new Set();
  try {
    const animas = await api("/api/animas");
    existingNames = new Set(animas.map((a) => a.name));
  } catch { /* ignore */ }
  await Promise.all([_fetchModels(), _fetchOrgInfo()]);

  const org = _orgInfo || { departments: [], titles: [], animas: [] };

  // ── Team-level settings ──
  const deptDatalist = _datalistHtml("tbDeptList", org.departments);
  const supervisorOptions = org.animas.map((a) =>
    `<option value="${escapeHtml(a.name)}" ${a.name === (team.reportTo || "") ? "selected" : ""}>${escapeHtml(a.name)}${a.department ? ` (${escapeHtml(a.department)})` : ""}</option>`
  ).join("");

  const teamSettingsHtml = `
    <div class="card" style="margin-bottom:1rem;">
      <div class="card-body" style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;">
        <div style="display:flex;align-items:center;gap:6px;">
          <label style="font-size:0.85em;color:var(--aw-color-text-muted,#888);white-space:nowrap;">${escapeHtml(t("tb.department") || "Department")}:</label>
          <input type="text" id="tbDepartment" list="tbDeptList" value="${escapeHtml(team.department || "")}" placeholder="${escapeHtml(t("tb.department_placeholder") || "Select or enter new")}" style="padding:4px 8px;border:1px solid var(--border-color,#ddd);border-radius:4px;width:180px;">
          ${deptDatalist}
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
          <label style="font-size:0.85em;color:var(--aw-color-text-muted,#888);white-space:nowrap;">${escapeHtml(t("tb.report_to") || "Reports to")}:</label>
          <select id="tbReportTo" style="padding:4px 8px;border:1px solid var(--border-color,#ddd);border-radius:4px;max-width:200px;">
            <option value="">${escapeHtml(t("tb.report_to_owner") || "Owner (direct)")}</option>
            ${supervisorOptions}
          </select>
        </div>
      </div>
    </div>
  `;

  // ── Member cards ──
  const titleDatalist = _datalistHtml("tbTitleList", org.titles);

  const memberCards = team.members.map((m, idx) => {
    const role = getRoleById(m.roleId);
    const roleName = role ? t(role.nameKey) : m.roleId;
    const color = AVATAR_COLORS[idx % AVATAR_COLORS.length];
    const initial = escapeHtml(m.displayName.charAt(0));
    const toolTags = m.tools.map((tid) =>
      `<span class="tb-tool-tag">${escapeHtml(getToolLabel(tid))}</span>`
    ).join("");

    const animaName = m.displayName.toLowerCase().replace(/\s+/g, "_").replace(/-/g, "_").replace(/[^a-z0-9_]/g, "");
    const conflict = existingNames.has(animaName);
    const conflictBadge = conflict
      ? `<span style="color:#dc2626;font-size:0.8em;margin-left:4px;">⚠ ${escapeHtml(t("tb.name_conflict") || "exists")}</span>`
      : "";

    const modelSelect = _modelSelectHtml(m.id, m.model || "");
    const leadChecked = m.isLead ? "checked" : "";

    return `
      <div class="tb-member-card" style="${m.isLead ? "border-left:3px solid #f59e0b;" : ""}">
        <div class="tb-member-avatar" style="background:${color}">${initial}</div>
        <div class="tb-member-info">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span class="tb-member-role-label" style="color:var(--aw-color-text-muted,#888);font-size:0.85em;">${escapeHtml(roleName)}:</span>
            <input type="text" class="tb-name-input" data-member-id="${escapeHtml(m.id)}" value="${escapeHtml(m.displayName)}" placeholder="${escapeHtml(t("tb.step5.name_placeholder") || "Name")}" style="font-weight:600;font-size:1rem;padding:2px 6px;border:1px solid ${conflict ? "#dc2626" : "var(--border-color,#ddd)"};border-radius:4px;width:140px;background:var(--aw-color-bg-secondary,#f8f8f8);">
            ${conflictBadge}
            <label style="display:flex;align-items:center;gap:3px;font-size:0.85em;color:#f59e0b;cursor:pointer;margin-left:auto;">
              <input type="radio" name="tbLead" class="tb-lead-radio" data-member-id="${escapeHtml(m.id)}" ${leadChecked}>
              ${escapeHtml(t("tb.lead") || "Lead")}
            </label>
          </div>
          <div style="display:flex;align-items:center;gap:6px;margin-top:4px;flex-wrap:wrap;">
            <span style="color:var(--aw-color-text-muted,#888);font-size:0.85em;">${escapeHtml(t("tb.title") || "Title")}:</span>
            <input type="text" class="tb-title-input" data-member-id="${escapeHtml(m.id)}" list="tbTitleList" value="${escapeHtml(m.title || "")}" placeholder="${escapeHtml(t("tb.title_placeholder") || "Select or enter new")}" style="font-size:0.85em;padding:2px 6px;border:1px solid var(--border-color,#ddd);border-radius:4px;width:160px;">
            <span style="color:var(--aw-color-text-muted,#888);font-size:0.85em;">${escapeHtml(t("tb.model") || "Model")}:</span>
            ${modelSelect}
          </div>
          <div class="tb-member-tools-row">${toolTags}</div>
        </div>
      </div>
    `;
  }).join("");

  const hasConflicts = team.members.some((m) => {
    const animaName = m.displayName.toLowerCase().replace(/\s+/g, "_").replace(/-/g, "_").replace(/[^a-z0-9_]/g, "");
    return existingNames.has(animaName);
  });

  _container.innerHTML = `
    <div class="tb-wizard">
      ${_stepIndicator(4)}
      <div class="tb-complete-icon">\uD83C\uDF89</div>
      <div class="tb-complete-title">${escapeHtml(t("tb.step4.title"))}</div>
      ${hasConflicts ? `<div style="text-align:center;color:#dc2626;font-size:0.9em;margin-bottom:0.5rem;">⚠ ${escapeHtml(t("tb.name_conflict_warn") || "Some names already exist. Please rename before deploying.")}</div>` : ""}
      ${teamSettingsHtml}
      ${titleDatalist}
      <div class="tb-member-list">
        ${memberCards}
      </div>
      <div id="tbDeployStatus" style="display:none;text-align:center;margin:1rem 0;"></div>
      <div class="tb-wizard-nav" style="justify-content:center;">
        <button class="btn-primary" data-action="deploy" style="background:#16a34a;">${escapeHtml(t("tb.step4.deploy") || "Deploy as Animas")}</button>
        <button class="btn-secondary" data-action="edit">${escapeHtml(t("tb.step4.edit"))}</button>
      </div>
    </div>
  `;

  // ── Event handlers ──

  // Team-level: department
  document.getElementById("tbDepartment")?.addEventListener("change", (e) => {
    updateTeamSettings(e.target.value.trim(), undefined);
    state.createdTeam = getTeam();
  });

  // Team-level: reportTo
  document.getElementById("tbReportTo")?.addEventListener("change", (e) => {
    updateTeamSettings(undefined, e.target.value);
    state.createdTeam = getTeam();
  });

  // Name editing
  _container.querySelectorAll(".tb-name-input").forEach((input) => {
    input.addEventListener("change", (e) => {
      const newName = e.target.value.trim();
      if (newName) {
        updateMemberName(input.dataset.memberId, newName);
        state.createdTeam = getTeam();
        _renderStep4();
      }
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    });
  });

  // Title editing
  _container.querySelectorAll(".tb-title-input").forEach((input) => {
    input.addEventListener("change", (e) => {
      updateMemberTitle(input.dataset.memberId, e.target.value);
      state.createdTeam = getTeam();
    });
  });

  // Lead radio
  _container.querySelectorAll(".tb-lead-radio").forEach((radio) => {
    radio.addEventListener("change", () => {
      setTeamLead(radio.dataset.memberId);
      state.createdTeam = getTeam();
      _renderStep4();
    });
  });

  // Model selection
  _container.querySelectorAll(".tb-model-select").forEach((sel) => {
    sel.addEventListener("change", (e) => {
      const opt = sel.options[sel.selectedIndex];
      const credential = opt?.dataset?.credential || "";
      updateMemberModel(sel.dataset.memberId, e.target.value, credential);
      state.createdTeam = getTeam();
    });
    sel.addEventListener("click", (e) => e.stopPropagation());
  });

  _container.querySelector("[data-action='deploy']")?.addEventListener("click", _deployTeam);

  _container.querySelector("[data-action='edit']")?.addEventListener("click", () => {
    state.currentStep = 5;
    _renderCurrentStep();
  });
}

async function _deployTeam() {
  const team = state.createdTeam;
  if (!team || !team.members || !team.members.length) return;

  const deployBtn = _container.querySelector("[data-action='deploy']");
  const statusEl = document.getElementById("tbDeployStatus");
  if (deployBtn) { deployBtn.disabled = true; deployBtn.textContent = t("tb.deploying") || "Deploying..."; }
  if (statusEl) { statusEl.style.display = ""; statusEl.innerHTML = `<span style="color:var(--aw-color-text-muted,#888);">${escapeHtml(t("tb.deploying") || "Deploying...")}</span>`; }

  try {
    let existingNames = new Set();
    try {
      const animas = await api("/api/animas");
      existingNames = new Set(animas.map((a) => a.name));
    } catch { /* ignore */ }

    const toAnimaName = (displayName) =>
      displayName.toLowerCase().replace(/\s+/g, "_").replace(/-/g, "_").replace(/[^a-z0-9_]/g, "");

    const newMembers = team.members.filter((m) => !existingNames.has(toAnimaName(m.displayName)));
    const skipped = team.members.filter((m) => existingNames.has(toAnimaName(m.displayName)));

    if (newMembers.length === 0) {
      if (statusEl) {
        statusEl.innerHTML = `<span style="color:#dc2626;">${escapeHtml(t("tb.deploy_all_exist") || "All members already exist. Please rename first.")}</span>`;
      }
      if (deployBtn) { deployBtn.disabled = false; deployBtn.textContent = t("tb.step4.deploy") || "Deploy as Animas"; }
      return;
    }

    const result = await api("/api/teams/deploy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        members: newMembers,
        department: team.department || "",
        reportTo: team.reportTo || "",
      }),
    });

    const created = result.created || [];
    const errors = result.errors || [];

    for (const m of skipped) {
      errors.push(`${m.displayName}: ${t("tb.deploy_skipped_exists") || "already exists — skipped"}`);
    }

    if (created.length > 0) {
      const names = created.map(c => c.displayName || c.name).join(", ");
      if (statusEl) {
        statusEl.innerHTML = `<span style="color:#16a34a;font-weight:bold;">${escapeHtml(t("tb.deploy_success") || "Deploy complete!")}</span><br><span style="font-size:0.9em;">${escapeHtml(names)}</span>`;
      }
      if (deployBtn) { deployBtn.textContent = t("tb.deploy_done") || "Deployed!"; }

      setTimeout(() => {
        if (deployBtn) {
          deployBtn.textContent = t("tb.deploy_go_animas") || "Go to Anima Management";
          deployBtn.disabled = false;
          deployBtn.onclick = () => navigateTo("#/animas");
        }
      }, 1500);
    }

    if (errors.length > 0) {
      const errHtml = errors.map(e => escapeHtml(e)).join("<br>");
      if (statusEl) {
        statusEl.innerHTML += `<br><span style="color:#dc2626;font-size:0.85em;">${errHtml}</span>`;
      }
    }

    if (created.length === 0 && errors.length > 0) {
      if (deployBtn) { deployBtn.disabled = false; deployBtn.textContent = t("tb.step4.deploy") || "Deploy as Animas"; }
    }

    // Invalidate org info cache so next render picks up new animas
    _orgInfo = null;

  } catch (err) {
    if (statusEl) {
      statusEl.innerHTML = `<span style="color:#dc2626;">${escapeHtml(t("tb.deploy_error") || "Deploy failed")}: ${escapeHtml(err.message)}</span>`;
    }
    if (deployBtn) { deployBtn.disabled = false; deployBtn.textContent = t("tb.step4.deploy") || "Deploy as Animas"; }
  }
}

// ══════════════════════════════════════════════
// Screen 5: Team Edit
// ══════════════════════════════════════════════

async function _renderStep5() {
  const team = state.createdTeam || getTeam();
  if (!team || !team.members) {
    state.currentStep = 1;
    _renderCurrentStep();
    return;
  }
  state.createdTeam = team;
  await Promise.all([_fetchModels(), _fetchOrgInfo()]);

  const org = _orgInfo || { departments: [], titles: [], animas: [] };
  const deptDatalist = _datalistHtml("tbDeptList5", org.departments);
  const titleDatalist = _datalistHtml("tbTitleList5", org.titles);
  const supervisorOptions = org.animas.map((a) =>
    `<option value="${escapeHtml(a.name)}" ${a.name === (team.reportTo || "") ? "selected" : ""}>${escapeHtml(a.name)}${a.department ? ` (${escapeHtml(a.department)})` : ""}</option>`
  ).join("");

  _container.innerHTML = `
    <div class="tb-wizard">
      <div class="tb-screen-title">${escapeHtml(t("tb.step5.title"))}</div>
      <div class="tb-screen-desc">${escapeHtml(t("tb.step5.desc"))}</div>
      <div class="card" style="margin-bottom:1rem;">
        <div class="card-body" style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;">
          <div style="display:flex;align-items:center;gap:6px;">
            <label style="font-size:0.85em;color:var(--aw-color-text-muted,#888);white-space:nowrap;">${escapeHtml(t("tb.department") || "Department")}:</label>
            <input type="text" id="tbDepartment5" list="tbDeptList5" value="${escapeHtml(team.department || "")}" placeholder="${escapeHtml(t("tb.department_placeholder") || "Select or enter new")}" style="padding:4px 8px;border:1px solid var(--border-color,#ddd);border-radius:4px;width:180px;">
            ${deptDatalist}
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <label style="font-size:0.85em;color:var(--aw-color-text-muted,#888);white-space:nowrap;">${escapeHtml(t("tb.report_to") || "Reports to")}:</label>
            <select id="tbReportTo5" style="padding:4px 8px;border:1px solid var(--border-color,#ddd);border-radius:4px;max-width:200px;">
              <option value="">${escapeHtml(t("tb.report_to_owner") || "Owner (direct)")}</option>
              ${supervisorOptions}
            </select>
          </div>
        </div>
      </div>
      ${titleDatalist}
      <div class="tb-edit-section" id="tbEditMembers"></div>
      <button class="tb-add-member-btn" id="tbAddMemberBtn">+ ${escapeHtml(t("tb.step5.add_member"))}</button>
      <div id="tbAddMemberPicker" style="display:none;"></div>
      <div class="tb-wizard-nav" style="margin-top:16px;">
        <button class="btn-secondary" data-action="back">${escapeHtml(t("tb.back"))}</button>
        <button class="btn-primary" data-action="done">${escapeHtml(t("tb.step5.done"))}</button>
      </div>
    </div>
  `;

  // Team-level events
  document.getElementById("tbDepartment5")?.addEventListener("change", (e) => {
    updateTeamSettings(e.target.value.trim(), undefined);
    state.createdTeam = getTeam();
  });
  document.getElementById("tbReportTo5")?.addEventListener("change", (e) => {
    updateTeamSettings(undefined, e.target.value);
    state.createdTeam = getTeam();
  });

  _renderEditMembers();

  document.getElementById("tbAddMemberBtn")?.addEventListener("click", () => {
    _toggleAddPicker();
  });

  _container.querySelector("[data-action='back']")?.addEventListener("click", () => {
    state.currentStep = 4;
    _renderCurrentStep();
  });

  _container.querySelector("[data-action='done']")?.addEventListener("click", () => {
    navigateTo("#/");
  });
}

/** Render the editable member list inside #tbEditMembers */
function _renderEditMembers() {
  const section = document.getElementById("tbEditMembers");
  if (!section) return;

  const team = state.createdTeam;
  if (!team || !team.members) { section.innerHTML = ""; return; }

  const allTools = getAllTools();

  const cards = team.members.map((m, idx) => {
    const color = AVATAR_COLORS[idx % AVATAR_COLORS.length];
    const initial = escapeHtml(m.displayName.charAt(0));

    const roleSelect = ROLES.map((r) =>
      `<option value="${escapeHtml(r.id)}" ${r.id === m.roleId ? "selected" : ""}>${escapeHtml(t(r.nameKey))}</option>`
    ).join("");

    const toolToggles = allTools.map((tool) => {
      const isActive = m.tools.includes(tool.id);
      return `<button class="tb-tool-toggle ${isActive ? "active" : ""}" data-tool="${escapeHtml(tool.id)}" data-member-id="${escapeHtml(m.id)}">${isActive ? "\u2713 " : ""}${escapeHtml(tool.label)}</button>`;
    }).join("");

    const modelSelect = _modelSelectHtml(m.id, m.model || "");
    const leadChecked = m.isLead ? "checked" : "";

    return `
      <div class="tb-member-card" style="${m.isLead ? "border-left:3px solid #f59e0b;" : ""}">
        <div class="tb-member-avatar" style="background:${color}">${initial}</div>
        <div class="tb-member-info">
          <div style="display:flex;align-items:center;gap:6px;">
            <input type="text" class="tb-name-input" data-member-id="${escapeHtml(m.id)}" value="${escapeHtml(m.displayName)}" placeholder="${escapeHtml(t("tb.step5.name_placeholder") || "Name")}" style="font-weight:600;font-size:1rem;padding:2px 6px;border:1px solid var(--border-color,#ddd);border-radius:4px;width:140px;background:var(--aw-color-bg-secondary,#f8f8f8);">
            <label style="display:flex;align-items:center;gap:3px;font-size:0.85em;color:#f59e0b;cursor:pointer;">
              <input type="radio" name="tbLead5" class="tb-lead-radio" data-member-id="${escapeHtml(m.id)}" ${leadChecked}>
              ${escapeHtml(t("tb.lead") || "Lead")}
            </label>
          </div>
          <div style="display:flex;align-items:center;gap:6px;margin-top:6px;flex-wrap:wrap;">
            <select class="tb-role-select" data-member-id="${escapeHtml(m.id)}">${roleSelect}</select>
            <input type="text" class="tb-title-input" data-member-id="${escapeHtml(m.id)}" list="tbTitleList5" value="${escapeHtml(m.title || "")}" placeholder="${escapeHtml(t("tb.title_placeholder") || "Title")}" style="font-size:0.85em;padding:2px 6px;border:1px solid var(--border-color,#ddd);border-radius:4px;width:140px;">
            <span style="color:var(--aw-color-text-muted,#888);font-size:0.85em;">${escapeHtml(t("tb.model") || "Model")}:</span>
            ${modelSelect}
          </div>
          <div class="tb-member-tools-row" style="margin-top:6px;">${toolToggles}</div>
        </div>
        <div class="tb-member-actions">
          <button class="tb-btn-icon danger" data-delete-id="${escapeHtml(m.id)}" title="${escapeHtml(t("tb.step5.delete"))}">\u2715</button>
        </div>
      </div>
    `;
  }).join("");

  section.innerHTML = `<div class="tb-member-list">${cards}</div>`;

  // Name changes
  section.querySelectorAll(".tb-name-input").forEach((input) => {
    input.addEventListener("change", (e) => {
      const newName = e.target.value.trim();
      if (newName) {
        updateMemberName(input.dataset.memberId, newName);
        state.createdTeam = getTeam();
        const card = input.closest(".tb-member-card");
        const avatar = card?.querySelector(".tb-member-avatar");
        if (avatar) avatar.textContent = newName.charAt(0);
      }
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    });
  });

  // Title changes
  section.querySelectorAll(".tb-title-input").forEach((input) => {
    input.addEventListener("change", (e) => {
      updateMemberTitle(input.dataset.memberId, e.target.value);
      state.createdTeam = getTeam();
    });
  });

  // Lead radio
  section.querySelectorAll(".tb-lead-radio").forEach((radio) => {
    radio.addEventListener("change", () => {
      setTeamLead(radio.dataset.memberId);
      state.createdTeam = getTeam();
      _renderEditMembers();
    });
  });

  // Role changes
  section.querySelectorAll(".tb-role-select").forEach((sel) => {
    sel.addEventListener("change", (e) => {
      updateMemberRole(sel.dataset.memberId, e.target.value);
      state.createdTeam = getTeam();
      _renderEditMembers();
    });
  });

  // Model changes
  section.querySelectorAll(".tb-model-select").forEach((sel) => {
    sel.addEventListener("change", (e) => {
      const opt = sel.options[sel.selectedIndex];
      const credential = opt?.dataset?.credential || "";
      updateMemberModel(sel.dataset.memberId, e.target.value, credential);
      state.createdTeam = getTeam();
    });
    sel.addEventListener("click", (e) => e.stopPropagation());
  });

  // Tool toggles
  section.querySelectorAll(".tb-tool-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      toggleMemberTool(btn.dataset.memberId, btn.dataset.tool);
      state.createdTeam = getTeam();
      _renderEditMembers();
    });
  });

  // Delete
  section.querySelectorAll("[data-delete-id]").forEach((btn) => {
    btn.addEventListener("click", () => {
      removeMember(btn.dataset.deleteId);
      state.createdTeam = getTeam();
      _renderEditMembers();
    });
  });
}

/** Toggle the add-member role picker */
function _toggleAddPicker() {
  const picker = document.getElementById("tbAddMemberPicker");
  if (!picker) return;

  if (picker.style.display !== "none") {
    picker.style.display = "none";
    picker.innerHTML = "";
    return;
  }

  let html = '<div class="card" style="margin-top:12px;"><div class="card-header">' +
    escapeHtml(t("tb.step5.pick_role")) +
    '</div><div class="card-body" style="display:flex;flex-wrap:wrap;gap:8px;">';

  for (const role of ROLES) {
    const icon = ROLE_ICONS[role.id] || "";
    html += `<button class="btn-secondary" data-add-role="${escapeHtml(role.id)}">${icon} ${escapeHtml(t(role.nameKey))}</button>`;
  }

  html += "</div></div>";
  picker.innerHTML = html;
  picker.style.display = "block";

  picker.querySelectorAll("[data-add-role]").forEach((btn) => {
    btn.addEventListener("click", () => {
      addMember(btn.dataset.addRole);
      state.createdTeam = getTeam();
      picker.style.display = "none";
      picker.innerHTML = "";
      _renderEditMembers();
    });
  });
}
