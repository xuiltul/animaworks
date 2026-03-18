/* ── Activity Report Page ─────────────────── */

import { api, apiStream } from "../modules/api.js";
import { renderMarkdown, escapeHtml } from "../modules/state.js";
import { t } from "/shared/i18n.js";

let _abortCtrl = null;

export async function render(container) {
  const todayStr = new Date().toISOString().slice(0, 10);

  container.innerHTML = `
    <div class="activity-report-page">
      <div class="page-header">
        <h2>${t("report.page_title")}</h2>
      </div>

      <div class="report-controls card">
        <div class="card-body report-controls-inner">
          <div class="report-control-row">
            <label class="report-label" for="reportDate">${t("report.date_label")}</label>
            <input type="date" id="reportDate" class="report-date-input" value="${todayStr}" max="${todayStr}" />
          </div>
          <div class="report-control-row">
            <label class="report-label" for="reportModel">${t("report.model_label")}</label>
            <select id="reportModel" class="report-model-select">
              <option value="">${t("report.model_default")}</option>
            </select>
          </div>
          <div class="report-btn-row">
            <button id="reportGenBtn" class="btn btn-primary">${t("report.generate_btn")}</button>
            <button id="reportRegenBtn" class="btn btn-secondary" style="display:none">${t("report.regenerate_btn")}</button>
          </div>
        </div>
      </div>

      <div id="reportStatus" class="report-status" style="display:none"></div>

      <div id="reportStructured" class="report-section" style="display:none">
        <div class="card">
          <div class="card-header">${t("report.structured_title")}</div>
          <div class="card-body" id="structuredContent"></div>
        </div>
      </div>

      <div id="reportAnimas" class="report-section" style="display:none">
        <div class="card">
          <div class="card-header">${t("report.anima_table_title")}</div>
          <div class="card-body" id="animaTableContent"></div>
        </div>
      </div>

      <div id="reportNarrative" class="report-section" style="display:none">
        <div class="card">
          <div class="card-header">${t("report.narrative_title")}</div>
          <div class="card-body markdown-body" id="narrativeContent"></div>
        </div>
      </div>
    </div>
  `;

  _loadModels(container);

  container.querySelector("#reportGenBtn").addEventListener("click", () => _onGenerate(container, false));
  container.querySelector("#reportRegenBtn").addEventListener("click", () => _onGenerate(container, true));
}

export function destroy() {
  if (_abortCtrl) {
    _abortCtrl.abort();
    _abortCtrl = null;
  }
}

// ── Model dropdown ───────────────────────────

async function _loadModels(container) {
  try {
    const data = await api("/api/activity-report/models");
    const sel = container.querySelector("#reportModel");
    if (!sel) return;
    const defaultModel = data.default_model || "";
    for (const m of data.available_models || []) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      if (m.id === defaultModel) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch {
    // models dropdown stays with default only
  }
}

// ── Generate ─────────────────────────────────

async function _onGenerate(container, forceRegenerate) {
  const dateInput = container.querySelector("#reportDate");
  const modelSelect = container.querySelector("#reportModel");
  const genBtn = container.querySelector("#reportGenBtn");
  const regenBtn = container.querySelector("#reportRegenBtn");
  const statusEl = container.querySelector("#reportStatus");

  if (!dateInput || !genBtn) return;

  const reportDate = dateInput.value;
  const model = modelSelect ? modelSelect.value : "";

  genBtn.disabled = true;
  regenBtn.disabled = true;
  regenBtn.style.display = "none";
  statusEl.style.display = "block";
  statusEl.className = "report-status report-status-loading";
  statusEl.textContent = t("report.generating");

  _abortCtrl = new AbortController();

  let resultData = null;

  try {
    await apiStream("/api/activity-report/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        date: reportDate,
        model: model,
        force_regenerate: forceRegenerate,
      }),
      signal: _abortCtrl.signal,
      onProgress: (data) => {
        const phase = data.phase || "";
        const key = `report.phase_${phase}`;
        statusEl.textContent = t(key) || t("report.generating");
      },
      onResult: (data) => {
        resultData = data;
      },
      onError: (data) => {
        statusEl.className = "report-status report-status-error";
        statusEl.textContent = t("report.generate_failed") + ": " + (data.message || data.code || "");
      },
    });

    if (resultData) {
      _renderReport(container, resultData);
      statusEl.className = "report-status report-status-success";
      statusEl.textContent = resultData.cached
        ? t("report.loaded_from_cache")
        : t("report.generated_success");
      regenBtn.style.display = "inline-block";
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    statusEl.className = "report-status report-status-error";
    statusEl.textContent = t("report.generate_failed") + ": " + (err.message || err);
  } finally {
    genBtn.disabled = false;
    regenBtn.disabled = false;
    _abortCtrl = null;
  }
}

// ── Render report ────────────────────────────

function _renderReport(container, data) {
  const structured = data.structured;

  // Summary cards
  const structuredSection = container.querySelector("#reportStructured");
  const structuredContent = container.querySelector("#structuredContent");
  if (structuredSection && structuredContent) {
    structuredSection.style.display = "block";
    structuredContent.innerHTML = `
      <div class="report-stat-grid">
        <div class="stat-card">
          <div class="stat-value">${structured.active_anima_count}</div>
          <div class="stat-label">${t("report.stat_active_animas")}</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${_fmt(structured.total_entries)}</div>
          <div class="stat-label">${t("report.stat_total_entries")}</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${_fmt(structured.total_messages)}</div>
          <div class="stat-label">${t("report.stat_total_messages")}</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${structured.total_errors}</div>
          <div class="stat-label">${t("report.stat_total_errors")}</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${structured.total_tasks_done}</div>
          <div class="stat-label">${t("report.stat_tasks_done")}</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${structured.disabled_anima_count}</div>
          <div class="stat-label">${t("report.stat_disabled")}</div>
        </div>
      </div>
    `;
  }

  // Anima table
  const animaSection = container.querySelector("#reportAnimas");
  const animaContent = container.querySelector("#animaTableContent");
  if (animaSection && animaContent && structured.animas && structured.animas.length > 0) {
    animaSection.style.display = "block";

    const sorted = [...structured.animas].sort((a, b) => b.total_entries - a.total_entries);
    let rows = "";
    for (const a of sorted) {
      const statusCls = a.enabled ? "anima-enabled" : "anima-disabled";
      const statusLabel = a.enabled ? "●" : "○";
      rows += `<tr>
        <td><span class="${statusCls}">${statusLabel}</span> ${escapeHtml(a.name)}</td>
        <td class="num">${_fmt(a.total_entries)}</td>
        <td class="num">${a.messages_sent + a.messages_received}</td>
        <td class="num">${a.errors}</td>
        <td class="num">${a.tasks_done}/${a.tasks_total}</td>
        <td>${escapeHtml(a.role || "-")}</td>
      </tr>`;
    }

    animaContent.innerHTML = `
      <div class="report-table-wrap">
        <table class="report-table">
          <thead>
            <tr>
              <th>${t("report.col_name")}</th>
              <th class="num">${t("report.col_entries")}</th>
              <th class="num">${t("report.col_messages")}</th>
              <th class="num">${t("report.col_errors")}</th>
              <th class="num">${t("report.col_tasks")}</th>
              <th>${t("report.col_role")}</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }

  // Narrative (LLM summary)
  const narrativeSection = container.querySelector("#reportNarrative");
  const narrativeContent = container.querySelector("#narrativeContent");
  if (narrativeSection && narrativeContent) {
    if (data.narrative_md) {
      narrativeSection.style.display = "block";
      narrativeContent.innerHTML = renderMarkdown(data.narrative_md);
    } else {
      narrativeSection.style.display = "none";
    }
  }
}

function _fmt(n) {
  return Number(n).toLocaleString();
}
