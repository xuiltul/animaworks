// ── Anima detail tab: Process ────────────────
// Single-anima process status + actions (absorbed from pages/processes.js).

import { api } from "../../modules/api.js";
import { escapeHtml, timeStr } from "../../modules/state.js";
import {
  healthIndicatorHtml,
  statusBadgeHtml,
  formatUptime,
  processActionButtonsHtml,
  bindProcessActionButtons,
  fetchProcessMap,
} from "../../modules/animas.js";
import { t } from "/shared/i18n.js";

let _refreshInterval = null;
let _animaName = null;
let _container = null;

/**
 * @param {HTMLElement} container
 * @param {{ animaName: string }} opts
 */
export function render(container, { animaName } = {}) {
  _container = container;
  _animaName = animaName;

  container.innerHTML = `
    <div id="animaProcessTabContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  _load();
  _refreshInterval = setInterval(_load, 10000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
  _container = null;
  _animaName = null;
}

async function _load() {
  const content = document.getElementById("animaProcessTabContent");
  if (!content || !_animaName) return;

  try {
    let proc = null;
    const processes = await fetchProcessMap();
    if (processes[_animaName]) {
      proc = { name: _animaName, ...processes[_animaName] };
    } else {
      // Fallback: look up this anima in /api/animas
      try {
        const animas = await api("/api/animas");
        const found = animas.find((a) => a.name === _animaName);
        if (found) proc = found;
      } catch {
        /* ignore */
      }
    }

    if (!proc) {
      content.innerHTML = `<div class="loading-placeholder">${t("processes.no_processes")}</div>`;
      return;
    }

    const status = proc.status || "unknown";
    const missedPings = proc.missed_pings || 0;
    const health = healthIndicatorHtml(status, missedPings);
    const uptime = proc.uptime_sec ? formatUptime(proc.uptime_sec) : "--";
    const restarts = proc.restart_count ?? "--";
    const lastPing = proc.last_ping ? timeStr(proc.last_ping) : "--";
    const pid = proc.pid || "--";

    content.innerHTML = `
      <div class="card" style="margin-bottom:1.5rem;">
        <div class="card-header">${t("processes.table_status")}</div>
        <div class="card-body">
          <table class="data-table">
            <tbody>
              <tr>
                <th style="width:10rem;">${t("processes.table_health")}</th>
                <td>${health}</td>
              </tr>
              <tr>
                <th>${t("processes.table_anima")}</th>
                <td style="font-weight:600;">${escapeHtml(_animaName)}</td>
              </tr>
              <tr>
                <th>${t("processes.table_pid")}</th>
                <td>${escapeHtml(String(pid))}</td>
              </tr>
              <tr>
                <th>${t("processes.table_status")}</th>
                <td>${statusBadgeHtml(status)}</td>
              </tr>
              <tr>
                <th>${t("processes.table_uptime")}</th>
                <td>${escapeHtml(uptime)}</td>
              </tr>
              <tr>
                <th>${t("processes.table_restarts")}</th>
                <td>${escapeHtml(String(restarts))}</td>
              </tr>
              <tr>
                <th>${t("processes.table_missed_pings")}</th>
                <td>${escapeHtml(String(missedPings))}</td>
              </tr>
              <tr>
                <th>${t("processes.table_last_ping")}</th>
                <td>${escapeHtml(lastPing)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <div class="card-header">${t("processes.table_actions")}</div>
        <div class="card-body">
          <div class="process-actions">
            ${processActionButtonsHtml(_animaName, status)}
          </div>
        </div>
      </div>
    `;

    bindProcessActionButtons(content, { onReload: _load });
  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">${t("processes.fetch_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}
