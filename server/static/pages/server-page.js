// ── Server Communication ────────────────────
import { api } from "../modules/api.js";
import { escapeHtml, timeStr } from "../modules/state.js";
import { t } from "/shared/i18n.js";

let _refreshInterval = null;

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>${t("nav.server")}</h2>
    </div>

    <div class="card-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 1.5rem;">
      <div class="stat-card">
        <div class="stat-label">${t("server.uptime_label")}</div>
        <div class="stat-value" id="serverUptime">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">${t("server.connections_label")}</div>
        <div class="stat-value" id="serverClients">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">${t("server.jobs_label")}</div>
        <div class="stat-value" id="serverJobs">--</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">${t("server.ws_connections")}</div>
      <div class="card-body" id="serverWsContent">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">${t("server.scheduler_status")}</div>
      <div class="card-body" id="serverSchedulerContent">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">${t("server.system_status")}</div>
      <div class="card-body" id="serverStatusContent">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>
    </div>
  `;

  _loadAll();
  _refreshInterval = setInterval(_loadAll, 15000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// ── Data Loading ───────────────────────────

async function _loadAll() {
  _loadStatus();
  _loadConnections();
  _loadScheduler();
}

async function _loadStatus() {
  const statusContent = document.getElementById("serverStatusContent");
  const uptimeEl = document.getElementById("serverUptime");
  if (!statusContent) return;

  try {
    const data = await api("/api/system/status");

    // Server is running if we got a response
    if (uptimeEl) {
      uptimeEl.textContent = t("server.running");
    }

    const rows = [];
    rows.push([t("home.anima_count"), data.animas ?? 0]);
    rows.push([t("server.scheduler"), data.scheduler_running ? t("server.running") : t("server.stopped")]);

    if (data.processes) {
      const processCount = Object.keys(data.processes).length;
      rows.push([t("server.process_count"), processCount]);
    }

    statusContent.innerHTML = `
      <table class="data-table">
        <tbody>
          ${rows.map(([k, v]) => `<tr><td style="font-weight:500;">${escapeHtml(String(k))}</td><td>${escapeHtml(String(v))}</td></tr>`).join("")}
        </tbody>
      </table>
    `;
  } catch (err) {
    if (uptimeEl) uptimeEl.textContent = t("server.stopped");
    statusContent.innerHTML = `<div class="loading-placeholder">${t("server.status_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

async function _loadConnections() {
  const content = document.getElementById("serverWsContent");
  const clientsEl = document.getElementById("serverClients");
  if (!content) return;

  try {
    const data = await api("/api/system/connections");

    // Read WebSocket client count from response structure
    const wsCount = data.websocket?.connected_clients ?? 0;
    if (clientsEl) clientsEl.textContent = wsCount;

    const rows = [];

    // WebSocket connections summary
    if (wsCount > 0) {
      rows.push(`<tr><td>WebSocket</td><td><code>--</code></td><td>${wsCount} ${t("server.connections_unit")}</td></tr>`);
    }

    // Process connections
    const processes = data.processes || {};
    for (const [name, info] of Object.entries(processes)) {
      const status = info.status || "unknown";
      const pid = info.pid || "--";
      rows.push(`<tr><td>${t("server.process")}</td><td><code>${escapeHtml(name)} (PID: ${pid})</code></td><td>${escapeHtml(status)}</td></tr>`);
    }

    if (rows.length > 0) {
      content.innerHTML = `
        <table class="data-table">
          <thead><tr><th>${t("server.conn_type")}</th><th>${t("server.conn_id")}</th><th>${t("server.conn_state")}</th></tr></thead>
          <tbody>${rows.join("")}</tbody>
        </table>
      `;
    } else {
      content.innerHTML = `<div class="loading-placeholder">${t("server.no_connections")}</div>`;
      if (clientsEl) clientsEl.textContent = "0";
    }
  } catch {
    content.innerHTML = `<div class="loading-placeholder">${t("server.api_unimplemented")}</div>`;
    if (clientsEl) clientsEl.textContent = "--";
  }
}

async function _loadScheduler() {
  const content = document.getElementById("serverSchedulerContent");
  const jobsEl = document.getElementById("serverJobs");
  if (!content) return;

  try {
    const data = await api("/api/system/scheduler");

    const jobs = data.jobs || [];
    if (jobsEl) jobsEl.textContent = jobs.length;

    if (jobs.length > 0) {
      content.innerHTML = `
        <table class="data-table">
          <thead><tr><th>${t("server.job_name")}</th><th>${t("server.job_person")}</th><th>${t("server.job_schedule")}</th><th>${t("server.job_last_run")}</th><th>${t("server.job_next_run")}</th></tr></thead>
          <tbody>
            ${jobs.map(j => `
              <tr>
                <td style="font-weight:500;">${escapeHtml(j.name || j.id || "--")}</td>
                <td>${escapeHtml(j.anima || "--")}</td>
                <td><code>${escapeHtml(j.schedule || j.trigger || "--")}</code></td>
                <td>${escapeHtml(j.last_run ? timeStr(j.last_run) : "--")}</td>
                <td>${escapeHtml(j.next_run ? timeStr(j.next_run) : "--")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    } else {
      content.innerHTML = `<div class="loading-placeholder">${t("server.no_jobs")}</div>`;
    }
  } catch {
    content.innerHTML = `<div class="loading-placeholder">${t("server.api_unimplemented")}</div>`;
    if (jobsEl) jobsEl.textContent = "--";
  }
}
