// ── Process Monitoring ──────────────────────
import { api } from "../modules/api.js";
import { escapeHtml, statusClass, timeStr } from "../modules/state.js";
import { t } from "/shared/i18n.js";

let _refreshInterval = null;

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>${t("processes.header")}</h2>
    </div>
    <div id="processesContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  _loadProcesses();
  _refreshInterval = setInterval(_loadProcesses, 10000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// ── Data Loading ───────────────────────────

async function _loadProcesses() {
  const content = document.getElementById("processesContent");
  if (!content) return;

  try {
    const data = await api("/api/system/status");
    const processes = data.processes || {};
    const entries = Object.entries(processes);

    if (entries.length === 0) {
      const animas = await api("/api/animas");
      if (animas.length === 0) {
        content.innerHTML = `<div class="loading-placeholder">${t("processes.no_processes")}</div>`;
        return;
      }
      _renderFromAnimas(content, animas);
      return;
    }

    _renderFromProcesses(content, entries);
  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">${t("processes.fetch_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

function _renderFromProcesses(container, entries) {
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>${t("processes.table_health")}</th>
          <th>${t("processes.table_anima")}</th>
          <th>${t("processes.table_pid")}</th>
          <th>${t("processes.table_status")}</th>
          <th>${t("processes.table_uptime")}</th>
          <th>${t("processes.table_restarts")}</th>
          <th>${t("processes.table_missed_pings")}</th>
          <th>${t("processes.table_last_ping")}</th>
          <th>${t("processes.table_actions")}</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const [name, proc] of entries) {
    const status = proc.status || "unknown";
    const missedPings = proc.missed_pings || 0;
    const health = _getHealthIndicator(status, missedPings);
    const uptime = proc.uptime_sec ? _formatUptime(proc.uptime_sec) : "--";
    const restarts = proc.restart_count ?? 0;
    const lastPing = proc.last_ping ? timeStr(proc.last_ping) : "--";
    const pid = proc.pid || "--";

    html += `
      <tr>
        <td>${health}</td>
        <td style="font-weight:600;">${escapeHtml(name)}</td>
        <td>${escapeHtml(String(pid))}</td>
        <td>
          <span class="status-badge ${status === 'running' ? 'success' : status === 'error' ? 'error' : 'warning'}">
            ${escapeHtml(status)}
          </span>
        </td>
        <td>${escapeHtml(uptime)}</td>
        <td>${restarts}</td>
        <td>${missedPings}</td>
        <td>${escapeHtml(lastPing)}</td>
        <td>
          <div class="process-actions">
            ${_buildActionButtons(name, status)}
          </div>
        </td>
      </tr>
    `;
  }

  html += "</tbody></table>";
  container.innerHTML = html;
  _bindAllButtons(container);
}

function _renderFromAnimas(container, animas) {
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>${t("processes.table_health")}</th>
          <th>${t("processes.table_anima")}</th>
          <th>${t("processes.table_pid")}</th>
          <th>${t("processes.table_status")}</th>
          <th>${t("processes.table_uptime")}</th>
          <th>${t("processes.table_actions")}</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const p of animas) {
    const status = p.status || "offline";
    const health = _getHealthIndicator(status, 0);
    const uptime = p.uptime_sec ? _formatUptime(p.uptime_sec) : "--";
    const pid = p.pid || "--";

    html += `
      <tr>
        <td>${health}</td>
        <td style="font-weight:600;">${escapeHtml(p.name)}</td>
        <td>${escapeHtml(String(pid))}</td>
        <td>
          <span class="status-badge ${status === 'running' || status === 'idle' ? 'success' : status === 'error' ? 'error' : ''}">
            ${escapeHtml(status)}
          </span>
        </td>
        <td>${escapeHtml(uptime)}</td>
        <td>
          <div class="process-actions">
            ${_buildActionButtons(p.name, status)}
          </div>
        </td>
      </tr>
    `;
  }

  html += "</tbody></table>";
  container.innerHTML = html;
  _bindAllButtons(container);
}

// ── Action Buttons ──────────────────────────

function _buildActionButtons(name, status) {
  const eName = escapeHtml(name);
  const btnStyle = 'style="font-size:0.8rem; padding:0.25rem 0.5rem;"';

  if (status === "running" || status === "idle") {
    return `
      <button class="btn-primary process-trigger-btn" data-name="${eName}" ${btnStyle}>Heartbeat</button>
      <button class="btn-warning process-interrupt-btn" data-name="${eName}" ${btnStyle}>${t("processes.interrupt")}</button>
      <button class="btn-warning process-restart-btn" data-name="${eName}" ${btnStyle}>${t("processes.restart")}</button>
      <button class="btn-danger process-stop-btn" data-name="${eName}" ${btnStyle}>${t("processes.stop")}</button>
    `;
  }

  if (status === "stopped" || status === "not_found" || status === "offline") {
    return `
      <button class="btn-success process-start-btn" data-name="${eName}" ${btnStyle}>${t("processes.start")}</button>
    `;
  }

  if (status === "starting") {
    return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">${t("processes.starting")}</span>`;
  }

  if (status === "restarting") {
    return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">${t("processes.restarting")}</span>`;
  }

  return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">--</span>`;
}

function _bindAllButtons(container) {
  // Heartbeat
  container.querySelectorAll(".process-trigger-btn").forEach(btn => {
    btn.addEventListener("click", () => _handleAction(btn, "trigger", {
      label: "Heartbeat", busyLabel: t("processes.running"), doneLabel: t("processes.done"),
    }));
  });

  // Stop
  container.querySelectorAll(".process-stop-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.name;
      if (!confirm(t("processes.confirm_stop", { name }))) return;
      _handleAction(btn, "stop", {
        label: t("processes.stop"), busyLabel: t("processes.stopping"), doneLabel: t("processes.stop_done"), reload: true,
      });
    });
  });

  // Start
  container.querySelectorAll(".process-start-btn").forEach(btn => {
    btn.addEventListener("click", () => _handleAction(btn, "start", {
      label: t("processes.start"), busyLabel: t("processes.starting"), doneLabel: t("processes.start_done"), reload: true,
    }));
  });

  // Restart
  container.querySelectorAll(".process-restart-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.name;
      if (!confirm(t("processes.confirm_restart", { name }))) return;
      _handleAction(btn, "restart", {
        label: t("processes.restart"), busyLabel: t("processes.restarting"), doneLabel: t("processes.restart_done"), reload: true,
      });
    });
  });

  // Interrupt
  container.querySelectorAll(".process-interrupt-btn").forEach(btn => {
    btn.addEventListener("click", () => _handleAction(btn, "interrupt", {
      label: t("processes.interrupt"), busyLabel: t("processes.interrupting"), doneLabel: t("processes.interrupt_done"), reload: true,
    }));
  });
}

async function _handleAction(btn, action, opts) {
  const name = btn.dataset.name;
  btn.disabled = true;
  btn.textContent = opts.busyLabel;

  try {
    await fetch(`/api/animas/${encodeURIComponent(name)}/${action}`, { method: "POST" });
    btn.textContent = opts.doneLabel;

    if (opts.reload) {
      setTimeout(_loadProcesses, 1000);
    }
    setTimeout(() => {
      btn.textContent = opts.label;
      btn.disabled = false;
    }, 2000);
  } catch {
    btn.textContent = "失敗";
    setTimeout(() => {
      btn.textContent = opts.label;
      btn.disabled = false;
    }, 2000);
  }
}

// ── Helpers ────────────────────────────────

function _getHealthIndicator(status, missedPings) {
  if (status === "error" || status === "down") {
    return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ef4444;" title="${t("processes.health_error")}"></span>`;
  }
  if (missedPings > 0) {
    return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f59e0b;" title="${t("processes.health_warning")}"></span>`;
  }
  if (status === "running" || status === "idle") {
    return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#22c55e;" title="${t("processes.health_ok")}"></span>`;
  }
  return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#9ca3af;" title="${t("common.unknown")}"></span>`;
}

function _formatUptime(seconds) {
  if (!seconds || seconds < 0) return "--";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return t("animas.uptime_hm", { h, m });
  return t("animas.uptime_m", { m });
}
