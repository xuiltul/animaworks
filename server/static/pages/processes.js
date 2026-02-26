// ── Process Monitoring ──────────────────────
import { api } from "../modules/api.js";
import { escapeHtml, statusClass, timeStr } from "../modules/state.js";

let _refreshInterval = null;

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>プロセス監視</h2>
    </div>
    <div id="processesContent">
      <div class="loading-placeholder">読み込み中...</div>
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
        content.innerHTML = '<div class="loading-placeholder">稼働中のプロセスはありません</div>';
        return;
      }
      _renderFromAnimas(content, animas);
      return;
    }

    _renderFromProcesses(content, entries);
  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">プロセス情報の取得に失敗しました: ${escapeHtml(err.message)}</div>`;
  }
}

function _renderFromProcesses(container, entries) {
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>ヘルス</th>
          <th>Anima名</th>
          <th>PID</th>
          <th>ステータス</th>
          <th>稼働時間</th>
          <th>再起動回数</th>
          <th>ミスping</th>
          <th>最終ping</th>
          <th>操作</th>
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
          <th>ヘルス</th>
          <th>Anima名</th>
          <th>PID</th>
          <th>ステータス</th>
          <th>稼働時間</th>
          <th>操作</th>
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
      <button class="btn-warning process-interrupt-btn" data-name="${eName}" ${btnStyle}>中断</button>
      <button class="btn-warning process-restart-btn" data-name="${eName}" ${btnStyle}>再起動</button>
      <button class="btn-danger process-stop-btn" data-name="${eName}" ${btnStyle}>停止</button>
    `;
  }

  if (status === "stopped" || status === "not_found" || status === "offline") {
    return `
      <button class="btn-success process-start-btn" data-name="${eName}" ${btnStyle}>開始</button>
    `;
  }

  if (status === "starting") {
    return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">起動中...</span>`;
  }

  if (status === "restarting") {
    return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">再起動中...</span>`;
  }

  return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">--</span>`;
}

function _bindAllButtons(container) {
  // Heartbeat
  container.querySelectorAll(".process-trigger-btn").forEach(btn => {
    btn.addEventListener("click", () => _handleAction(btn, "trigger", {
      label: "Heartbeat", busyLabel: "実行中...", doneLabel: "完了",
    }));
  });

  // Stop
  container.querySelectorAll(".process-stop-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.name;
      if (!confirm(`${name} を停止しますか？停止するとチャットやハートビートが停止します。`)) return;
      _handleAction(btn, "stop", {
        label: "停止", busyLabel: "停止中...", doneLabel: "停止完了", reload: true,
      });
    });
  });

  // Start
  container.querySelectorAll(".process-start-btn").forEach(btn => {
    btn.addEventListener("click", () => _handleAction(btn, "start", {
      label: "開始", busyLabel: "起動中...", doneLabel: "起動完了", reload: true,
    }));
  });

  // Restart
  container.querySelectorAll(".process-restart-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.name;
      if (!confirm(`${name} を再起動しますか？実行中の処理は中断されます。`)) return;
      _handleAction(btn, "restart", {
        label: "再起動", busyLabel: "再起動中...", doneLabel: "再起動完了", reload: true,
      });
    });
  });

  // Interrupt
  container.querySelectorAll(".process-interrupt-btn").forEach(btn => {
    btn.addEventListener("click", () => _handleAction(btn, "interrupt", {
      label: "中断", busyLabel: "中断中...", doneLabel: "中断完了", reload: true,
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
    return '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ef4444;" title="異常"></span>';
  }
  if (missedPings > 0) {
    return '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f59e0b;" title="警告"></span>';
  }
  if (status === "running" || status === "idle") {
    return '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#22c55e;" title="正常"></span>';
  }
  return '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#9ca3af;" title="不明"></span>';
}

function _formatUptime(seconds) {
  if (!seconds || seconds < 0) return "--";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}時間${m}分`;
  return `${m}分`;
}
