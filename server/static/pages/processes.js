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
      // Fallback: try anima list
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
          <button class="btn-primary process-trigger-btn" data-name="${escapeHtml(name)}" style="font-size:0.8rem; padding:0.25rem 0.5rem;">Heartbeat</button>
        </td>
      </tr>
    `;
  }

  html += "</tbody></table>";
  container.innerHTML = html;

  // Bind trigger buttons
  container.querySelectorAll(".process-trigger-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const name = btn.dataset.name;
      btn.disabled = true;
      btn.textContent = "実行中...";
      try {
        await fetch(`/api/animas/${encodeURIComponent(name)}/trigger`, { method: "POST" });
        btn.textContent = "完了";
        setTimeout(() => { btn.textContent = "Heartbeat"; btn.disabled = false; }, 2000);
      } catch {
        btn.textContent = "失敗";
        setTimeout(() => { btn.textContent = "Heartbeat"; btn.disabled = false; }, 2000);
      }
    });
  });
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
          <button class="btn-primary process-trigger-btn" data-name="${escapeHtml(p.name)}" style="font-size:0.8rem; padding:0.25rem 0.5rem;">Heartbeat</button>
        </td>
      </tr>
    `;
  }

  html += "</tbody></table>";
  container.innerHTML = html;

  container.querySelectorAll(".process-trigger-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const name = btn.dataset.name;
      btn.disabled = true;
      btn.textContent = "実行中...";
      try {
        await fetch(`/api/animas/${encodeURIComponent(name)}/trigger`, { method: "POST" });
        btn.textContent = "完了";
        setTimeout(() => { btn.textContent = "Heartbeat"; btn.disabled = false; }, 2000);
      } catch {
        btn.textContent = "失敗";
        setTimeout(() => { btn.textContent = "Heartbeat"; btn.disabled = false; }, 2000);
      }
    });
  });
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
