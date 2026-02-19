// ── Server Communication ────────────────────
import { api } from "../modules/api.js";
import { escapeHtml, timeStr } from "../modules/state.js";

let _refreshInterval = null;

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>サーバー通信</h2>
    </div>

    <div class="card-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 1.5rem;">
      <div class="stat-card">
        <div class="stat-label">サーバー稼働</div>
        <div class="stat-value" id="serverUptime">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">接続クライアント</div>
        <div class="stat-value" id="serverClients">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">スケジューラジョブ</div>
        <div class="stat-value" id="serverJobs">--</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">WebSocket接続</div>
      <div class="card-body" id="serverWsContent">
        <div class="loading-placeholder">読み込み中...</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">スケジューラ状況</div>
      <div class="card-body" id="serverSchedulerContent">
        <div class="loading-placeholder">読み込み中...</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">システムステータス</div>
      <div class="card-body" id="serverStatusContent">
        <div class="loading-placeholder">読み込み中...</div>
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
      uptimeEl.textContent = "稼働中";
    }

    const rows = [];
    rows.push(["Anima数", data.animas ?? 0]);
    rows.push(["スケジューラ", data.scheduler_running ? "稼働中" : "停止"]);

    if (data.processes) {
      const processCount = Object.keys(data.processes).length;
      rows.push(["プロセス数", processCount]);
    }

    statusContent.innerHTML = `
      <table class="data-table">
        <tbody>
          ${rows.map(([k, v]) => `<tr><td style="font-weight:500;">${escapeHtml(String(k))}</td><td>${escapeHtml(String(v))}</td></tr>`).join("")}
        </tbody>
      </table>
    `;
  } catch (err) {
    if (uptimeEl) uptimeEl.textContent = "停止";
    statusContent.innerHTML = `<div class="loading-placeholder">ステータス取得失敗: ${escapeHtml(err.message)}</div>`;
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
      rows.push(`<tr><td>WebSocket</td><td><code>--</code></td><td>${wsCount} 接続</td></tr>`);
    }

    // Process connections
    const processes = data.processes || {};
    for (const [name, info] of Object.entries(processes)) {
      const status = info.status || "unknown";
      const pid = info.pid || "--";
      rows.push(`<tr><td>プロセス</td><td><code>${escapeHtml(name)} (PID: ${pid})</code></td><td>${escapeHtml(status)}</td></tr>`);
    }

    if (rows.length > 0) {
      content.innerHTML = `
        <table class="data-table">
          <thead><tr><th>タイプ</th><th>ID</th><th>状態</th></tr></thead>
          <tbody>${rows.join("")}</tbody>
        </table>
      `;
    } else {
      content.innerHTML = '<div class="loading-placeholder">接続中のクライアントはありません</div>';
      if (clientsEl) clientsEl.textContent = "0";
    }
  } catch {
    content.innerHTML = '<div class="loading-placeholder">このAPIは未実装です</div>';
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
          <thead><tr><th>ジョブ名</th><th>パーソン</th><th>スケジュール</th><th>最終実行</th><th>次回実行</th></tr></thead>
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
      content.innerHTML = '<div class="loading-placeholder">スケジュールされたジョブはありません</div>';
    }
  } catch {
    content.innerHTML = '<div class="loading-placeholder">このAPIは未実装です</div>';
    if (jobsEl) jobsEl.textContent = "--";
  }
}
