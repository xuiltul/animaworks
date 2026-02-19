// ── Home Dashboard ──────────────────────────
import { api } from "../modules/api.js";
import { escapeHtml, timeStr, statusClass } from "../modules/state.js";
import { getIcon, getDisplaySummary } from "../shared/activity-types.js";

let _refreshInterval = null;

// ── Render ─────────────────────────────────

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>ダッシュボード</h2>
    </div>

    <div class="card-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 1.5rem;">
      <div class="stat-card" id="homeStatAnimas">
        <div class="stat-label">Anima数</div>
        <div class="stat-value" id="homeAnimaCount">--</div>
      </div>
      <div class="stat-card" id="homeStatScheduler">
        <div class="stat-label">スケジューラ</div>
        <div class="stat-value" id="homeSchedulerStatus">--</div>
      </div>
      <div class="stat-card" id="homeStatProcesses">
        <div class="stat-label">稼働プロセス</div>
        <div class="stat-value" id="homeProcessCount">--</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">Anima一覧</div>
      <div class="card-body">
        <div class="card-grid" id="homeAnimaCards" style="grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));">
          <div class="loading-placeholder">読み込み中...</div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">最近のアクティビティ</div>
      <div class="card-body">
        <div id="homeActivityTimeline">
          <div class="loading-placeholder">読み込み中...</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">クイックリンク</div>
      <div class="card-body" style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
        <a href="/workspace/" class="btn-primary" style="text-decoration:none;">Workspace</a>
        <a href="#/chat" class="btn-secondary" style="text-decoration:none;">チャット</a>
        <a href="#/animas" class="btn-secondary" style="text-decoration:none;">Anima管理</a>
        <a href="#/memory" class="btn-secondary" style="text-decoration:none;">記憶ブラウザ</a>
      </div>
    </div>
  `;

  _loadAll();
  _refreshInterval = setInterval(_loadAll, 30000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// ── Data Loading ───────────────────────────

async function _loadAll() {
  _loadSystemStatus();
  _loadAnimaCards();
  _loadActivity();
}

async function _loadSystemStatus() {
  try {
    const data = await api("/api/system/status");
    const animaCountEl = document.getElementById("homeAnimaCount");
    const schedulerEl = document.getElementById("homeSchedulerStatus");
    const processCountEl = document.getElementById("homeProcessCount");
    if (animaCountEl) animaCountEl.textContent = data.animas ?? 0;
    if (schedulerEl) schedulerEl.textContent = data.scheduler_running ? "稼働中" : "停止";
    const processes = data.processes || {};
    const runningCount = Object.values(processes).filter(p => p.status === "running").length;
    if (processCountEl) processCountEl.textContent = runningCount;
  } catch {
    const el = document.getElementById("homeAnimaCount");
    if (el) el.textContent = "取得失敗";
  }
}

async function _loadAnimaCards() {
  const grid = document.getElementById("homeAnimaCards");
  if (!grid) return;

  try {
    const animas = await api("/api/animas");
    if (animas.length === 0) {
      grid.innerHTML = '<div class="loading-placeholder">Animaが登録されていません</div>';
      return;
    }

    grid.innerHTML = "";
    for (const p of animas) {
      const card = document.createElement("div");
      card.className = "card";
      card.style.cssText = "cursor:pointer; transition: transform 0.15s;";
      card.addEventListener("mouseenter", () => { card.style.transform = "translateY(-2px)"; });
      card.addEventListener("mouseleave", () => { card.style.transform = ""; });

      const dotClass = statusClass(p.status);
      const statusLabel = p.status || "offline";

      // Avatar: try HEAD check, fallback to initial
      let avatarHtml = `<div class="anima-avatar-placeholder" style="width:48px;height:48px;font-size:1.2rem;margin:0 auto 0.5rem;">${escapeHtml(p.name.charAt(0).toUpperCase())}</div>`;

      card.innerHTML = `
        <div class="card-body" style="text-align:center; padding: 1rem;">
          <div id="homeAvatar_${escapeHtml(p.name)}">${avatarHtml}</div>
          <div style="font-weight:600; margin-bottom: 0.25rem;">${escapeHtml(p.name)}</div>
          <span class="status-badge ${statusLabel === 'running' || statusLabel === 'idle' ? 'success' : statusLabel === 'error' ? 'error' : ''}"">
            <span class="status-dot ${dotClass}" style="display:inline-block;"></span>
            ${escapeHtml(statusLabel)}
          </span>
        </div>
      `;

      card.addEventListener("click", () => {
        location.hash = "#/animas";
      });

      grid.appendChild(card);

      // Try loading avatar image
      _tryLoadAvatar(p.name);
    }
  } catch (err) {
    grid.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}

async function _tryLoadAvatar(name) {
  const container = document.getElementById(`homeAvatar_${name}`);
  if (!container) return;

  const candidates = ["avatar_bustup.png", "avatar_chibi.png"];
  for (const filename of candidates) {
    const url = `/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
    try {
      const resp = await fetch(url, { method: "HEAD" });
      if (resp.ok) {
        container.innerHTML = `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" style="width:48px;height:48px;border-radius:50%;object-fit:cover;margin:0 auto 0.5rem;display:block;">`;
        return;
      }
    } catch { /* try next */ }
  }
}

async function _loadActivity() {
  const timeline = document.getElementById("homeActivityTimeline");
  if (!timeline) return;

  try {
    const data = await api("/api/activity/recent?hours=12&limit=10");
    const events = data.events || [];
    if (events.length === 0) {
      timeline.innerHTML = '<div class="loading-placeholder">最近のアクティビティはありません</div>';
      return;
    }

    const eventsHtml = events.map(evt => {
      const icon = getIcon(evt.type);
      const ts = timeStr(evt.ts);
      const anima = evt.anima || "";
      const summary = getDisplaySummary(evt);
      return `
        <div style="display:flex; align-items:flex-start; gap:0.5rem; padding:0.4rem 0; border-bottom:1px solid var(--border-color, #eee);">
          <span style="flex-shrink:0;">${icon}</span>
          <span style="color:var(--text-secondary, #666); flex-shrink:0; min-width:3rem;">${escapeHtml(ts)}</span>
          <span style="font-weight:500; flex-shrink:0; max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(anima)}</span>
          <span style="color:var(--text-secondary, #666); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(summary)}</span>
        </div>
      `;
    }).join("");

    timeline.innerHTML = `
      <div id="homeActivityEvents">${eventsHtml}</div>
      <div style="text-align:right;margin-top:0.5rem;"><a href="#/activity" style="color:var(--accent-color,#2563eb);text-decoration:none;font-size:0.85rem;">アクティビティを見る →</a></div>
    `;
  } catch (err) {
    timeline.innerHTML = `<div class="loading-placeholder">アクティビティ取得失敗: ${escapeHtml(err.message)}</div>`;
  }
}
