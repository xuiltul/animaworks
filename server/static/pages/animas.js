// ── Anima Management ───────────────────────
import { api } from "../modules/api.js";
import { escapeHtml, statusClass, renderMarkdown } from "../modules/state.js";

let _viewMode = "list"; // "list" | "detail"
let _selectedName = null;
let _container = null;

export function render(container) {
  _container = container;
  _viewMode = "list";
  _selectedName = null;
  _renderList();
}

export function destroy() {
  _container = null;
  _viewMode = "list";
  _selectedName = null;
}

// ── List View ──────────────────────────────

async function _renderList() {
  if (!_container) return;

  _container.innerHTML = `
    <div class="page-header">
      <h2>Anima管理</h2>
    </div>
    <div id="animasListContent">
      <div class="loading-placeholder">読み込み中...</div>
    </div>
  `;

  const content = document.getElementById("animasListContent");
  if (!content) return;

  try {
    const animas = await api("/api/animas");

    if (animas.length === 0) {
      content.innerHTML = '<div class="loading-placeholder">Animaが登録されていません</div>';
      return;
    }

    content.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>名前</th>
            <th>ステータス</th>
            <th>PID</th>
            <th>稼働時間</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody id="animasTableBody"></tbody>
      </table>
    `;

    const tbody = document.getElementById("animasTableBody");
    for (const p of animas) {
      const dotClass = statusClass(p.status);
      const statusLabel = p.status || "offline";
      const uptime = p.uptime_sec ? _formatUptime(p.uptime_sec) : "--";
      const pid = p.pid || "--";

      // Determine visual state class
      let stateClass = "";
      if (p.status === "bootstrapping" || p.bootstrapping) {
        stateClass = "anima-item anima-item--loading";
      } else if (p.status === "not_found" || p.status === "stopped") {
        stateClass = "anima-item anima-item--sleeping";
      } else {
        stateClass = "anima-item";
      }

      const tr = document.createElement("tr");
      tr.className = stateClass;
      tr.dataset.anima = p.name;
      tr.style.cursor = "pointer";
      tr.innerHTML = `
        <td style="font-weight:600;">${escapeHtml(p.name)}</td>
        <td>
          <span class="status-dot ${dotClass}" style="display:inline-block;"></span>
          ${escapeHtml(statusLabel)}
        </td>
        <td>${escapeHtml(String(pid))}</td>
        <td>${escapeHtml(uptime)}</td>
        <td>
          <button class="btn-secondary anima-detail-btn" data-name="${escapeHtml(p.name)}" style="font-size:0.8rem; padding:0.25rem 0.5rem;">詳細</button>
          <button class="btn-primary anima-trigger-btn" data-name="${escapeHtml(p.name)}" style="font-size:0.8rem; padding:0.25rem 0.5rem;">Heartbeat</button>
        </td>
      `;

      tr.addEventListener("click", (e) => {
        if (e.target.classList.contains("anima-trigger-btn")) return;
        _showDetail(p.name);
      });

      tbody.appendChild(tr);
    }

    // Bind trigger buttons
    content.querySelectorAll(".anima-trigger-btn").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
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

  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}

// ── Detail View ────────────────────────────

async function _showDetail(name) {
  if (!_container) return;
  _viewMode = "detail";
  _selectedName = name;

  _container.innerHTML = `
    <div class="page-header" style="display:flex; align-items:center; gap:1rem;">
      <button class="btn-secondary" id="animasBackBtn" style="font-size:0.85rem;">&larr; 一覧に戻る</button>
      <h2>${escapeHtml(name)}</h2>
    </div>
    <div id="animasDetailContent">
      <div class="loading-placeholder">読み込み中...</div>
    </div>
  `;

  document.getElementById("animasBackBtn").addEventListener("click", () => {
    _viewMode = "list";
    _selectedName = null;
    _renderList();
  });

  const content = document.getElementById("animasDetailContent");
  if (!content) return;

  try {
    const detail = await api(`/api/animas/${encodeURIComponent(name)}`);

    // Try optional endpoints
    let animaConfig = null;
    let memoryStats = null;
    try { animaConfig = await api(`/api/animas/${encodeURIComponent(name)}/config`); } catch { /* 404 ok */ }
    try { memoryStats = await api(`/api/animas/${encodeURIComponent(name)}/memory/stats`); } catch { /* 404 ok */ }

    let html = '<div class="card-grid" style="grid-template-columns: 1fr 1fr; margin-bottom: 1.5rem;">';

    // Identity card
    html += `
      <div class="card">
        <div class="card-header">アイデンティティ</div>
        <div class="card-body" style="max-height:300px; overflow-y:auto;">
          ${detail.identity ? renderMarkdown(detail.identity) : '<span style="color:var(--text-secondary, #666);">未設定</span>'}
        </div>
      </div>
    `;

    // Injection card
    html += `
      <div class="card">
        <div class="card-header">インジェクション（役割指針）</div>
        <div class="card-body" style="max-height:300px; overflow-y:auto;">
          ${detail.injection ? renderMarkdown(detail.injection) : '<span style="color:var(--text-secondary, #666);">未設定</span>'}
        </div>
      </div>
    `;

    html += "</div>";

    // State + Pending
    html += '<div class="card-grid" style="grid-template-columns: 1fr 1fr; margin-bottom: 1.5rem;">';

    html += `
      <div class="card">
        <div class="card-header">現在の状態</div>
        <div class="card-body">
          <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">${escapeHtml(
            detail.state ? (typeof detail.state === "string" ? detail.state : JSON.stringify(detail.state, null, 2)) : "状態情報なし"
          )}</pre>
        </div>
      </div>
    `;

    html += `
      <div class="card">
        <div class="card-header">保留タスク</div>
        <div class="card-body">
          <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">${escapeHtml(
            detail.pending ? (typeof detail.pending === "string" ? detail.pending : JSON.stringify(detail.pending, null, 2)) : "保留タスクなし"
          )}</pre>
        </div>
      </div>
    `;

    html += "</div>";

    // Memory stats
    const epCount = detail.episode_files?.length ?? memoryStats?.episodes ?? 0;
    const knCount = detail.knowledge_files?.length ?? memoryStats?.knowledge ?? 0;
    const prCount = detail.procedure_files?.length ?? memoryStats?.procedures ?? 0;

    html += `
      <div class="card-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 1.5rem;">
        <div class="stat-card">
          <div class="stat-label">エピソード</div>
          <div class="stat-value">${epCount}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">知識</div>
          <div class="stat-value">${knCount}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">手順書</div>
          <div class="stat-value">${prCount}</div>
        </div>
      </div>
    `;

    // Model config
    if (animaConfig) {
      html += `
        <div class="card" style="margin-bottom: 1.5rem;">
          <div class="card-header">モデル設定</div>
          <div class="card-body">
            <pre style="white-space:pre-wrap; margin:0;">${escapeHtml(JSON.stringify(animaConfig, null, 2))}</pre>
          </div>
        </div>
      `;
    }

    // Action buttons
    html += `
      <div style="display:flex; gap:0.75rem;">
        <button class="btn-primary" id="animaDetailTrigger">Heartbeatトリガー</button>
      </div>
    `;

    content.innerHTML = html;

    // Bind trigger button
    document.getElementById("animaDetailTrigger")?.addEventListener("click", async (e) => {
      const btn = e.target;
      btn.disabled = true;
      btn.textContent = "実行中...";
      try {
        await fetch(`/api/animas/${encodeURIComponent(name)}/trigger`, { method: "POST" });
        btn.textContent = "完了";
        setTimeout(() => { btn.textContent = "Heartbeatトリガー"; btn.disabled = false; }, 2000);
      } catch {
        btn.textContent = "失敗";
        setTimeout(() => { btn.textContent = "Heartbeatトリガー"; btn.disabled = false; }, 2000);
      }
    });

  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">詳細の読み込みに失敗しました: ${escapeHtml(err.message)}</div>`;
  }
}

// ── Helpers ────────────────────────────────

function _formatUptime(seconds) {
  if (!seconds || seconds < 0) return "--";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}時間${m}分`;
  return `${m}分`;
}
