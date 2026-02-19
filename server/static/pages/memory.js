// ── Memory Browser (Full Page) ──────────────
import { api } from "../modules/api.js";
import { escapeHtml, renderMarkdown } from "../modules/state.js";

let _selectedAnima = null;
let _activeTab = "episodes";
let _viewMode = "list"; // "list" | "content"
let _container = null;
let _animas = [];

export function render(container) {
  _container = container;
  _selectedAnima = null;
  _activeTab = "episodes";
  _viewMode = "list";

  container.innerHTML = `
    <div class="page-header">
      <h2>記憶ブラウザ</h2>
    </div>

    <div style="display:flex; gap:1rem; align-items:center; margin-bottom:1rem;">
      <label style="font-weight:500;">Anima:</label>
      <select id="memoryAnimaSelect" class="anima-dropdown" style="flex:1; max-width:300px;">
        <option value="">Animaを選択...</option>
      </select>
    </div>

    <div class="page-tabs" style="margin-bottom:1rem;">
      <button class="page-tab active" data-tab="episodes">エピソード</button>
      <button class="page-tab" data-tab="knowledge">知識</button>
      <button class="page-tab" data-tab="procedures">手順書</button>
    </div>

    <div class="card-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom:1.5rem;" id="memoryStatsBar">
    </div>

    <div class="card">
      <div class="card-body" id="memoryMainContent">
        <div class="loading-placeholder">Animaを選択してください</div>
      </div>
    </div>
  `;

  _loadAnimaList();
  _bindEvents();
}

export function destroy() {
  _container = null;
  _animas = [];
}

// ── Event Binding ──────────────────────────

function _bindEvents() {
  if (!_container) return;

  // Anima selector
  const select = document.getElementById("memoryAnimaSelect");
  if (select) {
    select.addEventListener("change", (e) => {
      _selectedAnima = e.target.value || null;
      _viewMode = "list";
      _loadStats();
      _loadFileList();
    });
  }

  // Tab buttons
  _container.querySelectorAll(".page-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      _activeTab = btn.dataset.tab;
      _container.querySelectorAll(".page-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === _activeTab));
      _viewMode = "list";
      _loadFileList();
    });
  });
}

// ── Data Loading ───────────────────────────

async function _loadAnimaList() {
  const select = document.getElementById("memoryAnimaSelect");
  if (!select) return;

  try {
    _animas = await api("/api/animas");
    let opts = '<option value="">Animaを選択...</option>';
    for (const p of _animas) {
      opts += `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)}</option>`;
    }
    select.innerHTML = opts;
  } catch {
    select.innerHTML = '<option value="">取得失敗</option>';
  }
}

async function _loadStats() {
  const bar = document.getElementById("memoryStatsBar");
  if (!bar || !_selectedAnima) {
    if (bar) bar.innerHTML = "";
    return;
  }

  let epCount = 0, knCount = 0, prCount = 0;

  // Try memory stats endpoint first
  try {
    const stats = await api(`/api/animas/${encodeURIComponent(_selectedAnima)}/memory/stats`);
    epCount = stats.episodes ?? 0;
    knCount = stats.knowledge ?? 0;
    prCount = stats.procedures ?? 0;
  } catch {
    // Fallback: count from file list endpoints
    try {
      const [ep, kn, pr] = await Promise.all([
        api(`/api/animas/${encodeURIComponent(_selectedAnima)}/episodes`).catch(() => ({ files: [] })),
        api(`/api/animas/${encodeURIComponent(_selectedAnima)}/knowledge`).catch(() => ({ files: [] })),
        api(`/api/animas/${encodeURIComponent(_selectedAnima)}/procedures`).catch(() => ({ files: [] })),
      ]);
      epCount = (ep.files || []).length;
      knCount = (kn.files || []).length;
      prCount = (pr.files || []).length;
    } catch { /* ignore */ }
  }

  bar.innerHTML = `
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
  `;
}

async function _loadFileList() {
  const content = document.getElementById("memoryMainContent");
  if (!content) return;

  if (!_selectedAnima) {
    content.innerHTML = '<div class="loading-placeholder">Animaを選択してください</div>';
    return;
  }

  content.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  const endpoint = `/api/animas/${encodeURIComponent(_selectedAnima)}/${_activeTab}`;

  try {
    const data = await api(endpoint);
    const files = data.files || [];

    if (files.length === 0) {
      content.innerHTML = '<div class="loading-placeholder">ファイルがありません</div>';
      return;
    }

    content.innerHTML = files.map(f =>
      `<div class="memory-file-item" data-file="${escapeHtml(f)}" style="padding:0.5rem 0.75rem; border-bottom:1px solid var(--border-color, #eee); cursor:pointer;">
        ${escapeHtml(f)}
      </div>`
    ).join("");

    content.querySelectorAll(".memory-file-item").forEach(item => {
      item.addEventListener("click", () => {
        _loadFileContent(item.dataset.file);
      });
      item.addEventListener("mouseenter", () => { item.style.background = "var(--hover-bg, #f5f5f5)"; });
      item.addEventListener("mouseleave", () => { item.style.background = ""; });
    });
  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">読み込み失敗: ${escapeHtml(err.message)}</div>`;
  }
}

async function _loadFileContent(file) {
  const content = document.getElementById("memoryMainContent");
  if (!content || !_selectedAnima) return;

  _viewMode = "content";

  content.innerHTML = `
    <div>
      <button class="btn-secondary" id="memoryBackToList" style="margin-bottom:1rem;">&larr; 一覧に戻る</button>
      <h3 style="margin-bottom:0.75rem;">${escapeHtml(file)}</h3>
      <div id="memoryFileBody" class="loading-placeholder">読み込み中...</div>
    </div>
  `;

  document.getElementById("memoryBackToList")?.addEventListener("click", () => {
    _viewMode = "list";
    _loadFileList();
  });

  const endpoint = `/api/animas/${encodeURIComponent(_selectedAnima)}/${_activeTab}/${encodeURIComponent(file)}`;

  try {
    const data = await api(endpoint);
    const body = document.getElementById("memoryFileBody");
    if (body) {
      const raw = data.content || "(内容なし)";
      body.className = "";
      body.innerHTML = `<div style="background:var(--bg-secondary, #f8f9fa); padding:1rem; border-radius:0.5rem;">${renderMarkdown(raw)}</div>`;
    }
  } catch (err) {
    const body = document.getElementById("memoryFileBody");
    if (body) {
      body.className = "loading-placeholder";
      body.textContent = `読み込み失敗: ${err.message}`;
    }
  }
}
