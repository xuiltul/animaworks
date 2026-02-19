// ── Asset Management ────────────────────────
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";

let _container = null;
let _animas = [];
let _selectedAnima = null;
let _metadata = null;
let _previewBackupId = null;
let _previewGenerated = false;
let _rebuildInProgress = false;

// ── Render / Destroy ────────────────────────

export async function render(container) {
  _container = container;
  _selectedAnima = null;
  _metadata = null;
  _previewBackupId = null;
  _previewGenerated = false;
  _rebuildInProgress = false;

  _installWsHandler();

  container.innerHTML = `
    <div class="page-header">
      <h2>アセット管理</h2>
      <p>キャラクターアセットの閲覧とリメイク</p>
    </div>

    <div class="assets-anima-selector" id="assetsAnimaSelector">
      <div class="loading-placeholder">Anima一覧を読み込み中...</div>
    </div>

    <div id="assetsGalleryContent">
      <div class="loading-placeholder">Animaを選択してください</div>
    </div>
  `;

  await _loadAnimaList();
}

export function destroy() {
  _removeWsHandler();
  _container = null;
  _animas = [];
  _selectedAnima = null;
  _metadata = null;
  _previewBackupId = null;
  _previewGenerated = false;
  _rebuildInProgress = false;
}

// ── WebSocket Handler ───────────────────────

function _installWsHandler() {
  window.__assetsWsHandler = _handleWsEvent;
}

function _removeWsHandler() {
  delete window.__assetsWsHandler;
}

function _handleWsEvent(eventType, data) {
  switch (eventType) {
    case "anima.remake_preview_ready":
      _onPreviewReady(data);
      break;
    case "anima.remake_progress":
      _onRemakeProgress(data);
      break;
    case "anima.remake_complete":
      _onRemakeComplete(data);
      break;
  }
}

// ── Anima Selector ─────────────────────────

async function _loadAnimaList() {
  const selector = document.getElementById("assetsAnimaSelector");
  if (!selector) return;

  try {
    _animas = await api("/api/animas");

    if (_animas.length === 0) {
      selector.innerHTML = '<div class="loading-placeholder">Animaが登録されていません</div>';
      return;
    }

    selector.innerHTML = "";
    for (const p of _animas) {
      const btn = document.createElement("button");
      btn.className = "assets-anima-btn";
      btn.dataset.name = p.name;
      btn.textContent = p.name;
      btn.addEventListener("click", () => _selectAnima(p.name));
      selector.appendChild(btn);
    }
  } catch (err) {
    selector.innerHTML = `<div class="loading-placeholder">Anima取得失敗: ${escapeHtml(err.message)}</div>`;
  }
}

function _selectAnima(name) {
  _selectedAnima = name;
  _metadata = null;
  _previewBackupId = null;
  _previewGenerated = false;

  // Update active button state
  const selector = document.getElementById("assetsAnimaSelector");
  if (selector) {
    selector.querySelectorAll(".assets-anima-btn").forEach(btn => {
      btn.classList.toggle("assets-anima-btn--active", btn.dataset.name === name);
    });
  }

  _loadGallery();
}

// ── Asset Gallery ───────────────────────────

async function _loadGallery() {
  const content = document.getElementById("assetsGalleryContent");
  if (!content || !_selectedAnima) return;

  content.innerHTML = '<div class="loading-placeholder">アセットを読み込み中...</div>';

  const enc = encodeURIComponent(_selectedAnima);

  try {
    _metadata = await api(`/api/animas/${enc}/assets/metadata`);
  } catch (err) {
    content.innerHTML = `<div class="loading-placeholder">アセット情報の取得に失敗しました: ${escapeHtml(err.message)}</div>`;
    return;
  }

  const assets = _metadata.assets || {};
  const animations = _metadata.animations || {};
  const modelCount = [assets.model_chibi, assets.model_rigged].filter(Boolean).length;
  const animCount = Object.keys(animations).length;

  let html = "";

  // Thumbnail gallery
  html += '<div class="assets-gallery">';
  html += _renderThumbnail("Fullbody", assets.avatar_fullbody);
  html += _renderThumbnail("Bustup", assets.avatar_bustup);
  html += _renderThumbnail("Chibi", assets.avatar_chibi);

  // 3D model badge
  html += `
    <div class="assets-thumb-card">
      <div class="assets-thumb-placeholder">
        <span class="assets-badge-icon">3D</span>
      </div>
      <div class="assets-thumb-label">3Dモデル</div>
      <span class="assets-badge">${modelCount}</span>
    </div>
  `;

  // Animation badge
  html += `
    <div class="assets-thumb-card">
      <div class="assets-thumb-placeholder">
        <span class="assets-badge-icon">Anim</span>
      </div>
      <div class="assets-thumb-label">アニメーション</div>
      <span class="assets-badge">${animCount}</span>
    </div>
  `;

  html += "</div>";

  // Remake button
  html += `
    <div style="margin-top: 1.5rem;">
      <button class="btn-primary" id="assetsRemakeBtn">Remake Assets</button>
    </div>
  `;

  content.innerHTML = html;

  // Bind remake button
  document.getElementById("assetsRemakeBtn")?.addEventListener("click", () => {
    _openRemakeModal();
  });
}

function _renderThumbnail(label, assetInfo) {
  if (assetInfo && assetInfo.url) {
    return `
      <div class="assets-thumb-card">
        <img class="assets-thumb-img" src="${escapeHtml(assetInfo.url)}" alt="${escapeHtml(label)}" loading="lazy">
        <div class="assets-thumb-label">${escapeHtml(label)}</div>
      </div>
    `;
  }
  return `
    <div class="assets-thumb-card">
      <div class="assets-thumb-placeholder">
        <span class="assets-thumb-empty">No Image</span>
      </div>
      <div class="assets-thumb-label">${escapeHtml(label)}</div>
    </div>
  `;
}

// ── Remake Modal ────────────────────────────

function _openRemakeModal() {
  if (!_selectedAnima || !_metadata) return;

  // Remove existing modal if any
  _closeRemakeModal(false);

  const enc = encodeURIComponent(_selectedAnima);
  const assets = _metadata.assets || {};
  const currentFullbodyUrl = assets.avatar_fullbody?.url || "";

  // Build style-from options (other animas)
  let styleOptions = '<option value="">-- 選択してください --</option>';
  for (const p of _animas) {
    if (p.name !== _selectedAnima) {
      styleOptions += `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)}</option>`;
    }
  }

  const overlay = document.createElement("div");
  overlay.id = "assetsRemakeOverlay";
  overlay.className = "assets-modal-overlay";
  overlay.innerHTML = `
    <div class="assets-modal">
      <div class="assets-modal-header">
        <h3>Remake Assets - ${escapeHtml(_selectedAnima)}</h3>
        <button class="assets-modal-close" id="assetsModalCloseBtn">&times;</button>
      </div>

      <div class="assets-modal-body">
        <div class="assets-modal-columns">
          <!-- Left: Current image -->
          <div class="assets-modal-col">
            <div class="assets-modal-col-label">Current</div>
            ${currentFullbodyUrl
              ? `<img class="assets-modal-preview-img" src="${escapeHtml(currentFullbodyUrl)}" alt="Current fullbody">`
              : '<div class="assets-modal-preview-placeholder">No fullbody image</div>'
            }
          </div>

          <!-- Right: Preview image -->
          <div class="assets-modal-col">
            <div class="assets-modal-col-label">Preview</div>
            <div id="assetsPreviewContainer" class="assets-modal-preview-placeholder">
              プレビュー生成待ち
            </div>
          </div>
        </div>

        <!-- Controls -->
        <div class="assets-modal-controls">
          <div class="assets-modal-control-row">
            <label class="assets-modal-label" for="assetsStyleFrom">Style From:</label>
            <select id="assetsStyleFrom" class="assets-modal-select">
              ${styleOptions}
            </select>
          </div>

          <div class="assets-modal-control-row">
            <label class="assets-modal-label" for="assetsVibeStrength">
              Vibe Strength: <span id="assetsVibeStrengthVal">0.6</span>
            </label>
            <input type="range" id="assetsVibeStrength" min="0" max="1" step="0.05" value="0.6" class="assets-modal-range">
          </div>

          <div class="assets-modal-control-row">
            <label class="assets-modal-label" for="assetsInfoExtract">
              Info Extract: <span id="assetsInfoExtractVal">0.8</span>
            </label>
            <input type="range" id="assetsInfoExtract" min="0" max="1" step="0.05" value="0.8" class="assets-modal-range">
          </div>
        </div>

        <!-- Action buttons -->
        <div class="assets-modal-actions">
          <button class="btn-primary" id="assetsGeneratePreviewBtn">Generate Preview</button>
          <button class="btn-primary" id="assetsAcceptBtn" style="display:none; background:#16a34a;">Accept &amp; Rebuild All</button>
          <button class="btn-secondary" id="assetsRetryBtn" style="display:none;">Retry</button>
          <button class="btn-secondary" id="assetsCancelBtn">Cancel</button>
        </div>

        <!-- Progress section (hidden by default) -->
        <div id="assetsProgressSection" class="assets-progress-section" style="display:none;">
          <div class="assets-progress-header">Rebuild Progress</div>
          <div id="assetsProgressList" class="assets-progress-list"></div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // Trigger active state after DOM insertion for CSS transition
  requestAnimationFrame(() => overlay.classList.add("active"));

  // Bind slider live updates
  const vibeSlider = document.getElementById("assetsVibeStrength");
  const vibeVal = document.getElementById("assetsVibeStrengthVal");
  if (vibeSlider && vibeVal) {
    vibeSlider.addEventListener("input", () => { vibeVal.textContent = vibeSlider.value; });
  }

  const infoSlider = document.getElementById("assetsInfoExtract");
  const infoVal = document.getElementById("assetsInfoExtractVal");
  if (infoSlider && infoVal) {
    infoSlider.addEventListener("input", () => { infoVal.textContent = infoSlider.value; });
  }

  // Bind action buttons
  document.getElementById("assetsGeneratePreviewBtn")?.addEventListener("click", _generatePreview);
  document.getElementById("assetsAcceptBtn")?.addEventListener("click", _acceptAndRebuild);
  document.getElementById("assetsRetryBtn")?.addEventListener("click", _generatePreview);
  document.getElementById("assetsCancelBtn")?.addEventListener("click", () => _cancelRemake());
  document.getElementById("assetsModalCloseBtn")?.addEventListener("click", () => _cancelRemake());

  // Close on overlay background click
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) _cancelRemake();
  });
}

async function _generatePreview() {
  if (!_selectedAnima) return;

  const styleFrom = document.getElementById("assetsStyleFrom")?.value;
  const vibeStrength = parseFloat(document.getElementById("assetsVibeStrength")?.value || "0.6");
  const infoExtracted = parseFloat(document.getElementById("assetsInfoExtract")?.value || "0.8");

  if (!styleFrom) {
    _showModalError("Style FromのAnimaを選択してください");
    return;
  }

  const generateBtn = document.getElementById("assetsGeneratePreviewBtn");
  const retryBtn = document.getElementById("assetsRetryBtn");
  const previewContainer = document.getElementById("assetsPreviewContainer");

  // Disable buttons during generation
  if (generateBtn) { generateBtn.disabled = true; generateBtn.textContent = "Generating..."; }
  if (retryBtn) { retryBtn.disabled = true; }

  // Show loading state in preview
  if (previewContainer) {
    previewContainer.innerHTML = '<div class="assets-spinner"></div><div style="margin-top:0.5rem;">プレビュー生成中...</div>';
    previewContainer.className = "assets-modal-preview-placeholder";
  }

  const enc = encodeURIComponent(_selectedAnima);

  try {
    const result = await api(`/api/animas/${enc}/assets/remake-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        style_from: styleFrom,
        vibe_strength: vibeStrength,
        vibe_info_extracted: infoExtracted,
      }),
    });

    _previewBackupId = result.backup_id || null;

    // If preview URL is returned immediately (sync mode)
    if (result.preview_url) {
      _showPreviewImage(result.preview_url);
    }
    // Otherwise, wait for WebSocket event (anima.remake_preview_ready)

  } catch (err) {
    if (previewContainer) {
      previewContainer.innerHTML = `<div class="assets-error">プレビュー生成失敗: ${escapeHtml(err.message)}</div>`;
      previewContainer.className = "assets-modal-preview-placeholder";
    }
    if (generateBtn) { generateBtn.disabled = false; generateBtn.textContent = "Generate Preview"; }
    if (retryBtn) { retryBtn.disabled = false; }
  }
}

function _showPreviewImage(url) {
  const previewContainer = document.getElementById("assetsPreviewContainer");
  if (previewContainer) {
    const img = document.createElement("img");
    img.className = "assets-modal-preview-img";
    img.alt = "Preview";
    img.src = url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
    img.addEventListener("error", () => {
      previewContainer.innerHTML = '<div class="assets-error">プレビュー画像の読み込みに失敗しました</div>';
      previewContainer.className = "assets-modal-preview-placeholder";
    });
    previewContainer.innerHTML = "";
    previewContainer.className = "";
    previewContainer.appendChild(img);
  }

  _previewGenerated = true;

  // Show accept and retry buttons
  const generateBtn = document.getElementById("assetsGeneratePreviewBtn");
  const acceptBtn = document.getElementById("assetsAcceptBtn");
  const retryBtn = document.getElementById("assetsRetryBtn");
  if (generateBtn) { generateBtn.disabled = false; generateBtn.textContent = "Generate Preview"; generateBtn.style.display = "none"; }
  if (acceptBtn) { acceptBtn.style.display = ""; }
  if (retryBtn) { retryBtn.style.display = ""; retryBtn.disabled = false; }
}

function _showModalError(message) {
  const previewContainer = document.getElementById("assetsPreviewContainer");
  if (previewContainer) {
    previewContainer.innerHTML = `<div class="assets-error">${escapeHtml(message)}</div>`;
    previewContainer.className = "assets-modal-preview-placeholder";
  }
}

async function _acceptAndRebuild() {
  if (!_selectedAnima || !_previewBackupId) return;

  const acceptBtn = document.getElementById("assetsAcceptBtn");
  const retryBtn = document.getElementById("assetsRetryBtn");
  const cancelBtn = document.getElementById("assetsCancelBtn");

  if (acceptBtn) { acceptBtn.disabled = true; acceptBtn.textContent = "Rebuilding..."; }
  if (retryBtn) { retryBtn.disabled = true; }
  if (cancelBtn) { cancelBtn.disabled = true; }

  _rebuildInProgress = true;

  // Show progress section
  const progressSection = document.getElementById("assetsProgressSection");
  if (progressSection) { progressSection.style.display = ""; }

  const enc = encodeURIComponent(_selectedAnima);

  try {
    const result = await api(`/api/animas/${enc}/assets/remake-confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backup_id: _previewBackupId }),
    });

    // If result contains steps, initialize progress bars
    if (result.steps) {
      _initProgressBars(result.steps);
    }

    // Otherwise wait for WebSocket progress updates

  } catch (err) {
    _showModalError(`リビルド失敗: ${err.message}`);
    if (acceptBtn) { acceptBtn.disabled = false; acceptBtn.textContent = "Accept & Rebuild All"; }
    if (retryBtn) { retryBtn.disabled = false; }
    if (cancelBtn) { cancelBtn.disabled = false; }
    _rebuildInProgress = false;
  }
}

async function _cancelRemake() {
  if (_rebuildInProgress) return; // Don't close during active rebuild

  if (_selectedAnima && _previewBackupId) {
    const enc = encodeURIComponent(_selectedAnima);
    try {
      await fetch(`/api/animas/${enc}/assets/remake-preview`, { method: "DELETE" });
    } catch {
      // Ignore cleanup errors
    }
  }

  _closeRemakeModal(true);
}

function _closeRemakeModal(resetState) {
  const overlay = document.getElementById("assetsRemakeOverlay");
  if (overlay) overlay.remove();

  if (resetState) {
    _previewBackupId = null;
    _previewGenerated = false;
    _rebuildInProgress = false;
  }
}

// ── Progress Tracking ───────────────────────

function _initProgressBars(steps) {
  const list = document.getElementById("assetsProgressList");
  if (!list) return;

  list.innerHTML = "";
  for (const step of steps) {
    const stepName = typeof step === "string" ? step : step.name || step;
    const item = document.createElement("div");
    item.className = "assets-progress-item";
    item.dataset.step = stepName;
    item.innerHTML = `
      <div class="assets-progress-item-label">${escapeHtml(stepName)}</div>
      <div class="assets-progress-bar-track">
        <div class="assets-progress-bar-fill" style="width: 0%;"></div>
      </div>
      <span class="assets-progress-pct">0%</span>
    `;
    list.appendChild(item);
  }
}

function _updateProgressStep(stepName, percent, status) {
  const list = document.getElementById("assetsProgressList");
  if (!list) return;

  let item = list.querySelector(`[data-step="${CSS.escape(stepName)}"]`);

  // Create step item if it doesn't exist yet
  if (!item) {
    item = document.createElement("div");
    item.className = "assets-progress-item";
    item.dataset.step = stepName;
    item.innerHTML = `
      <div class="assets-progress-item-label">${escapeHtml(stepName)}</div>
      <div class="assets-progress-bar-track">
        <div class="assets-progress-bar-fill" style="width: 0%;"></div>
      </div>
      <span class="assets-progress-pct">0%</span>
    `;
    list.appendChild(item);
  }

  const fill = item.querySelector(".assets-progress-bar-fill");
  const pct = item.querySelector(".assets-progress-pct");

  const clampedPercent = Math.max(0, Math.min(100, percent));
  if (fill) fill.style.width = `${clampedPercent}%`;
  if (pct) pct.textContent = `${Math.round(clampedPercent)}%`;

  // Update visual state based on status
  if (status === "completed" || clampedPercent >= 100) {
    item.classList.add("assets-progress-item--done");
    if (fill) fill.style.background = "#16a34a";
  } else if (status === "error") {
    item.classList.add("assets-progress-item--error");
    if (fill) fill.style.background = "#ef4444";
  }
}

// ── WebSocket Event Handlers ────────────────

function _onPreviewReady(data) {
  const animaName = data.name || data.anima;
  if (animaName !== _selectedAnima) return;

  const previewUrl = data.preview_url || data.url;
  if (previewUrl) {
    _previewBackupId = data.backup_id || _previewBackupId;
    _showPreviewImage(previewUrl);
  }
}

function _onRemakeProgress(data) {
  const animaName = data.name || data.anima;
  if (animaName !== _selectedAnima) return;

  // Show progress section if hidden
  const progressSection = document.getElementById("assetsProgressSection");
  if (progressSection) progressSection.style.display = "";

  const stepName = data.step || data.stage || "unknown";
  const percent = data.progress_pct ?? data.percent ?? data.progress ?? 0;
  const status = data.status || (percent >= 100 ? "completed" : "in_progress");

  _updateProgressStep(stepName, percent, status);
}

function _onRemakeComplete(data) {
  const animaName = data.name || data.anima;
  if (animaName !== _selectedAnima) return;

  _rebuildInProgress = false;

  const acceptBtn = document.getElementById("assetsAcceptBtn");
  const cancelBtn = document.getElementById("assetsCancelBtn");

  if (data.success !== false) {
    // Success state
    if (acceptBtn) { acceptBtn.textContent = "Complete"; acceptBtn.style.background = "#16a34a"; }
    if (cancelBtn) { cancelBtn.disabled = false; cancelBtn.textContent = "Close"; }

    // Append completion message
    const progressList = document.getElementById("assetsProgressList");
    if (progressList) {
      const msg = document.createElement("div");
      msg.className = "assets-progress-complete";
      msg.textContent = "All assets rebuilt successfully.";
      progressList.appendChild(msg);
    }

    // Refresh gallery after short delay
    setTimeout(() => {
      _loadGallery();
    }, 1500);
  } else {
    // Failure state
    const errorMsg = data.error || "Rebuild failed";
    _showModalError(errorMsg);
    if (acceptBtn) { acceptBtn.disabled = false; acceptBtn.textContent = "Accept & Rebuild All"; }
    if (cancelBtn) { cancelBtn.disabled = false; }
  }
}
